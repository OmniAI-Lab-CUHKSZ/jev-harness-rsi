"""Execute precompiled workflows; no online question/candidate generation path."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time

from .expressions import build_candidates, resolve
from .transport import WorkerUnavailable


# One of AutoTrust's 16 options is reserved for NONE.
MAX_REAL_OPTIONS = 15
STEP_REVIEW = "R_review"
REVIEW_MAX_TOKENS = 128
REVIEW_THINKING_BUDGET = 96


def is_open_generation(operation, request):
    return operation == "generate" and request.get("purpose") != STEP_REVIEW


class Uncovered(RuntimeError):
    """Continue the actual task through generation, without synthesizing decisions."""


class NativeNone(Uncovered):
    def __init__(self, response):
        super().__init__("Jev selected the uncovered option")
        self.response = response


def parse_json(text):
    text = text.split("</think>")[-1].strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            return value
        except json.JSONDecodeError:
            pass
    raise ValueError("Generation did not return JSON")


class Trace:
    def __init__(self, path=None):
        self.events = []
        self.stream = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.stream = Path(path).open("w", encoding="utf-8")

    def add(self, event_type, **values):
        event = {"event": event_type, **values}
        self.events.append(event)
        if self.stream:
            self.stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.stream.flush()
        return event

    def close(self):
        if self.stream:
            self.stream.close()


class Executor:
    def __init__(self, model, environment, max_calls=100, output_budget=65536, generation_tokens=1024,
                 generation_thinking=False, history_chars=None, step_review_threshold=None,
                 step_review_mode="generated_index", step_review_thinking_budget=None):
        self.model = model
        self.environment = environment
        self.max_calls = max_calls
        self.output_budget = output_budget
        self.generation_tokens = generation_tokens
        self.generation_thinking = generation_thinking
        self.history_chars = history_chars
        self.step_review_threshold = step_review_threshold
        self.step_review_mode = step_review_mode
        self.step_review_thinking_budget = step_review_thinking_budget

    def program_context(self):
        # Generation output is not input to question/candidate construction.
        return {"task": self.context["task"], "vars": self.context["vars"]}

    def _model(self, operation, **values):
        calls = [e for e in self.trace.events if e["event"] == "model"]
        spent = sum(e["response"].get("output_tokens", 0) for e in calls)
        if len(calls) >= self.max_calls or spent >= self.output_budget:
            raise RuntimeError("Execution model budget exhausted")
        if operation == "generate":
            values["max_tokens"] = min(values.get("max_tokens", self.generation_tokens), self.generation_tokens,
                                       self.output_budget-spent)
        elif operation == "decide" and values.get("thinking") == "auto":
            values["think_budget"] = min(values["think_budget"], self.output_budget-spent)
        try:
            result = self.model.request(operation, **values)
        except WorkerUnavailable:
            raise
        except Exception as exc:
            self.trace.add("model", operation=operation, request=values,
                           response={"mode": values.get("purpose", "D"), "output_tokens": 0, "error": str(exc)})
            raise
        if operation == "generate":
            result["mode"] = values.get("purpose", result["mode"])
        elif result.get("thinking_used"):
            values["purpose"] = STEP_REVIEW
        self.trace.add("model", operation=operation, request=values, response=result)
        return result

    def call(self, app, api, arguments, node=None):
        response = self.environment.request("call", app=app, api=api, arguments=arguments)
        self.trace.add("api", node=node, app=app, api=api, arguments=arguments, response=response)
        self.context["history"].append({"app": app, "api": api, "arguments": arguments, "response": response})
        self.done = bool(response.get("done"))
        return response.get("value", response.get("output"))

    def decision(self, question, evidence, candidates, skill="", policy=None, node=None,
                 review_step=False, widen_on_none=False, native_response=None):
        policy = policy or {}
        if not candidates:
            raise Uncovered("Candidate builder returned no objects")
        if len(candidates) > MAX_REAL_OPTIONS:
            raise Uncovered("Candidate set needs an offline decomposition; no silent truncation")
        choices = candidates + [{"label": "NONE: no listed option is supported by the evidence", "value": None}]
        labels = [str(item["label"]) for item in choices]
        if policy.get("mode", "D") == "R":
            prompt = (skill + "\n" + question + "\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False)
                      + "\nOptions:\n" + json.dumps(dict(enumerate(labels)), ensure_ascii=False)
                      + '\nReturn JSON {"index": <zero-based option index>}.')
            response = self._model("generate", messages=[{"role": "user", "content": prompt}],
                                   thinking=True, max_tokens=policy.get("max_tokens", 2048))
            index = int(parse_json(response["text"])["index"])
        else:
            request = dict(question=question, state=json.dumps(evidence, ensure_ascii=False, default=str),
                           options=labels, skill=skill)
            step = review_step and node == "fragment/step" and self.step_review_threshold is not None
            official = step and self.step_review_mode == "official_adaptive"
            response = native_response
            final_call = getattr(self, "requests", 0) + 1 >= self.max_calls
            if response is None and (not official or widen_on_none or final_call):
                response = self._model("decide", **request)
            # Focused menus can first abstain and widen. For a complete menu the official server
            # handles both the high-confidence exit and the low-confidence review in one request.
            if official and (response is None or (
                    max(response["probabilities"]) < self.step_review_threshold
                    and not (widen_on_none and response["index"] == len(candidates)))):
                response = self._model("decide", **request, thinking="auto", threshold=self.step_review_threshold,
                                       think_budget=self.step_review_thinking_budget,
                                       native_prechecked=response is not None)
            index = response["index"]
            confidence = max(response["probabilities"])
            if official and response.get("thinking_used"):
                prior = response["native_probabilities"]
                initial = prior.index(max(prior))
                self.trace.add("step_review", node=node, policy="official_adaptive", native_index=initial,
                               native_confidence=max(prior), threshold=self.step_review_threshold, final_index=index,
                               outcome="kept" if initial == index else "changed", output_tokens=response["output_tokens"],
                               reasoning_tokens=response["reasoning_tokens"], native_probabilities=prior,
                               reasoned_probabilities=response["reasoned_probabilities"], probabilities=response["probabilities"],
                               finished_within_budget=response["finished_within_budget"])
            elif (step and not official and confidence < self.step_review_threshold
                  and not (widen_on_none and index == len(candidates))):
                index = self.review_step_choice(question, evidence, labels, skill, index, confidence, node)
            else:
                step = False
            if step:
                self.trace.add("decision", node=node, question=question, evidence=evidence,
                               candidates=choices, selected=index, skill=skill, mode=STEP_REVIEW)
                if index == len(candidates):
                    raise Uncovered("Reasoned fragment review handed off to open generation")
                return choices[index]["value"]
            if confidence < policy.get("minimum_probability", 0.0):
                raise Uncovered("Decision policy requested generation for this uncertain state")
        if not 0 <= index < len(choices):
            raise Uncovered("Reasoned decision returned an invalid index")
        self.trace.add("decision", node=node, question=question, evidence=evidence,
                       candidates=choices, selected=index, skill=skill)
        if choices[index]["value"] is None:
            if policy.get("mode", "D") == "D":
                raise NativeNone(response)
            raise Uncovered("Jev selected the uncovered option")
        return choices[index]["value"]

    def review_step_choice(self, question, evidence, labels, skill, initial, confidence, node):
        messages = [{"role": "system", "content": (
            "Select the next executable step. Consider the current subgoal, what is already known, and "
            "each option's prerequisites and effect. A partial step can be useful. Choose NONE if none "
            "of the listed steps helps. Give only the zero-based option number or NONE as your final answer.")},
            {"role": "user", "content": json.dumps({"question": question, "skill": skill, "state": evidence,
              "options": dict(enumerate(labels))}, ensure_ascii=False, default=str)}]
        none = len(labels) - 1
        response = self._model("generate", messages=messages, purpose=STEP_REVIEW, thinking=True,
                               thinking_budget=REVIEW_THINKING_BUDGET, max_tokens=REVIEW_MAX_TOKENS)
        answer = response["text"].rsplit("</think>", 1)[-1].strip()
        index = none
        if answer.upper() == "NONE":
            valid = True
        else:
            valid = bool(re.fullmatch(r"[0-9]{1,2}", answer)) and int(answer) <= none
            if valid:
                index = int(answer)
        valid = valid and not response.get("truncated", False)
        if not valid:
            index = none
        self.trace.add("step_review", node=node, native_index=initial, native_confidence=confidence,
                       threshold=self.step_review_threshold, final_index=index,
                       outcome="invalid" if not valid else "kept" if initial == index else "changed",
                       output_tokens=response.get("output_tokens", 0), reasoning_tokens=response.get("reasoning_tokens", 0))
        return index

    def steps(self, steps, harness):
        for node in steps:
            if self.done:
                break
            identifier = node.get("id", node["kind"])
            kind = node["kind"]
            self.trace.add("node", node=identifier, kind=kind)
            if kind == "call":
                binding = resolve(harness["bindings"][node["binding"]], self.context)
                if node.get("paginate"):
                    result = []
                    for page in range(node.get("max_pages", 100)):
                        arguments = {**binding.get("arguments", {}), "page_index": page, "page_limit": 20}
                        chunk = self.call(binding["app"], binding["api"], arguments, identifier)
                        if not isinstance(chunk, list):
                            raise Uncovered("Pagination returned an unexpected response")
                        result.extend(chunk)
                        if len(chunk) < 20:
                            break
                    else:
                        raise Uncovered("Pagination budget exhausted")
                else:
                    result = self.call(binding["app"], binding["api"], binding.get("arguments", {}), identifier)
                self.context["vars"][node.get("store", identifier)] = result
            elif kind == "compute":
                self.context["vars"][node.get("store", identifier)] = resolve(
                    harness["bindings"][node["binding"]], self.program_context())
            elif kind == "decision":
                key = node["decision"]
                value = self.decision(resolve(harness["questions"][key], self.program_context()),
                    resolve(harness["evidence"][key], self.program_context()),
                    build_candidates(harness["candidates"][key], self.program_context()),
                    harness.get("decision_skills", {}).get(key, ""),
                    harness.get("mode_policy", {}).get(key), identifier)
                self.context["vars"][node.get("store", key)] = value
            elif kind == "foreach":
                items = resolve(harness["bindings"][node["binding"]], self.program_context())
                for item in items:
                    self.context["vars"][node.get("item", "item")] = item
                    self.steps(node["body"], harness)
                    if self.done:
                        break
            elif kind == "if":
                condition = resolve(harness["bindings"][node["binding"]], self.program_context())
                self.steps(node.get("then", []) if condition else node.get("else", []), harness)
            elif kind == "generate":
                skill = harness["generation_skills"][node["skill"]]
                prompt = resolve(skill, self.context)
                response = self._model("generate", messages=[{"role": "user", "content": prompt}],
                    max_tokens=harness.get("mode_policy", {}).get(node["skill"], {}).get("max_tokens", self.generation_tokens))
                self.context["generated"][node.get("store", identifier)] = (
                    parse_json(response["text"]) if node.get("json", True) else response["text"])
            elif kind == "agent":
                self.agent(harness.get("generation_skills", {}).get(node.get("skill", "fallback"), ""))
            elif kind == "finish":
                arguments = resolve(harness.get("bindings", {}).get(node.get("binding", "complete"), {}), self.context)
                self.call("supervisor", "complete_task", arguments, identifier)
            else:
                raise ValueError("Unknown compiled operation: " + kind)

    def select_workflow(self, library):
        workflows = library.get("workflows", [])
        if not workflows:
            return None
        instruction = self.context["task"]["instruction"]
        definitions = []
        for entry in workflows:
            workflow = entry["workflow"]
            if "applicable_if" in workflow and not resolve(workflow["applicable_if"], self.program_context()):
                continue
            description = workflow.get("description", entry.get("description", ""))
            definitions.append({"label": description, "value": entry["id"]})
        if not definitions:
            raise Uncovered("No workflow satisfies its applicability conditions")
        if len(definitions) > MAX_REAL_OPTIONS:
            # This shortlist uses the instruction and compiled descriptions only.
            stop_words = {"the", "and", "for", "with", "from", "this", "that", "all", "any"}
            def terms(text):
                return set(re.findall(r"[a-z0-9]{3,}", text.lower())) - stop_words
            query = terms(instruction)
            ranked = [(len(query & terms(option["label"])), option) for option in definitions]
            shortlisted = [option for score, option in sorted(ranked, key=lambda pair: pair[0], reverse=True)
                           if score > 0][:MAX_REAL_OPTIONS]
            self.trace.add("workflow_shortlist", eligible=len(definitions),
                           retained=[option["value"] for option in shortlisted])
            if not shortlisted:
                raise Uncovered("No compiled workflow matched the instruction during retrieval")
            definitions = shortlisted
        selected = self.decision(
            "Which reusable workflow can carry out this request? Choose NONE if its operations or requirements do not fit.",
            {"instruction": instruction}, definitions,
            library.get("router_skill", "Match the requested operation, not just the app or shared keywords."), node="workflow_router")
        return next(h for h in workflows if h["id"] == selected)

    def preferred_tool_docs(self):
        return set()

    def _function_docs(self):
        task = self.context["task"]
        text = task["instruction"] + " " + " ".join(
            e.get("app", "") + " " + e.get("api", "") for e in self.context["history"][-4:])
        terms = {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower())} - {"the", "my", "and", "all", "with", "from", "thi", "that"}
        apps = {row.get("function", row)["name"].split("__", 1)[0] for row in task["functions"]}
        focus = {app for app in apps if all(word in terms for word in app.split("_"))}
        functions = []
        forced = {"supervisor__show_account_passwords", "supervisor__complete_task", "api_docs__show_api_doc",
                  "api_docs__show_api_descriptions", "api_docs__search_api_docs"}
        preferred = self.preferred_tool_docs()
        for row in task["functions"]:
            f = row.get("function", row)
            app = f["name"].split("__", 1)[0]
            name_words = {w.rstrip("s") for w in re.findall(r"[a-z]+", f["name"].replace("_", " ").lower())}
            description_words = {w.rstrip("s") for w in re.findall(r"[a-z]+", f.get("description", "").lower())}
            score = 6*len(terms & name_words) + len(terms & description_words)
            if focus:
                score += 30 if app in focus else -100
            if f["name"] in forced:
                score += 1000
            if "login" in f["name"] and any(app in terms for app in f["name"].split("__")[:1]):
                score += 50
            functions.append((score, f))
        result = []
        def rank(item):
            score, f = item
            return (f["name"] in forced, f["name"] in preferred, score) if preferred else (score,)
        for _, function in sorted(functions, key=rank, reverse=True)[:18]:
            app, api = function["name"].split("__", 1)
            result.append({"app": app, "api": api, "description": function.get("description", ""),
                           "parameters": function["parameters"]})
        return result

    def generation_history(self):
        history = self.context["history"]
        if self.history_chars is None:
            return history
        def chars(value):
            return len(json.dumps(value, ensure_ascii=False, default=str))
        if chars(history) <= self.history_chars:
            return history
        history = [dict(row) for row in history]
        # Preserve action arguments and recent observations; drop older observations first.
        for row in history[:-2]:
            if "response" in row:
                row["response"] = "[older observation omitted]"
            if chars(history) <= self.history_chars:
                return history
        while len(history) > 2 and chars(history) > self.history_chars:
            del history[0]
        if chars(history) > self.history_chars:
            for row in history:
                for key in ("response", "arguments", "generated"):
                    if key in row and chars(row[key]) > self.history_chars // 5:
                        row[key] = json.dumps(row[key], ensure_ascii=False, default=str)[:self.history_chars // 5] + " [truncated]"
        return history

    def agent(self, skill=""):
        """Residual generation performs task actions directly, never makes Q/C for Jev."""
        task = self.context["task"]
        for step in range(self.max_calls):
            if self.done:
                return
            system = (
                "You are an AppWorld tool-calling agent. Complete the user's task by calling APIs. "
                "Return ONLY one JSON object: {\"app\":\"app_name\",\"api\":\"api_name\",\"arguments\":{...}}. "
                "Use actual identifiers and values from observations. The schemas list app and api separately. "
                "To get credentials call supervisor.show_account_passwords; apps may then require login. "
                "Explore api_docs for missing APIs. End by calling supervisor.complete_task; supply an answer only for questions. "
                "The records below show actions already executed. Continue from their results. " + str(skill))
            visible = {"instruction": task["instruction"], "supervisor": task["supervisor"],
                       "datetime": task["datetime"], "compiled_bindings": self.context["vars"],
                       "generated_parameters": self.context["generated"],
                       "api_schemas": self._function_docs(), "executed_actions": self.generation_history()}
            response = self._model("generate", messages=[{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(visible, ensure_ascii=False, default=str)}],
                thinking=self.generation_thinking, purpose="G", max_tokens=self.generation_tokens, seed=self.seed)
            try:
                action = parse_json(response["text"])
                if not isinstance(action, dict) or not all(k in action for k in ("app", "api")):
                    raise ValueError("Expected app, api and arguments")
                app, api = action["app"], action["api"]
                if "__" in api:
                    app, api = api.split("__", 1)
                known = [r.get("function", r)["name"] for r in task["functions"]]
                if app+"__"+api not in known:
                    matches = [name for name in known if name.endswith("__"+api)]
                    if len(matches) == 1:
                        app, api = matches[0].split("__", 1)
                self.call(app, api, action.get("arguments", {}), "residual_agent")
            except (ValueError, KeyError, TypeError) as exc:
                self.trace.add("generation_parse_error", error=str(exc))
                self.context["history"].append({"format_error": str(exc), "generated": response["text"]})
        raise RuntimeError("Residual agent step budget exhausted")

    def execute_task(self, library):
        try:
            workflow = self.select_workflow(library)
            if workflow:
                self.workflow_id = workflow["id"]
                self.trace.add("workflow", workflow_id=self.workflow_id)
                self.steps(workflow["workflow"]["steps"], workflow)
            if not self.done:
                self.agent()
        except (Uncovered, KeyError, TypeError, ValueError) as exc:
            self.trace.add("fallback", reason=str(exc))
            self.agent()

    def run(self, task_id, library=None, trace_path=None, seed=0, label="harness"):
        self.trace = Trace(trace_path)
        self.seed = seed
        self.done = False
        started = time.perf_counter()
        try:
            task = self.environment.request("prepare", task_id=task_id, seed=seed,
                                             experiment_name=f"jev_compiled_{label}_{task_id}",
                                             **self.prepare_options())
            self.context = {"task": task, "vars": {}, "generated": {}, "history": []}
            self.trace.add("task", task_id=task_id, instruction=task["instruction"],
                           supervisor=task["supervisor"], datetime=task["datetime"], seed=seed)
            self.workflow_id, error = None, None
            try:
                self.execute_task(library or {})
            except WorkerUnavailable:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.trace.add("execution_error", error=error)
            evaluation = self.environment.request("evaluate")
            summary = self.summarize(task_id, label, seed, started, error, evaluation)
            self.trace.add("summary", **summary)
            return summary
        finally:
            self.trace.close()

    def prepare_options(self):
        return {}

    def summarize(self, task_id, label, seed, started, error, evaluation):
        model_events = [e for e in self.trace.events if e["event"] == "model"]
        modes = Counter(e["response"]["mode"] for e in model_events)
        for event in model_events:
            if event["response"].get("thinking_used") and not event["request"].get("native_prechecked"):
                modes["D"] += 1
        summary = {"task_id": task_id, "label": label, "seed": seed, "workflow": self.workflow_id,
            "success": evaluation["success"], "error": error, "failures": evaluation["failures"],
            "input_tokens": sum(e["response"].get("input_tokens", 0) for e in model_events),
            "output_tokens": sum(e["response"].get("output_tokens", 0) for e in model_events),
            "reasoning_tokens": sum(e["response"].get("reasoning_tokens", 0) for e in model_events),
            "model_calls": dict(modes), "api_calls": sum(e["event"] == "api" for e in self.trace.events),
            "seconds": time.perf_counter()-started,
            "question_generation_calls": sum(e["request"].get("purpose") == "question_generation" for e in model_events),
            "candidate_generation_calls": sum(e["request"].get("purpose") == "candidate_generation" for e in model_events)}
        summary["execution_settings"] = {"max_model_calls": self.max_calls, "output_budget": self.output_budget,
                "generation_tokens": self.generation_tokens, "generation_thinking": self.generation_thinking,
                **({"history_chars": self.history_chars} if self.history_chars is not None else {})}
        return summary
