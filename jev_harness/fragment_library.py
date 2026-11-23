"""Editable fragment data and the task-time composer used in the formal run."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from .evolution import COMPONENTS
from .expressions import build_candidates, resolve
from .formal_runtime import ContextTooLong, FormalExecutor
from .runtime import Uncovered, parse_json


def new_library():
    ref = lambda path: {"ref": path}
    return {key: {} for key in COMPONENTS} | {"workflow": {
        "rules": {},
        "fragments": {
            "account_passwords": {"inputs": [], "output": "account_passwords", "app": "supervisor",
                                  "source_tasks": [], "steps": [{"kind": "call", "binding": "account_passwords", "store": "rows"}],
                                  "result": ref("vars.rows")},
        },
    }, "bindings": {"account_passwords": {"app": "supervisor", "api": "show_account_passwords", "arguments": {}}}}


def apply_fragment_patch(parent, patch, open_tasks):
    changes = patch.get("changes")
    if not isinstance(changes, dict) or not (1 <= len(changes) <= 2) or changes.keys()-set(COMPONENTS):
        raise ValueError("An edit names one or two components")
    result = deepcopy(parent)
    touched = []
    for key, entries in changes.items():
        if not isinstance(entries, dict):
            raise ValueError(f"{key} edits must be a map of changed entries")
        if key == "workflow":
            if entries.keys()-{"fragments", "rules"}:
                raise ValueError("Workflow edits name fragments or rules")
            for collection, edits in entries.items():
                for name, edit in edits.items():
                    if not isinstance(edit, dict) or set(edit.get("source_tasks", ())) - set(open_tasks):
                        raise ValueError(f"Unrecognized source for {name}")
                    result[key][collection][name] = edit
        else:
            result[key].update(entries)
        if result[key] != parent[key]:
            touched.append(key)
    if not touched:
        raise ValueError("The candidate does not change the library")
    if patch.get("operation") == "register":
        old = parent["workflow"]["fragments"]
        new = result["workflow"]["fragments"]
        old_rules = parent["workflow"]["rules"]
        if (len(new.keys()-old.keys()) != 1 or any(new[k] != v for k, v in old.items())
                or any(result["workflow"]["rules"][k] != v for k, v in old_rules.items())):
            raise ValueError("Register exactly one new independent fragment")
    return result, touched


def check_library(library, functions=(), *, allow_alternatives=False):
    known = {f.get("function", f)["name"] for f in functions}
    wf = library["workflow"]
    choices = ("questions", "candidates")

    def check_steps(steps):
        for step in steps:
            kind = step["kind"]
            if kind == "decision":
                for component in choices:
                    if step["decision"] not in library[component]:
                        raise ValueError(f"Missing {component} entry for {step['decision']}")
                continue
            binding = library["bindings"][step["binding"]]
            if kind == "call":
                api = f"{binding['app']}__{binding['api']}"
                if api == "supervisor__complete_task" or (known and api not in known):
                    raise ValueError(f"Unknown or reserved API: {api}")
            elif kind == "foreach":
                check_steps(step["body"])
            elif kind == "if":
                check_steps(step.get("then", ()))
                check_steps(step.get("else", ()))
            elif kind != "compute":
                raise ValueError(f"Unknown fragment node {kind}")

    for name, route in wf["rules"].items():
        if not route["patterns"] or not route["goals"]:
            raise ValueError(f"Empty route {name}")
        for pattern in (*route["patterns"], *route.get("exclude", ())):
            re.compile(pattern)
    outputs = set()
    for name, frag in wf["fragments"].items():
        if not isinstance(frag["inputs"], list) or any(not isinstance(x, str) for x in frag["inputs"]):
            raise ValueError(f"Invalid input list for {name}")
        output = frag["output"]
        if output in outputs and not allow_alternatives:
            raise ValueError(f"More than one producer for {output}")
        outputs.add(output)
        check_steps(frag["steps"])
        if "result" not in frag:
            raise ValueError(f"Missing result for {name}")
    return library


def matching_routes(rules, instruction):
    text = instruction.casefold()
    for name, route in rules.items():
        if all(re.search(pattern, text) for pattern in route["patterns"]) and not any(
                re.search(pattern, text) for pattern in route.get("exclude", [])):
            yield name, route


class FragmentExecutor(FormalExecutor):
    def program_context(self):
        result = super().program_context()
        # task_id and evaluator metadata are unavailable to harness predicates.
        return {**result, "inputs": getattr(self, "fragment_inputs", {}),
                "task": {key: result["task"][key] for key in ("instruction", "supervisor", "datetime")}}

    def execute_task(self, library):
        self.fragment_inputs = {}
        ins = self.context["task"]["instruction"]
        workflow = library.get("workflow", {})
        fragments = workflow.get("fragments", {})
        rules = workflow.get("rules", {})
        initial = {}
        goals = []
        matched, applicable = [], []
        for name, route in matching_routes(rules, ins):
            matched.append(name)
            try:
                values = resolve(route.get("inputs", {}), self.program_context())
            except (KeyError, ValueError, TypeError):
                values = {}
            conflict = [k for k, v in values.items() if k in initial and initial[k] != v]
            if conflict:
                self.trace.add("composition_missing", goal=name, reason="Conflicting task parameters")
                continue
            initial.update(values)
            goals.extend(route["goals"])
            applicable.append(name)
        goals = list(dict.fromkeys(goals))
        self.trace.add("composition_goals", goals=goals, matched_rules=matched)
        providers = {spec["output"]: (name, spec) for name, spec in fragments.items()}
        live = initial.copy()
        broken = {}
        recovered = set()
        self.generated_inputs = recovered
        self.write_progress = []
        completed = []
        skipped = []

        def need(role, chain=()):
            if role in live:
                return live[role]
            if role in broken:
                raise Uncovered(broken[role])
            if role in chain or role not in providers:
                raise Uncovered(f"No usable producer for {role}")
            name, spec = providers[role]
            args = [need(key, (*chain, role)) for key in spec["inputs"]]
            self.trace.add("fragment_start", fragment=name, input_roles=spec["inputs"], goal=chain[0] if chain else role)
            try:
                value = self.run_fragment(name, spec, args, library)
            except (Uncovered, KeyError, TypeError, ValueError) as exc:
                broken[role] = str(exc)[:300]
                self.trace.add("fragment_failure", fragment=name, reason=broken[role])
                raise
            live[role] = value
            if name in self.context["generated"]:
                recovered.add(role)
            self.trace.add("fragment_complete", fragment=name, input_roles=spec["inputs"], output_role=role,
                           source_tasks=spec.get("source_tasks", []))
            return value

        for goal in goals:
            try:
                need(goal)
            except (Uncovered, ValueError, KeyError, TypeError) as exc:
                skipped.append(goal)
                self.trace.add("composition_missing", goal=goal, reason=str(exc)[:300])
            else:
                completed.append(goal)
            if self.done:
                break
        complete = any(route.get("complete") and name in applicable
                       and set(route["goals"]) <= set(completed) for name, route in rules.items())
        if not self.done and not skipped and completed and complete:
            arguments = {"answer": live["answer"]} if "answer" in live else {}
            self.call("supervisor", "complete_task", arguments, "composition_finish")
        if not self.done and not goals:
            self.trace.add("fallback", scope="generation_only", completed_goals=[])
            self.agent()
        elif not self.done:
            names = self.prime_generated_session(live)
            visible = {k: v for k, v in live.items()
                       if k in names and not any(part in k for part in ("token", "session", "password"))}
            progress = {"completed_goals": completed, "missing_goals": skipped,
                        "recent_writes": self.write_progress[-15:],
                        "variables": names, "live_values": visible}
            self.trace.add("fallback", scope="unfinished_goals" if completed else "generation_only", completed_goals=completed)
            self.agent(progress=progress)

    def run_fragment(self, name, spec, values, library):
        parent = self.context["vars"]
        parent_inputs = self.fragment_inputs
        self.fragment_inputs = dict(zip(spec["inputs"], values))
        self.context["vars"] = self.fragment_inputs.copy()
        inherited = self.generated_inputs
        self.generated_inputs = set(spec["inputs"]) & inherited
        try:
            self.fragment_steps(spec["steps"], library, name)
            if self.done:
                raise Uncovered("A fragment cannot finish the outer task")
            output = resolve(spec["result"], self.program_context())
            if output is None:
                raise Uncovered("The fragment returned no value")
            return output
        except (Uncovered, ValueError, KeyError, TypeError, ContextTooLong) as exc:
            recovery = library["generation_skills"].get(name)
            if not recovery or self.done:
                raise Uncovered(str(exc)) from exc
            return self.recover(name, recovery, library["mode_policy"].get(name, {}), str(exc))
        finally:
            self.context["vars"] = parent
            self.fragment_inputs = parent_inputs
            self.generated_inputs = inherited

    def fragment_steps(self, steps, library, name):
        for step in steps:
            self.check_time()
            if self.done:
                return
            kind = step["kind"]
            if kind == "decision":
                node = step["decision"]
                ctx = self.program_context()
                ctx = {**ctx, **{namespace: {k: v for k, v in ctx[namespace].items() if k not in self.generated_inputs}
                                for namespace in ("vars", "inputs")}}
                question = resolve(library["questions"][node], ctx)
                evidence = resolve(library["evidence"].get(node, {"instruction": {"ref": "task.instruction"}}), ctx)
                choices = build_candidates(library["candidates"][node], ctx)
                policy = library["mode_policy"].get(node, {})
                if any(row["value"] is None for row in choices):
                    raise Uncovered("Choice payload is missing")
                try:
                    result = self.decision(question, evidence, choices, library["decision_skills"].get(node, ""), policy, node=node)
                except Uncovered:
                    if (not policy.get("reason_on_none") or policy.get("mode") == "R"
                            or not 1 <= len(choices) <= 15):
                        raise
                    result = self.decision(question, evidence, choices, library["decision_skills"].get(node, ""),
                                           {"mode": "R", "max_tokens": policy.get("max_tokens", 512)}, node=node)
                self.context["vars"][step["store"]] = result
                continue
            binding = resolve(library["bindings"][step["binding"]], self.program_context())
            if kind == "foreach":
                rows = []
                for item in binding:
                    self.context["vars"][step.get("item", "item")] = item
                    self.fragment_steps(step["body"], library, name)
                    if "collect" in step:
                        rows.append(resolve(step["collect"], self.program_context()))
                if "store" in step:
                    self.context["vars"][step["store"]] = rows
                continue
            if kind == "if":
                self.fragment_steps(step.get("then" if binding else "else", []), library, name)
                continue
            if kind == "call":
                app, api = binding["app"], binding["api"]
                if app == "supervisor" and api == "complete_task":
                    raise Uncovered("A fragment does not own task completion")
                args = binding.get("arguments", {})
                if step.get("paginate"):
                    result = []
                    for page in range(100):
                        rows = self.call(app, api, {**args, "page_index": page, "page_limit": 20}, name)
                        if not isinstance(rows, list):
                            raise Uncovered(f"{app}.{api} did not return list rows")
                        result.extend(rows)
                        if len(rows) < 20:
                            break
                    else:
                        raise Uncovered(f"{app}.{api} pagination did not finish")
                else:
                    result = self.call(app, api, args, name)
                if "expect" in step and (not isinstance(result, dict) or any(result.get(k) != v for k, v in step["expect"].items())):
                    raise Uncovered(f"Unexpected response to {app}.{api}")
                if step.get("require") and (not isinstance(result, dict) or any(k not in result for k in step["require"])):
                    raise Uncovered(f"Expected fields missing from {app}.{api}")
                if step.get("expect") and not api.startswith(("show_", "search_", "list_")) and api != "login":
                    self.write_progress.append({"app": app, "api": api, "args": {k: v for k, v in args.items()
                                                if k not in ("access_token", "password", "content")}})
            elif kind == "compute":
                result = binding
            else:
                raise Uncovered(f"Unknown fragment node {kind}")
            if "store" in step:
                self.context["vars"][step["store"]] = result

    def recover(self, name, skill, policy, reason):
        self.trace.add("local_recovery", fragment=name, reason=reason[:500])
        focus = {"fragment": name, "failure": reason[:500], "observed_values": self.context["vars"]}
        # Tool-free slots stay separate from observations used to construct any later choices.
        if policy.get("recovery", "value") == "value":
            response = self.generate_message("Complete only this local step, given the evidence. "
                "Return one JSON object {\"value\": result}. Do not generate questions, choices, or a new workflow. " + skill,
                focus, purpose="G_local", max_tokens=policy.get("local_tokens", 384))
            result = parse_json(response["text"])["value"]
        else:
            names = self.prime_generated_session(self.context["vars"])
            focus["variables"] = names
            response = self.generate_message("Complete only this local AppWorld step with one Python block. "
                "Use public apis, do not complete the full task, and print a final compact JSON object "
                "{\"value\": result}. Existing inputs are also available in the Python dict compiled. "
                "Do not repeat completed actions. " + skill,
                focus, purpose="G_local", max_tokens=policy.get("local_tokens", 768))
            output = self.code(self.code_from(response["text"]), name, permit_finish=False)["output"]
            if output.startswith("Execution failed."):
                raise Uncovered("Local tool recovery failed")
            result = parse_json(output.splitlines()[-1])["value"]
        expected = policy.get("local_type")
        types = {"int": int, "number": (int, float), "str": str, "dict": dict, "list": list, "bool": bool}
        if result is None or (expected in types and not isinstance(result, types[expected])):
            raise Uncovered("Local generation returned an invalid result")
        self.context["generated"][name] = result
        self.trace.add("local_recovery_return", fragment=name, result=result)
        return result

    def summarize(self, task_id, label, seed, started, error, evaluation):
        row = super().summarize(task_id, label, seed, started, error, evaluation)
        row["fragments"] = [e["fragment"] for e in self.trace.events if e["event"] == "fragment_complete"]
        row["local_recoveries"] = sum(e["event"] == "local_recovery_return" for e in self.trace.events)
        row["decisions_with_multiple_objects"] = sum(len(e["candidates"]) > 2 for e in self.trace.events if e["event"] == "decision")
        row["fallback"] = [e["scope"] for e in self.trace.events if e["event"] == "fallback"]
        return row
