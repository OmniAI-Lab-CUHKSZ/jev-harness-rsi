"""Check the first changed decision on each previously solved, explicitly opened Train task."""

from copy import deepcopy
import json

from .artifacts import cleaned_text, read, read_events
from .composed_assembly import ComposedAssembly
from .formal_runtime import BudgetExhausted
from .runtime import STEP_REVIEW, Trace, Uncovered
from .schema_tools import catalog, public_value


class Boundary(Exception):
    """A recorded response cannot be supplied to this request."""


def different_fields(left, right):
    return sorted(key for key in left.keys() | right.keys()
                  if key not in left or key not in right or
                  json.dumps(left[key], sort_keys=True) != json.dumps(right[key], sort_keys=True))


def recorded_requests(events):
    for index, e in enumerate(events):
        kind = e["event"]
        if kind == "model":
            args = dict(e["request"])
            if e["operation"] == "decide" and e["response"].get("thinking_used") and args.get("purpose") == STEP_REVIEW:
                args.pop("purpose")
            yield index, e["operation"], args, e["response"]
        elif kind in ("api", "tool_batch"):
            keys = ("app", "api", "arguments") if kind == "api" else ("code", "permit_finish", "trace_actions")
            yield index, "call" if kind == "api" else "execute", {k:e[k] for k in keys if k in e}, e["response"]
        elif kind == "generation_resume_bindings":
            yield index, "bind_progress", {"names":e["names"]}, {}


def run_on_tape(events, library, functions, settings, client, executor_type):
    start = events[0]
    task = {k:start[k] for k in ("task_id", "instruction", "datetime", "supervisor")}
    engine = executor_type(client, client, **settings)
    engine.prepare_options()
    engine.trace, engine.seed, engine.done = Trace(), start["seed"], False
    engine.context = {"task":{**task, "functions":functions}, "history":[]}
    try:
        engine.execute_task(library)
        return "returned"
    except Boundary:
        return "boundary"
    except (BudgetExhausted, Uncovered) as exc:
        return str(exc)


def fragment_question(node, values, evidence):
    if node != "native/fragment_choice" or not values or not isinstance(values[0], dict):
        return None
    first = values[0]
    if "role" not in first:
        return None
    return {
        "fragment": first["fragment"], "role": first["role"],
        "using": [(r["source"], r["ref"], r["row"]) for r in evidence.get("reference_records", [])],
    }


class RecordedComposition(ComposedAssembly):
    def decision(self, question, evidence, candidates, *args, **kwargs):
        self.model.offered = [c["value"] for c in candidates] + [None]
        self.model.menu_node = kwargs.get("node")
        self.model.fragment_question = fragment_question(self.model.menu_node, self.model.offered, evidence)
        self.model.calculations = self.transforms
        self.model.card_offers = self.card_offers
        return super().decision(question, evidence, candidates, *args, **kwargs)

def first_change(events, candidate, functions, execution, model=None):
    apps = [name.split("__", 1)[0] for name in catalog(functions) if name.endswith("__login")]
    tape = list(recorded_requests(events))
    position = 0
    report = {}
    recorded_menus = {}
    recorded_nodes = {}
    recorded_questions = {}
    pending = []
    for i, event in enumerate(events):
        if event["event"] == "model" and event["operation"] == "decide":
            pending.append(i)
        elif event["event"] == "decision":
            for index in pending:
                recorded_menus[index] = [c["value"] for c in event["candidates"]]
                recorded_nodes[index] = event.get("node")
                recorded_questions[index] = fragment_question(
                    event.get("node"), recorded_menus[index], event.get("evidence", {})
                )
            pending.clear()
        elif event["event"] in {"api", "native_operation", "native_feedback"}:
            pending.clear()

    def identity(choice):
        if isinstance(choice, dict) and "arguments" in choice:
            return {"tool": choice.get("tool"), "arguments": choice["arguments"]}
        return choice

    class Client:
        def request(self, op, **args):
            nonlocal position, report
            i, expected, previous, response = tape[position] if position < len(tape) else (None, None, {}, {})
            comparable = public_value(args) if op == "call" else args
            if op == "call":
                comparable.get("arguments", {}).pop("raise_on_failure", None)
            elif op == "bind_progress":
                comparable = {"names": sorted(args["values"])}
            elif op == "execute":
                comparable = {**args, "code": cleaned_text(args["code"])}
            changed = different_fields(previous, comparable) if op == expected else []
            old_values = recorded_menus.get(i)
            comparable_menu = old_values and not any(type(v) is int for v in old_values)
            if op == expected == "decide" and comparable_menu:
                if json.dumps([identity(v) for v in old_values], sort_keys=True) != json.dumps(
                    [identity(v) for v in self.offered], sort_keys=True
                ):
                    changed.append("bound_options")
            if op == expected and not changed and not response.get("error"):
                position += 1
                replayed = deepcopy(response)
                if (op == "call" and events[i].get("node") == "session"
                        and replayed.get("value") == {"session_data_received": True}):
                    if args["api"] == "show_account_passwords":
                        replayed["value"] = [{"account_name": app, "password": "<session>"} for app in apps]
                    else:
                        replayed["value"] = {"access_token": "<session>"}
                return replayed
            if "continuation" in report:
                report["continuation"].update(status="changed", event=i, before=expected, after=op,
                                               changed_fields=changed, matched_requests=position)
                raise Boundary
            report = {"event": i, "before": expected, "after": op, "changed_fields": changed}
            if op == "decide":
                previous_node, new_node = recorded_nodes.get(i), self.menu_node
                old_question, new_question = recorded_questions.get(i), self.fragment_question
                different_roles = "native/fragment_choice" in (previous_node, new_node) and (
                    previous_node != new_node or old_question != new_question
                )
                prior = (
                    previous["options"][response["index"]]
                    if expected == "decide" and "index" in response
                    else None
                )
                report.update(status="needs_real_choice", old_choice=prior)
                if different_roles:
                    report.update(comparison="different_decision_roles", previous_node=previous_node, new_node=new_node)
                    if old_question or new_question:
                        report.update(previous_question=old_question, new_question=new_question)
                if model is not None:
                    result = model.request("decide", **args)
                    choice = args["options"][result["index"]]
                    report.update(
                        status="reselected",
                        new_choice=choice,
                        same_displayed_choice=choice == prior if prior is not None and not different_roles else None,
                        output_tokens=result.get("output_tokens", 0),
                        input_tokens=result.get("input_tokens", 0),
                    )
                    if new_node == "native/fragment_choice":
                        report["selected_record"] = self.offered[result["index"]]
                    if comparable_menu and expected == "decide" and "index" in response and not different_roles:
                        old, new = old_values[response["index"]], self.offered[result["index"]]
                        report["same_semantic_choice"] = json.dumps(
                            identity(old), sort_keys=True
                        ) == json.dumps(identity(new), sort_keys=True)
                        if (
                            isinstance(old, (list, tuple))
                            and old[0] == "calculate"
                            and isinstance(new, (list, tuple))
                            and new[0] == "calculate"
                        ):
                            for event in events[i + 1 :]:
                                if event["event"] == "model":
                                    break
                                if event["event"] == "native_observation" and event["scope"] == old[1]:
                                    report["same_local_result"] = event["value"] == self.calculations[new[1]]
                                    break
                    selected = self.offered[result["index"]]
                    is_card = isinstance(selected, (tuple, list)) and tuple(selected) in self.card_offers
                    declined_card = selected is None and any(
                        isinstance(value, (tuple, list)) and tuple(value) in self.card_offers
                        for value in self.offered
                    )
                    same_choice = report.get("same_semantic_choice")
                    if is_card:
                        report["proposed_reads"] = [
                            {"scope": plan["scope"], "choices": plan.get("choices", {}),
                             "sources": [read["source"] for read in plan["reads"].values()]}
                            for plan in self.card_offers[tuple(selected)]
                        ]
                    if same_choice or is_card or declined_card:
                        report["continuation"] = {"status": "replaying", "first_choice_matched": bool(same_choice)}
                        if same_choice:
                            position += 1
                        return result
            else:
                report["status"] = "changed_" + op
            raise Boundary

    stop = run_on_tape(events, candidate, functions, execution, Client(), RecordedComposition)
    if not report:
        report = {"status": "unchanged" if position == len(tape) else "unresolved", "stop": stop}
    if report.get("continuation", {}).get("status") == "replaying":
        report["continuation"].update(status="unchanged" if position == len(tape) else "unresolved",
                                       matched_requests=position, stop=stop)
    report["matched_requests"] = position
    return report


def solved_preview(study, candidate, opened, *, budget=24):
    results = {}
    used = 0
    for task, (trace, catalogue, execution) in opened.items():
        events = read_events(trace)
        functions = read(catalogue)
        outcome = first_change(events, candidate, functions, execution)
        if outcome["status"] == "needs_real_choice" and used < budget:
            study.start_workers()
            model = study.workers[used % len(study.workers)].worker("model")
            outcome = first_change(events, candidate, functions, execution, model)
            used += 1
        elif outcome["status"] == "needs_real_choice":
            outcome["reason"] = f"{budget} historical reselections already used this round"
        results[task] = outcome
    return results
