"""Check the first changed decision on each previously solved, explicitly opened Train task."""

from copy import deepcopy
import json

from formal_run import read
from .formal_optimizer import cleaned_text, read_events
from .native_assembly import NativeAssembly
from .schema_tools import catalog, public_value
from .trace_comparison import Boundary, different_fields, recorded_requests, run_on_tape


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


class RecordedChoices:
    def decision(self, question, evidence, candidates, *args, **kwargs):
        self.model.offered = [c["value"] for c in candidates] + [None]
        self.model.menu_node = kwargs.get("node")
        self.model.fragment_question = fragment_question(self.model.menu_node, self.model.offered, evidence)
        self.model.calculations = self.transforms
        return super().decision(question, evidence, candidates, *args, **kwargs)

    def session_token(self, app):
        # Original traces contain no credentials. Never create a live environment in this check.
        return "<session>"


class RecordedNative(RecordedChoices, NativeAssembly):
    pass


class RecordedGeneration(RecordedNative):
    generation_only = True


def first_change(events, candidate, functions, execution, model=None, *, generation_only=False, executor_type=None):
    apps = [name.split("__", 1)[0] for name in catalog(functions) if name.endswith("__login")]
    tape = [
        (i, op, request, response)
        for i, op, request, response in recorded_requests(events)
        if executor_type is not None or events[i].get("node") != "session"
    ]
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
            elif op == "execute" and executor_type is not None:
                comparable = {**args, "code": cleaned_text(args["code"])}
            changed = different_fields(previous, comparable)
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
            else:
                report["status"] = "changed_" + op
            raise Boundary

    if executor_type is not None:
        class Recorded(RecordedChoices, executor_type):
            # Reconstruct the public history of authentication with inert values.
            session_token = executor_type.session_token
        executor = Recorded
    else:
        executor = RecordedGeneration if generation_only else RecordedNative
    stop = run_on_tape(events, candidate, functions, execution, Client(), executor)
    if not report:
        report = {"status": "unchanged" if position == len(tape) else "unresolved", "stop": stop}
    report["matched_requests"] = position
    return report


def historical_preview(study, candidate, history):
    opened = {}
    for directory in history:
        plan = read(directory / "pilot_plan.json")
        for task in plan["train"]:
            for stage in ("h0_decision", "candidate"):
                path = directory / "runs" / stage / task / "record.json"
                if path.is_file() and (row := read(path))["success"]:
                    opened[task] = (directory / row["trace_path"], directory / "offline/public_functions.json",
                                    plan["execution"])
    return solved_preview(study, candidate, opened)


def solved_preview(study, candidate, opened, *, budget=24, executor_type=None):
    results = {}
    used = 0
    for task, (trace, catalogue, execution) in opened.items():
        events = read_events(trace)
        functions = read(catalogue)
        runtime = {"executor_type": executor_type} if executor_type is not None else {}
        outcome = first_change(events, candidate, functions, execution, **runtime)
        if outcome["status"] == "needs_real_choice" and used < budget:
            study.start_workers()
            model = study.workers[used % len(study.workers)].worker("model")
            outcome = first_change(events, candidate, functions, execution, model, **runtime)
            used += 1
        elif outcome["status"] == "needs_real_choice":
            outcome["reason"] = f"{budget} historical reselections already used this round"
        results[task] = outcome
    return results
