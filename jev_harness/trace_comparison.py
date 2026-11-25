"""Compare two harnesses on one recorded path, stopping at the first unanswered request."""
from collections import Counter
from copy import deepcopy
import json

from .formal_runtime import BudgetExhausted
from .runtime import STEP_REVIEW, Trace, Uncovered
from .semantic_assembly import SemanticAssembly


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


def run_on_tape(events, library, functions, settings, client, executor_type=SemanticAssembly):
    start = events[0]
    task = {k:start[k] for k in ("task_id", "instruction", "datetime", "supervisor")}
    engine = executor_type(client, client, **settings)
    engine.prepare_options()
    engine.trace, engine.seed, engine.done = Trace(), start["seed"], False
    engine.context = {"task":{**task, "functions":functions}, "history":[], "vars":{}, "generated":{}}
    try:
        engine.execute_task(library)
        return "returned"
    except Boundary:
        return "boundary"
    except (BudgetExhausted, Uncovered) as exc:
        return str(exc)


class ParentPath:
    """Keep full rerendered requests in memory; reports only contain public locations/field names."""
    def __init__(self, events, library, functions, settings, executor_type=SemanticAssembly):
        self.tape = list(recorded_requests(events))
        self.requests = []
        self.matched = 0
        self.reason = None
        self.stop = run_on_tape(events, library, functions, settings, self, executor_type)

    def request(self, op, **args):
        index, expected, recorded, result = (self.tape[self.matched] if self.matched < len(self.tape)
                                             else (None, None, {}, {}))
        self.requests.append((index, op, args))
        observed = {"names":sorted(args["values"])} if op == "bind_progress" else args
        if expected != op or different_fields(recorded, observed):
            self.reason = "recording ended" if expected is None else "parent differs from recording"
            raise Boundary
        if op in ("decide", "generate") and result.get("error"):
            self.reason = "recorded model error"
            raise Boundary
        self.matched += 1
        return deepcopy(result)

    @property
    def complete(self):
        return self.reason is None and self.matched == len(self.tape)


def compare_path(events, parent_path, candidate, functions, settings, *, inspect_change=None,
                 executor_type=SemanticAssembly, compare_menu=None, inspect_boundary=None):
    position = 0
    difference = None

    def change(index, before, after, fields):
        return {"event":index, "request":position + 1, "parent":before, "candidate":after, "fields":fields}

    class Candidate:
        def request(self, op, **args):
            nonlocal position, difference
            if inspect_boundary and position == parent_path.matched and op == "decide":
                inspect_boundary(args, self.menu)
            if position == len(parent_path.requests):
                difference = change(None, "stop", op, [])
                raise Boundary
            index, expected, source = parent_path.requests[position]
            fields = different_fields(source, args) if op == expected else []
            if compare_menu and op == expected == "decide" and position < parent_path.matched:
                fields = compare_menu(position, args, fields, self.menu)
            if expected != op or fields:
                if inspect_change is not None:
                    inspect_change(expected,source,op,args)
                # Never serialize bound records, credentials or generated code in the short report.
                if fields == ["values"] and op == "bind_progress":
                    fields = ["values." + k for k in different_fields(source["values"], args["values"])]
                difference = change(index, expected, op, fields)
                raise Boundary
            if position == parent_path.matched:
                raise Boundary
            result = parent_path.tape[position][3]
            position += 1
            return deepcopy(result)

    stop = run_on_tape(events, candidate, functions, settings, Candidate(), executor_type)
    if difference is None and stop != "boundary":
        if position < len(parent_path.requests):
            index, op, _ = parent_path.requests[position]
            difference = change(index, op, "stop", [])
        elif stop != parent_path.stop:
            difference = change(None, "stop", "different termination", [])
    status = "changed" if difference else "equal_on_record" if parent_path.complete else "unresolved"
    return {"status":status, "matched":position, "recorded":len(parent_path.tape),
            "first_change":difference, "parent_coverage":parent_path.reason or
            ("complete" if parent_path.complete else "parent stopped before end of recording")}


def replay_requests(events, lib, functions, execution, executor_type=SemanticAssembly):
    """Historical audit interface: compare through the last recorded request."""
    path = ParentPath(events, lib, functions, execution, executor_type)
    return {"matched_all_recorded_requests":path.matched == len(path.tape), "matched":path.matched,
            "recorded":len(path.tape), "matched_by_kind":dict(Counter(op for _,op,_,_ in path.tape[:path.matched])),
            "mismatch":path.reason if path.matched < len(path.tape) else None}
