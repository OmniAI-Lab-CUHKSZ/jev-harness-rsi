"""Run a proposed card on a fresh AppWorld state reconstructed from a current Train trace."""
import ast
import json
import re

from .card_library import ports
from .formal_optimizer import cleaned_text
from .formal_runtime import BudgetExhausted
from .runtime import Trace, Uncovered, is_open_generation
from .semantic_assembly import SemanticAssembly


class AtAnchor(Exception):
    pass


class ReplayDiverged(Exception):
    """The recorded decision no longer corresponds to the fresh parent state."""


class RecordedModel:
    def __init__(self, events):
        self.responses = iter(e for e in events if e["event"] == "model")

    def request(self, op, **args):
        event = next(self.responses, None)
        if event is None or event["operation"] != op:
            raise ReplayDiverged("Parent replay diverged at a model request")
        if op == "decide" and (args["options"] != event["request"]["options"] or
                               args["question"] != event["request"]["question"]):
            raise ReplayDiverged("Parent replay reached a different decision menu")
        if op == "generate" and args.get("purpose") != event["request"].get("purpose"):
            raise ReplayDiverged("Parent replay reached a different generation mode")
        return event["response"]


class LocalReplay(SemanticAssembly):
    def reach(self, task, parent, events, anchor, folder, seed, label):
        self.anchor = anchor
        self.source_events = iter((i,e) for i,e in enumerate(events)
                                  if e["event"] == "tool_batch" and e.get("node") == "generation_agent")
        self.replaying = True
        self.observed_names = set()
        self.seed, self.done = seed, False
        self.trace = Trace(folder/"trace.jsonl")
        public = self.environment.request("prepare", task_id=task, seed=seed, experiment_name=label,
                                          **self.prepare_options())
        self.context = {"task":public, "vars":{}, "generated":{}, "history":[]}
        if anchor == "start":
            self.initialize(parent)
            return
        try:
            self.execute_task(parent)
        except AtAnchor:
            return
        raise ValueError("Requested location was not reached in the fresh parent replay")

    def _model(self, operation, **values):
        if self.replaying and self.anchor == "generation" and is_open_generation(operation, values):
            raise AtAnchor()
        return super()._model(operation, **values)

    def perform_card(self, name):
        super().perform_card(name)
        if self.replaying and self.anchor == {"after_card":name}:
            raise AtAnchor()

    def tool_request(self, op, event, node, **params):
        if self.replaying and event == "tool_batch" and node == "generation_agent":
            position, original = next(self.source_events, (None, {}))
            if original.get("code") != params.get("code"):
                raise ReplayDiverged("Parent replay diverged from the recorded public code")
            if isinstance(self.anchor, dict) and self.anchor.get("event") == position:
                through = self.anchor.get("through_line", 0)
                if through:
                    program = original["code"]
                    if through not in {s.end_lineno for s in ast.parse(program).body}:
                        raise ValueError("through_line must end a complete top-level statement of the recorded cell")
                    prefix = "\n".join(program.splitlines()[:through])
                    result = super().tool_request("execute", event, node, code=prefix, permit_finish=False,
                                                  trace_actions=True)
                    if "Execution failed." in str(result.get("output", "")):
                        raise ValueError("Recorded cell prefix failed in the replay: "+str(result["output"])[-220:])
                    self.remember_names(prefix)
                raise AtAnchor()
            self.remember_names(original["code"])
        return super().tool_request(op, event, node, **params)

    def remember_names(self, code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        self.observed_names.update(n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store))

    def variable(self, name):
        if (not isinstance(name, str) or name not in self.observed_names or name.startswith("_")
                or re.search("password|credential|token|secret",name,re.I)):
            raise ValueError("Expected a public variable from the recorded generated code")
        result = self.environment.request("execute", permit_finish=False, code=(
            "import json\nprint(json.dumps("+name+", default=lambda x: sorted(x) if isinstance(x, set) else str(x)))"))
        try:
            return json.loads(result["output"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Recorded variable was unavailable or its output was incomplete: "+name) from exc

    def choose_inline(self, question, choices, skill, bindings):
        if self.replaying:
            return super().choose_inline(question, choices, skill, bindings)
        slot = next((name for name, definition in self.registry["cards"][self.active_card].get("parameters",{}).items()
                     if name not in bindings and definition["question"] == question), None)
        overrides = self.fixed.get(self.active_card,{})
        if slot in overrides:
            for choice in choices:
                if choice["value"] == overrides[slot]:
                    return choice["value"]
            raise ValueError("Fixed local parameter is absent from its live candidates: "+slot)
        return super().choose_inline(question, choices, skill, bindings)


def preview(value):
    s = cleaned_text(json.dumps(value, ensure_ascii=False, default=str))
    return {"type":type(value).__name__, "length":len(value) if isinstance(value,(list,dict,str)) else None,
            "preview":s[:360]}


def run_local(study, worker, parent, candidate, events, case, folder, task, parent_success):
    def engine_at(anchor, part):
        dest = folder/part
        dest.mkdir(parents=True, exist_ok=True)
        engine = LocalReplay(RecordedModel(events), worker.worker("environment"), **study.execution)
        try:
            label = re.sub(r"\W", "_", "_".join((study.root.name,*dest.relative_to(study.root).parts,task)))
            engine.reach(task, parent, events, anchor, dest, study.seed, label)
        except BaseException:
            engine.trace.close()
            raise
        return engine

    report = {"task":task,"parent_task_passed":parent_success,"forced_cards":case["cards"],
              "anchor":case.get("anchor","generation"),"executions":[]}
    reference = case.get("reference")
    if reference:
        try:
            ref_engine = engine_at({k:reference[k] for k in ("event","through_line")},"reference")
            try:
                expected = ref_engine.variable(reference["variable"])
            finally:
                ref_engine.trace.close()
            report["reference"] = {**reference, **preview(expected)}
        except (ValueError, KeyError, TypeError, Uncovered, BudgetExhausted, SyntaxError, ReplayDiverged) as exc:
            report["reference_unavailable"] = cleaned_text(str(exc))[:300]
    try:
        engine = engine_at(report["anchor"],"replacement")
        try:
            engine.replaying = False
            engine.registry, engine.pool = candidate, set(candidate["cards"])
            engine.model, engine.fixed = worker.worker("model"), case.get("parameters",{})
            engine.trace.add("local_replacement_start", cards=case["cards"])
            for role, var in case.get("bindings",{}).items():
                engine.public[role] = engine.variable(var)
            for name in case["cards"]:
                card = candidate["cards"][name]
                missing = set(ports(card)[1]) - engine.public.keys()
                if missing:
                    raise ValueError("Missing live inputs for "+name+": "+", ".join(sorted(missing)))
                engine.perform_card(name)
                role = ports(card)[2]
                value = engine.public[role]
                bindings = engine.provenance[role]["parameters"]
                report["executions"].append({"card":name,"role":role,"bound_parameters":bindings,**preview(value)})
                if reference and "reference" in report and name == reference.get("card",case["cards"][-1]):
                    if reference.get("order","ordered") == "unordered" and isinstance(value,list) and isinstance(expected,list):
                        def ordered(items):
                            return sorted(json.dumps(v, sort_keys=True) for v in items)
                        equal = ordered(value) == ordered(expected)
                    else:
                        equal = value == expected
                    report["matches_recorded_value"] = equal
        finally:
            engine.trace.close()
    except (ValueError, KeyError, TypeError, Uncovered, BudgetExhausted, SyntaxError, ReplayDiverged) as exc:
        report["stopped"] = cleaned_text(str(exc))[:300]
    return report
