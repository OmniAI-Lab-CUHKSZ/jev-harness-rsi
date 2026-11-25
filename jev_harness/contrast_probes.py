"""Compare alternative patches at real saved Train states under one exploration budget."""
from collections import Counter
from copy import deepcopy
import json
import re

from formal_run import read, save
from .card_assembly import public_apps
from .formal_optimizer import read_events
from .formal_runtime import BudgetExhausted
from .fragment_replay import LocalReplay, ReplayDiverged
from .fragment_trials import natural_use
from .history_impact import historical_report
from .observed_bindings import ObserveBindings
from .phase1_optimizer import ProbeComplete
from .runtime import STEP_REVIEW, Uncovered, is_open_generation
from .transport import WorkerError, WorkerUnavailable
from .trace_comparison import different_fields


def request_key(operation, values):
    return json.dumps([operation,values],sort_keys=True,ensure_ascii=False,separators=(",",":"))


def recorded_input(event):
    request = dict(event["request"])
    if event["operation"] == "decide" and event["response"].get("thinking_used") and request.get("purpose") == STEP_REVIEW:
        request.pop("purpose")
    return request_key(event["operation"],request)


class ExactParentReplay:
    def __init__(self, events):
        self.model_events = iter(e for e in events if e["event"] == "model")
        self.count = 0

    def request(self, op, **values):
        source = next(self.model_events,None)
        if not source or recorded_input(source) != request_key(op,values) or "error" in source["response"]:
            raise ReplayDiverged("Fresh parent replay differs at model request "+str(self.count+1))
        self.count += 1
        return deepcopy(source["response"])


class MeteredModel:
    def __init__(self, model, ballots, cost, calls, allow_generation, on_charge=None):
        self.model,self.ballots,self.cost = model,ballots,cost
        self.generations_left = int(allow_generation)
        self.remaining = calls
        self.on_charge = on_charge
        self.events = []

    def request(self, op, **values):
        key = request_key(op,values)
        if op == "decide" and key in self.ballots:
            self.cost["cached_decisions"] += 1
            self.events.append({"source":"exact_request_cache","op":op})
            return deepcopy(self.ballots[key])
        balance = 4096-self.cost["output_tokens"]
        if op == "decide":
            if self.remaining <= 0 or self.cost["fresh_decisions"] >= 12:
                raise ProbeComplete("Local decision budget reached")
            if values.get("thinking") == "auto" and balance < values.get("think_budget",256):
                raise ProbeComplete("Insufficient remaining output budget for an official adaptive choice")
            self.remaining -= 1
            self.cost["fresh_decisions"] += 1
        else:
            if self.generations_left <= 0 or self.cost["open_generations"] >= 3 or balance <= 0:
                raise ProbeComplete("Generation boundary or shared generation budget reached")
            self.generations_left -= 1
            self.cost["open_generations"] += 1
            values["max_tokens"] = min(1024,balance,values["max_tokens"])
        if self.on_charge:
            self.on_charge()
        response = self.model.request(op,**values)
        self.cost["output_tokens"] += response.get("output_tokens",0)
        self.cost["reasoning_tokens"] += response.get("reasoning_tokens",0)
        self.cost["decision_output_tokens" if op=="decide" else "generation_output_tokens"] += response.get("output_tokens",0)
        self.cost["seconds"] += response.get("seconds",0)
        self.cost["input_tokens"] += response.get("input_tokens",0)
        if op == "decide":
            self.cost["decision_backend_requests"] += response.get("backend_requests",1)
            self.ballots[key] = deepcopy(response)
        self.events.append({"source":"live","op":op,"output_tokens":response.get("output_tokens",0),
                            "thinking_used":response.get("thinking_used"),"seconds":response.get("seconds",0)})
        if self.on_charge:
            self.on_charge()
        return response


class MatchedPrefix:
    """Spend fresh probe calls starting at the first change in a clean candidate world."""
    def __init__(self, parent, limit, meter):
        self.parent, self.limit, self.meter = parent, limit, meter
        self.position = 0

    def match(self, op, args):
        if self.position >= self.limit:
            return None
        _, expected, values = self.parent.requests[self.position]
        if op != expected or different_fields(values,args):
            self.limit = self.position
            return None
        return self.parent.tape[self.position][3]

    def tool_result(self, op, args, result):
        saved = self.match(op,args)
        if saved is None:
            return
        if op != "bind_progress" and different_fields(saved,result):
            self.limit = self.position
            return
        self.position += 1

    def request(self, op, **values):
        saved = self.match(op,values)
        if saved is None:
            return self.meter.request(op,**values)
        self.position += 1
        self.meter.cost["replayed_parent_requests"] += 1
        self.meter.cost["prefix_"+op] += 1
        self.meter.events.append({"source":"matched_parent_prefix","op":op})
        return deepcopy(saved)


class ConditionalReplay(ObserveBindings, LocalReplay):
    prefix = None

    def initialize(self, lib):
        if getattr(self,"resume_existing",False):
            self.registry,self.pool = lib,set(lib["cards"])
            self.app_hints = public_apps(self.request,lib["cards"])
        else:
            super().initialize(lib)

    def assemble_available(self):
        if getattr(self,"resume_handoff",None) is not None:
            reason,self.resume_handoff = self.resume_handoff,None
            raise Uncovered(reason)
        return super().assemble_available()

    def _model(self, operation, **values):
        if not self.replaying and is_open_generation(operation,values):
            self.trace.add("probe_generation_boundary",messages=values["messages"])
        return super()._model(operation,**values)

    def prime_generated_session(self,values):
        if self.replaying:
            return LocalReplay.prime_generated_session(self,values)
        bound = super().prime_generated_session(values)
        if self.prefix and bound:
            self.prefix.tool_result("bind_progress",{"values":{k:values[k] for k in bound}}, {})
        return bound

    def tool_request(self, op, event, node, **params):
        result = super().tool_request(op,event,node,**params)
        if self.prefix:
            self.prefix.tool_result(op,params,result)
        return result


def observations(events,names):
    values = []
    for i,e in enumerate(events):
        kind = e["event"]
        if kind=="model" and is_open_generation(e["operation"],e["request"]) and not e["response"].get("error"):
            values.append({"index":i,"event":"open_generation","output_tokens":e["response"].get("output_tokens",0),
                           "reasoning_tokens":e["response"].get("reasoning_tokens",0),"truncated":e["response"].get("truncated")})
        elif kind == "decision":
            vals = e["candidates"]
            row = {"event":i,"node":e["node"],"selected":vals[e["selected"]]["value"]}
            if e["node"]=="fragment/card_parameter":
                evidence = e.get("evidence",{})
                row.update(question=e.get("question"),grounded_options=len(vals)-1,
                           rendered_evidence_fields=list(evidence),rendered_views=evidence.get("views",{}))
            values.append(row)
        elif kind=="parameter_state":
            values.append({"index":i,"event":kind,"card":e.get("card"),"available":e.get("available",[]),
                           "policy_inputs":{"datetime":e.get("datetime"),"active_card":e.get("card")}})
        elif kind in ("card_executed","card_failed","handoff_selection","program_error","fragment_handoff",
                      "fragment_reentry","generated_read_observation","generation_parse_error","parameter_bindings","result_source","api_action"):
            row = {k:v for k,v in e.items() if k in ("event","card","output","reason","guide","policy","error","api","roles","role","producer","available","bound","method","url")}
            if "url" in row:
                row["url"] = row["url"].partition("?")[0]
            values.append({"index":i,**row})
        elif kind == "tool_batch" and e.get("node")=="generation_agent":
            from .formal_optimizer import cleaned_text
            values.append({"index":i,"event":"generated_cell","result":cleaned_text(e["response"].get("output",""))[-400:]})
    if len(values)>42:
        important = [v for v in values[10:-10] if v.get("event") in
                     ("card_failed","card_executed","generated_read_observation","generation_parse_error","generated_cell","result_source","parameter_bindings","parameter_state")
                     or v.get("node")=="fragment/card_parameter"]
        values = values[:10]+important[-22:]+values[-10:]
    return {"events":values,"natural_use":natural_use(events,names)}


def compare_candidates(bench,draft,worker,*,folder=None,prior_cost=None,matched_paths=None,history_docs=None):
    study = bench.study
    resumable = folder is not None
    folder = folder or bench.folder
    folder.mkdir(parents=True,exist_ok=True)
    if resumable:
        spec = {"objects":[[h["id"],h["objects"]] for h in draft["alternatives"]],"cases":draft["cases"]}
        source = folder/"tested_patch.json"
        if source.exists() and read(source)!=spec:
            raise ValueError("Saved observations belong to different objects or cases in "+str(folder))
        save(source,spec)
    target = folder/"local_evidence.json"
    if target.is_file():
        return read(target)
    hypotheses = draft["alternatives"]
    archive = {doc:bench.trace_data(doc) for doc in (bench.archived if history_docs is None else history_docs)}
    candidates,impact = {},{}
    for j,h in enumerate(hypotheses):
        lib,_ = bench.install(bench.lib,h,bench.sources,study.functions)
        candidates[h["id"]] = lib
        summary,details = historical_report(archive,bench.lib,lib,h["objects"])
        save(folder/f"historical_{j+1}.json",{"summary":summary,"details":details})
        impact[h["id"]] = summary
    results = []
    cost = Counter({k:0 for k in ("fresh_decisions","cached_decisions","replayed_parent_requests","open_generations",
                                   "decision_backend_requests","output_tokens","decision_output_tokens","generation_output_tokens",
                                   "reasoning_tokens","seconds","input_tokens")})
    cost.update(prior_cost or {})
    meter_file = folder/"meter.json"
    if resumable and meter_file.exists():
        cost = Counter(read(meter_file))
    cases = draft["cases"]
    pending = len(cases)*len(hypotheses)
    for i,case in enumerate(cases):
        doc = case["trace"]
        task = doc.rsplit("/",1)[-1]
        es = read_events(bench.paths[doc])
        ballots = {recorded_input(e):e["response"] for e in es if e["event"]=="model" and e["operation"]=="decide" and "error" not in e["response"]}
        source = bench.lib
        if not doc.startswith("current/") and case["anchor"]!="start":
            stage = doc.split("/")[0]
            revision = next((r for r in bench.previous if stage==r["candidate"] or stage.startswith(r["candidate"]+"_")),None)
            if revision and stage==revision["candidate"]:
                source = read(bench.paths[doc].parents[4]/"versions"/(stage+".json"))
            elif revision:
                source = read(bench.paths[doc].parents[4]/"versions"/(revision["parent"]+".json"))
            elif "_entry" in stage:
                version = stage.split("_")[1]
                source = read(bench.paths[doc].parents[4]/"versions"/(version+".json"))
        for j,h in enumerate(hypotheses):
            trial = folder/f"case_{i+1}_alt_{j+1}"
            trial.mkdir(exist_ok=True)
            if resumable and (trial/"observation.json").exists():
                results.append(read(trial/"observation.json"))
                pending -= 1
                continue
            matched = matched_paths.get(task) if matched_paths else None
            if matched and matched["status"] == "equal_on_record":
                row = {"id":h["id"], **case, "reached_anchor":False,"scope":"recorded_requests",
                       "stop":"All comparable requests match through the complete recorded task",
                       "model_calls":[],"events":[],"natural_use":{}}
                results.append(row)
                save(trial/"observation.json",row)
                pending -= 1
                continue
            reader = ExactParentReplay(es)
            allowance = case.get("decisions",max(1,(12-cost["fresh_decisions"])//pending))
            cells = case.get("cells",case.get("generation") is True)
            if resumable:
                start = trial/"cost_at_start.json"
                if not start.exists():
                    save(start,dict(cost))
                initial = read(start)
                allowance = max(0,allowance-cost["fresh_decisions"]+initial["fresh_decisions"])
                cells = max(0,cells-cost["open_generations"]+initial["open_generations"])
            meter = MeteredModel(worker.worker("model"),ballots,cost,allowance,cells,
                                 (lambda:save(meter_file,dict(cost))) if resumable else None)
            pending -= 1
            engine = ConditionalReplay(reader,worker.worker("environment"),**study.execution)
            begin = None
            try:
                trial_path = trial.relative_to(study.root) if trial.is_relative_to(study.root) else trial
                label = re.sub(r"\W","_",f"{study.root.name}_{trial_path}_{task}")
                engine.reach(task,source,es,case["anchor"],trial,study.seed,label)
                begin = len(engine.trace.events)
                engine.resume_existing,engine.replaying,engine.fixed,engine.model = True,False,{},meter
                if matched and case["anchor"] == "start":
                    ref = bench.reference_paths[task][2]
                    engine.prefix = MatchedPrefix(ref,matched["matched"],meter)
                    engine.model = engine.prefix
                if case["anchor"]=="generation":
                    prior = next((e for e in reversed(engine.trace.events) if e["event"]=="fragment_handoff"),{})
                    engine.resume_handoff = prior.get("reason","Reached recorded generation handoff")
                engine.execute_task(candidates[h["id"]])
                stop = "execution ended"
            except WorkerUnavailable:
                raise
            except WorkerError as exc:
                if any(s in str(exc).lower() for s in ("connection refused","connection reset","remote end closed","timed out","broken pipe")):
                    raise WorkerUnavailable("Probe service interrupted: "+str(exc)) from exc
                stop = type(exc).__name__+": "+str(exc)[:240]
            except (ProbeComplete,BudgetExhausted,ReplayDiverged,ValueError,KeyError,TypeError) as exc:
                stop = type(exc).__name__+": "+str(exc)[:240]
            finally:
                if getattr(engine,"trace",None):
                    engine.trace.close()
            cost["replayed_parent_requests"] += reader.count
            fragment = engine.trace.events[begin:] if begin is not None else []
            names = [o["name"] for o in h["objects"] if o["kind"]=="card"]
            results.append({"id":h["id"],**case,"reached_anchor":begin is not None,"stop":stop,
                            "model_calls":meter.events,**observations(fragment,names)})
            save(trial/"observation.json",results[-1])
            if resumable:
                save(meter_file,dict(cost))
    report = {"cases":results,"historical_impact":impact,"cost":dict(cost),
              "stage_cost":dict(cost-Counter(prior_cost or {}))}
    save(target,report)
    return report
