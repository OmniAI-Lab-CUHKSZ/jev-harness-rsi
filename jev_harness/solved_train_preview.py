"""Inspect the first differing request on one successful recording per earlier Train task."""
from collections import Counter
import json
from pathlib import Path
import re

from formal_run import read, save
from .contrast_probes import request_key
from .formal_optimizer import cleaned_text
from .historical_choice import ballot_outcome, official_request, recorded_decision
from .semantic_assembly import SemanticAssembly
from .trace_comparison import Boundary, ParentPath, compare_path, different_fields
from .transport import WorkerError, WorkerUnavailable


MODE = "first_change_per_solved_task"
USAGE = ("model_calls", "backend_requests", "input_tokens", "output_tokens", "reasoning_tokens", "seconds")


def trace_document(path):
    return "/".join(Path(str(path).replace("\\", "/")).parts[-4:-2])


def recorded_library(bench, doc):
    """Resolve the harness that bound a historical page's local indices to real actions."""
    if not hasattr(bench, "trace_versions"):
        versions = {}
        checkpoints = bench.study.root/"checkpoints.json"
        if checkpoints.is_file():
            from phase1_run import completed_train_entries
            for checkpoint in read(checkpoints):
                visit = checkpoint.get("visit", f"b{checkpoint['batch']:02d}")
                for path in completed_train_entries(bench.study.root, visit, checkpoint, bench.study.settings):
                    versions[trace_document(path)] = checkpoint["entry_version"]
        for revision in bench.previous:
            for arm, field in (("parent","parent_traces"), ("candidate","train_traces")):
                if revision.get(arm):
                    for path in revision.get(field, ()):
                        versions.setdefault(trace_document(path), revision[arm])
            for pair in revision.get("paired_selection", {}).get("repeats", {}).values():
                for row in pair:
                    if version := revision.get(row["arm"]):
                        versions.setdefault(trace_document(row["trace_path"]), version)
        bench.trace_versions = versions
    version = bench.trace_versions.get(doc)
    if version:
        file = bench.study.root/"versions"/(version+".json")
        if file.is_file():
            return read(file)


def solved_documents(bench):
    """Use scored Train provenance, including confirmations. Reusing a parent is not a new run."""
    records = bench.precedents.records
    eligible = {doc for doc in bench.archived if records[doc]["task_success"] is True
                and records[doc]["task"] not in bench.tasks}
    visits = {}
    for path in getattr(bench.study, "prior_train_entries", ()):
        stage = trace_document(path).split("/",1)[0]
        label = re.match(r"^b\d+(?:_revisit\d+)?", stage)
        visits.setdefault(label.group() if label else stage, []).append(path)
    for revision in bench.previous:
        visit = revision.get("visit") or f"b{revision.get('batch',0):02d}"
        paths = visits.setdefault(visit, [])
        paths.extend(revision.get("parent_traces", ()))
        paths.extend(revision.get("train_traces", ()))
        for pair in revision.get("paired_selection", {}).get("repeats", {}).values():
            paths.extend(row["trace_path"] for row in pair)
    ordered = (trace_document(p) for paths in visits.values() for p in paths)
    result = {records[doc]["task"]:doc for doc in sorted(eligible)}
    for doc in dict.fromkeys(ordered):
        if doc in eligible:
            result[records[doc]["task"]] = doc
    return result


class MenuReplay(SemanticAssembly):
    def pick_page(self, part, pick, **mode):
        self.page_options = part
        try:
            return super().pick_page(part, pick, **mode)
        finally:
            self.page_options = None

    def decision(self, question, evidence, candidates, skill="", policy=None, node=None, **mode):
        spent = sum(e["response"].get("output_tokens", 0) for e in self.trace.events if e["event"] == "model")
        self.model.menu = dict(node=node, raw=candidates, choices=getattr(self, "page_options", None) or candidates,
                               budget=self.output_budget-spent, calls_left=self.max_calls-self.requests,
                               review=node=="fragment/step" and mode.get("review_step", False),
                               widen=mode.get("widen_on_none", False), minimum=(policy or {}).get("minimum_probability", 0))
        return super().decision(question, evidence, candidates, skill, policy, node, **mode)


def values(choices):
    # JSON normalizes tuples written by the executor and lists loaded from its trace.
    return [json.dumps(c["value"], sort_keys=True) for c in choices]


def before_ballot(events, event):
    # A second adaptive request for the same page can follow its native request.
    pos = event - 1
    while pos >= 0 and events[pos]["event"] == "model" and events[pos]["operation"] == "decide":
        pos -= 1
    return pos


def saved_page(events, event, choices):
    pos = before_ballot(events,event)
    if pos >= 0 and events[pos]["event"] == "decision_rules":
        pos -= 1
    if pos >= 0 and events[pos]["event"] == "page_bindings":
        data = events[pos]["values"]
        if len(data) == len(choices):
            return [dict(c,value=v) for c,v in zip(choices,data)]


def different_rules(events, event, page):
    pos = before_ballot(events,event)
    return pos >= 0 and events[pos]["event"] == "decision_rules" and any(
        events[pos][k] != page[k] for k in ("review","widen","minimum"))


class SolvedReference(ParentPath):
    def __init__(self, events, library, functions, settings):
        self.events, self.menus = events, {}
        super().__init__(events, library, functions, settings, executor_type=MenuReplay)

    def request(self, op, **args):
        position = self.matched
        if op == "decide" and position < len(self.tape):
            before = recorded_decision(self.events, self.tape[position][0]) if self.tape[position][1] == "decide" else None
            self.menus[position] = (self.menu, before)
        result = super().request(op, **args)
        if op == "decide":
            _, before = self.menus[position]
            bound = saved_page(self.events,self.tape[position][0],before["candidates"][:-1]) if before else None
            if before and (before["node"] != self.menu["node"] or
                           values(before["candidates"][:-1]) != values(self.menu["raw"]) or
                           bound is not None and values(bound) != values(self.menu["choices"]) or
                           different_rules(self.events,self.tape[position][0],self.menu)):
                self.matched = position
                self.reason = "parent decision bindings differ from recording"
                raise Boundary
        return result


def menu_ballot(before, old_choices, old_request, page, request, settings, *, force_live=False):
    none = before["candidates"][-1]
    choices = [*page["choices"], none]
    if [str(c["label"]) for c in choices] != request.get("options"):
        return {}
    selected = before.get("selected")
    if type(selected) is not int or not 0 <= selected <= len(old_choices):
        return {}
    prior = [*old_choices, none][selected]
    old_values, new_values = values(old_choices), values(page["choices"])
    ballot = dict(node=page["node"], question=request["question"], parent_choice=prior,
                  parent_offered=values([prior])[0] in values(choices), choices=choices, page=page,
                  added_values=[c["value"] for c,v in zip(page["choices"],new_values) if v not in old_values][:4],
                  added_options=[s for s in request["options"] if s not in old_request["options"]],
                  removed_options=[s for s in old_request["options"] if s not in request["options"]])
    if force_live or different_fields(old_request, request):
        ballot["request"] = official_request(page, request, settings)
    else:
        ballot["recorded"] = dict(candidate_choice=choices[selected], same_choice=values([choices[selected]])==values([prior]))
    return ballot


def new_route_ballot(operation, page, request, settings, before=None):
    choices = [*page["choices"], {"label":request["options"][-1],"value":None}]
    if [str(c["label"]) for c in choices] != request["options"]:
        return {}
    return dict(route_from=operation, previous_node=before and before["node"],
                node=page["node"], question=request["question"], page=page, choices=choices,
                request=official_request(page, request, settings))


def recording_boundary_ballot(events, parent, page, request, functions, settings, source_library, *, retrieval=False):
    """All requests before this position matched. Compare at this position without supplying its old response."""
    position = parent.matched
    if position >= len(parent.tape):
        return {}
    event, op, old_request, _ = parent.tape[position]
    before = recorded_decision(events, event) if op=="decide" else None
    if retrieval and (op != "decide" or before and page["node"] != before["node"]):
        result = new_route_ballot(op,page,request,settings,before)
        if result:
            result["recording_comparison"] = {"event":event,"fields":["decision_route"]}
        return result
    if not before and retrieval:
        result = new_route_ballot(op,page,request,settings)
        if result:
            result["recording_comparison"] = {"event":event,"fields":["unrecorded_final_selection"]}
        return result
    if not before or page["node"] != before["node"]:
        return {}
    old_choices = before["candidates"][:-1]
    # Paged decisions record indices instead of the underlying rows. Recover those
    # rows with their originating harness. A coincident index cannot identify an action.
    bound = saved_page(events,event,old_choices)
    if bound is not None:
        old_choices = bound
    elif old_choices and [c["value"] for c in old_choices] == list(range(len(old_choices))):
        original = source_library() if source_library is not None else None
        archive = SolvedReference(events, original, functions, settings) if original is not None else None
        if archive is None or archive.matched <= position:
            # The candidate input is grounded at the shared prefix, even if the old row ID was lost.
            if retrieval:
                result = new_route_ballot(op,page,request,settings,before)
                if result:
                    result["recording_comparison"] = {"event":event,"fields":["unavailable_recorded_binding"]}
                return result
            return {}
        old_choices = archive.menus[position][0]["choices"]
    policy_changed = different_rules(events,event,page)
    ballot = menu_ballot(before, old_choices, old_request, page, request, settings, force_live=policy_changed)
    if not ballot:
        return {}
    parent_page = parent.menus.get(position, (None,None))[0]
    matches = False
    if parent_page and parent_page["node"] == page["node"]:
        parent_request = parent.requests[position][2]
        matches = (parent_request == request and values(parent_page["choices"]) == values(page["choices"])
                   and all(parent_page[k] == page[k] for k in ("review","widen","minimum"))
                   and official_request(dict(parent_page),parent_request,settings) == official_request(dict(page),request,settings))
    fields = different_fields(old_request, request)
    if policy_changed:
        fields.append("decision_policy")
    if values(old_choices) != values(page["choices"]):
        fields.append("option_values")
    ballot["recording_comparison"] = {"event":event, "fields":fields, "candidate_menu_matches_parent":matches}
    return ballot


def first_change(events, parent, candidate, functions, settings, *, source_library=None):
    ballot = {}
    request_detail = {}
    retrieval = isinstance(candidate, dict) and bool(candidate.get("menu_retrieval"))

    def locate(before, old, after, new):
        if before == after == "call":
            request_detail["endpoints"] = [str(a.get("app")) + "." + str(a.get("api")) for a in (old,new)]
            request_detail["argument_fields"] = different_fields(old.get("arguments",{}),new.get("arguments",{}))
        elif before == after == "generate" and old.get("messages") != new.get("messages"):
            for left,right in zip(old.get("messages",()),new.get("messages",())):
                if left.get("role") == right.get("role") == "user" and left.get("content") != right.get("content"):
                    try:
                        a,b = (json.loads(m["content"]) for m in (left,right))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(a,dict) and isinstance(b,dict):
                        request_detail["generation_fields"] = different_fields(a,b)

    def compare_menu(position, args, fields, menu):
        previous, before = parent.menus[position]
        same_node = previous["node"] == menu["node"]
        old_values, new_values = values(previous["choices"]), values(menu["choices"])
        extra = [] if same_node else ["decision_node"]
        if old_values != new_values:
            extra.append("option_values")
        if not same_node and retrieval:
            ballot.update(new_route_ballot("decide",menu,args,settings,before))
        policy_changed = any(previous[k] != menu[k] for k in ("review","widen","minimum"))
        if policy_changed:
            extra.append("decision_policy")
        if not (fields or extra) or not before or not same_node:
            return fields + extra
        ballot.update(menu_ballot(before, previous["choices"], parent.requests[position][2], menu, args, settings,
                                  force_live=policy_changed))
        return fields + extra

    def inspect_boundary(request, page):
        ballot.update(recording_boundary_ballot(events, parent, page, request, functions, settings, source_library,
                                                retrieval=retrieval))

    def inspect_route(op, request, page):
        ballot.update(new_route_ballot(op,page,request,settings))

    path = compare_path(events, parent, candidate, functions, settings, executor_type=MenuReplay,
                        compare_menu=compare_menu, inspect_change=locate, inspect_boundary=inspect_boundary,
                        inspect_route=inspect_route if retrieval else None)
    if request_detail:
        path["first_change"].update(request_detail)
    if "recording_comparison" in ballot:
        path["recording_comparison"] = ballot["recording_comparison"]
    return path, ballot


def public_value(value):
    text = cleaned_text(json.dumps(value, ensure_ascii=False, default=str))
    return json.loads(text) if len(text) <= 350 else text[:350] + "…"


def public_choice(choice):
    return {key:public_value(choice[key]) for key in ("label","value")}


def saved_requests(folder):
    return [read(p) for p in sorted(folder.glob("historical_requests/*.json"))]


def fetch_choice(job, workers):
    path, request = job
    record = {"request":request, "status":"outcome_unconfirmed", "model_calls":0}
    try:
        model = workers.worker("model")
        record["model_calls"] = 1
        save(path, record)
        answer = model.request("decide", **request)
    except (WorkerError, WorkerUnavailable, OSError) as exc:
        record["error"] = cleaned_text(str(exc))[:240]
    else:
        record.update(status="received", answer={k:answer[k] for k in
                      ("index","probabilities","native_probabilities","thinking_used") if k in answer})
        record.update({k:answer[k] for k in USAGE[1:] if k in answer})
    save(path, record)
    return record


def task_observation(task, path, reference, ballot, cache):
    _, events, parent = reference
    row = {"task":task, "trace":path["trace"], "goal":cleaned_text(events[0]["instruction"])[:300]}
    change = path["first_change"]
    if change:
        row["first_change"] = change
    recording = ballot.get("recording_comparison")
    if "route_from" in ballot:
        row.update(previous_operation=ballot["route_from"], previous_node=ballot["previous_node"], node=ballot["node"])
        if recording:
            row["recording_comparison"] = recording
        req = ballot.get("request")
        if not req:
            return {**row, "status":"outcome_unconfirmed","reason":"candidate decision exceeds the recorded budget"}
        record = cache[request_key("decide",req)]
        if record["status"] != "received":
            return {**row, "status":"outcome_unconfirmed","reason":record.get("error","no reply")}
        outcome = ballot_outcome(record["answer"],ballot["choices"],{"value":None},ballot["page"])
        if outcome["status"] != "observed":
            return {**row,"status":outcome["status"],"source":"historical_27b"}
        status = "observed_without_old_binding" if recording and any(field in recording["fields"] for field in
                 ("unavailable_recorded_binding","unrecorded_final_selection")) else "new_decision"
        return {**row,"status":status,"source":"historical_27b",
                "candidate_choice":public_choice(outcome["candidate_choice"]), "confidence":outcome["confidence"]}
    if recording:
        row["recording_comparison"] = recording
    elif not parent.complete and path["matched"] >= parent.matched:
        if parent.matched < len(parent.requests):
            row["alignment_event"] = parent.requests[parent.matched][0]
        return {**row, "status":"unaligned", "reason":path["parent_coverage"]}
    elif not change:
        if path["status"] == "unresolved":
            return {**row, "status":"unaligned", "reason":path["parent_coverage"]}
        return {**row, "status":path["status"]}
    elif change["parent"] != "decide" or change["candidate"] != "decide":
        return {**row, "status":"changed_request"}
    if not ballot.get("request") and "recorded" not in ballot:
        return {**row, "status":"unaligned", "reason":"no comparable final Jev decision at the first change"}
    row["recorded_choice" if recording else "parent_choice"] = public_choice(ballot["parent_choice"])
    row["recorded_choice_offered" if recording else "parent_offered"] = ballot["parent_offered"]
    row.update(node=ballot["node"], question=cleaned_text(ballot["question"])[:320],
               added_values=[public_value(v) for v in ballot["added_values"]],
               added_options=[cleaned_text(s)[:220] for s in ballot["added_options"][:4]],
               removed_options=[cleaned_text(s)[:220] for s in ballot["removed_options"][:4]])
    if "recorded" in ballot:
        outcome = {"status":"observed", **ballot["recorded"]}
        row["source"] = "recorded_index"
    else:
        record = cache[request_key("decide", ballot["request"])]
        if record["status"] != "received":
            return {**row, "status":"outcome_unconfirmed", "reason":record.get("error","dispatch recorded without reply")}
        outcome = ballot_outcome(record["answer"], ballot["choices"], ballot["parent_choice"], ballot["page"])
        row["source"] = "historical_27b"
    if outcome["status"] == "observed":
        outcome["status"] = "same_first_choice" if outcome.pop("same_choice") else "changed_choice"
    if "candidate_choice" in outcome:
        outcome["candidate_choice"] = public_choice(outcome["candidate_choice"])
    return {**row, **outcome}


def review_solved(bench, old, ballots, folder):
    """Batch distinct requests across sessions, then bind each response on its own page."""
    stage = folder.name.removeprefix("trial_")
    document = "historical_choices/" + stage
    target = folder/"historical_choice.json"
    if target.is_file():
        report = read(target)
    else:
        earlier = saved_requests(folder.parent/"trial_draft") if stage == "revised" else []
        current = saved_requests(folder)
        cache = {request_key("decide", row["request"]):row for row in [*earlier,*current]}
        needed = {request_key("decide", b["request"]):b["request"] for b in ballots.values() if b.get("request")}
        missing = [req for key,req in needed.items() if key not in cache]
        last = max((int(p.stem) for p in folder.glob("historical_requests/*.json")), default=0)
        jobs = [(folder/"historical_requests"/f"{i:04d}.json", req) for i,req in enumerate(missing, last+1)]
        if jobs:
            for record in bench.study.execute_jobs(jobs, fetch_choice):
                cache[request_key("decide", record["request"])] = record
                current.append(record)
        rows = [task_observation(task,path,bench.reference_paths[task],ballots.get(task,{}),cache)
                for task,path in sorted(old.items())]
        summary = {"mode":MODE, "previous_train_tasks":len({s.split("/",1)[1] for s in bench.archived}),
                   "previously_solved_tasks":len(rows), "counts":dict(Counter(r["status"] for r in rows)),
                   "compared_to_recorded_menu":sum("recording_comparison" in r and r["status"] in
                                                   ("same_first_choice","changed_choice","policy_abstained") for r in rows),
                   "distinct_requests":len(needed), **{k:sum(x.get(k,0) for x in current) for k in USAGE},
                   "unconfirmed_calls":sum(x["model_calls"] for x in current if x["status"]=="outcome_unconfirmed"),
                   "details":document, "exceptions":[{k:v for k,v in r.items() if k not in
                       ("question","added_options","removed_options")} for r in rows
                       if r["status"] not in ("equal_on_record","same_first_choice")]}
        report = {"summary":summary, "tasks":rows}
        save(target, report)
    bench.docs[document] = [json.dumps(r, ensure_ascii=False) for r in report["tasks"]]
    return report["summary"]
