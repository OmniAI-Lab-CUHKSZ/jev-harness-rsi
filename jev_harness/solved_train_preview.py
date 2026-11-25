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


def solved_documents(bench):
    """Use scored Train provenance, including confirmations. Reusing a parent is not a new run."""
    records = bench.precedents.records
    eligible = {doc for doc in bench.archived if records[doc]["task_success"] is True
                and records[doc]["task"] not in bench.tasks}
    visits = {}
    def document(path):
        return "/".join(Path(str(path).replace("\\", "/")).parts[-4:-2])

    for path in getattr(bench.study, "prior_train_entries", ()):
        stage = document(path).split("/",1)[0]
        label = re.match(r"^b\d+(?:_revisit\d+)?", stage)
        visits.setdefault(label.group() if label else stage, []).append(path)
    for revision in bench.previous:
        visit = revision.get("visit") or f"b{revision.get('batch',0):02d}"
        paths = visits.setdefault(visit, [])
        paths.extend(revision.get("parent_traces", ()))
        paths.extend(revision.get("train_traces", ()))
        for pair in revision.get("paired_selection", {}).get("repeats", {}).values():
            paths.extend(row["trace_path"] for row in pair)
    ordered = (document(p) for paths in visits.values() for p in paths)
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


class SolvedReference(ParentPath):
    def __init__(self, events, library, functions, settings):
        self.events, self.menus = events, {}
        super().__init__(events, library, functions, settings, executor_type=MenuReplay)

    def request(self, op, **args):
        position = self.matched
        result = super().request(op, **args)
        if op == "decide":
            before = recorded_decision(self.events, self.tape[position][0])
            if before and (before["node"] != self.menu["node"] or
                           values(before["candidates"][:-1]) != values(self.menu["raw"])):
                self.matched = position
                self.reason = "parent decision bindings differ from recording"
                raise Boundary
            self.menus[position] = (self.menu, before)
        return result


def first_change(events, parent, candidate, functions, settings):
    ballot = {}
    request_detail = {}

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
        if not (fields or extra) or not before or not same_node:
            return fields + extra
        none = before["candidates"][-1]
        choices = [*menu["choices"], none]
        index = before["selected"]
        prior = [*previous["choices"], none][index]
        ballot.update(node=menu["node"], question=args["question"], parent_choice=prior,
                      parent_offered=values([prior])[0] in values(choices), choices=choices, page=menu,
                      added_values=[c["value"] for c,v in zip(menu["choices"],new_values) if v not in old_values][:4],
                      added_options=[s for s in args["options"] if s not in parent.requests[position][2]["options"]],
                      removed_options=[s for s in parent.requests[position][2]["options"] if s not in args["options"]])
        if fields:
            ballot["request"] = official_request(menu, args, settings)
        else:
            # Same visible input and recorded decision. Bind that exact index in the new menu.
            ballot["recorded"] = dict(candidate_choice=choices[index], same_choice=values([choices[index]])==values([prior]))
        return fields + extra

    path = compare_path(events, parent, candidate, functions, settings,
                        executor_type=MenuReplay, compare_menu=compare_menu, inspect_change=locate)
    if request_detail:
        path["first_change"].update(request_detail)
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
    if not parent.complete and path["matched"] >= parent.matched:
        if parent.matched < len(parent.requests):
            row["alignment_event"] = parent.requests[parent.matched][0]
        return {**row, "status":"unaligned", "reason":path["parent_coverage"]}
    if not change:
        if path["status"] == "unresolved":
            return {**row, "status":"unaligned", "reason":path["parent_coverage"]}
        return {**row, "status":path["status"]}
    if change["parent"] != "decide" or change["candidate"] != "decide":
        return {**row, "status":"changed_request"}
    if not ballot.get("request") and "recorded" not in ballot:
        return {**row, "status":"unaligned", "reason":"no comparable final Jev decision at the first change"}
    row.update(node=ballot["node"], question=cleaned_text(ballot["question"])[:320],
               parent_choice=public_choice(ballot["parent_choice"]), parent_offered=ballot["parent_offered"],
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
                   "distinct_requests":len(needed), **{k:sum(x.get(k,0) for x in current) for k in USAGE},
                   "unconfirmed_calls":sum(x["model_calls"] for x in current if x["status"]=="outcome_unconfirmed"),
                   "details":document, "exceptions":[{k:v for k,v in r.items() if k not in
                       ("question","added_options","removed_options")} for r in rows
                       if r["status"] not in ("equal_on_record","same_first_choice")]}
        report = {"summary":summary, "tasks":rows}
        save(target, report)
    bench.docs[document] = [json.dumps(r, ensure_ascii=False) for r in report["tasks"]]
    return report["summary"]
