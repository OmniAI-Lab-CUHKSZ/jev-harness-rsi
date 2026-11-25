"""Probe one patch on a fixed failure and a preserved task, using the real executor."""
from collections import Counter
import json
from pathlib import Path

from formal_run import read,save

from .contrast_probes import FRESH_DECISION_LIMIT, compare_candidates
from .formal_optimizer import read_events
from .history_impact import assess, parameter_views, trace_sites
from .menu_preview import initial_change
from . import solved_train_preview
from .train_evidence import serialized
from .trace_comparison import ParentPath, compare_path


def probe_cases(draft, revised=False, protect_generation=False, *, spent=0, matched_preservation=False):
    available = max(0, FRESH_DECISION_LIMIT - spent) if revised else FRESH_DECISION_LIMIT // 2
    reserve = min(2, available) if draft.get("preserve_task") and not matched_preservation else 0
    cases = [{"trace":"current/"+draft["focus"]["task"],"anchor":"start","relation":"focus",
              "expect":draft["focus"]["observable"],"decisions":available-reserve,
              "cells":1 if revised or protect_generation else 2}]
    if draft.get("preserve_task"):
        cases.append({"trace":"current/"+draft["preserve_task"],"anchor":"start","relation":"preserve",
                      "expect":"Retain the appropriate path on the different request","decisions":reserve,
                      "cells":1 if protect_generation else 0})
    return cases


def exercised_objects(events):
    """Installed objects observed on this path; eligibility alone does not count as use."""
    used = set()
    for event in events:
        kind = event["event"]
        if kind=="card_executed":
            used.add(("card",event["card"]))
        elif kind=="handoff_selection":
            used.add(("handoff",event.get("guide")))
        elif kind=="assembly_selection":
            used.add(("assembly",event.get("family")))
        elif kind=="decision":
            used.add(("decision",event["node"].rsplit("/",1)[-1]))
            used.update(("state",k) for k in event.get("evidence",{}).get("views",{}))
        elif kind=="api":
            used.add(("adapter",event["app"]+"__"+event["api"]))
            if event.get("node")=="session" and event["api"]=="login":
                used.add(("adapter",event["app"]))
    return used


def preservation_case(bench, proposal, current, *, fixed=False):
    focus, suggested = proposal["focus"]["task"], proposal.get("preserve_task")
    edits = {(o["kind"],o["name"]) for o in proposal["objects"]}
    changed = {row["task"] for row in bench.index if row["success"] and row["task"] != focus
               and current[row["task"]]["status"]=="changed"}
    direct = {t:sorted("/".join(obj) for obj in edits & exercised_objects(bench.current_events[t])) for t in changed}
    candidates = {t for t in direct if direct[t]} or changed
    if candidates and not fixed:
        # Sol breaks ties when its proposed case is among the most exposed successes.
        selected = min(candidates, key=lambda t:(t != suggested,
                       t.rsplit("_",1)[0] == focus.rsplit("_",1)[0],t))
    else:
        selected = suggested
    return {"task":selected, "suggested_task":suggested,"changed_solved_tasks":sorted(changed),
            "exercised_edits":direct.get(selected,[])}


def compare_wording(lib, events, original, rewrite):
    """Change only a Train instruction at recorded parameter states; run the real candidate builders."""
    comparisons = []
    for state in trace_sites(events)["parameters"]:
        name = state["card"]
        if name not in lib["cards"]:
            continue
        before,after = (parameter_views(lib["cards"][name],state,request) for request in (original,rewrite))
        for slot in sorted(before.keys()|after.keys()):
            a,b = (v.get(slot,{}) for v in (before,after))
            row = {"event":state["index"],"card":name,"slot":slot}
            if "choices" in a and "choices" in b:
                left,right = ({json.dumps(c["value"],sort_keys=True,ensure_ascii=False) for c in v["choices"]} for v in (a,b))
                row.update(distinct_values=[len(left),len(right)],shared_values=len(left&right))
                if "menu_sizes" in a or "menu_sizes" in b:
                    row["menu_sizes"] = [v.get("menu_sizes",[len(v["choices"])]) for v in (a,b)]
                if slot in state["bound"]:
                    observed = json.dumps(state["bound"][slot],sort_keys=True,ensure_ascii=False)
                    row["observed_binding_offered"] = [observed in left,observed in right]
            else:
                row["uncaptured"] = [v.get("unavailable","slot not reached") for v in (a,b)]
            comparisons.append(row)
    return comparisons


def compare_train_paths(bench, proposal, candidate, *, solved_ballots=None):
    """Render current Train paths and the historical references selected by this experiment."""
    solved_mode = getattr(bench.study, "settings", {}).get("optimizer", {}).get("historical_choice") == solved_train_preview.MODE
    if not hasattr(bench,"reference_paths"):
        docs = {task:"current/"+task for task in bench.tasks}
        if solved_mode:
            docs.update(solved_train_preview.solved_documents(bench))
        else:
            for revision in reversed(bench.previous):
                for field in (("train_traces","parent_traces") if revision.get("accepted") else ("parent_traces","train_traces")):
                    for name in revision.get(field,[]):
                        path = Path(name)
                        task = path.parts[-3]
                        doc = path.parts[-4]+"/"+task
                        if doc in bench.paths:
                            docs.setdefault(task,doc)
            for doc in reversed(bench.archived):
                docs.setdefault(doc.split("/", 1)[1], doc)
        bench.reference_paths = {}
        for task,doc in sorted(docs.items()):
            events = bench.current_events[task] if task in bench.tasks else read_events(bench.paths[doc])
            ref_type = solved_train_preview.SolvedReference if solved_mode and task not in bench.tasks else ParentPath
            ref = ref_type(events,bench.lib,bench.study.functions,bench.study.execution)
            bench.reference_paths[task] = (doc,events,ref)

    def rerender(task,lib):
        doc,events,ref = bench.reference_paths[task]
        return {"trace":doc, **compare_path(events,ref,lib,bench.study.functions,bench.study.execution)}

    current = {task:rerender(task,candidate) for task in sorted(bench.tasks)}
    old = {}
    for task,(doc,events,ref) in bench.reference_paths.items():
        if task in current:
            continue
        if solved_mode:
            path, ballot = solved_train_preview.first_change(events,ref,candidate,bench.study.functions,bench.study.execution)
            old[task] = {"trace":doc, **path}
            if ballot and solved_ballots is not None:
                solved_ballots[task] = ballot
        else:
            old[task] = rerender(task,candidate)
    openings = Counter()
    for task,row in old.items():
        if row["status"] != "unresolved" or solved_mode:
            continue
        events = bench.reference_paths[task][1]
        opening = initial_change(bench.lib,candidate,events,bench.study.functions)
        row["conditional_opening"] = opening
        openings["tasks"] += 1
        openings["family_menus_rendered"] += int(opening["family_menu_rendered"])
        openings["changed_family_menus"] += bool(opening["changed_fields"]["fragment/assembly"])
        openings["comparable_action_menus"] += int(opening["action_menu_comparable"])
        openings["changed_action_menus"] += bool(opening["changed_fields"]["fragment/step"])
    focal = proposal["focus"]["task"]
    edits = {}
    for obj in proposal["objects"]:
        label = obj["kind"]+"/"+obj["name"]
        if len(proposal["objects"]) == 1:
            edits[label] = current[focal]
            continue
        try:
            single,_ = bench.install(bench.lib,{"objects":[obj]},bench.sources,bench.study.functions)
        except ValueError as exc:
            edits[label] = {"status":"depends_on_other_edit","reason":str(exc)}
            continue
        edits[label] = rerender(focal,single)
    report = {"focus":current[focal],"individual_edits_at_focus":edits,
              "historical_paths":dict(Counter(row["status"] for row in old.values())),
              "conditional_openings":dict(openings),
              "interpretation":"First differing request on the recorded trajectory. Separate edits are tested alone; later effects and new task outcomes require execution."}
    paths_by_doc = {row["trace"]:row for row in (*current.values(), *old.values())}
    comparisons = {}
    def historical_path(doc):
        if doc in comparisons:
            return comparisons[doc]
        # A previous version of this task may have succeeded by a different route.
        # Replaying the most recent record for that task would test the wrong example.
        if doc in paths_by_doc:
            result = dict(paths_by_doc[doc])
        elif solved_mode:
            return {"status":"source_only","reason":"Outside the selected first-change preservation references."}
        else:
            events = read_events(bench.paths[doc])
            ref = ParentPath(events, bench.lib, bench.study.functions, bench.study.execution)
            result = compare_path(events, ref, candidate, bench.study.functions, bench.study.execution)
            if result["status"] == "unresolved":
                result["conditional_opening"] = initial_change(bench.lib, candidate, events, bench.study.functions)
        result["recorded_exposure"] = assess(bench.trace_data(doc), bench.lib, candidate, proposal["objects"])["exposures"]
        comparisons[doc] = result
        return result

    related = bench.precedents.related(focal, proposal["focus"]["event"])
    for match in related["matches"]:
        match["replay"] = historical_path(match["trace"])
    report["focus_history"] = related
    cited = []
    for source in proposal.get("history_use", []):
        doc = source["trace"]
        record = bench.precedents.records[doc]
        cited.append({"trace":doc, "cited_event":source["event"], "past_task_success":record["task_success"],
                      "past_trace_output_tokens":record["trace_output_tokens"], "replay":historical_path(doc)})
    report["cited_train_paths"] = cited
    return current, old, report


def preview_batch(bench,proposal,worker,key,prior_cost=None):
    candidate,_ = bench.install(bench.lib,proposal,bench.sources,bench.study.functions)
    choice_mode = bench.study.settings.get("optimizer",{}).get("historical_choice")
    ballots = {}
    current,old,paths = compare_train_paths(bench,proposal,candidate,solved_ballots=ballots)
    batch = [{"task":item["task"],"parent_success":item["success"],"parent_tokens":item["output_tokens"],
              **current[item["task"]]} for item in bench.index]
    folder = bench.folder/("trial_"+key)
    saved_spec = folder/"tested_patch.json"
    resuming = saved_spec.is_file()
    case_proposal = proposal
    if resuming:
        kept = next((case for case in read(saved_spec)["cases"] if case["relation"]=="preserve"),None)
        case_proposal = {**proposal,"preserve_task":kept["trace"].split("/",1)[1] if kept else None}
    preservation = preservation_case(bench,case_proposal,current,fixed=key!="draft" or resuming)
    preservation["suggested_task"] = proposal.get("preserve_task")
    selected = preservation["task"]
    protect = bool(kept and kept["cells"]) if resuming else selected in preservation["changed_solved_tasks"]
    cases = probe_cases({**proposal,"preserve_task":selected},revised=prior_cost is not None,
                        protect_generation=protect,spent=(prior_cost or {}).get("fresh_decisions",0),
                        matched_preservation=current.get(selected,{}).get("status")=="equal_on_record")
    report = compare_candidates(bench,{"alternatives":[{**proposal,"id":key}],
                                      "cases":cases},
                                worker,folder=folder,prior_cost=prior_cost,matched_paths=current,
                                history_docs=[r["trace"] for r in old.values()] if choice_mode==solved_train_preview.MODE else None)
    if choice_mode == solved_train_preview.MODE:
        report["historical_choice"] = solved_train_preview.review_solved(bench, old, ballots, folder)
    elif key == "draft" and choice_mode:
        from .historical_choice import historical_choice
        report["historical_choice"] = historical_choice(bench,proposal,candidate,old,paths,worker,
                                                        folder/"historical_choice.json")
    save(folder/"path_comparison.json",{"batch":batch,"history":old,**paths})
    documents = []
    if choice_mode == solved_train_preview.MODE:
        documents.append(report["historical_choice"]["details"])
    for i,case in enumerate(report["cases"],1):
        name = f"trial/{key}/{case['relation']}"
        trace = folder/f"case_{i}_alt_1"/"trace.jsonl"
        if trace.exists():
            bench.paths[name] = trace
            documents.append(name)
    impact_doc = "impact/"+key
    bench.docs[impact_doc] = [serialized(row) for row in read(folder/"historical_1.json")["details"]]
    path_doc = "paths/"+key
    bench.docs[path_doc] = [serialized({"task":t,**r}) for t,r in {**current,**old}.items()]
    bench.docs["trace_directory"] = sorted(bench.paths)
    wording = None
    check = proposal.get("wording_check")
    rewrite = check.get("request") if isinstance(check,dict) else check
    if isinstance(rewrite,str) and rewrite.strip():
        task = proposal["focus"]["task"]
        focal_trace = folder/"case_1_alt_1"/"trace.jsonl"
        events = read_events(focal_trace) if focal_trace.exists() else bench.current_events[task]
        original = bench.current_events[task][0]["instruction"]
        wording = {"rewrite":rewrite,"relation":check.get("scope") if isinstance(check,dict) else "whole_task",
                   "scope":"Sol-authored Train wording, same recorded state; no model choice or grade.",
                   "bindings":compare_wording(candidate,events,original,rewrite)}
        alternate = check.get("alternate") if isinstance(check,dict) else None
        if isinstance(alternate,str) and alternate.strip() and alternate != rewrite:
            wording["alternate"] = {"rewrite":alternate,"bindings":compare_wording(candidate,events,original,alternate)}
        save(folder/"wording.json",wording)
    return {**report,"batch":batch,"path_comparison":paths,"wording":wording,"preservation":preservation,
            "documents":[*documents,impact_doc,path_doc]}
