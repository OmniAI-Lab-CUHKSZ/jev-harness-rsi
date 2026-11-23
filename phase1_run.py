"""Run the Phase 1 Train revisions, Dev selection and paired Test-N evaluation."""
from pathlib import Path

from formal_run import Study, final_report, quality, read, save
from jev_harness.formal_v2 import gain
from jev_harness.metrics import aggregate
from jev_harness.semantic_assembly import SemanticAssembly
from jev_harness.semantic_library import PROTOCOL

RESOURCES = Path(__file__).resolve().parent / "jev_harness" / "resources"
MAX_REVISIONS = 2
BATCHES = [row["scenarios"] for row in read(RESOURCES / "task_split.json")["train_batches"]]


def batch_ids(groups):
    return [f"{group}_{i}" for i in (1, 2, 3) for group in groups]


def design_manifest(protocol=PROTOCOL):
    manifest = read(RESOURCES / "task_split.json")
    manifest["protocol"] = protocol
    return manifest


class PhaseOneStudy(Study):
    compiled_executor = SemanticAssembly

    def record_public_catalog(self, stage):
        return not self.test_started and stage.startswith("b") and stage[1:3].isdigit()

    def run_task(self, *args, **kwargs):
        row = super().run_task(*args, **kwargs)
        from jev_harness.formal_v2 import include_retries
        return include_retries(self.root, row)


def summary(rows):
    fragment_scenarios = {}
    for row in rows:
        for fragment in row.get("fragments",[]):
            fragment_scenarios.setdefault(fragment,set()).add(row["task_id"].rsplit("_",1)[0])
    uses_g = lambda r: any(n for m,n in r.get("model_calls",{}).items() if m.startswith("G") or m=="R")
    ran_fresh = lambda r: sum(not e["reused"] for e in r.get("fragment_selections",[]))
    result = {**aggregate(rows), "fragment_D_calls":sum(r.get("fragment_decisions",0) for r in rows),
            "fragment_D_used":sum(ran_fresh(r) for r in rows),
            "fragment_D_cache_hits":sum(e["reused"] for r in rows for e in r.get("fragment_selections",[])),
            "object_D_used":sum(len(r.get("consumed_native_choices",[])) for r in rows),
            "tasks_using_fragment_D":sum(bool(ran_fresh(r)) for r in rows),
            "tasks_using_object_D":sum(bool(r.get("consumed_native_choices")) for r in rows),
            "tasks_without_generation":sum(not uses_g(r) for r in rows),
            "tasks_program_only":sum(not uses_g(r) and not r.get("model_calls",{}).get("D",0) for r in rows),
            "tasks_with_partial_generation":sum(uses_g(r) and bool(r.get("fragments")) for r in rows),
            "tasks_using_generation_only":sum(uses_g(r) and not r.get("fragments") and not r.get("model_calls",{}).get("D",0) for r in rows),
            "tasks_with_D_then_uncovered_generation":sum(uses_g(r) and not r.get("fragments") and bool(r.get("model_calls",{}).get("D",0)) for r in rows),
            "fragment_scenarios":{k:sorted(v) for k,v in fragment_scenarios.items()},
            "G_calls":sum(n for r in rows for m,n in r.get("model_calls",{}).items() if m.startswith("G") or m=="R"),
            "online_question_calls":sum(r.get("question_generation_calls",0) for r in rows),
            "online_candidate_calls":sum(r.get("candidate_generation_calls",0) for r in rows)}
    if any("step_review" in row for row in rows):
        result["step_review"] = {k:sum(r.get("step_review", {}).get(k,0) for r in rows)
                                 for k in ("calls","output_tokens","reasoning_tokens","kept","changed","invalid")}
        result["tasks_without_open_generation"] = result["tasks_without_generation"]
        result["tasks_without_generation"] = sum(not r["output_tokens"] for r in rows)
        result["open_generation_output_tokens"] = result["total_output_tokens"] - result["step_review"]["output_tokens"]
    if any('semantic_handoff' in row for row in rows):
        result['semantic_handoff'] = {k:sum(r.get('semantic_handoff',{}).get(k,0) for r in rows)
                                      for k in ('handoffs','eligible','selected','calls','input_tokens','output_tokens','model_seconds')}
        scenarios = {}
        for row in rows:
            for name in row.get('semantic_handoff',{}).get('guides',[]):
                scenarios.setdefault(name,set()).add(row['task_id'].rsplit('_',1)[0])
        result['semantic_handoff']['scenarios_per_guide'] = {name:sorted(groups) for name,groups in scenarios.items()}
    if any("card_menus" in r for r in rows):
        successors={}
        for row in rows:
            path=row.get("card_path",[])
            for left,right in zip(path,path[1:]):
                if "G" not in (left,right) and not left.startswith("submit_") and not right.startswith("submit_"):
                    successors.setdefault(left,set()).add(right)
        result["cards"]={"native_calls":sum(r.get("model_calls",{}).get("D",0) for r in rows),
            "chosen_parameters":sum(r.get("parameter_choices",0) for r in rows),
            "menus":sum(r.get("card_menus",0) for r in rows),
            "tasks_using_cards":sum(bool(r.get("fragments")) for r in rows),
            "tasks_using_cards_and_generation":sum(bool(r.get("fragments")) and uses_g(r) for r in rows),
            "tasks_with_no_card_and_generation":sum(not r.get("fragments") and uses_g(r) for r in rows),
            "tasks_done_with_cards_only":sum(r["success"] and bool(r.get("fragments")) and not uses_g(r) for r in rows),
            "public_read_reentries":sum(r.get("public_read_reentries",0) for r in rows),
            "reused_in_multiple_scenarios":sorted(k for k,v in fragment_scenarios.items()
                                                    if len(v)>1 and not k.startswith("submit_")),
            "multiple_observed_successors":{k:sorted(v) for k,v in successors.items() if len(v)>1}}
        # Object metrics from the staged whole-operation selector have no equivalent in this executor.
        for key in ("object_D_used","tasks_using_object_D","fragment_D_cache_hits"):
            result.pop(key)
        if any("assembly_decisions" in row for row in rows):
            result["assembly"] = {key:sum(row.get(key,0) for row in rows) for key in
                                  ("assembly_decisions","assembly_focused_menus","assembly_widens")}
    if any("staged" in r for r in rows):
        stages=[r["staged"] for r in rows if "staged" in r]
        outcome=sorted({s["complete"] for s in stages})
        cost_names=sorted({k for s in stages for k in s["cost_by_stage"]})
        paths={}
        for row in rows:
            stage=row.get("staged")
            if stage is None:
                continue
            if stage["complete"]=="completed":
                path="complete_operation"
            elif uses_g(row):
                path="partial_then_generation" if stage["local_fragments"] else "generation_without_selected_partial"
            else:
                path="stopped_before_generation"
            group=paths.setdefault(path,{"tasks":0,"successes":0,"output_tokens":0})
            group["tasks"]+=1
            group["successes"]+=int(row["success"])
            group["output_tokens"]+=row["output_tokens"]
        result["staged"]={"complete_outcomes":{k:sum(s["complete"]==k for s in stages) for k in outcome},
            "execution_paths":paths,
            "tasks_with_executed_local_choice":sum(bool(s["local_fragments"]) for s in stages),
            "tasks_reaching_generation":sum(s["handoff"] for s in stages),
            "tasks_with_cross_scenario_choice":sum(bool(s.get("cross_scenario_choices")) for s in stages),
            "complete_menu":{k:sum(s.get("entry_cost",{}).get(k,0) for s in stages) for k in ("calls","input_tokens")},
            "by_stage":{k:{m:sum(s["cost_by_stage"].get(k,{}).get(m,0) for s in stages)
                           for m in ("calls","input_tokens","output_tokens")} for k in cost_names}}
    return result


def paired_holdout(study, ids, version, library):
    """An arm pair shares a worker, not simulator state. Alternate order across the panel."""
    stages = {"A": "test_generation", "B": "test_" + version}
    cached = {}
    if hasattr(study, "root"):
        for task in ids:
            cached[task] = {}
            for arm, stage in stages.items():
                path = study.root / "runs" / stage / task / "record.json"
                if path.is_file():
                    cached[task][arm] = read(path).get("gpu")

    def pair(job, worker):
        i,task=job
        out={}
        previous = cached.get(task, {})
        complete_pair = len(previous) == 2 and previous["A"] == previous["B"]
        device = getattr(worker, "gpu", None)
        for arm in (("A","B") if i%2==0 else ("B","A")):
            stage = stages[arm]
            if device and arm in previous and previous[arm] != device and not complete_pair:
                stage += "_paired_" + device.replace(":", "_")
            out[arm]=study.run_task(task,stage,arm,None if arm=="A" else library,workers=worker)
        return out["A"],out["B"]
    jobs = list(enumerate(ids))
    if hasattr(study, "execute_preferred"):
        preferred = [next(iter(cached.get(task, {}).values()), None) for task in ids]
        pairs = study.execute_preferred(jobs, pair, preferred if any(preferred) else None)
    else:
        pairs = study.execute_jobs(jobs, pair)
    return [a for a,b in pairs],[b for a,b in pairs]


def run(study, count, rounds, formal, *, revisit_after_six=(), evaluate_test=True, reference_dev=None):
    """A callable reference_dev scores the retained parent when a candidate first reaches Dev."""
    if not 1 <= rounds <= MAX_REVISIONS:
        raise ValueError(f"Phase one allows at most {MAX_REVISIONS} revisions per batch")
    root = study.root
    first = root/"versions"/"h00.json"
    from jev_harness import card_library, semantic_library
    protocol = study.settings.get("protocol", PROTOCOL)
    is_cards = protocol in (card_library.PROTOCOL, semantic_library.PROTOCOL)
    semantic = protocol == semantic_library.PROTOCOL
    if semantic:
        install = semantic_library.patch_library
    elif is_cards:
        install = card_library.patch_cards
    else:
        raise ValueError(f"Unknown fragment harness protocol: {protocol}")
    if not first.exists():
        if semantic:
            seed_library = (read(RESOURCES / "h00.json") if study.settings.get("semantic_handoffs")
                            else semantic_library.initial_library())
        else:
            seed_library = card_library.initial_cards()
        save(first, seed_library)
    checkpoint_file = root/"checkpoints.json"
    checkpoints = read(checkpoint_file) if checkpoint_file.is_file() else []
    version = checkpoints[-1]["next_version"] if checkpoints else "h00"
    lib = read(root/"versions"/(version+".json"))
    visits = []
    for batch in range(1, count + 1):
        label = "b07_after_revisits" if batch == 7 and revisit_after_six else f"b{batch:02d}"
        visits.append((batch, label))
        if batch == 6:
            visits.extend((source, f"b06_revisit{source:02d}") for source in revisit_after_six)
    completed = [item.get("visit", f"b{item['batch']:02d}") for item in checkpoints]
    if completed != [label for _, label in visits[:len(completed)]]:
        raise ValueError("The saved batch visits are not a prefix of the requested schedule")
    history = []
    for label, checkpoint in zip(completed, checkpoints):
        for rnd in range(1, checkpoint["revisions_scored"]+1):
            history.append(read(root/"batches"/label/f"{label}r{rnd:02d}"/"result.json"))
    manifest=design_manifest(protocol if is_cards else PROTOCOL)
    if revisit_after_six:
        manifest["train_visits"] = [{"visit": label, "source_batch": batch} for batch, label in visits]
    save(root/"task_split.json",manifest)
    seed_sources=set(manifest["historical_seed_train_ids"])
    sources=seed_sources.copy()
    dev=manifest["dev_tasks"] if formal else []
    def score_dev(ver, library):
        return study.batch(dev, "dev_"+ver, library=library)
    if formal and reference_dev is None:
        if study.settings.get('optimizer',{}).get('history_preview'):
            reference_dev = lambda current=version, library=lib: score_dev(current,library)
        else:
            reference_dev = score_dev(version, lib)
    for position, (batch, label) in enumerate(visits):
        ids = batch_ids(BATCHES[batch - 1])
        sources.update(ids)
        if position < len(completed):
            continue
        if hasattr(study, "begin_inference_phase"):
            study.begin_inference_phase("train", batch)
        entry_ver, entry_lib = version, lib
        entry_stage = label+"_"+version+"_entry"
        continuation = study.settings.get("execution_from_next_batch", {})
        if batch == continuation.get("from_batch"):
            # Isolate the first new-policy Train attempt when taking over a live controller.
            threshold = str(continuation["step_review_threshold"]).replace(".", "p")
            entry_stage = f"{label}_{version}_review_{threshold}_entry"
        rows = study.batch(ids, entry_stage, library=lib)
        entrance = summary(rows)
        entrance_by_seed={name:summary(part) for name,part in (
            ("historical_seed",[r for r in rows if r["task_id"] in seed_sources]),
            ("outside_seed",[r for r in rows if r["task_id"] not in seed_sources])) if part}
        revisions_scored = 0
        save(root/"batches"/label/"entry.json", {"version":version, **entrance})
        from jev_harness.phase1_optimizer import propose_one
        for rnd in range(1, rounds+1):
            if quality(rows) == (len(ids),0):
                break
            candidate_version = f"{label}r{rnd:02d}"
            folder = root/"batches"/label/candidate_version
            result_file = folder/"result.json"
            if result_file.is_file():
                # The recorded pair can come from GPUs outside today's worker pool.
                record = read(result_file)
                history.append(record)
                revisions_scored += 1
                if record["accepted"]:
                    version = candidate_version
                    lib = read(root/"versions"/(version+".json"))
                traces = record["train_traces" if record["accepted"] else "parent_traces"]
                rows = [read((root/trace.replace("\\", "/")).parents[1]/"record.json") for trace in traces]
                continue
            prior_ver = version
            proposal_file = folder/"proposal.json"
            study.status("offline_sol", batch=batch, visit=label, round=rnd, parent=version)
            if not proposal_file.exists():
                proposal = propose_one(study, lib, rows, history, folder, sources)
                save(proposal_file, proposal)
            proposal = read(proposal_file)
            candidate, edits = install(lib, proposal, sources, study.functions)
            save(root/"versions"/(candidate_version+".json"), candidate)
            parent_rows, candidate_rows = study.candidate_batch(ids, candidate_version, candidate, rows, lib)
            comparison = gain(parent_rows, candidate_rows)
            keep_tie = comparison["tied"] and (study.settings.get("optimizer", {}).get("fragment_trials") is True
                                               or bool(revisit_after_six) and position >= 6)
            accepted = quality(candidate_rows) > quality(parent_rows) or keep_tie
            record = {"batch":batch, "visit":label, "round":rnd, "parent":prior_ver, "candidate":candidate_version,
                      "edits":edits, "prediction":proposal.get("prediction"), "diagnosis":proposal.get("reason"),
                      "accepted":accepted, "comparison":comparison,
                      "parent_metrics":summary(parent_rows), "candidate_metrics":summary(candidate_rows),
                      "train_traces":[r["trace_path"] for r in candidate_rows],"parent_traces":[r["trace_path"] for r in parent_rows]}
            impact_file = folder/'optimizer_impact.json'
            if study.settings.get('optimizer',{}).get('history_preview') and impact_file.exists():
                record['historical_impact'] = read(impact_file)
            if study.settings.get('optimizer',{}).get('history_preview') and proposal.get('capability_gap'):
                record['capability_gap'] = proposal['capability_gap']
            save(result_file, record)
            history.append(record)
            revisions_scored += 1
            rows = parent_rows
            if accepted:
                version, lib, rows = candidate_version, candidate, candidate_rows
            study.status("revision_scored", batch=batch, visit=label, round=rnd, accepted=accepted, version=version,
                         successes=summary(rows)["successes"], tokens=summary(rows)["total_output_tokens"])
        train_version = version
        dev_choice=None
        if formal and version != entry_ver:
            if callable(reference_dev):
                reference_dev = reference_dev()
            ev = score_dev(version, lib)
            reference_score, evaluated_score = quality(reference_dev), quality(ev)
            carry=evaluated_score >= reference_score
            dev_choice={"compared_to":entry_ver,"candidate":version,"reference":reference_score,
                        "evaluated":evaluated_score,"carry":carry,"tied":evaluated_score == reference_score}
            if study.settings.get("inference_schedule"):
                dev_choice["reference_lanes_per_gpu"] = sorted({r.get("lanes_per_gpu", 1) for r in reference_dev})
                dev_choice["candidate_lanes_per_gpu"] = sorted({r.get("lanes_per_gpu", 1) for r in ev})
            if carry:
                reference_dev = ev
            else:
                version, lib = entry_ver, entry_lib
        checkpoints.append({"batch":batch, "visit":label, "entry_version":entry_ver, "train_version":train_version,
                            "next_version":version, "revisions_scored":revisions_scored,
                            "entry":entrance, "entry_by_seed":entrance_by_seed,
                            "train":summary(rows),"dev_selection":dev_choice})
        save(checkpoint_file, checkpoints)
    if formal and evaluate_test:
        if study.settings.get("adaptive_decision") and callable(reference_dev):
            reference_dev = reference_dev()
        study.close_account()
        study.test_started = True
        if hasattr(study, "begin_inference_phase"):
            study.begin_inference_phase("test")
        ids = manifest["normal_holdout_tasks"]
        a,b = paired_holdout(study,ids,version,lib)
        report=final_report(a,b,study.seed)
        report["arms"]={"A":"unchanged 27B generation agent","B":"same 27B with Sol-evolved " +
                        ("Jev-assembled menus and six-component harness" if semantic else
                         "fragment-card harness" if is_cards else "staged harness")}
        usages=[read(p) for p in root.glob("batches/b*/b*r*/sol*/turn_*/usage.json")]
        if study.settings.get('optimizer',{}).get('history_preview'):
            authors = [read(p) for p in root.glob('batches/b*/b*r*/sol*/probe_*/wording_author_*/usage.json')]
            usages.extend(authors)
            report['offline_wording_author'] = {'calls':len(authors), **{k:sum(u.get(k,0) for u in authors)
                for k in ('inputTokens','outputTokens','reasoningOutputTokens','seconds')}}
            previews = [read(p)['summary']['local_native_cost'] for p in root.glob('batches/b*/b*r*/sol*/probe_*/historical_impact.json')]
            report['offline_historical_checks'] = {k:sum(u.get(k,0) for u in previews) for k in ('calls','input_tokens','output_tokens')}
        report["offline_sol_usage"]={"turns_with_usage":len(usages),
                                    **{k:sum(u.get(k,0) for u in usages) for k in
                                       ("inputTokens","outputTokens","reasoningOutputTokens","seconds")}}
        report["paired_tasks"]=[{"task":x["task_id"],"A":{k:x.get(k) for k in ("success","output_tokens","gpu","trace_path")},
                                 "B":{k:y.get(k) for k in ("success","output_tokens","gpu","trace_path")}} for x,y in zip(a,b)]
        save(root/"test_report.json",report)
        save(root/"test_mechanisms.json", summary(b))
    save(root/"result.json", {"batches":checkpoints,"final_version":version,"formal":formal})
    study.status("completed", batches=count, train_visits=len(visits), rounds=rounds, version=version, formal=formal)
