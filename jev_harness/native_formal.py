"""Cold-start native-tool experiment: two ordered Dev measurements, Train evolution, paired Test-N."""

from pathlib import Path

from .allocation import SharedStudy
from .artifacts import include_retries, read, save
from .composed_assembly import initial_library
from .metrics import aggregate, final_report
from .native_assembly import blank_library
from .protocol import PROTOCOL, design_manifest
from .native_revision import DRAFT_REPLY_LIMIT, EditRound, TrainSample, revise
from .paired_selection import TWO_REPEATS_ONLY, paired_rollouts, review_versions


def configuration(shared):
    optimizer = {k: shared["optimizer"][k] for k in ("model", "reasoning_effort", "rounds", "max_objects")}
    optimizer["draft_reply_limit"] = DRAFT_REPLY_LIMIT
    return {**{k: shared[k] for k in ("executor", "model_path", "seed", "execution")},
            "protocol": PROTOCOL, "cuda_graph_from_start": True, "lanes_per_gpu": 2,
            "train_sampling": "entry_harness_only", "optimizer": optimizer,
            "confirmation_scoring": TWO_REPEATS_ONLY}


def measure(study, ids, stage, arm, library, reference=()):
    paths = [study.root / "runs" / stage / task / "record.json" for task in ids]
    if all(path.is_file() for path in paths):
        return [include_retries(study.root, read(path)) for path in paths]
    affinity = {r["task_id"]: r["gpu"] for r in reference}
    rows = study.execute_preferred(ids, lambda task, worker: study.run_task(task, stage, arm, library, workers=worker),
                                   [affinity.get(task) for task in ids] if affinity else None)
    save(study.root / "runs" / stage / "aggregate.json", panel_metrics(rows))
    return rows


def panel_metrics(rows):
    return {**aggregate(rows),
            **{k: sum(r[k] for r in rows) for k in ("input_tokens", "open_output_tokens", "decision_output_tokens",
                                                    "open_calls", "native_ballots", "api_calls")},
            "retry_output_tokens": sum(r.get("retry_output_tokens", 0) for r in rows),
            "tasks_without_open_generation": sum(r["open_calls"] == 0 for r in rows)}


def train_samples(root, rows, library, label):
    return {label + "/" + r["task_id"]: TrainSample(root / r["trace_path"], r, library) for r in rows}


def confirmed_train_samples(root, path, parent, child):
    samples = {}
    for task, runs in read(path)["repeats"].items():
        for run in runs:
            trace = root / run["trace_path"]
            row = read(trace.parents[1] / "record.json")
            arm = run["arm"]
            name = f"{path.parent.name}_confirmation{run['confirmation_pass']}_{arm}/{task}"
            samples[name] = TrainSample(trace, row, parent if arm == "parent" else child)
    return samples


def selection(study, stage, folder, parent_rows, child_rows, parent, child):
    path = folder / "paired_selection.json"
    saved = read(path) if path.is_file() else {}
    if "accepted" not in saved:
        saved = review_versions(study, stage, path, parent_rows, child_rows, parent, child)
    return {k: saved[k] for k in ("accepted", "parent_score", "candidate_score", "accuracy_review_tasks",
                                  "cost_review_tasks", "repeat_rollouts", "repeat_output_tokens")}


def optimization_costs(root, reviews):
    usage = [read(p) for p in (root / "offline").glob("b*r*/sol/*/usage.json")]
    drafts = [read(p) for p in (root / "runs").glob("b*r*_draft_train_*/*/record.json")]
    choices = []
    for path in (root / "offline").glob("b*r*/draft_preview.json"):
        preview = read(path)
        for group in ("current_train", "previously_solved_train"):
            choices.extend(r for r in preview[group].values() if r["status"] == "reselected")
    return {"sol": {"turns": len(usage), **{k: sum(r.get(k, 0) for r in usage)
                                           for k in ("inputTokens", "outputTokens", "reasoningOutputTokens", "seconds")}},
            "draft_episodes": len(drafts), "draft_output_tokens": sum(r["output_tokens"] for r in drafts),
            "preview_reselections": len(choices), "preview_input_tokens": sum(r["input_tokens"] for r in choices),
            "preview_output_tokens": sum(r["output_tokens"] for r in choices),
            "confirmation_episodes": sum(r["repeat_rollouts"] for r in reviews),
            "confirmation_output_tokens": sum(r["repeat_output_tokens"] for r in reviews)}


def run_formal(study, manifest, through):
    root = study.root
    h0_path = root / "versions/h00.json"
    h0 = read(h0_path) if h0_path.is_file() else initial_library()
    save(h0_path, h0)
    react_library = blank_library()
    save(root / "task_split.json", manifest)
    dev = manifest["dev_tasks"]
    study.status("dev_react")
    react = measure(study, dev, "dev_react", "A", react_library)
    if through == "react-dev":
        study.status("react_dev_complete", tasks=len(react))
        return
    study.status("dev_h00")
    dev_reference = measure(study, dev, "dev_h00", "B", h0, react)
    fields = ("success", "output_tokens", "open_calls", "native_ballots", "trace_path")
    save(root / "initial_dev.json", {"react": panel_metrics(react), "h00": panel_metrics(dev_reference),
                                    "paired": {a["task_id"]: {"react": {k: a[k] for k in fields},
                                                               "h00": {k: b[k] for k in fields}}
                                               for a, b in zip(react, dev_reference, strict=True)}})
    if through == "h0-dev":
        study.status("h0_dev_complete", tasks=len(dev_reference))
        return

    version, library = "h00", h0
    past_samples, past_solved = {}, {}
    checkpoints, reviews = [], []
    checkpoint_file = root / "checkpoints.json"
    recorded_batches = len(read(checkpoint_file)) if checkpoint_file.is_file() else 0
    catalogue = root / "offline/public_functions.json"
    for batch in manifest["train_batches"]:
        label, ids = f"b{batch['batch']:02d}", batch["tasks"]
        folder = root / "batches" / label
        entry_version, entry_library = version, library
        study.status("train_entry", batch=batch["batch"], parent=version)
        rows = measure(study, ids, label + "_entry", "B", library)
        batch_samples = train_samples(root, rows, library, label + "_entry")
        outcomes, other_previous_arm = [], {}
        for rnd in range(1, study.settings["optimizer"]["rounds"] + 1):
            name = f"{label}r{rnd:02d}"
            study.status("native_sol_revision", batch=batch["batch"], round=rnd, parent=version)
            samples = {**past_samples, **other_previous_arm,
                       **train_samples(root, rows, library, "h0_decision")}
            current = EditRound(root / "offline" / name, library, samples, past_solved, name + "_",
                                outcomes[-1] if outcomes else {})
            child = revise(study, ids, current=current)
            proposal = read(current.folder / "proposal.json")
            result = {"round": rnd, "parent": version, "reason": proposal.get("reason", ""),
                      "objects": [[edit["kind"], edit["name"]] for edit in proposal["objects"]]}
            if child == library:
                result["outcome"] = "no_change"
            else:
                save(root / "versions" / (name + ".json"), child)
                child_rows = measure(study, ids, name, "B", child, rows)
                reviewed = selection(study, name, folder / name, rows, child_rows, library, child)
                reviews.append(reviewed)
                result.update(outcome="accepted" if reviewed["accepted"] else "rejected", candidate=name,
                              selection=reviewed, initial_by_task={
                                  a["task_id"]: {"parent": [a["success"], a["output_tokens"]],
                                                 "candidate": [b["success"], b["output_tokens"]]}
                                  for a, b in zip(rows, child_rows, strict=True)})
                batch_samples.update(train_samples(root, child_rows, child, name))
                confirmations = (confirmed_train_samples(root, folder / name / "paired_selection.json", library, child)
                                 if reviewed["repeat_rollouts"] else {})
                batch_samples.update(confirmations)
                if reviewed["accepted"]:
                    other_previous_arm = train_samples(root, rows, library, "previous_parent")
                    version, library, rows = name, child, child_rows
                else:
                    other_previous_arm = train_samples(root, child_rows, child, "previous_candidate")
                other_previous_arm.update({"confirmation/" + k: s for k, s in confirmations.items()})
            save(folder / name / "result.json", result)
            outcomes.append(result)

        train_version = version
        dev_result = None
        if version != entry_version:
            study.status("dev_selection", batch=batch["batch"], parent=entry_version, candidate=version)
            candidates = measure(study, dev, "dev_" + version, "B", library, dev_reference)
            dev_result = selection(study, "dev_" + version, folder / "dev", dev_reference, candidates,
                                   entry_library, library)
            reviews.append(dev_result)
            if dev_result["accepted"]:
                dev_reference = candidates
            else:
                version, library = entry_version, entry_library
        checkpoints.append({"batch": batch["batch"], "entry_version": entry_version, "train_version": train_version,
                            "round_outcomes": [r["outcome"] for r in outcomes], "dev_selection": dev_result,
                            "next_version": version})
        if len(checkpoints) > recorded_batches:
            save(checkpoint_file, checkpoints)
        for name, sample in batch_samples.items():
            past_samples["history/" + name] = sample
            if sample.row["variant"] == "B" and sample.row["success"]:
                past_solved[sample.row["task_id"]] = (sample.trace, catalogue, study.execution)

    study.close_account()
    costs = optimization_costs(root, reviews)
    if through == "test":
        study.test_started = True
        study.status("test_normal", version=version)
        arms = {"A": "test_react", "B": "test_" + version}
        a, b = paired_rollouts(study, manifest["normal_holdout_tasks"], arms, {"A": "A", "B": "B"},
                               {"A": react_library, "B": library})
        report = final_report(a, b, study.seed)
        report.update(version=version, arms={"A": "all-generation native-tool ReAct", "B": "retained Jev harness"},
                      optimization_costs=costs,
                      native_metrics={"A": panel_metrics(a), "B": panel_metrics(b)})
        save(root / "test_report.json", report)
    save(root / "result.json", {"batches": checkpoints, "final_version": version, "through": through,
                                "optimization_costs": costs})
    study.status("native_" + through + "_complete", version=version)


def configure(parser):
    parser.add_argument("--connections", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-gpus", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--through", choices=("react-dev", "h0-dev", "train", "test"), default="test",
                        help="Finish the named stage, including preceding stages in order")


def launch(args, shared, connections):
    root = args.root.resolve()
    settings = configuration(shared)
    model_args = next(iter(connections["model_nodes"].values()))["args"]
    settings["model_path"] = model_args[model_args.index("--model") + 1]
    path = root / "configuration.json"
    if path.is_file() and read(path) != settings:
        raise ValueError("Use a fresh output directory or resume the same composition experiment")
    if not path.is_file() and (root / "runs").exists():
        raise ValueError("Choose a fresh output directory for the composition experiment")
    save(path, settings)
    study = SharedStudy(root, settings, connections, max_gpus=args.max_gpus, lanes_per_gpu=2)
    try:
        study.start_workers()
        study.paired_model_info()
        run_formal(study, design_manifest(), args.through)
    except BaseException as exc:
        study.status("interrupted", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        study.close()
