"""Isolated Train-only comparison on the official structured-tool interface."""

from pathlib import Path

from formal_run import read, save, scenario
from phase1_run import design_manifest
from .allocation import SharedStudy
from .native_assembly import NativeAssembly, NativeGeneration, PROTOCOL, blank_library


class NativeStudy(SharedStudy):
    compiled_executor = NativeAssembly
    generation_executor = NativeGeneration

    def record_public_catalog(self, stage):
        return not self.test_started


def configure(command):
    command.add_argument("--connections", required=True, type=Path)
    command.add_argument("--root", required=True, type=Path)
    command.add_argument("--train", nargs="+", default=[], help="Official Train tasks shown to Sol")
    command.add_argument("--check", nargs="+", default=[], help="Additional Train tasks withheld from Sol")
    command.add_argument(
        "--composed", action="store_true", help="Use composable cards and collect only the decision harness's Train traces"
    )
    command.add_argument("--gpus", help="Fixed devices; otherwise select eligible GPUs")
    command.add_argument("--max-gpus", type=int, default=2, choices=range(1, 7))
    command.add_argument(
        "--history-run",
        type=Path,
        action="append",
        default=[],
        help="Earlier native pilot. Open its optimization Train tasks only; last candidate is the entry harness.",
    )
    command.add_argument("--phase", choices=("initial", "revise", "evaluate", "all", "score"), default="all")
    command.add_argument("--entry-library", type=Path, help="Entry JSON for --phase score")
    command.add_argument("--candidate-library", type=Path, help="Final JSON for --phase score")


def run_pilot(args, settings, connections):
    root = args.root.resolve()
    train, check = args.train, args.check
    tasks = train + check
    scoring = args.phase == "score"
    if not tasks or (not scoring and not train):
        raise ValueError("Supply --train for evolution, or --train or --check for fixed-library scoring")
    manifest = design_manifest()
    allowed = {t for b in manifest["train_batches"] for t in b["tasks"]}
    if not set(tasks) <= allowed or len(set(tasks)) != len(tasks):
        raise ValueError("Use distinct tasks from official Train in the optimization and check panels")
    history = [p.resolve() for p in args.history_run]
    if scoring:
        if history or not args.entry_library or not args.candidate_library:
            raise ValueError("Score requires two saved libraries: --entry-library and --candidate-library")
    elif args.entry_library or args.candidate_library:
        raise ValueError("Use --phase score to measure two supplied libraries")
    past = set()
    for previous in history:
        history_plan = read(previous / "pilot_plan.json")
        if (
            previous == root
            or history_plan["protocol"] not in {"appworld_native_tools_v1", PROTOCOL}
            or not set(history_plan["train"]) <= allowed
        ):
            raise ValueError("Choose an earlier native Train pilot")
        past.update(history_plan["train"])
    if past & set(check):
        raise ValueError("A withheld task must not appear in opened historical Train")
    plan = dict(
        protocol=PROTOCOL,
        train=train,
        check=check,
        seed=settings["seed"],
        execution=settings["execution"],
        novel_check_scenarios=[t for t in check if scenario(t) not in {scenario(x) for x in [*past, *train]}],
        cuda_graph=True,
        lanes_per_gpu=2,
    )
    composed = getattr(args, "composed", False)
    if composed:
        from .composed_assembly import ComposedAssembly, initial_library

        plan["execution_arm"] = "composed_cards"
    if history:
        plan["history_runs"] = [str(p) for p in history]
    if scoring:
        plan["fixed_libraries"] = True
    config = root / "pilot_plan.json"
    if config.exists() and read(config) != plan:
        raise ValueError("Use a fresh pilot root for a different task panel or execution settings")
    save(config, plan)
    parent = (
        read(args.entry_library)
        if scoring
        else read(history[-1] / "offline/candidate.json")
        if history
        else initial_library() if composed else blank_library()
    )
    if scoring:
        for name, lib in (
            ("entry_harness.json", parent),
            ("offline/candidate.json", read(args.candidate_library)),
        ):
            if (root / name).exists() and read(root / name) != lib:
                raise ValueError("Choose a new score directory when changing the supplied libraries")
            save(root / name, lib)
    save(root / "entry_harness.json", parent)
    devices = tuple(args.gpus.split(",")) if args.gpus else ()
    if len(set(devices)) != len(devices) or any(
        d.rpartition(":")[0] not in connections["model_nodes"] or not d.rpartition(":")[2].isdecimal()
        for d in devices
    ):
        raise ValueError("Choose distinct configured node:GPU devices")
    study = NativeStudy(
        root,
        {**settings, "cuda_graph_from_start": True},
        connections,
        devices=devices,
        max_gpus=args.max_gpus,
        lanes_per_gpu=2,
    )
    if composed:
        study.compiled_executor = ComposedAssembly
    entry_arms = (("h0_decision", "B"),) if composed else (("h0_generation", "A"), ("h0_decision", "B"))

    def panel(stage, task_list, variant, lib):
        affinity = [read(root / "runs/h0_decision" / t / "record.json")["gpu"] for t in task_list]
        return study.execute_preferred(
            task_list,
            lambda task, worker: study.run_task(task, stage, variant, lib, workers=worker),
            affinity,
        )

    try:
        if args.phase in ("all", "initial", "score"):
            study.start_workers()
            study.paired_model_info()
            jobs = [
                (task, stage, arm)
                for task in tasks
                for stage, arm in entry_arms
            ]
            devices = list(study.groups)
            affinity = [devices[tasks.index(t) % len(devices)] for t, stage, arm in jobs]
            study.execute_preferred(
                jobs,
                lambda job, worker: study.run_task(
                    *job, parent if job[2] == "B" else blank_library(), workers=worker
                ),
                affinity,
            )
        if args.phase in ("all", "revise"):
            from .native_revision import pilot_round, revise

            study.status("native_sol_revision", train_tasks=len(train))
            if composed:
                current = pilot_round(root, train, history, entry_only=True)
                revise(study, train, history, current=current, composed=True)
            else:
                revise(study, train, history)
        if args.phase in ("all", "evaluate", "score"):
            library = read(root / "offline" / "candidate.json")
            study.start_workers()
            panel("candidate", tasks, "B", library)
            from .formal_optimizer import read_events
            from .native_evidence import fragment_evidence

            for task in tasks:
                for stage, lib in (
                    ("h0_generation", blank_library()),
                    ("h0_decision", parent),
                    ("candidate", library),
                ):
                    directory = root / "runs" / stage / task
                    if not (directory / "record.json").is_file():
                        continue
                    row = read(directory / "record.json")
                    save(
                        directory / "fragment_flow.json",
                        fragment_evidence(read_events(root / row["trace_path"]), lib),
                    )
        report = []
        for task in tasks:
            for stage in ("h0_generation", "h0_decision", "candidate"):
                path = root / "runs" / stage / task / "record.json"
                if path.exists():
                    row = read(path)
                    report.append(
                        {
                            "stage": stage,
                            "panel": "optimization" if task in train else "withheld Train",
                            **{
                                k: row[k]
                                for k in (
                                    "task_id",
                                    "success",
                                    "output_tokens",
                                    "open_calls",
                                    "native_ballots",
                                    "open_output_tokens",
                                    "decision_output_tokens",
                                    "api_calls",
                                    "error",
                                    "seconds",
                                    "trace_path",
                                )
                            },
                        }
                    )
        save(root / "pilot_results.json", report)
        if composed and args.phase in ("all", "evaluate", "score"):
            rows = {stage: [r for r in report if r["stage"] == stage] for stage in ("h0_decision", "candidate")}
            totals = {stage: {key: sum(row[key] for row in entries) for key in (
                "success", "output_tokens", "open_output_tokens", "decision_output_tokens", "open_calls",
                "native_ballots", "api_calls",
            )} for stage, entries in rows.items()}
            comparison = {"totals": totals}
            if train:
                def train_score(entries):
                    train_rows = [row for row in entries if row["panel"] == "optimization"]
                    return sum(row["success"] for row in train_rows), -sum(row["output_tokens"] for row in train_rows)

                entry, child = (train_score(rows[stage]) for stage in ("h0_decision", "candidate"))
                comparison.update(retained="candidate" if child >= entry else "h0_decision", retention_panel="Train")
            save(root / "comparison.json", comparison)
        study.status("pilot_" + args.phase + "_complete", episodes=len(report))
    except BaseException as exc:
        study.status("interrupted", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        study.close()
