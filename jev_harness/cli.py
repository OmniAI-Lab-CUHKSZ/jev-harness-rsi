"""Command-line entry points for the published AppWorld experiment."""
import argparse
import json
import os
from pathlib import Path
from string import Template

from formal_run import read, save
from phase1_run import design_manifest, initial_harness, run
from .allocation import SharedStudy
from .paired_selection import INITIAL_PLUS_ONE, TWO_REPEATS_ONLY


PACKAGE = Path(__file__).resolve().parent
RESOURCES = PACKAGE / "resources"


def load_connections(path):
    locations = {**os.environ, "PACKAGE_DIR": str(PACKAGE)}
    config = read(path)
    # Expand each parsed field separately. Windows backslashes stay ordinary path characters.
    def resolve(value):
        if isinstance(value, str):
            return Template(value).substitute(locations)
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {name: resolve(item) for name, item in value.items()}
        return value
    try:
        connections = resolve(config)
    except KeyError as exc:
        raise ValueError(f"Set {exc.args[0]} for {path}") from exc
    if not connections.get("model_nodes"):
        raise ValueError("List at least one model worker in model_nodes")
    return connections


def initialize(root, settings, *, confirmation_scoring=None, revisit_after=None, cuda_graph_after=None, restart_revisit=None):
    config_path = root / "configuration.json"
    if config_path.is_file():
        saved = read(config_path)
        if "menu_retrieval" not in saved:
            settings.pop("menu_retrieval", None)
        if saved.get("cuda_graph_from_start"):
            settings["cuda_graph_from_start"] = True
        if "confirmation_scoring" in saved:
            settings["confirmation_scoring"] = saved["confirmation_scoring"]
        if "revisit_after" in saved:
            settings["revisit_after"] = saved["revisit_after"]
        if "cuda_graph_after" in saved:
            settings["cuda_graph_after"] = saved["cuda_graph_after"]
        if "restart_revisit" in saved:
            settings["restart_revisit"] = saved["restart_revisit"]
        if "historical_choice" not in saved.get("optimizer", {}):
            settings["optimizer"].pop("historical_choice", None)
        elif isinstance(saved["optimizer"]["historical_choice"], bool):
            # Existing experiments keep their recorded single-choice or disabled protocol.
            settings["optimizer"]["historical_choice"] = saved["optimizer"]["historical_choice"]
        if "menu_format" not in saved:
            # Earlier releases used the history-preview editor and single-session batch one.
            settings.pop("menu_format", None)
            settings["optimizer"].pop("batch_revision", None)
            settings["optimizer"].update(history_preview=True, fragment_trials=True, same_batch_feedback_from=1)
            settings["inference_schedule"]["first_batch"] = 1
        if "inference_schedule" not in saved:
            settings.pop("inference_schedule", None)
        if saved != settings:
            raise ValueError("Choose a separate output directory for a different experiment configuration")
    elif settings.get("paired_selection_from_batch") is not None:
        settings["confirmation_scoring"] = TWO_REPEATS_ONLY
    if confirmation_scoring is not None:
        settings["confirmation_scoring"] = confirmation_scoring
    if revisit_after is not None:
        after, *sources = revisit_after
        if not sources or not 1 <= after <= len(settings["batches"]) or any(not 1 <= source <= after for source in sources):
            raise ValueError("Choose one or more Train batches already visited at the insertion point")
        if len(sources) != len(set(sources)):
            raise ValueError("List each Train batch once in the revisit sequence")
        requested = [after, *sources]
        saved_visits = settings.get("revisit_after", [])
        if requested[:len(saved_visits)] != saved_visits:
            raise ValueError("An existing revisit sequence can only be extended")
        checkpoints = root / "checkpoints.json"
        if (saved_visits != requested and checkpoints.is_file()
                and any(row["batch"] > after for row in read(checkpoints))):
            raise ValueError("The run has already scored a batch after the requested insertion point")
        settings["revisit_after"] = requested
    if restart_revisit is not None:
        if "revisit_after" not in settings or not (restart_revisit.isascii() and restart_revisit.isalnum()):
            raise ValueError("Use --revisit-after and an alphanumeric retry name with --restart-revisit")
        after, source = settings["revisit_after"][:2]
        previous_tag = settings.get("restart_revisit")
        previous_visit = f"b{after:02d}" + (f"_{previous_tag}" if previous_tag else "") + f"_revisit{source:02d}"
        checkpoints = root/"checkpoints.json"
        if restart_revisit != previous_tag and checkpoints.is_file() and any(row.get("visit")==previous_visit for row in read(checkpoints)):
            raise ValueError(f"{previous_visit} already has an official selection")
        settings["restart_revisit"] = restart_revisit
    if cuda_graph_after is not None:
        if not 1 <= cuda_graph_after <= len(settings["batches"]):
            raise ValueError("Choose a scheduled Train batch for the CUDA Graph switch")
        if settings.get("cuda_graph_after", cuda_graph_after) != cuda_graph_after:
            raise ValueError("The saved run already uses a different decode switch")
        checkpoints = root / "checkpoints.json"
        if "cuda_graph_after" not in settings and checkpoints.exists():
            visits = [c.get("visit", f"b{c['batch']:02d}") for c in read(checkpoints)]
            if f"b{cuda_graph_after:02d}" in visits[:-1]:
                raise ValueError("A later visit has already been scored with the original engine")
        settings["cuda_graph_after"] = cuda_graph_after
    save(config_path, settings)
    reference = root / "dev_reference.json"
    if not reference.exists():
        adaptive = settings["adaptive_decision"]
        save(reference, {"version": "h00", "source_stage": str(root / "runs/dev_h00"),
             "mode": "measure_new_h0_then_reuse", "tasks": 0, "successes": None, "output_tokens": None,
             "threshold": adaptive["threshold"], "thinking": "official_adaptive", "think_budget": adaptive["think_budget"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Print the exact protocol, H0 and task IDs without loading a model")
    from .native_pilot import configure, run_pilot
    configure(commands.add_parser("native-pilot", help="Pilot schema-driven tools and a Train-only Sol revision"))
    from . import native_formal
    native_formal.configure(commands.add_parser("native-run", help="Run or resume the cold-start native-tool experiment"))
    native_plan = commands.add_parser("native-plan", help="Print the native split, settings and H0 without starting workers")
    native_plan.add_argument("--composed", action="store_true", help="Show the composed card experiment")
    for command, help_text in (("smoke", "Run one H0 Train task with the real model and simulator"),
                               ("run", "Run or resume the ten-batch experiment and paired Test-N evaluation")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--connections", type=Path, required=True)
        sub.add_argument("--root", type=Path, required=True, help="Output directory for this run")
        if command == "smoke":
            sub.add_argument("--gpu", required=True, help="One worker, for example local:0")
            sub.add_argument("--task", default="6ea6792_2")
        else:
            allocation = sub.add_mutually_exclusive_group()
            allocation.add_argument("--gpus", help="Use fixed devices, e.g. local:0,local:1")
            allocation.add_argument("--max-gpus", type=int, choices=range(1, 7), default=4)
            sub.add_argument("--gpu-name", default="A800", help="GPU name substring for automatic allocation")
            sub.add_argument("--paired-selection-from-batch", type=int, choices=range(1, 11),
                             help="First Train batch and Dev selection to use conditional paired confirmation")
            sub.add_argument("--confirmation-scoring", choices=(INITIAL_PLUS_ONE, TWO_REPEATS_ONLY),
                             help="Scoring for pending and future confirmations; completed selections are retained")
            sub.add_argument("--revisit-after", type=int, nargs="+", metavar="BATCH",
                             help="After the first batch number, revisit the remaining batch numbers in order")
            sub.add_argument("--restart-revisit", metavar="NAME",
                             help="Restart optimization of the pending revisit under a new name, reusing measured executions")
            sub.add_argument("--cuda-graph-after", type=int, metavar="BATCH",
                             help="Switch after this regular visit and measure its retained Dev parent on the new engine")
        sub.add_argument("--cuda-graph", action="store_true", help="Use CUDA Graph from the first model request")
    args = parser.parse_args(argv)
    settings = read(RESOURCES / "experiment.json")
    if args.command == "native-plan":
        print(json.dumps({"configuration": native_formal.configuration(settings, composed=args.composed),
                          "sequence": ["react-dev", "h0-dev", "train with per-batch dev selection", "test-normal"],
                          "default_allocation": {"gpu": "A800", "max_gpus": 6, "lanes_per_gpu": 2},
                          "split": design_manifest(native_formal.PROTOCOL),
                          "h0": native_formal.initial_library(args.composed)},
                         ensure_ascii=False, indent=2))
        return
    if args.command == "native-run":
        return native_formal.launch(args, settings, load_connections(args.connections))
    if args.command == "native-pilot":
        return run_pilot(args,settings,load_connections(args.connections))
    manifest = design_manifest()
    if args.command == "plan":
        print(json.dumps({"configuration": settings, "split": manifest, "h0": initial_harness(settings)},
                         ensure_ascii=False, indent=2))
        return
    connections = load_connections(args.connections)
    devices = ([args.gpu] if args.command == "smoke" else
               [name.strip() for name in args.gpus.split(",")] if args.gpus else [])
    for device in devices:
        node, separator, index = device.rpartition(":")
        if not separator or not index.isdecimal() or node not in connections["model_nodes"]:
            parser.error(f"Unknown GPU {device!r}. Use a model_nodes name followed by : and its GPU index.")
    if len(set(devices)) != len(devices) or len(devices) > 6:
        parser.error("Specify up to six distinct GPU devices")
    if args.command == "smoke" and not any(args.task in b["tasks"] for b in manifest["train_batches"]):
        parser.error("Choose a Train task from the plan for the H0 smoke run")
    first_node = next(iter(connections["model_nodes"].values()))
    model_args = first_node["args"]
    settings["model_path"] = model_args[model_args.index("--model") + 1]
    root = args.root.resolve()
    if args.cuda_graph:
        if getattr(args, "cuda_graph_after", None) is not None:
            parser.error("Choose either --cuda-graph or --cuda-graph-after")
        settings["cuda_graph_from_start"] = True
    if args.command == "run":
        first_review = args.paired_selection_from_batch
        if first_review is None and (root / "configuration.json").is_file():
            first_review = read(root / "configuration.json").get("paired_selection_from_batch")
        if first_review is not None:
            settings["paired_selection_from_batch"] = first_review
        elif args.confirmation_scoring is not None:
            parser.error("--confirmation-scoring requires conditional paired selection")
    initialize(root, settings, confirmation_scoring=getattr(args, "confirmation_scoring", None),
               revisit_after=getattr(args, "revisit_after", None), cuda_graph_after=getattr(args, "cuda_graph_after", None),
               restart_revisit=getattr(args, "restart_revisit", None))
    study = SharedStudy(root, settings, connections, devices=devices, max_gpus=getattr(args, "max_gpus", 1),
                        gpu_name=getattr(args, "gpu_name", "A800"))
    try:
        study.start_workers()
        if args.command == "smoke":
            row = study.run_task(args.task, "smoke_h00", "B", initial_harness(settings), workers=study.workers[0])
            save(root / "smoke_result.json", {k: row[k] for k in
                 ("task_id", "success", "output_tokens", "gpu", "trace_path", "step_review", "execution_settings")})
            study.status("smoke_completed", task=row["task_id"], success=row["success"])
        else:
            catalog = root / "offline/public_functions.json"
            if catalog.exists():
                study.functions = read(catalog)
            run(study, len(manifest["train_batches"]), settings["optimizer"]["rounds"], True)
    except BaseException as exc:
        study.status("interrupted", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        study.close()
