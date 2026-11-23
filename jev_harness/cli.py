"""Command-line entry points for the published AppWorld experiment."""
import argparse
import json
import os
from pathlib import Path
from string import Template

from formal_run import read, save
from phase1_run import design_manifest, run
from .allocation import SharedStudy


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


def initialize(root, settings):
    config_path = root / "configuration.json"
    if config_path.is_file() and read(config_path) != settings:
        raise ValueError("Choose a separate output directory for a different experiment configuration")
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
            allocation.add_argument("--max-gpus", type=int, choices=range(1, 5), default=4)
            sub.add_argument("--gpu-name", default="A800", help="GPU name substring for automatic allocation")
    args = parser.parse_args(argv)
    settings = read(RESOURCES / "experiment.json")
    manifest = design_manifest()
    if args.command == "plan":
        print(json.dumps({"configuration": settings, "split": manifest, "h0": read(RESOURCES / "h00.json")},
                         ensure_ascii=False, indent=2))
        return
    connections = load_connections(args.connections)
    devices = ([args.gpu] if args.command == "smoke" else
               [name.strip() for name in args.gpus.split(",")] if args.gpus else [])
    for device in devices:
        node, separator, index = device.rpartition(":")
        if not separator or not index.isdecimal() or node not in connections["model_nodes"]:
            parser.error(f"Unknown GPU {device!r}. Use a model_nodes name followed by : and its GPU index.")
    if len(set(devices)) != len(devices) or len(devices) > 4:
        parser.error("Specify up to four distinct GPU devices")
    if args.command == "smoke" and not any(args.task in b["tasks"] for b in manifest["train_batches"]):
        parser.error("Choose a Train task from the plan for the H0 smoke run")
    first_node = next(iter(connections["model_nodes"].values()))
    model_args = first_node["args"]
    settings["model_path"] = model_args[model_args.index("--model") + 1]
    root = args.root.resolve()
    initialize(root, settings)
    study = SharedStudy(root, settings, connections, devices=devices, max_gpus=getattr(args, "max_gpus", 1),
                        gpu_name=getattr(args, "gpu_name", "A800"))
    try:
        study.start_workers()
        if args.command == "smoke":
            row = study.run_task(args.task, "smoke_h00", "B", read(RESOURCES / "h00.json"), workers=study.workers[0])
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
