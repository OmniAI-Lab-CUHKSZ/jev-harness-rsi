"""Command-line entry points for the published AppWorld experiment."""
import argparse
import json
import os
from pathlib import Path
from string import Template

from . import native_formal
from .artifacts import read
from .protocol import RESOURCES, design_manifest


PACKAGE = Path(__file__).resolve().parent


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Print the protocol, H0 and task split without loading a model")
    native_formal.configure(commands.add_parser("run", help="Run or resume the composition experiment"))
    args = parser.parse_args(argv)
    settings = read(RESOURCES / "experiment.json")
    if args.command == "plan":
        print(json.dumps({
            "configuration": native_formal.configuration(settings),
            "sequence": ["react-dev", "h0-dev", "train", "test"],
            "default_allocation": {"gpu": "A800", "max_gpus": 6, "lanes_per_gpu": 2},
            "split": design_manifest(),
            "h0": native_formal.initial_library(),
        }, ensure_ascii=False, indent=2))
        return
    return native_formal.launch(args, settings, load_connections(args.connections))
