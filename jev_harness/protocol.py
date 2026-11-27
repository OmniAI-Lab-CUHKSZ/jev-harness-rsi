"""Packaged configuration and the official AppWorld task split."""

from pathlib import Path

from .artifacts import read


PROTOCOL = "appworld_composition_v1"
RESOURCES = Path(__file__).with_name("resources")


def design_manifest():
    return read(RESOURCES / "task_split.json")
