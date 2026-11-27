import json
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from jev_harness.artifacts import include_retries, read
from jev_harness.cli import main


def test_plan_uses_only_composition_and_current_schedule(capsys):
    main(["plan"])
    content = json.loads(capsys.readouterr().out)
    assert content["sequence"] == ["react-dev", "h0-dev", "train", "test"]
    settings = content["configuration"]
    assert settings["protocol"] == content["split"]["protocol"] == "appworld_composition_v1"
    assert settings["optimizer"]["rounds"] == 2 and settings["optimizer"]["max_objects"] == 3
    assert settings["optimizer"]["draft_reply_limit"] == 12
    assert settings["execution"]["step_review_mode"] == "official_adaptive"
    assert settings["execution"]["step_review_thinking_budget"] == 256
    assert settings["confirmation_scoring"] == "two_repeats_only"
    assert content["default_allocation"] == {"gpu": "A800", "max_gpus": 6, "lanes_per_gpu": 2}
    assert content["h0"] == json.loads(files("jev_harness").joinpath("resources", "composed_h0.json").read_text())
    cards = content["h0"]["fragment"]
    assert len(cards) == 5
    assert all("composition" in card and "sources" not in card for card in cards.values())
    split = content["split"]
    assert len(split["train_batches"]) == 10 and all(len(b["tasks"]) == 9 for b in split["train_batches"])
    sets = [{task for b in split["train_batches"] for task in b["tasks"]},
            set(split["dev_tasks"]), set(split["normal_holdout_tasks"])]
    assert [len(arm) for arm in sets] == [90, 57, 90]
    assert len(set.union(*sets)) == sum(map(len, sets))


@pytest.mark.parametrize("name", ["smoke", "native-plan", "native-pilot", "native-run"])
def test_legacy_cli_is_retired(name):
    with pytest.raises(SystemExit) as failure:
        main([name])
    assert failure.value.code == 2


def test_interrupted_attempt_output_is_charged_once_after_restart():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        base = root / "runs/b01_h00_entry/task_1"
        for number in (1, 2):
            (base / f"attempt_{number}").mkdir(parents=True)
        event = {"event": "model", "request": {}, "response": {"output_tokens": 4, "input_tokens": 8}}
        (base / "attempt_1/trace.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        row = {"trace_path": "runs/b01_h00_entry/task_1/attempt_2/trace.jsonl",
               "input_tokens": 30, "reasoning_tokens": 0, "output_tokens": 11}
        assert include_retries(root, row)["output_tokens"] == 15
        assert include_retries(root, read(base / "record.json"))["output_tokens"] == 15
