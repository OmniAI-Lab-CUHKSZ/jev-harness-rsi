import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from formal_run import read
from jev_harness.cli import RESOURCES
from jev_harness.formal_v2 import include_retries
from jev_harness.semantic_library import initial_library
from phase1_run import design_manifest, run


def scored(tasks, stage, successes, cost):
    return [{"task_id": task, "seed": 1, "success": index < successes, "output_tokens": cost,
             "seconds": 1, "model_calls": {}, "trace_path": f"runs/{stage}/{task}/attempt_1/trace.jsonl"}
            for index, task in enumerate(tasks)]


class PublicRunTests(unittest.TestCase):
    def test_distributed_h0_matches_constructor_and_plan(self):
        config = read(RESOURCES / "experiment.json")
        split = design_manifest()
        self.assertEqual(read(RESOURCES / "h00.json"), initial_library(semantic_handoffs=True))
        self.assertEqual(config["batches"], [batch["scenarios"] for batch in split["train_batches"]])
        self.assertEqual(config["execution"]["step_review_mode"], "official_adaptive")
        self.assertEqual(config["execution"]["step_review_thinking_budget"], 256)

    def test_dev_tie_carries_candidate_and_parent_score_is_reused_across_batches(self):
        with TemporaryDirectory() as folder:
            study = Mock(root=Path(folder), settings=read(RESOURCES / "experiment.json"), functions=[])
            def batch(tasks, stage, library):
                successes = 10 if stage.startswith("dev_") else 1
                return scored(tasks, stage, successes, 50)
            study.batch.side_effect = batch
            study.candidate_batch.side_effect = lambda tasks, stage, candidate, parent, library: (
                parent, scored(tasks, stage, 2, 55))
            patch_data = {"objects": [{"kind": "handoff", "name": "default", "value": {"skill": "Use the live values."}}]}
            with patch("jev_harness.phase1_optimizer.propose_one", return_value=patch_data) as optimize:
                run(study, 2, 1, True, evaluate_test=False)
                snapshots = read(study.root / "checkpoints.json")
                self.assertEqual([r["next_version"] for r in snapshots], ["b01r01", "b02r01"])
                self.assertTrue(all(r["dev_selection"]["tied"] for r in snapshots))
                dev_stages = [c.args[1] for c in study.batch.call_args_list if c.args[1].startswith("dev_")]
                self.assertEqual(dev_stages, ["dev_h00", "dev_b01r01", "dev_b02r01"])
                study.batch.reset_mock()
                run(study, 2, 1, True, evaluate_test=False)
                study.batch.assert_not_called()
                self.assertEqual(optimize.call_count, 2)

    def test_interrupted_attempt_output_is_charged_once_after_restart(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "runs/b01_h00_entry/task_1"
            for number in (1, 2):
                (base / f"attempt_{number}").mkdir(parents=True)
            event = {"event": "model", "request": {}, "response": {"output_tokens": 4, "input_tokens": 8}}
            (base / "attempt_1/trace.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            row = {"trace_path": "runs/b01_h00_entry/task_1/attempt_2/trace.jsonl",
                   "input_tokens": 30, "reasoning_tokens": 0, "output_tokens": 11}
            self.assertEqual(include_retries(root, row)["output_tokens"], 15)
            self.assertEqual(include_retries(root, read(base / "record.json"))["output_tokens"], 15)
