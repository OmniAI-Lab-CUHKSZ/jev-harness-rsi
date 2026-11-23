import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from formal_run import Study, save
from jev_harness.batch_feedback import BatchFeedbackWorkbench, prefix_changes, prefix_view
from jev_harness.phase1_optimizer import propose_one
from jev_harness.semantic_library import initial_library
from jev_harness.semantic_workbench import SemanticProbe, SemanticWorkbench


def generation(reads):
    return [{"role": "user", "content": json.dumps({"compiled_progress": {"available_public_reads": reads}})}]


class BatchFeedbackTests(unittest.TestCase):
    def test_workbench_switches_at_the_configured_batch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            study = Mock(root=root, functions=[], settings={"optimizer":{"same_batch_feedback_from":6}})
            with patch.object(SemanticWorkbench, "work", autospec=True, side_effect=type):
                for batch, expected in ((5, SemanticWorkbench), (6, BatchFeedbackWorkbench)):
                    folder = root/"batches"/f"b{batch:02d}"/f"b{batch:02d}r01"
                    self.assertIs(propose_one(study, initial_library(), [], [], folder, set()), expected)
                folder = root/"batches/b06_revisit03/b06_revisit03r01"
                self.assertIs(propose_one(study, initial_library(), [], [], folder, set()), BatchFeedbackWorkbench)

    def test_changed_prompt_with_unchanged_choice_and_real_reselection(self):
        def events(options, choice, reads, trial=False):
            return [{"event": "model", "operation": "decide", "request": {"options": options}},
                    {"event": "decision", "node": "step", "candidates": [{"value": choice}], "selected": 0},
                    ({"event": "probe_generation_boundary", "messages": generation(reads)} if trial else
                     {"event": "model", "operation": "generate", "request": {"messages": generation(reads)}})]
        parent = prefix_view(events(["first"], None, ["old"]))
        changed = prefix_changes(parent, prefix_view(events(["first", "extra"], None, ["old", "extra"], True)))
        self.assertEqual(changed["first_changed_Jev_request"]["fields"], ["options"])
        self.assertNotIn("first_changed_Jev_choice", changed)
        self.assertEqual(changed["public_reads"], {"added": ["extra"], "removed": []})
        selected = prefix_changes(parent, prefix_view(events(["first", "extra"], "first", ["old"], True)))
        self.assertEqual(selected["first_changed_Jev_choice"]["candidate"], ["step", "first"])
        self.assertNotIn("first_generation_input", selected)

    def test_all_sibling_prefixes_are_real_choices_with_no_generation_and_restore_after_restart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            task_ids = ["a_1", "b_1"]
            functions = [{"name": "phone__search_contacts", "parameters": {}}]
            lib = initial_library()
            study = Study(root, {"execution": {}, "seed": 3}, {}, [])
            study.functions = functions
            worker = Mock()
            def environment(op, **args):
                if op == "prepare":
                    return {"instruction": "Check my contacts.", "supervisor": {}, "datetime": "2026-01-01",
                            "functions": functions}
                raise AssertionError(op)
            def model(op, **args):
                self.assertEqual(op, "decide")
                n = len(args["options"])
                return {"index": n-1, "probabilities": [0.0]*(n-1)+[1.0], "mode": "D", "output_tokens": 0}
            env, head = Mock(), Mock()
            env.request.side_effect, head.request.side_effect = environment, model
            worker.worker.side_effect = {"model": head, "environment": env}.__getitem__
            study.workers = [worker]
            matched, rejected, rows = [], [], []
            for task in task_ids:
                outputs = []
                for stage, passed in (("parent", True), ("candidate", False)):
                    trace = Path("runs")/stage/task/"attempt_1"/"trace.jsonl"
                    engine = SemanticProbe(head, env)
                    engine.set_probe_target(None)
                    engine.run_prefix(task, lib, (root/trace).parent, 3, task)
                    row = {"task_id": task, "success": passed, "output_tokens": 10, "trace_path": str(trace)}
                    save((root/trace).parents[1]/"record.json", row)
                    outputs.append(row)
                rows.append(outputs[0])
                matched.append(outputs[0]["trace_path"])
                rejected.append(outputs[1]["trace_path"])
            history = [{"parent": "old", "candidate": "failed_trial", "accepted": False,
                        "train_traces": rejected, "parent_traces": matched}]
            folder = root/"batches/b06/b06r02/sol"
            bench = BatchFeedbackWorkbench(study, lib, rows, history, folder, set(task_ids))
            self.assertEqual(bench.index[1]["last_measured_round"]["success_before_after"], [True, False])
            self.assertFalse(bench.index[1]["last_measured_round"]["accepted"])
            revisit = root/"batches/b06_revisit03/b06_revisit03r01/sol"
            old_round = dict(history[0], batch=3)
            first = BatchFeedbackWorkbench(study, lib, rows, [old_round], revisit, set(task_ids))
            self.assertNotIn("last_measured_round", first.index[1])
            measured = dict(old_round, visit="b06_revisit03")
            second = BatchFeedbackWorkbench(study, lib, rows, [old_round, measured], revisit, set(task_ids))
            self.assertEqual(second.index[1]["last_measured_round"]["success_before_after"], [True, False])
            proposal = {"objects":[{"kind":"card","name":"read_contact_names","value":{
                "app":"phone","inputs":[],"output":"contact_records","description":"Read contacts.",
                "skill":"Choose for contact records.","api":"phone__search_contacts","public_read":True}}]}
            env.reset_mock()
            result = bench.probe(proposal, task_ids[0])
            self.assertEqual(env.request.call_count, 2)
            self.assertEqual([r["task"] for r in result["current_batch_prefixes"]], task_ids)
            for r in result["current_batch_prefixes"]:
                self.assertIn("phone__search_contacts", r["changes"]["public_reads"]["added"])
                self.assertEqual(r["prefix_generated_tokens"], 0)
                self.assertIn(r["candidate_trace"], bench.paths)
                self.assertNotIn("candidate_success", r)
            reopened = BatchFeedbackWorkbench(study, lib, rows, history, folder, set(task_ids))
            env.reset_mock()
            second = reopened.probe(proposal, task_ids[1])
            self.assertEqual(env.request.call_count, 1)
            self.assertEqual(second["current_batch_prefixes"], result["current_batch_prefixes"])
            self.assertTrue(all(reopened.paths[r["candidate_trace"]].is_file() for r in second["current_batch_prefixes"]))
            matches = reopened.find_evidence({"docs":["events/a_1", "events/b_1"], "query":"decision"})
            self.assertEqual([r["doc"] for r in matches], ["events/a_1", "events/b_1"])
            self.assertTrue(all(r["hits"] for r in matches))
