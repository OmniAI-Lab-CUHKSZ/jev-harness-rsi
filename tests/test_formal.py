import tempfile
import unittest
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from formal_run import Study, read, save
from jev_harness.fragment_library import FragmentExecutor, apply_fragment_patch, new_library
from jev_harness.generalization import classify


class FakeEnvironment:
    def __init__(self):
        self.writes = []
        self.finished = False
        self.total = 0

    def request(self, op, **p):
        if op == "prepare":
            return {"task_id": p["task_id"], "instruction": "Adjust the wake-up alarm by forty minutes",
                    "datetime": "2023-05-18", "supervisor": {}, "functions": [
                    {"name": "phone__apply_delta", "parameters": {}}, {"name": "supervisor__complete_task", "parameters": {}}]}
        if op == "call":
            self.total += 1
            if p["api"] == "apply_delta":
                self.writes.append(p["arguments"])
                out = {"message": "ok"}
            else:
                self.finished = True
                out = {"message": "done"}
            return {"value": out, "done": self.finished, "api_total": self.total}
        if op == "evaluate":
            return {"success": self.finished and self.writes == [{"minutes": -40}], "failures": []}
        raise AssertionError(op)


class FakeModel:
    def request(self, op, **p):
        if op != "generate" or p.get("purpose") != "G_local":
            raise AssertionError((op, p.get("purpose")))
        return {"mode": "G", "text": '{"value":-40}', "output_tokens": 8, "reasoning_tokens": 0}


class FormalTests(unittest.TestCase):

    def test_parallel_jobs_use_distinct_workers_and_return_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            connections = {"model": {"args": ["--gpu", "4"]}}
            study = Study(directory, {"execution": {}, "seed": 1}, connections, ("4", "1"))
            simultaneous = Barrier(2)

            def task(number, workers):
                if number < 2:
                    simultaneous.wait(timeout=5)
                return number, workers.gpu, workers.connections["model"]["args"][-1]

            rows = study.execute_jobs([0, 1, 2, 3], task)
            self.assertEqual([row[0] for row in rows], [0, 1, 2, 3])
            self.assertEqual({row[1] for row in rows[:2]}, {"4", "1"})
            self.assertTrue(all(gpu == selected for _, gpu, selected in rows))
            self.assertEqual(connections["model"]["args"], ["--gpu", "4"])

    def test_gpus_on_different_nodes_use_their_own_model_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            assigned = ("node-a:4", "node-a:5", "node-b:2", "node-b:4")
            connections = {"model_nodes": {"node-a": {"port": 2201, "args": ["--gpu", "4"]},
                                            "node-b": {"port": 2202, "args": ["--gpu", "1"]}}}
            study = Study(directory, {"execution": {}, "seed": 1}, connections, assigned)
            simultaneous = Barrier(4)

            def task(number, workers):
                simultaneous.wait(timeout=5)
                model = workers.connections["model"]
                return workers.gpu, model["port"], model["args"][-1]

            rows = study.execute_jobs(range(4), task)
            self.assertEqual({row[0] for row in rows}, set(assigned))
            for label, port, device in rows:
                node, index = label.split(":")
                self.assertEqual(port, connections["model_nodes"][node]["port"])
                self.assertEqual(device, index)

    def test_cross_node_candidate_replays_the_parent_on_the_candidate_card(self):
        with tempfile.TemporaryDirectory() as directory:
            connections = {"model_nodes": {"node-a": {"args": []}, "node-c": {"args": []}}}
            study = Study(directory, {"execution": {}, "seed": 1}, connections, ("node-a:4", "node-c:0"))
            parent = [{"task_id": f"task_{i}", "seed": 1, "success": True, "output_tokens": 475,
                       "seconds": 1.0, "gpu": "4"} for i in (1, 2)]
            old, new = new_library(), new_library()
            new["decision_skills"]["x"] = "Use observed facts."
            replayed = []

            def run_task(task, stage, variant, library, *, workers):
                if stage.endswith("_matched_parent"):
                    self.assertIs(library, old)
                    replayed.append((task, workers.gpu))
                else:
                    self.assertIs(library, new)
                tokens = 605 if workers.gpu == "node-c:0" else 475
                return {"task_id": task, "success": True, "seed": 1, "output_tokens": tokens,
                        "seconds": 1.0, "gpu": workers.gpu}

            def assign(tasks, run):
                return [run(task, worker) for task, worker in zip(tasks, study.workers)]

            with patch.object(study, "run_task", side_effect=run_task), patch.object(study, "execute_jobs", side_effect=assign):
                reference, candidate = study.candidate_batch(["task_1", "task_2"], "dev_case", new, parent, old)
            self.assertEqual(replayed, [("task_2", "node-c:0")])
            self.assertEqual([r["output_tokens"] for r in reference], [r["output_tokens"] for r in candidate])

    def test_generalization_groups_use_plans_and_keep_scenario_variants_together(self):
        lib = new_library()
        lib["workflow"]["rules"].update(
            list_a={"patterns": ["list a"], "goals": ["rows_a"], "complete": True},
            list_b={"patterns": ["list b"], "exclude": ["skip b"], "goals": ["rows_b"], "complete": True},
            missing={"patterns": ["missing"], "goals": ["no_provider"], "complete": True})
        lib["workflow"]["fragments"].update(
            a={"inputs": ["account_passwords"], "output": "rows_a"},
            b={"inputs": ["account_passwords"], "output": "rows_b"})
        train = [{"task_id": "train_1", "instruction": "List a"},
                 {"task_id": "train_2", "instruction": "List b"}]
        test = [{"task_id": "seen_1", "instruction": "List a"},
                {"task_id": "new_1", "instruction": "List a and list b"},
                {"task_id": "mixed_1", "instruction": "List a and list b"},
                {"task_id": "mixed_2", "instruction": "List a and list b, with the missing step"},
                {"task_id": "mixed_3", "instruction": "List a and list b, skip b"}]
        labels = classify(lib, train, test)
        self.assertEqual(labels["scenarios"], {"seen": "seen_structure", "new": "new_composition",
                                                "mixed": "partial_or_uncovered"})
        self.assertEqual(labels["tasks"]["mixed_3"]["matching_train_tasks"], ["train_1"])
        self.assertEqual(labels["tasks"]["mixed_2"]["missing_roles"], ["no_provider"])

    def test_typed_local_recovery_continues_to_a_real_downstream_call(self):
        lib = new_library()
        lib["workflow"]["rules"]["alarm"] = {"patterns": ["alarm"], "goals": ["alarm_updated"], "complete": True}
        lib["workflow"]["fragments"].update(
            parse_delta={"inputs": [], "output": "minutes", "steps": [{"kind": "compute", "binding": "delta", "store": "delta"}],
                         "result": {"ref": "vars.delta"}},
            update_alarm={"inputs": ["minutes"], "output": "alarm_updated", "steps": [
                {"kind": "call", "binding": "update", "store": "row", "expect": {"message": "ok"}}],
                "result": {"ref": "vars.row"}})
        lib["bindings"].update(delta={"ref": "vars.missing"}, update={"app": "phone", "api": "apply_delta",
                                                                    "arguments": {"minutes": {"ref": "vars.minutes"}}})
        lib["generation_skills"]["parse_delta"] = "Return the signed number of minutes"
        lib["mode_policy"]["parse_delta"] = {"local_type": "int"}
        with tempfile.TemporaryDirectory() as d:
            row = FragmentExecutor(FakeModel(), FakeEnvironment()).run("train_1", lib, Path(d)/"trace.jsonl")
        self.assertTrue(row["success"], row)
        self.assertEqual(row["local_recoveries"], 1)
        self.assertEqual(row["fragments"], ["parse_delta", "update_alarm"])
        self.assertEqual(row["api_calls"], 2)
        self.assertEqual(row["output_tokens"], 8)

    def test_optimizer_cannot_bundle_three_components_or_read_future_trace(self):
        lib = new_library()
        with self.assertRaises(ValueError):
            apply_fragment_patch(lib, {"changes": {"questions": {}, "evidence": {}, "bindings": {}}}, {"train_1"})
        with self.assertRaises(ValueError):
            apply_fragment_patch(lib, {"operation": "register", "changes": {"workflow": {"fragments": {
                "future": {"source_tasks": ["test_1"]}}}}}, {"train_1"})
