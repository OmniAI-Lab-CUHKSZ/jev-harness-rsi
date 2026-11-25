import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from formal_run import read, save
from jev_harness.allocation import SharedStudy
from jev_harness.cli import RESOURCES, initialize
from jev_harness.formal_v2 import include_retries
from jev_harness.scoped_menu import scoped_library
from jev_harness.semantic_library import initial_library
from phase1_run import design_manifest, initial_harness, run


def scored(tasks, stage, successes, cost):
    return [{"task_id": task, "seed": 1, "success": index < successes, "output_tokens": cost,
             "seconds": 1, "model_calls": {}, "trace_path": f"runs/{stage}/{task}/attempt_1/trace.jsonl"}
            for index, task in enumerate(tasks)]


class PublicRunTests(unittest.TestCase):
    def test_distributed_h0_matches_constructor_and_plan(self):
        config = read(RESOURCES / "experiment.json")
        split = design_manifest()
        self.assertEqual(initial_harness(config), scoped_library(initial_library(semantic_handoffs=True)))
        self.assertEqual(read(RESOURCES / "h00.json"), initial_harness(config))
        self.assertTrue(config["optimizer"]["batch_revision"])
        with TemporaryDirectory() as directory:
            study = SharedStudy(directory, config, {})
            for phase, batch in (("train", 1), ("dev", 1), ("train", 2), ("dev", 10), ("test", None)):
                self.assertEqual(study.width_for(phase, batch), 2)
        self.assertEqual(config["batches"], [batch["scenarios"] for batch in split["train_batches"]])
        self.assertEqual(config["execution"]["step_review_mode"], "official_adaptive")
        self.assertEqual(config["execution"]["step_review_thinking_budget"], 256)

    def test_resume_preserves_older_public_optimizer_and_batch_one_width(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = read(RESOURCES / "experiment.json")
            old.pop("menu_format")
            old["optimizer"].pop("batch_revision")
            old["optimizer"].update(history_preview=True, fragment_trials=True, same_batch_feedback_from=1)
            old["inference_schedule"]["first_batch"] = 1
            for configured_width in (True, False):
                if not configured_width:
                    old.pop("inference_schedule")
                save(root / "configuration.json", old)
                current = read(RESOURCES / "experiment.json")
                initialize(root, current)
                self.assertEqual(current, old)
                self.assertEqual(initial_harness(current), initial_library(semantic_handoffs=True))

    def test_dev_tie_carries_candidate_and_parent_score_is_reused_across_batches(self):
        with TemporaryDirectory() as folder:
            study = Mock(root=Path(folder), settings=read(RESOURCES / "experiment.json"), functions=[])
            def batch(tasks, stage, library):
                successes = 10 if stage.startswith("dev_") else 1
                rows = scored(tasks, stage, successes, 50)
                if stage == "dev_b02r01":
                    for row in rows:
                        row["lanes_per_gpu"] = 2
                return rows
            study.batch.side_effect = batch
            study.candidate_batch.side_effect = lambda tasks, stage, candidate, parent, library: (
                parent, scored(tasks, stage, 2, 55))
            patch_data = {"objects": [{"kind": "handoff", "name": "default", "value": {"skill": "Use the live values."}}]}
            with patch("jev_harness.phase1_optimizer.propose_one", return_value=patch_data) as optimize:
                run(study, 2, 1, True, evaluate_test=False)
                snapshots = read(study.root / "checkpoints.json")
                self.assertEqual([r["next_version"] for r in snapshots], ["b01r01", "b02r01"])
                self.assertTrue(all(r["dev_selection"]["tied"] for r in snapshots))
                self.assertEqual(snapshots[1]["dev_selection"]["reference_lanes_per_gpu"], [1])
                self.assertEqual(snapshots[1]["dev_selection"]["candidate_lanes_per_gpu"], [2])
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

    def test_optimizer_restart_reuses_only_matching_ordered_harness_and_parent_pair(self):
        from phase1_run import previous_revisit_evaluation, train_entry_stage
        with TemporaryDirectory() as directory:
            root=Path(directory)
            original,new="b04_revisit03","b04_24j_revisit03"
            settings=read(RESOURCES/"experiment.json")
            initialize(root,settings,revisit_after=(4,3),restart_revisit="24j")
            resumed=read(RESOURCES/"experiment.json")
            initialize(root,resumed)
            self.assertEqual(train_entry_stage(new,"b02r02",3,resumed),original+"_b02r02_entry")
            save(root/"batches"/original/"entry.json",{"version":"b02r02"})
            old_round,new_round=original+"r01",new+"r01"
            for name,value in ((old_round,{"cards":{"a":1,"b":2}}),(new_round,{"cards":{"a":1,"b":2}}),
                               ("b02r02",{"cards":{}}),("other",{"cards":{"c":3}})):
                save(root/"versions"/(name+".json"),value)
            self.assertEqual(previous_revisit_evaluation(root,original,new_round,"b02r02"),(old_round,old_round))
            self.assertEqual(previous_revisit_evaluation(root,original,new_round,"other"),(old_round,new_round))
            save(root/"versions"/(new_round+".json"),{"cards":{"b":2,"a":1}})
            self.assertEqual(previous_revisit_evaluation(root,original,new_round,"b02r02"),(new_round,new_round))

    def test_extend_completed_third_revisit_with_fourth_before_pending_fifth(self):
        for wins_dev in (True, False):
            with self.subTest(wins_dev=wins_dev), TemporaryDirectory() as directory:
                root = Path(directory)
                config = lambda: read(RESOURCES / "experiment.json")
                initial = config()
                initialize(root, initial, revisit_after=(4, 3), restart_revisit="24j", cuda_graph_after=4)
                study = Mock(root=root, settings=initial, functions=[])
                measured = {}
                def panel(tasks, stage, library):
                    if stage not in measured:
                        successes = 2 if stage == "dev_h00_cuda_graph" else 3 if wins_dev else 1
                        measured[stage] = scored(tasks, stage, successes, 10)
                    return measured[stage]
                study.batch.side_effect = panel
                study.candidate_batch.side_effect = lambda tasks, stage, candidate, parents, lib: (
                    parents, scored(tasks, stage, len(tasks), 8))
                keep = {"keep_parent":True, "reason":"Retain this round's parent"}
                with patch("jev_harness.phase1_optimizer.propose_one", return_value=keep):
                    run(study, 4, 2, True, evaluate_test=False)
                prefix = read(root / "checkpoints.json")
                self.assertEqual(prefix[-1]["visit"], "b04_24j_revisit03")
                fifth_tasks = design_manifest()["train_batches"][4]["tasks"]
                panel(fifth_tasks, "b05_h00_entry", {})
                # A pending Sol decision was made before the additional fourth visit.
                save(root / "batches/b05/b05r01/proposal.json", keep)
                initialize(root, config(), revisit_after=(4, 3, 4))
                resumed = config()
                initialize(root, resumed)
                study.settings = resumed
                attempts = []
                def propose(study, lib, rows, history, folder, sources):
                    attempts.append(folder.name)
                    if folder.name == "b04_revisit04r02":
                        return {"objects":[{"kind":"handoff","name":"default","value":{"skill":"Use the scoped result."}}]}
                    return keep
                study.batch.reset_mock()
                with patch("jev_harness.phase1_optimizer.propose_one", side_effect=propose):
                    run(study, 5, 2, True, evaluate_test=False)
                    run(study, 5, 2, True, evaluate_test=False)
                visits = read(root / "checkpoints.json")
                self.assertEqual(visits[:-2], prefix)
                self.assertEqual([v["visit"] for v in visits[-2:]], ["b04_revisit04", "b05_after_repeat04"])
                self.assertEqual(attempts, ["b04_revisit04r01","b04_revisit04r02",
                                            "b05_after_repeat04r01","b05_after_repeat04r02"])
                winner = "b04_revisit04r02" if wins_dev else "h00"
                self.assertEqual(visits[-1]["entry_version"], winner)
                stages = [call.args[1] for call in study.batch.call_args_list]
                self.assertEqual(stages, ["b04_revisit04_h00_entry", "dev_h00_cuda_graph",
                                          "dev_b04_revisit04r02_cuda_graph", f"b05_{winner}_entry"])
                with self.assertRaisesRegex(ValueError, "already scored"):
                    initialize(root, config(), revisit_after=(4, 3, 4, 2))

    def test_cuda_graph_measures_actual_dev_winner_once_before_revisit_and_reuses_same_mode_on_resume(self):
        for old_candidate_wins in (True, False):
            with self.subTest(old_candidate_wins=old_candidate_wins), TemporaryDirectory() as directory:
                root = Path(directory)
                settings = read(RESOURCES/"experiment.json")
                initialize(root,settings,revisit_after=(4,3),cuda_graph_after=4)
                resumed = read(RESOURCES/"experiment.json")
                initialize(root,resumed)
                self.assertEqual(resumed["cuda_graph_after"],4)
                study = Mock(root=root,settings=resumed,functions=[])
                parent = "b04r02" if old_candidate_wins else "h00"
                graph_parent = "dev_"+parent+"_cuda_graph"
                dev = {"dev_h00":10,"dev_b04r02":11 if old_candidate_wins else 9,
                       graph_parent:13,"dev_b04_revisit03r02_cuda_graph":12,"dev_b05r01_cuda_graph":14}
                calls, measured = [], {}
                partial = True
                def panel(tasks,stage,library):
                    nonlocal partial
                    calls.append(stage)
                    rows = scored(tasks,stage,dev.get(stage,1),10)
                    for row in rows:
                        key = (stage,row["task_id"])
                        if key not in measured:
                            measured[key] = row
                            if stage == graph_parent and partial:
                                partial = False
                                raise RuntimeError("interrupted after one graph Dev task")
                    return [measured[stage,t] for t in tasks]
                study.batch.side_effect = panel
                study.candidate_batch.side_effect = lambda tasks,s,c,p,lib: (p,scored(tasks,s,2,8))
                def proposal(study, lib, rows, history, folder, sources):
                    if folder.name in ("b04r02","b04_revisit03r02","b05r01"):
                        return {"objects":[{"kind":"handoff","name":"default","value":{"skill":"Use the scoped result."}}]}
                    return {"keep_parent":True,"reason":"Retain the parent"}
                with patch("jev_harness.phase1_optimizer.propose_one", side_effect=proposal):
                    with self.assertRaisesRegex(RuntimeError,"one graph Dev"):
                        run(study,4,2,True,evaluate_test=False)
                    prior = read(root/"checkpoints.json")
                    self.assertEqual(prior[-1]["next_version"],parent)
                    self.assertFalse(any(s.startswith("b04_revisit03") for s in calls))
                    self.assertFalse((root/"cuda_graph_dev_reference.json").exists())
                    run(study,4,2,True,evaluate_test=False)
                    self.assertEqual(sum(s == graph_parent for s,_ in measured),57)
                    revisited = read(root/"checkpoints.json")[-1]
                    self.assertEqual(revisited["entry_version"],parent)
                    self.assertFalse(revisited["dev_selection"]["carry"])
                    self.assertEqual(revisited["dev_selection"]["reference_stage"],graph_parent)
                    run(study,5,2,True,evaluate_test=False)
                reference = read(root/"cuda_graph_dev_reference.json")
                self.assertEqual((reference["version"],reference["tasks"],reference["successes"]),(parent,57,13))
                final = read(root/"checkpoints.json")
                self.assertEqual(final[:4],prior)
                self.assertEqual(final[-1]["dev_selection"]["reference_stage"],graph_parent)
                self.assertTrue(final[-1]["dev_selection"]["carry"])
                self.assertLess(calls.index(graph_parent),calls.index("b04_revisit03_"+parent+"_entry"))

    def test_revisit_after_four_carries_dev_winner_and_resumes_nonconsecutive_scored_rounds(self):
        for win in (True, False):
            with self.subTest(revisit_wins_dev=win), TemporaryDirectory() as directory:
                root = Path(directory)
                config = lambda: read(RESOURCES/"experiment.json")
                initialize(root, config())
                initialize(root, config(), revisit_after=(4, 3))
                settings = config()
                initialize(root, settings)
                self.assertEqual(settings["revisit_after"], [4, 3])
                study = Mock(root=root, settings=settings, functions=[])
                dev = {"dev_h00":1, "dev_b04r02":2, "dev_b04_revisit03r02":3 if win else 1}
                study.batch.side_effect = lambda tasks,stage,library: scored(tasks,stage,dev.get(stage,1),10)
                study.candidate_batch.side_effect = lambda tasks,stage,lib,parents,old: (parents,scored(tasks,stage,2,8))
                versions = {"b04r02":"Use the live result.", "b04_revisit03r02":"Read the scoped result."}
                def propose(study, lib, rows, history, folder, sources):
                    if folder.name.startswith("b05"):
                        self.assertEqual([v["candidate"] for v in history], list(versions))
                    if folder.name in versions:
                        return {"objects":[{"kind":"handoff","name":"default","value":{"skill":versions[folder.name]}}]}
                    return {"keep_parent":True,"reason":"Retain this round's parent"}
                with patch("jev_harness.phase1_optimizer.propose_one",side_effect=propose):
                    run(study,4,2,True,evaluate_test=False)
                    prefix = read(root/"checkpoints.json")
                    run(study,5,2,True,evaluate_test=False)
                all_visits = read(root/"checkpoints.json")
                self.assertEqual(all_visits[:len(prefix)], prefix)
                self.assertEqual([c["visit"] for c in all_visits],["b01","b02","b03","b04","b04_revisit03","b05"])
                self.assertEqual(prefix[-1]["entry_version"],"b04r02")
                self.assertEqual(prefix[-1]["round_outcomes"],["retained_parent","scored"])
                self.assertEqual(prefix[-1]["revisions_scored"],1)
                self.assertEqual(all_visits[-1]["entry_version"],"b04_revisit03r02" if win else "b04r02")
                calls = {call.args[1]:call.args[0] for call in study.batch.call_args_list}
                self.assertEqual(calls["b03_h00_entry"],calls["b04_revisit03_b04r02_entry"])
