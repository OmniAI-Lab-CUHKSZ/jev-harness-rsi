import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from formal_run import read, save
from jev_harness.cli import RESOURCES
from jev_harness.paired_selection import accuracy_review, cost_review, first_fork, review_scores
from phase1_run import run


def generation(prompt, text):
    return {"event":"model", "operation":"generate", "request":{"messages":[prompt], "seed":1},
            "response":{"text":text}}


def choice(options, index):
    return {"event":"model", "operation":"decide", "request":{"options":options},
            "response":{"index":index}}


class PairedSelectionTests(TestCase):
    def test_late_generation_matches_actual_choices_and_the_entire_request(self):
        x = [choice(["A","B"],0), generation("q","one"), generation("r","two")]
        y = [choice(["X","A"],1), generation("q","one"), generation("r","three")]
        self.assertEqual(first_fork(x, y), "generation_2")
        y[-1]["request"]["seed"] = 9
        self.assertIsNone(first_fork(x, y))
        self.assertEqual(first_fork([choice(["A","B"],0)], [choice(["A","B"],1)]), "decision")
        self.assertIsNone(first_fork(x, [choice(["A","C"],1), *x[1:]]))

    def test_clear_margin_and_cost_only_repeats(self):
        pairs = {t:{"success":[0,1],"tokens":[4,4],"same_request_fork":None} for t in "abc"}
        pairs["d"] = {"success":[1,0], "tokens":[2,4], "same_request_fork":"generation_1"}
        self.assertEqual(accuracy_review(pairs), ([2,4], []))
        self.assertEqual(cost_review(pairs, {}), [])
        costs = {"a":{"success":[1,1], "tokens":[15,7], "same_request_fork":"generation_2"},
                 "b":{"success":[1,1], "tokens":[3,4], "same_request_fork":None}}
        self.assertEqual(cost_review(costs, {}), ["a"])
        repeated = {"a":[{"success":True,"output_tokens":4},{"success":True,"output_tokens":10}]}
        self.assertEqual(review_scores(costs, repeated), [[2, -12.5], [2, -12.5]])
        self.assertEqual(cost_review(costs, repeated), [])

    def test_train_and_dev_use_repeats_once_and_resume_the_selected_version(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = read(RESOURCES/"experiment.json")
            config["paired_selection_from_batch"] = 1
            study = Mock(root=root, settings=config, functions=[])
            def panel(tasks, stage, candidate):
                rows = []
                for i, t in enumerate(tasks):
                    solved = i == 0 if candidate else i in (1,2)
                    row = {"task_id":t, "seed":1, "success":solved, "output_tokens":10,
                           "seconds":1,"model_calls":{}, "trace_path":f"runs/{stage}/{t}/attempt_1/trace.jsonl"}
                    folder = root/"runs"/stage/t
                    save(folder/"record.json", row)
                    trace = root/row["trace_path"]
                    trace.parent.mkdir()
                    prompt = "changed" if candidate and i == 0 else "q"
                    trace.write_text(json.dumps(generation(prompt, str(solved)))+"\n",encoding="utf-8")
                    rows.append(row)
                return rows
            study.batch.side_effect = lambda tasks,stage,library: panel(tasks, stage, stage=="dev_b01r01")
            study.candidate_batch.side_effect = lambda tasks,stage,lib,parents,old: (parents,panel(tasks,stage,True))
            def repeat(study, tasks, stages, variants, libraries):
                self.assertEqual(variants, {"A":"B","B":"B"})
                self.assertEqual(len(tasks), 3)
                return [[{"task_id":t, "seed":1, "success":arm=="B", "output_tokens":20,
                          "gpu":"node:0", "trace_path":f"runs/{stages[arm]}/{t}/attempt_1/trace.jsonl"}
                         for t in tasks] for arm in ("A","B")]
            prop={"objects":[{"kind":"handoff","name":"default","value":{"skill":"Use the current values."}}]}
            with patch("jev_harness.phase1_optimizer.propose_one", return_value=prop), patch("phase1_run.paired_rollouts",side_effect=repeat) as rerun:
                run(study, 1, 1, True, evaluate_test=False)
                result = read(root/"batches/b01/b01r01/result.json")
                checkpoint = read(root/"checkpoints.json")[0]
                self.assertTrue(result["accepted"])
                self.assertEqual(result["candidate_metrics"]["successes"], 1)
                self.assertEqual(result["paired_selection"]["candidate_score"][0], 2)
                self.assertEqual(checkpoint["next_version"], "b01r01")
                self.assertEqual(checkpoint["dev_selection"]["reference"][0], 1)
                self.assertEqual(checkpoint["dev_selection"]["paired_selection"]["repeat_output_tokens"],120)
                run(study, 1, 1, True, evaluate_test=False)
                self.assertEqual(rerun.call_count,2)
