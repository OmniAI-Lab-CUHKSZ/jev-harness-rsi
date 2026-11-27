import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import Mock, patch

from jev_harness.artifacts import read, save
from jev_harness.paired_selection import TWO_REPEATS_ONLY, accuracy_review, cost_review, first_fork, review_scores
from jev_harness.paired_selection import review_versions


def generation(prompt, text):
    return {"event":"model", "operation":"generate", "request":{"messages":[prompt], "seed":1},
            "response":{"text":text}}


def choice(options, index):
    return {"event":"model", "operation":"decide", "request":{"options":options},
            "response":{"index":index}}


class PairedSelectionTests(TestCase):
    def test_native_read_result_and_local_effect_must_match_before_stochastic_fork(self):
        start = {"event": "native_protocol"}
        call = {"event": "api", "app": "store", "api": "show", "arguments": {},
                "response": {"value": [{"id": 2}], "done": False, "api_total": 1}}
        effect = {"event": "native_effect", "tool": "store__send", "effect": {"id": 2}}
        left = [start, call, effect, generation("q", "one")]
        right = [start, call, effect, generation("q", "two")]
        self.assertEqual(first_fork(left, right), "generation_1")
        right[1] = {**call, "response": {**call["response"], "value": [{"id": 4}]}}
        self.assertIsNone(first_fork(left, right))
        right[1] = call
        right[2] = {**effect, "effect": {"id": 4}}
        self.assertIsNone(first_fork(left, right))

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

    def test_two_pass_cost_scores_and_initially_identical_tasks(self):
        pairs = {"a":{"success":[1,1], "tokens":[15,7], "same_request_fork":"generation_1"},
                 "b":{"success":[1,1], "tokens":[3,4], "same_request_fork":None}}
        self.assertEqual(cost_review(pairs, {}), ["a"])
        repeats = {"a":[{"success":True,"output_tokens":4}, {"success":True,"output_tokens":10},
                        {"success":True,"output_tokens":8}, {"success":True,"output_tokens":2}]}
        self.assertEqual(review_scores(pairs, repeats), [[2,-9], [2,-10]])
        self.assertEqual(cost_review(pairs, repeats), [])

    def test_pending_dev_reuses_first_pass_reverses_arms_and_resumes_second_pass(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            study = Mock(root=root, settings={"confirmation_scoring":TWO_REPEATS_ONLY})
            worker = Mock(gpu="node:1")
            study.execute_preferred.side_effect = lambda jobs, op, pref: [op(job, worker) for job in jobs]
            def record(stage, task, success, tokens):
                trace = Path("runs", stage, task, "attempt_1", "trace.jsonl")
                row = {"task_id":task, "seed":1, "gpu":worker.gpu, "success":success,
                       "output_tokens":tokens, "trace_path":trace.as_posix()}
                save((root/trace).parents[1]/"record.json", row)
                (root/trace).parent.mkdir()
                (root/trace).write_text(json.dumps(generation("same", str(success)))+"\n",encoding="utf-8")
                return row
            parent = [record("p","flip",True,8000), record("p","fixed",True,10)]
            candidate = [record("c","flip",False,13), record("c","fixed",True,10)]
            record("c_review_parent","flip",False,130)
            record("c_review_candidate","flip",True,50)
            executed, warming = [], True
            def rollout(task, stage, variant, library, *, workers):
                nonlocal warming
                path = root/"runs"/stage/task/"record.json"
                if path.exists():
                    return read(path)
                if stage == "c_review_parent_2" and warming:
                    warming = False
                    raise InterruptedError()
                executed.append(stage)
                return record(stage, task, True, 110 if library=="parent" else 70)
            study.run_task.side_effect = rollout
            report_path = root/"selection.json"
            def select():
                return review_versions(study, "c", report_path, parent, candidate, "parent", "candidate")
            with self.assertRaises(InterruptedError):
                select()
            self.assertNotIn("accepted", read(report_path))
            result = select()
            self.assertEqual(executed, ["c_review_candidate_2","c_review_parent_2"])
            self.assertEqual(result["parent_score"], [1.5,-130])
            self.assertEqual(result["candidate_score"], [2,-70])
            self.assertTrue(result["accepted"])
            self.assertEqual(result["repeat_rollouts"], 4)
            self.assertEqual(result["repeat_output_tokens"], 360)
            observations = result["repeats"]["flip"]
            self.assertEqual([(r["confirmation_pass"],r["arm"]) for r in observations],
                             [(1,"parent"),(1,"candidate"),(2,"parent"),(2,"candidate")])
            self.assertEqual(len({r["trace_path"] for r in observations}), 4)

    def test_fast_task_starts_its_second_confirmation_while_slow_task_is_in_its_first(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            study = Mock(root=root, settings={"confirmation_scoring": TWO_REPEATS_ONLY})
            tasks = ["slow", "fast"]
            fast_second_pass = Event()
            calls = {task: [] for task in tasks}

            def panel(stage, success):
                rows = []
                for task in tasks:
                    trace = Path("runs", stage, task, "attempt_1", "trace.jsonl")
                    (root/trace).parent.mkdir(parents=True)
                    (root/trace).write_text(json.dumps(generation("same", str(success)))+"\n", encoding="utf-8")
                    rows.append({"task_id": task, "seed": 17, "success": success,
                                 "output_tokens": 9, "trace_path": trace.as_posix()})
                return rows

            def dispatch(jobs, operation, preferred):
                with ThreadPoolExecutor(2) as pool:
                    work = [pool.submit(operation, job, Mock(gpu=f"node:{job[0]}")) for job in jobs]
                    return [future.result() for future in work]

            def rollout(task, stage, variant, library, *, workers):
                calls[task].append((stage, workers.gpu))
                if task == "slow" and len(calls[task]) == 1:
                    self.assertTrue(fast_second_pass.wait(timeout=3), "The fast task was blocked by the slow one")
                if task == "fast" and stage.endswith("_2"):
                    fast_second_pass.set()
                return {"task_id": task, "seed": 17, "gpu": workers.gpu, "success": True,
                        "output_tokens": 4, "trace_path": f"runs/{stage}/{task}/attempt_1/trace.jsonl"}

            study.execute_preferred.side_effect = dispatch
            study.run_task.side_effect = rollout
            result = review_versions(study, "c", root/"selection.json", panel("p", True),
                                     panel("c", False), "parent", "candidate")
            self.assertEqual(result["repeat_rollouts"], 8)
            self.assertEqual(result["parent_score"], [2, -8])
            self.assertEqual(result["candidate_score"], [2, -8])
            self.assertTrue(result["accepted"])
            for task, arms, gpu in (("slow", ["parent","candidate","candidate_2","parent_2"], "node:0"),
                                    ("fast", ["candidate","parent","parent_2","candidate_2"], "node:1")):
                self.assertEqual(calls[task], [(f"c_review_{arm}", gpu) for arm in arms])
