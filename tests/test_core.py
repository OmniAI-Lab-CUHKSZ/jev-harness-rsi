import copy
import json
from pathlib import Path
import tempfile
import unittest

from jev_harness.metrics import aggregate, comparison
from jev_harness.evolution import apply_patch, evaluate_patch
from jev_harness.expressions import resolve, build_candidates
from jev_harness.runtime import Executor, parse_json


class FakeModel:
    def __init__(self):
        self.calls = []

    def request(self, operation, **values):
        self.calls.append((operation, values))
        if operation != "decide":
            raise AssertionError("Covered task must not call the generation path")
        options = values["options"]
        index = next(i for i, label in enumerate(options) if "bedtime" in label.lower())
        return {"index": index, "probabilities": [1.0 if i == index else 0.0 for i in range(len(options))],
                "mode": "D", "input_tokens": 90, "output_tokens": 0, "seconds": 0.01}


class FakeEnvironment:
    def __init__(self, alarms):
        self.alarms = copy.deepcopy(alarms)
        self.done = False

    def request(self, op, **values):
        if op == "prepare":
            self.done = False
            return {"task_id": values["task_id"], "instruction": "Move bedtime one hour later and disable the others.",
                    "supervisor": {}, "datetime": "2023-01-01", "functions": []}
        if op == "evaluate":
            success = self.done and all(a["time"] == "00:30" if "bedtime" in a["label"].lower() else not a["enabled"] for a in self.alarms)
            return {"success": success, "failures": []}
        if op == "call":
            api, args = values["api"], values["arguments"]
            if api == "show_alarms":
                value = copy.deepcopy(self.alarms)
            elif api == "update_alarm":
                row = next(a for a in self.alarms if a["alarm_id"] == args["alarm_id"])
                row.update({k: v for k, v in args.items() if k != "alarm_id"})
                value = {"success": True}
            elif api == "complete_task":
                self.done, value = True, {}
            else:
                raise AssertionError(api)
            return {"value": value, "output": json.dumps(value), "done": self.done}
        raise AssertionError(op)


def alarm_workflow():
    ref = lambda name: {"ref": name}
    return {"id": "alarm", "description": "Move the bedtime alarm and disable the other alarms.",
        "workflow": {"steps": [
            {"kind": "call", "binding": "list", "store": "alarms"},
            {"kind": "decision", "decision": "choose", "store": "selected"},
            {"kind": "call", "binding": "move"},
            {"kind": "foreach", "binding": "others", "item": "other", "body": [{"kind": "call", "binding": "disable"}]},
            {"kind": "finish"}]},
        "questions": {"choose": "Which alarm is the bedtime alarm?"},
        "evidence": {"choose": {"instruction": ref("task.instruction")}},
        "candidates": {"choose": {"source": ref("vars.alarms"), "label": {"template": "{item[label]} {item[time]}"}, "value": ref("item")}},
        "decision_skills": {"choose": "Choose the alarm matching the requested activity."},
        "generation_skills": {}, "mode_policy": {},
        "bindings": {
            "list": {"app": "phone", "api": "show_alarms", "arguments": {}},
            "move": {"app": "phone", "api": "update_alarm", "arguments": {"alarm_id": ref("vars.selected.alarm_id"),
                "time": {"op": "add_minutes", "args": [ref("vars.selected.time"), 60]}}},
            "others": {"filter": {"source": ref("vars.alarms"), "where": {"op": "ne", "args": [ref("item.alarm_id"), ref("vars.selected.alarm_id")]}}},
            "disable": {"app": "phone", "api": "update_alarm", "arguments": {"alarm_id": ref("vars.other.alarm_id"), "enabled": False}}
        }}


class RuntimeTests(unittest.TestCase):
    def test_unseen_ids_reordered_rows_and_midnight_are_bound_without_generation(self):
        for selected_id, other_id, reverse in [(992, 7, False), (3, 640, True)]:
            alarms = [{"alarm_id": selected_id, "label": "Bedtime", "time": "23:30", "enabled": True},
                      {"alarm_id": other_id, "label": "Lunch", "time": "12:00", "enabled": True}]
            if reverse:
                alarms.reverse()
            model, env = FakeModel(), FakeEnvironment(alarms)
            with tempfile.TemporaryDirectory() as tmp:
                executor = Executor(model, env)
                # Skip workflow routing here to isolate execution/binding of the graph.
                executor.select_workflow = lambda _: alarm_workflow()
                summary = executor.run("unseen_2", {"workflows": [alarm_workflow()]}, Path(tmp)/"trace.jsonl")
            self.assertTrue(summary["success"])
            self.assertEqual(summary["output_tokens"], 0)
            self.assertEqual([x[0] for x in model.calls], ["decide"])
            self.assertIn("requested activity", model.calls[0][1]["skill"])

    def test_question_and_candidates_are_program_expressions(self):
        context = {"vars": {"rows": [{"id": 12, "label": "New person"}]}}
        values = build_candidates({"source": {"ref": "vars.rows"},
            "label": {"template": "Person: {item[label]}"}, "value": {"ref": "item.id"}}, context)
        self.assertEqual(values, [{"label": "Person: New person", "value": 12}])
        self.assertEqual(resolve({"op": "shift_minutes", "args": ["Move it two hours earlier"]}, {}), -120)

    def test_reasoning_and_fences_do_not_break_json(self):
        self.assertEqual(parse_json('<think>reason</think>```json\n{"index":2}\n```'), {"index": 2})

    def test_decision_constructors_cannot_read_generated_candidate_text(self):
        executor = Executor(FakeModel(), FakeEnvironment([]))
        executor.context = {"task": {}, "vars": {}, "generated": {"choices": ["invented"]}, "history": []}
        with self.assertRaises(KeyError):
            build_candidates({"source": {"ref": "generated.choices"}}, executor.program_context())


class EvolutionTests(unittest.TestCase):
    def test_three_component_edit_is_rejected(self):
        library = {"workflows": [alarm_workflow()]}
        with self.assertRaises(ValueError):
            apply_patch(library, {"workflow_id": "alarm", "changes": {
                "questions": {}, "evidence": {}, "bindings": {}}})

    def test_two_component_edit_runs_and_unhelpful_edit_is_not_adopted(self):
        library = {"workflows": [alarm_workflow()]}
        patch = {"workflow_id": "alarm", "changes": {"questions": {"choose": "Different question"}, "decision_skills": {"choose": "Different skill"}}}
        calls = []
        def run(task, version, label):
            calls.append((task, label))
            return {"task_id": task, "seed": 1, "success": True,
                    "output_tokens": 20 if label == "parent" else 30, "seconds": 1}
        chosen, report = evaluate_patch(library, patch, ["task_1", "task_2"], run)
        self.assertEqual(len(calls), 4)
        self.assertFalse(report["evaluation"]["beneficial"])
        self.assertIs(chosen, library)

    def test_token_saving_cannot_compensate_for_a_failed_task(self):
        parent = [{"task_id": f"task_{i}", "seed": 1, "success": True, "output_tokens": 10000, "seconds": 1} for i in range(3)]
        candidate = copy.deepcopy(parent)
        for row in candidate:
            row["output_tokens"] = 0
        candidate[0]["success"] = False
        self.assertFalse(comparison(parent, candidate)["beneficial"])

    def test_incomplete_scenario_does_not_report_official_sgc(self):
        values = [{"task_id": "scenario_1", "seed": 1, "success": True, "output_tokens": 0, "seconds": 1}]
        self.assertIsNone(aggregate(values)["sgc"])

    def test_different_task_multiplicities_cannot_be_compared_as_matched_batches(self):
        row = lambda name: {"task_id": name, "seed": 1, "success": True, "output_tokens": 1, "seconds": 1}
        with self.assertRaises(ValueError):
            comparison([row("a"), row("a"), row("b")], [row("a"), row("b"), row("b")])


if __name__ == "__main__":
    unittest.main()
