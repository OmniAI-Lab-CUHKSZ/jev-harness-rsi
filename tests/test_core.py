import copy
import unittest

from jev_harness.metrics import aggregate, comparison
from jev_harness.runtime import parse_json


class CoreTests(unittest.TestCase):
    def test_reasoning_and_fences_do_not_break_json(self):
        self.assertEqual(parse_json('<think>reason</think>```json\n{"index":2}\n```'), {"index": 2})

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
