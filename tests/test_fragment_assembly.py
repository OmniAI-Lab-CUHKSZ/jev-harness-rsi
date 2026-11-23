import unittest
from unittest.mock import patch

from jev_harness.formal_runtime import FormalExecutor
from jev_harness.fragment_assembly import FragmentAssembly, literal_names
from jev_harness.runtime import Trace


class AssemblyTests(unittest.TestCase):
    def test_generated_data_rejoins_dispatch_with_new_scope(self):
        agent = FragmentAssembly(None, None)
        agent.done = False
        agent.trace, agent.sequence, agent.resumed = Trace(), [], set()
        agent.public = {"scope": ["previous collection"], "song_ids": [2]}
        agent.criteria = {"field": "like_count"}
        def finish():
            agent.done = True
        with patch.object(FormalExecutor, "code", side_effect=[
            {"output": "", "done": False}, {"output": '{"scope":"named playlist","song_ids":[8,9,8]}'},
        ]), patch.object(agent, "assemble_available", side_effect=finish) as resume:
            agent.code("fragment_resume = result", "generation_agent")
        resume.assert_called_once()
        self.assertEqual(agent.public, {"scope": ["named playlist"], "song_ids": [8, 9]})
        self.assertEqual(agent.criteria, {})
        self.assertTrue(agent.done)

    def test_installed_sort_preserves_identity_across_downstream_effects(self):
        agent = FragmentAssembly(None, None)
        agent.trace = Trace()
        agent.registry = {"pieces": {"ordered_records": {"name": "by_property", "result": {
            "op": "sorted", "args": [{"ref": "rows"}, {"ref": "field"}, {"ref": "descending"}]}}}}
        rows = [{"song_id": 7, "title": "Alpha", "play_count": 2},
                {"song_id": 3, "title": "Beta", "play_count": 4}]
        ordered = agent.apply_data("ordered_records", rows=rows, field="play_count", descending=True)
        self.assertEqual(ordered[0]["song_id"], 3)
        self.assertEqual(agent.apply_data("title_text", rows=ordered[:1]), "Beta")
        self.assertEqual(agent.apply_data("unique_ids", ids=[7, 3, 7]), [7, 3])
        self.assertEqual(agent.trace.events[0]["port"], "ordered_records")

    def test_unquoted_lowercase_name_needs_a_public_lookup(self):
        self.assertEqual(literal_names("play something from the album echo chamber chronicles."), [])
        self.assertIn("Echo Chamber Chronicles", literal_names("Use the album Echo Chamber Chronicles."))
