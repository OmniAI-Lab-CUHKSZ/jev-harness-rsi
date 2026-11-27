from unittest import TestCase
from unittest.mock import Mock

from jev_harness.composed_assembly import ComposedAssembly
from jev_harness.runtime import STEP_REVIEW, Trace


def head(index, probabilities):
    return {"mode":"D", "index":index, "probabilities":probabilities, "output_tokens":0}


def review(answer, tokens=12, truncated=False):
    return {"mode":"R", "text":"Reasoning.\n</think>\n\n"+answer, "output_tokens":tokens,
            "reasoning_tokens":tokens-2, "truncated":truncated}


def engine(responses, *, threshold, **limits):
    e = ComposedAssembly(Mock(request=Mock(side_effect=responses)), None, step_review_threshold=threshold, **limits)
    e.trace, e.seed, e.done = Trace(), 10, False
    e.prepare_options()
    e.context = {"task": {"instruction": "Find the referenced item", "functions": [], "supervisor": {}}, "history": []}
    e.initialize_native({})
    return e


class ReviewTests(TestCase):
    def test_official_review_uses_fused_index_at_the_thinking_limit(self):
        # A capped thought still produces an option distribution from the published service.
        fused = {**head(1, [.34, .60, .06]), "mode":STEP_REVIEW, "thinking_used":True,
                 "native_probabilities":[.53, .40, .07], "reasoned_probabilities":[.15, .80, .05],
                 "output_tokens":16, "reasoning_tokens":16, "finished_within_budget":False}
        e = engine([fused], threshold=.6, step_review_mode="official_adaptive",
                   step_review_thinking_budget=256, output_budget=16)
        selected = e.ask("native/action", "Which read helps?", [("Read songs", "songs"), ("Read notes", "notes")], "")
        self.assertEqual(selected, "notes")
        call = e.model.request.call_args
        self.assertEqual((call.args, call.kwargs["thinking"],call.kwargs["think_budget"]), (("decide",), "auto", 16))
        self.assertEqual((e.requests, e.model.request.call_count), (2, 1))
        event = next(row for row in e.trace.events if row["event"] == "step_review")
        self.assertEqual((event["native_index"], event["final_index"], event["outcome"]), (0, 1, "changed"))
        self.assertFalse(event["finished_within_budget"])
