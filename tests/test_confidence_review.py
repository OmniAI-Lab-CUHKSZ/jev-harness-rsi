import json
import unittest
from unittest.mock import Mock, patch

from jev_harness.batch_feedback import prefix_view
from jev_harness.formal_runtime import BudgetExhausted
from jev_harness.fragment_assembly import option
from jev_harness.phase1_optimizer import ProbeComplete
from jev_harness.runtime import STEP_REVIEW, Trace, Uncovered
from jev_harness.semantic_assembly import SemanticAssembly
from jev_harness.semantic_library import initial_library
from jev_harness.semantic_workbench import SemanticProbe


def head(index, probabilities):
    return {"mode":"D", "index":index, "probabilities":probabilities, "output_tokens":0}


def review(answer, tokens=12, truncated=False):
    return {"mode":"R", "text":"Reasoning.\n</think>\n\n"+answer, "output_tokens":tokens,
            "reasoning_tokens":tokens-2, "truncated":truncated}


def engine(responses, *, cls=SemanticAssembly, threshold=.70, **limits):
    model = Mock()
    model.request.side_effect = responses
    e = cls(model, None, step_review_threshold=threshold, **limits)
    e.trace, e.seed, e.done = Trace(), 10, False
    e.prepare_options()
    e.context = {"task":{"instruction":"Find the referenced item", "functions":[], "supervisor":{}},
                 "vars":{}, "history":[], "generated":{}}
    e.initialize(initial_library())
    return e


class ReviewTests(unittest.TestCase):
    def test_official_review_uses_fused_index_at_the_thinking_limit(self):
        # A capped thought still produces an option distribution from the published service.
        fused = {**head(1, [.34, .60, .06]), "mode":STEP_REVIEW, "thinking_used":True,
                 "native_probabilities":[.53, .40, .07], "reasoned_probabilities":[.15, .80, .05],
                 "output_tokens":16, "reasoning_tokens":16, "finished_within_budget":False}
        e = engine([fused], threshold=.6, step_review_mode="official_adaptive",
                   step_review_thinking_budget=256, output_budget=16)
        selected = e.choose("step", [option("Read songs","songs"), option("Read notes","notes")])
        self.assertEqual(selected, "notes")
        call = e.model.request.call_args
        self.assertEqual((call.args, call.kwargs["thinking"],call.kwargs["think_budget"]), (("decide",), "auto", 16))
        self.assertEqual((e.requests, e.model.request.call_count), (2, 1))
        event = next(row for row in e.trace.events if row["event"] == "step_review")
        self.assertEqual((event["native_index"], event["final_index"], event["outcome"]), (0, 1, "changed"))
        self.assertFalse(event["finished_within_budget"])

    def test_official_focused_none_expands_before_spending_thinking_tokens(self):
        e = engine([head(0,[.9, .05, .05]), head(1,[.45,.55]), head(1,[.1,.8,.1])],
                   threshold=.6, step_review_mode="official_adaptive", step_review_thinking_budget=256)
        names = ["library","album"]
        families = {"focused":("first",names[:1]), "other":("second",names[1:])}
        with patch.object(e, "unavailable", side_effect=lambda n,c:None if n in names else "unavailable"), \
             patch.object(e, "families", return_value=families):
            selected = e.next_card()
        calls = e.model.request.call_args_list
        self.assertEqual([c.kwargs.get("thinking","off") for c in calls], ["off","off","auto"])
        self.assertEqual([len(c.kwargs["options"]) for c in calls], [3,2,3])
        self.assertEqual(calls[-1].kwargs["think_budget"], 256)
        self.assertFalse(any(ev["event"] == "step_review" for ev in e.trace.events))
        self.assertEqual(selected, [ev for ev in e.trace.events if ev["event"] == "card_menu"][-1]["offered"][1])

    def test_threshold_is_strict_and_other_choices_and_legacy_tasks_stay_native(self):
        for name, threshold, probability in (("step",.7,.7), ("step",.6,.6), ("step",.6,.65),
                                             ("assembly",.7,.52), ("entity",.7,.52), ("step",None,.52)):
            with self.subTest(name=name, threshold=threshold):
                e = engine([head(0,[probability,1-probability])], threshold=threshold)
                self.assertEqual(e.choose(name,[option("Known object","id")]), "id")
                self.assertEqual(e.model.request.call_count,1)
        e = engine([head(0,[.59,.41]),review("0")],threshold=.6)
        self.assertEqual(e.choose("step",[option("Known object","id")]), "id")
        self.assertEqual(e.model.request.call_count,2)

    def test_widen_before_review_and_charge_the_same_option_menu(self):
        e = engine([head(0,[.9,.05,.05]), head(1,[.45,.55]), head(2,[.2,.3,.5]), review("1")])
        names = ["library","album"]
        families = {"focused":("first",names[:1]), "other":("second",names[1:])}
        with patch.object(e, "unavailable", side_effect=lambda n,c:None if n in names else "unavailable"), \
             patch.object(e, "families", return_value=families):
            selected = e.next_card()
        menus = [event for event in e.trace.events if event["event"] == "card_menu"]
        self.assertEqual(selected, menus[-1]["offered"][1])
        self.assertEqual(len(menus),2)
        model = [event for event in e.trace.events if event["event"] == "model"]
        scored = [event for event in e.trace.events if event["event"] == "step_review"]
        self.assertEqual([(r["native_index"],r["final_index"],r["outcome"]) for r in scored], [(2,1,"changed")])
        offered = json.loads(model[-1]["request"]["messages"][-1]["content"])["options"]
        self.assertEqual(list(offered.values()), model[-2]["request"]["options"])
        self.assertEqual((model[-1]["response"]["mode"],model[-1]["request"]["max_tokens"],
                          model[-1]["request"]["thinking_budget"]), (STEP_REVIEW,128,96))
        requests, decisions, _, boundary = prefix_view(e.trace.events)
        self.assertEqual((len(requests),decisions[-1][1][1],boundary), (3,selected,None))

    def test_reviewed_none_invalid_or_truncated_hands_off_once(self):
        for answer, truncated, outcome in (("NONE",False,"changed"), ("17",False,"invalid"),
                                          ("first",False,"invalid"), ("0",True,"invalid")):
            e = engine([head(0,[.55,.45]),review(answer, truncated=truncated)])
            with self.assertRaisesRegex(Uncovered,"handed off"):
                e.choose("step",[option("Partial read","read")],widen_on_none=True)
            receipt = next(r for r in e.trace.events if r["event"]=="step_review")
            self.assertEqual((receipt["outcome"],receipt["final_index"]), (outcome,1))
            self.assertEqual(e.model.request.call_count,2)

    def test_paged_abstentions_review_only_the_original_least_certain_page(self):
        probs = [.48/15]*15 + [.52]
        e = engine([head(15,probs), head(1,[.17,.83]), review("12")])
        choices = [option(f"Option {i}", f"card_{i}") for i in range(16)]
        self.assertEqual(e.choose("step", choices),"card_12")
        self.assertEqual(e.model.request.call_count,3)

    def test_review_shares_task_budgets_and_can_run_before_a_probe_generation_boundary(self):
        e = engine([head(1,[.48,.52]), review("0",tokens=5)], cls=SemanticProbe, max_calls=2, output_budget=5)
        e.set_probe_target(None)
        self.assertEqual(e.choose("step",[option("Read","read")]),"read")
        self.assertEqual(e.model.request.call_args.kwargs["max_tokens"],5)
        self.assertEqual(e.probe_g,0)
        with self.assertRaises(ProbeComplete):
            e._model("generate", purpose="G", messages=[])
        self.assertEqual(e.requests,2)
        e = engine([head(0,[.55,.45])], max_calls=1)
        with self.assertRaises(BudgetExhausted):
            e.choose("step",[option("Read","read")])
        self.assertEqual(e.model.request.call_count,1)
