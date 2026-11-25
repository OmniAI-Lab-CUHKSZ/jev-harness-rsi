from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from formal_run import read
from jev_harness.allocation import SharedStudy
from jev_harness.cli import RESOURCES
from jev_harness.expressions import parameter_groups
from jev_harness.retrieval import card_scores, family_groups, step_groups
from jev_harness.semantic_handoff import guide_groups, menu
from jev_harness.solved_train_preview import different_rules, saved_page


def operation(app, inputs, output, description):
    return dict(app=app,inputs=inputs,output=output,description=description)


class RetrievalTests(unittest.TestCase):
    def test_two_rounds_ignore_unrelated_new_card_and_retain_previous_consumer(self):
        cards = read(RESOURCES/"h00.json")["cards"]
        extra = operation("phone", [], "messages", "Read phone text messages")
        request = "What is the title of the most liked song in my Spotify playlist?"
        live = {"song_ids":[1], "ranked_songs":[1], "limited_songs":[1]}
        eligible = ["count_songs","titles","play","read_notes"]
        def retrieve(lib, available):
            scores = card_scores(lib,request,live,{"spotify"})
            return step_groups(eligible[:3],available,scores,lib,live,request)
        old = retrieve(cards,eligible)
        new = retrieve({**cards,"texts":extra},eligible+["texts"])
        self.assertEqual(old,new)
        self.assertIn("titles",new[-1])
        self.assertNotIn("read_notes",new[-1])

    def test_cross_application_provider_without_any_goal_word_is_retrievable(self):
        cards = {"pay":operation("venmo",["people"],"receipt","Pay coworkers"),
                 "lookup":operation("phone",[],"people","Load directory"),
                 "notes":operation("notes",[],"pages","Read notebook")}
        scores = card_scores(cards,"Pay my coworkers",{},{"venmo"})
        menus = {"phone":("Phone",["lookup"]), "notes":("Notes",["notes"])}
        self.assertEqual(family_groups(menus,scores,"Pay my coworkers",{"venmo"}),[["phone"]])
        self.assertEqual(step_groups(["lookup"],["lookup","notes"],scores,cards,{},"Pay my coworkers"),[["lookup"]])

    def test_refused_entity_can_reappear_with_discriminating_evidence(self):
        rows = [{"label":t,"value":{"title":t,"note_id":i}} for i,t in enumerate(
            ["Work dinner expense shares","Work dinner at the cafe"]+[f"Garden journal {i}" for i in range(34)])]
        slot = {"candidates":rows,"prefer":{"op":"contains","args":[{"ref":"label"},"Work dinner"]}}
        ctx = {"task":{"instruction":"Use the work dinner note about individual shares"}}
        old = parameter_groups(slot,ctx)
        new = parameter_groups(slot,ctx,retrieve=True)
        self.assertEqual([len(g) for g in old],[2,34])
        self.assertEqual([len(g) for g in new],[2,1])
        self.assertEqual(new[1][0]["value"],rows[0]["value"])
        self.assertIn("share",new[1][0]["label"])
        self.assertNotEqual(new[1][0]["label"],new[0][0]["label"])

    def test_guide_and_ordinary_alternatives_travel_together_with_their_evidence(self):
        lib = {"menu_retrieval":"state","handoffs":{
            "pay":{"label":"Pay the bill","requires":["receipt"],"alternatives":["Report the bill only"]},
            "music":{"label":"Play music","requires":["songs"]},
            "calendar":{"label":"Reschedule reservation","requires":["booking"]}}}
        state = {"request":"Pay the bill","values":{"receipt":3,"songs":[2]}}
        self.assertEqual(guide_groups(lib,state),[["pay"]])
        recent = {**state, "completed":["lookup_track"], "producers":{"songs":"lookup_track"}}
        self.assertEqual(guide_groups(lib,recent),[["pay"],["music"]])
        names,opts,facts = menu(lib,state)
        self.assertEqual(names,["pay"])
        self.assertEqual(len(opts),2)
        self.assertIn("receipt",facts)
        self.assertNotIn("songs",facts)

    def test_archived_paging_keeps_the_real_row_even_if_the_current_policy_changes(self):
        events = [{"event":"page_bindings","values":[100,200]},
                  {"event":"decision_rules","widen":True,"review":True,"minimum":0},
                  {"event":"model","operation":"decide"},{"event":"model","operation":"decide"}]
        choices = [{"label":"First","value":0},{"label":"Second","value":1}]
        self.assertEqual([c["value"] for c in saved_page(events,3,choices)],[100,200])
        self.assertTrue(different_rules(events,3,dict(widen=False,review=True,minimum=0)))

    def test_fresh_six_gpu_run_requests_graphs_for_both_lanes_and_keeps_prior_args(self):
        with TemporaryDirectory() as folder:
            spec = {"model_nodes":{"node":{"args":["--model","model","--cuda-graph"]}}}
            config = {**read(RESOURCES/"experiment.json"),"cuda_graph_from_start":True}
            study = SharedStudy(Path(folder),config,spec,max_gpus=6)
            self.assertEqual((study.max_gpus,study.lanes_per_gpu,study.decode_mode),(6,2,"cuda_graph"))
            self.assertEqual(study.connections["model_nodes"]["node"]["args"].count("--cuda-graph"),1)
