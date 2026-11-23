import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jev_harness.card_assembly import CardAssembly, public_apps, reachable_cards
from jev_harness.card_library import initial_cards, patch_cards
from jev_harness.phase1_optimizer import Workbench
from jev_harness.card_workbench import CardWorkbench
from jev_harness.runtime import Trace, Uncovered


def engine(request="Play a Spotify song",library=None):
    executor=CardAssembly(None,None)
    executor.context={"task":{"instruction":request,"functions":[],"datetime":"2023-05-18"},
                      "vars":{},"history":[],"generated":{}}
    executor.trace,executor.done=Trace(),False
    executor.initialize(library or initial_cards())
    return executor


class CardTests(unittest.TestCase):
    def test_actual_menu_carries_scope_count_read_capability_and_mutable_skill(self):
        lib=initial_cards()
        obj={"kind":"card","name":"filter_genre","value":{**lib["cards"]["filter_genre"],
             "skill":"Match eligibility before ordering."}}
        new,edits=patch_cards(lib,{"objects":[obj]},[])
        e=engine(library=new)
        e.public={"scope":["saved albums"],"song_ids":[2,3,5]}
        def choose(name,choices,**state):
            labels={c["value"]:c["label"] for c in choices}
            self.assertIn("song_ids (3 records)",labels["filter_genre"])
            self.assertIn("genre for each current song ID",labels["filter_genre"])
            self.assertIn("Match eligibility before ordering.",labels["filter_genre"])
            self.assertNotIn("submit_answer",labels)
            self.assertNotIn("take_first",labels)
            self.assertEqual(state["scope"],["saved albums"])
            return "filter_genre"
        with patch.object(e,"choose",side_effect=choose):
            self.assertEqual(e.next_card(),"filter_genre")
        self.assertEqual(edits,["card/filter_genre"])
        self.assertNotEqual(lib["cards"]["filter_genre"]["skill"],new["cards"]["filter_genre"]["skill"])

    def test_filter_sort_take_support_answer_or_action_without_repeating_lookup(self):
        live={1:{"song_id":1,"title":"A","genre":"jazz","play_count":90},
              2:{"song_id":2,"title":"B","genre":"EDM","play_count":30},
              3:{"song_id":3,"title":"C","genre":"EDM","play_count":69}}
        e=engine("Play the most played EDM song on Spotify")
        e.public={"scope":["library"],"song_ids":[1,2,3]}
        def choose(name,choices,**facts):
            return {"genre":"EDM","order":("play_count",True),"quantity":1}[name]
        def api(app,name,args,node):
            if name=="show_song":
                return live[args["song_id"]]
            if name=="play_music":
                self.assertEqual(args["song_id"],3)
                return {"playing":True}
            self.assertEqual(name,"complete_task")
            e.done=True
        with patch.object(e,"choose",side_effect=choose),patch.object(e,"call",side_effect=api) as calls,patch.object(e,"token",return_value="session"):
            e.perform_card("count_songs")
            e.perform_card("filter_genre")
            self.assertNotIn("answer",e.public)
            e.perform_card("sort_songs")
            self.assertEqual(e.public["ranked_songs"],[3,2])
            e.perform_card("take_first")
            e.perform_card("titles")
            self.assertEqual(e.public["answer"],"C")
            self.assertFalse(e.done)
            e.perform_card("play")
            self.assertFalse(e.done)
            e.perform_card("submit_effect")
        self.assertTrue(e.done)
        self.assertEqual(sum(c.args[1]=="show_song" for c in calls.call_args_list),3)

    def test_new_fragment_is_discovered_through_roles_across_apps_without_menu_edit(self):
        lib=initial_cards()
        card={"app":"venmo","inputs":["alarms"],"output":"answer",
              "description":"Count the observed objects.","skill":"Use when their count answers the request.",
              "compute":{"op":"length","args":[{"ref":"state.alarms"}]}}
        new,_=patch_cards(lib,{"objects":[{"kind":"card","name":"count_observed","value":card}]},[])
        e=engine("On Venmo use my personal schedule",new)
        self.assertIn("read_alarms",reachable_cards(new["cards"],public_apps(e.request,new["cards"])))
        self.assertIsNotNone(e.unavailable("count_observed",card))
        e.public["alarms"]=[{"id":101},{"id":204}]
        self.assertIsNone(e.unavailable("count_observed",card))
        e.perform_card("count_observed")
        self.assertEqual(e.public["answer"],2)
        self.assertFalse(e.done)

    def test_return_to_fragments_uses_environment_response_and_preserves_scope(self):
        e=engine("Which notes are pinned?")
        e.schemas={"simple_note__search_notes":{"properties":{"pinned":{},"access_token":{},"page_index":{},"page_limit":{}}}}
        e.residual_progress={}
        with patch.object(e,"call",return_value=[{"note_id":42}]) as api,patch.object(e,"token",return_value="session"),patch.object(e,"assemble_available") as again:
            e.code("yield_read('simple_note__search_notes', {'pinned': True})","generation_agent")
        self.assertEqual(e.public["note_summaries"],[{"note_id":42}])
        self.assertEqual(e.state_view()["read_scopes"]["note_summaries"]["arguments"],{"pinned":True})
        again.assert_called_once()
        self.assertEqual(api.call_args.args[2]["pinned"],True)
        self.assertEqual(e.trace.events[-1]["event"],"fragment_reentry")

    def test_slot_and_handoff_skill_follow_one_card(self):
        card={"app":"general","inputs":["entries"],"output":"answer","description":"Select one entry",
              "parameters":{"entry":{"question":"Which entry?","candidates":{"source":{"ref":"state.entries"},
              "label":{"ref":"item.name"},"value":{"ref":"item.id"}}}},
              "skill":"Match the live entry name.","handoff_skill":"Reuse the real entry ID.",
              "compute":{"ref":"params.entry"}}
        lib,_=patch_cards(initial_cards(),{"objects":[{"kind":"card","name":"select_entry","value":card}]},[])
        e=engine(library=lib)
        e.public={"entries":[{"id":7,"name":"Blue"}]}
        with patch.object(e,"decision",return_value=7) as choice,patch.object(e,"prime_generated_session"):
            e.perform_card("select_entry")
            self.assertEqual(e.handoff("gap")["fragment_guidance"],["Reuse the real entry ID."])
        self.assertIn(card["skill"],choice.call_args.kwargs["skill"])
        self.assertEqual(e.public["answer"],7)

    def test_rejects_hidden_completion_and_handles_ambiguous_read_names(self):
        lib=initial_cards()
        bad={**lib["cards"]["read_alarms"],"app":"supervisor","api":"supervisor__complete_task"}
        with self.assertRaises(ValueError):
            patch_cards(lib,{"objects":[{"kind":"card","name":"arbitrary_finish","value":bad}]},[])
        other={**lib["cards"]["read_alarms"],"output":"other_alarm_records"}
        lib,_=patch_cards(lib,{"objects":[{"kind":"card","name":"other_alarms","value":other}]},[])
        e=engine("Show my phone alarms",lib)
        e.schemas={"phone__show_alarms":{"properties":{}}}
        with self.assertRaises(Uncovered):
            e.delegate_read("phone__show_alarms",{})
        self.assertIn("other_alarms",e.read_offers())
        self.assertIsNot(CardWorkbench.probe_type,Workbench.probe_type)

    def test_formal_batch_runner_uses_card_seed_and_stops_at_perfect_zero(self):
        from phase1_run import run,read
        from jev_harness.card_library import PROTOCOL
        class FinishedBatch:
            settings={"protocol":PROTOCOL}
            def batch(self,ids,stage,library):
                assert "take_first" in library["cards"] and "workflow" not in library
                return [{"task_id":task,"success":True,"output_tokens":0,"seconds":1,
                         "model_calls":{"D":1},"fragments":["take_first"],"card_menus":1}
                        for task in ids]
            def status(self,*args,**kwargs):
                pass
        with TemporaryDirectory() as temp:
            study=FinishedBatch()
            study.root=Path(temp)
            run(study,1,2,False)
            self.assertEqual(read(study.root/"task_split.json")["protocol"],PROTOCOL)
            record=read(study.root/"checkpoints.json")[0]
            self.assertEqual((record["next_version"],record["revisions_scored"]),("h00",0))
