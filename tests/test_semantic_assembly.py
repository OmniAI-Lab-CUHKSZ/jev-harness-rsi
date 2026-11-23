import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from jev_harness.formal_runtime import FormalExecutor
from jev_harness.phase1_optimizer import ProbeComplete
from jev_harness.runtime import Trace, Uncovered
from jev_harness.semantic_assembly import SemanticAssembly
from jev_harness.semantic_library import initial_library, patch_library
from jev_harness.semantic_workbench import SemanticProbe, SemanticWorkbench, revision_attempt_folder


def engine(library=None, cls=SemanticAssembly):
    e = cls(None, None)
    e.trace, e.done = Trace(), False
    e.context = {"task":{"instruction":"Resolve the Venmo request for my coworkers.",
                         "supervisor":{"email":"user@example.com","phone_number":"1234567890"},
                         "functions":[]}, "vars":{}, "history":[], "generated":{}}
    e.initialize(library or initial_library())
    return e


class SemanticTests(unittest.TestCase):
    def test_retry_after_last_turn_read_keeps_evidence_and_requests_final_patch(self):
        with TemporaryDirectory() as tmp:
            revision = Path(tmp)
            spent = revision/"sol"/"turn_12"/"response.txt"
            spent.parent.mkdir(parents=True)
            spent.write_text('{"read":["dsl"]}', encoding="utf-8")
            attempt = revision_attempt_folder(revision)
            self.assertEqual(attempt.name, "sol_retry_01")

            study = Mock(root=revision, functions=[])
            bench = SemanticWorkbench(study, initial_library(), [], [], attempt, set())
            candidate = {"probe_task":"train_1", "objects":[{"kind":"decision", "name":"step",
                         "value":{"question":"Which action follows from the evidence?"}}]}
            account = study.solver.return_value
            account.turn.side_effect = [json.dumps(candidate), json.dumps(candidate)]
            def probe(proposal, task):
                bench.last_probed = bench.probe_key(proposal, task)
                return {"stop":"Reached card_executed"}
            with patch.object(bench, "continue_prompt", side_effect=lambda prompt:(11, prompt)), \
                 patch.object(bench, "probe", side_effect=probe) as trial:
                self.assertEqual(bench.work(), candidate)
            self.assertEqual(trial.call_count, 1)
            review = account.turn.call_args_list[-1].args[1]
            self.assertIn("Reached card_executed", review)
            self.assertIn("final interaction (12 of 12)", review)
            self.assertEqual(json.loads(spent.read_text(encoding="utf-8")), {"read":["dsl"]})

    def test_jev_selects_an_unmentioned_app_before_the_concrete_action(self):
        e = engine()
        def pick(name, options, **state):
            values = [x["value"] for x in options]
            if name == "assembly":
                self.assertIn("phone", values)
                self.assertIn("venmo", values)
                return "phone"
            self.assertEqual(values, ["read_alarms"])
            return values[0]
        with patch.object(e,"choose",side_effect=pick) as head:
            self.assertEqual(e.next_card(), "read_alarms")
        self.assertEqual([c.args[0] for c in head.call_args_list], ["assembly","step"])
        self.assertEqual(e.trace.events[0]["app_hints"], ["venmo"])

    def test_rejected_menu_expands_before_handoff(self):
        for family_abstains in (False, True):
            e = engine()
            def choose(name, options, **state):
                if name == "assembly" and not family_abstains:
                    return "phone"
                values = [row["value"] for row in options]
                if name == "step" and "read_pending_requests" in values:
                    return "read_pending_requests"
                raise Uncovered("Jev selected the uncovered option")
            with patch.object(e,"choose",side_effect=choose):
                self.assertEqual(e.next_card(), "read_pending_requests")
            self.assertIn("read_pending_requests", e.trace.events[-1]["offered"])

    def test_cross_app_dependency_is_compiled_before_its_consumer(self):
        lib = initial_library()
        card = {"app":"venmo","inputs":["alarms"],"output":"answer","compute":{"op":"length","args":[{"ref":"state.alarms"}]},
                "description":"Count supplied records","skill":"Use the requested scope."}
        rules = [{"kind":"card","name":"consumer","value":card},
                 {"kind":"assembly","name":"combined","value":{"description":"Process the outside data","cards":["consumer"]}}]
        lib, _ = patch_library(lib, {"objects":rules}, set())
        e = engine(lib)
        eligible = [n for n,c in lib["cards"].items() if e.unavailable(n,c) is None]
        self.assertIn("read_alarms", e.families(eligible)["venmo"][1])
        self.assertNotIn("consumer", eligible)
        e.public["alarms"] = []
        eligible = [n for n,c in lib["cards"].items() if e.unavailable(n,c) is None]
        self.assertEqual(e.families(eligible)["combined"][1], ["consumer"])

    def test_login_and_api_adapters_use_the_documented_identity_and_result(self):
        lib, _ = patch_library(initial_library(), {"objects":[
            {"kind":"adapter","name":"phone__search_contacts",
             "value":{"argument_map":{"group":"relationship"},"result_path":"rows","page_size":1}}]}, set())
        e = engine(lib)
        e.schemas = {"phone__search_contacts":{"properties":{"access_token":{},"relationship":{},"page_index":{},"page_limit":{}}}}
        def api(app, name, args, node=None):
            if name == "show_account_passwords":
                return [{"account_name":"phone","password":"pw"}]
            if name == "login":
                self.assertEqual(args["username"], "1234567890")
                return {"access_token":"session"}
            self.assertEqual((args["relationship"],args["access_token"],args["page_limit"]), ("roommate","session",1))
            return {"rows":[{"id":1}] if args["page_index"] == 0 else []}
        with patch.object(FormalExecutor,"call",side_effect=api):
            self.assertEqual(e.call_endpoint("phone__search_contacts",{"group":"roommate"},True), [{"id":1}])

    def test_selected_parameter_reaches_state_and_generation(self):
        lib, edits = patch_library(initial_library(), {"objects":[
            {"kind":"card","name":"choose_bound","value":{"app":"general","inputs":[],"output":"bound",
             "description":"Bind boundary","skill":"Respect request","compute":{"ref":"params.boundary"},
             "parameters":{"boundary":{"question":"Include the boundary?","candidates":[{"label":"Include","value":"inclusive"}]}}}},
            {"kind":"state","name":"boundary","value":{"requires":["bound"],"program":"return state['values']['bound']"}},
            {"kind":"handoff","name":"after_bound","value":{"requires":["bound"],"skill":"Use the verified boundary."}}
        ]}, set())
        self.assertEqual(len(edits), 3)
        e = engine(lib)
        with patch.object(e,"choose_inline",return_value="inclusive"), patch.object(e,"prime_generated_session"):
            e.perform_card("choose_bound")
            self.assertEqual(e.state_view()["bindings"]["bound"]["parameters"], {"boundary":"inclusive"})
            self.assertEqual(e.handoff("missing write")["decision_evidence"]["views"]["boundary"], "inclusive")
            self.assertIn("Use the verified boundary.", e.handoff("missing write")["fragment_guidance"])

    def test_decision_policy_reorders_only_grounded_choices_and_probe_can_follow_g(self):
        lib, _ = patch_library(initial_library(), {"objects":[{"kind":"decision","name":"step",
             "value":{"question":"Which step?","program":"return [1, 0]"}}]}, set())
        e = engine(lib)
        with patch.object(e, "decision", side_effect=lambda q,st,opts,**kw:opts[0]["value"]):
            self.assertEqual(e.choose("step",[{"value":"a","label":"A"},{"value":"b","label":"B"}]),"b")
        trial = engine(lib, SemanticProbe)
        trial.set_probe_target({"event":"api_action","url":"/phone/contacts","max_g":1})
        with patch.object(SemanticAssembly,"_model",return_value={"text":"test"}):
            self.assertEqual(trial._model("generate")["text"], "test")
        trial.trace.add("api_action",url="/phone/contacts")
        with self.assertRaisesRegex(ProbeComplete,"Reached api_action"):
            trial.check_goal()

    def test_replaced_builtin_uses_its_declared_public_read_on_reentry(self):
        card = {"app":"spotify","inputs":[],"output":"album_rows","description":"Read saved albums.",
                "skill":"Use a saved album scope.","api":"spotify__show_album_library","public_read":True}
        lib, _ = patch_library(initial_library(), {"objects":[{"kind":"card","name":"album","value":card}]},set())
        e = engine(lib)
        e.schemas = {"spotify__search_albums":{},"spotify__show_album_library":{}}
        self.assertEqual(e.read_offers()["album"], card["api"])
        with patch.object(e, "call_endpoint", return_value=[{"album_id":12}]) as call:
            e.delegate_read("album", {})
        call.assert_called_once_with("spotify__show_album_library", {}, False)
        self.assertEqual(e.public["album_rows"], [{"album_id":12}])
        self.assertEqual(e.provenance["album_rows"]["producer"], "album")
        self.assertIn("album_rows", e.derived)

    def test_take_consumes_ranked_ids_from_an_independent_provider(self):
        e = engine()
        e.public["ranked_songs"] = [42, 7]
        self.assertIsNone(e.unavailable("take_first",e.registry["cards"]["take_first"]))
        with patch.object(e,"choose",return_value=1):
            e.perform_card("take_first")
        self.assertEqual(e.public["song_ids"], [42])
