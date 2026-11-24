from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from jev_harness.contrast_probes import MatchedPrefix
from jev_harness.batch_revision_trials import compare_wording
from jev_harness.history_impact import assess, trace_sites
from jev_harness.revision_evidence import batch_patterns, request_context, trace_incidents
from jev_harness.semantic_library import patch_library
from jev_harness.trace_comparison import compare_path
from test_scoped_revision import actor, library


class RevisionEvidenceTests(TestCase):
    def test_generated_and_compiled_login_failures_have_different_owners(self):
        es = [{"event":"adapter_login","app":"phone","success":False,"identity_field":"email"},
              {"event":"tool_batch","node":"generation_agent","response":{"output":"Execution failed.\nKeyError: access_token"}},
              {"event":"api","node":"session","app":"phone","api":"login","arguments":{"password":"secret"}}]
        self.assertNotEqual(trace_incidents(es)[0]["observation"],trace_incidents(es)[1]["observation"])
        self.assertFalse(request_context(es,1,library())["compiled_endpoint_adapters_run"])
        ctx = request_context(es,2,library())
        self.assertEqual(ctx["login_identity_field"],"phone_number")
        self.assertNotIn("secret",str(ctx))

    def test_patterns_count_scenarios_include_successes_and_keep_different_decision_sites(self):
        def none(node):
            return {"event":"decision","node":"fragment/"+node,"selected":1,"candidates":[{"value":"list"},{"value":None}]}
        items = [{"task":"a_1","success":False},{"task":"a_2","success":True},{"task":"b_1","success":True}]
        incidents = {r["task"]:trace_incidents([none("step"),none("assembly")]) for r in items}
        table = batch_patterns(items,incidents)
        self.assertEqual(len(table),2)
        self.assertEqual(table[0]["distinct_scenarios"],2)
        self.assertEqual(table[0]["also_on_successes"],["a_2","b_1"])

    def test_parameter_only_views_and_policy_reach_the_actual_grounded_slot(self):
        lib = library()
        objs = [{"kind":"state","name":"clock","value":{"at":"parameter","requires":["rows"],"program":"return state['datetime']"}},
                {"kind":"decision","name":"card_parameter","value":{"skill":"Resolve the time range.","program":"return [1, 0]"}}]
        child = patch_library(lib,{"objects":objs},set())[0]
        e = actor(child)
        options = [{"label":"first","value":101},{"label":"second","value":202}]
        def select(q,evidence,candidates,**kw):
            self.assertEqual(evidence["views"]["clock"],"2025-01-01")
            self.assertEqual(kw["skill"],"Local. Resolve the time range.")
            self.assertEqual(q,"Which row?")
            return candidates[0]["value"]
        self.assertEqual(e.project_views("parameter"),{})
        e.public["rows"] = [101,202]
        self.assertNotIn("views",e.state_view())
        with patch.object(e,"decision",side_effect=select):
            self.assertEqual(e.choose_inline("Which row?",options,"Local.",{}),202)
        self.assertNotIn("card_parameter",lib["skills"])

    def test_prefix_after_a_different_tool_result_can_never_reuse_old_generation(self):
        ref = SimpleNamespace(requests=[(1,"call",{"api":"read"}),(2,"generate",{"messages":[]})],
                              tape=[(1,"call",{},{"value":[12]}),(2,"generate",{},{"text":"old"})])
        for result,expected in [({"value":[12]},"old"),({"value":[21]},"fresh")]:
            meter = Mock(cost=Counter(),events=[])
            meter.request.return_value={"text":"fresh"}
            prefix=MatchedPrefix(ref,2,meter)
            prefix.tool_result("call",{"api":"read"},result)
            self.assertEqual(prefix.request("generate",messages=[])["text"],expected)
            self.assertEqual(meter.cost["prefix_generate"],int(expected=="old"))
        prefix=MatchedPrefix(ref,2,meter)
        prefix.tool_result("call",{"api":"different"},{"value":[12]})
        self.assertEqual(prefix.request("generate",messages=[])["text"],"fresh")

    def test_changed_bound_values_stop_even_when_the_names_match(self):
        ref = SimpleNamespace(requests=[(5,"bind_progress",{"values":{"rows":[1]}})],matched=1,
                              tape=[(5,"bind_progress",{"names":["rows"]},{})],complete=True,reason=None,stop="returned")
        def rerender(es,lib,func,settings,client):
            from jev_harness.trace_comparison import Boundary
            try:
                client.request("bind_progress",values={"rows":lib})
                return "returned"
            except Boundary:
                return "boundary"
        with patch("jev_harness.trace_comparison.run_on_tape",side_effect=rerender):
            self.assertEqual(compare_path([],ref,[1],[],{})["status"],"equal_on_record")
            change=compare_path([],ref,[2],[],{})["first_change"]
            self.assertEqual(change["fields"],["values.rows"])
            ref.complete=False
            self.assertEqual(compare_path([],ref,[1],[],{})["status"],"unresolved")

    def test_parameter_view_exposure_uses_parameter_sites(self):
        parent = library()
        obj = {"kind":"state","name":"clock","value":{"at":"parameter","requires":["paths"],"program":"return state['datetime']"}}
        child = patch_library(parent,{"objects":[obj]},set())[0]
        handoff = {"event":"handoff_state","values":{"paths":[]},"completed":[],"failed":None,"available":["paths"],"instruction":"Select file"}
        bind = {"event":"parameter_state","card":"read","available":["paths"],"values":{"paths":["p"]}}
        for es,expected in [([handoff],False),([bind],True)]:
            result = assess(trace_sites(es),parent,child,[obj])
            self.assertEqual(bool(result["exposures"]),expected)

    def test_wording_check_finds_empty_candidate_set_without_changing_public_state(self):
        parent = library()
        parent["cards"]["reader"] = {"inputs":["files"],"skill":"Choose a file.","parameters":{
            "name":{"question":"Which file?", "candidates":{"source":{"ref":"state.files"},
                    "where":{"op":"contains","args":[{"ref":"task.instruction"},"cable bill"]},
                    "value":{"ref":"item"},"label":{"ref":"item"}}}}}
        es=[{"event":"parameter_state","card":"reader","values":{"files":["april","may"]},"available":["files"]}]
        out=compare_wording(parent,es,"Read the cable bill","Read the bill for cable")
        self.assertEqual(out[0]["distinct_values"],[2,0])
        self.assertEqual(out[0]["shared_values"],0)
        self.assertNotIn("observed_binding_offered",out[0])
        self.assertEqual(es[0]["values"]["files"],["april","may"])

    def test_wording_check_identifies_lost_binding_among_shared_distractors(self):
        parent = library()
        parent["cards"]["reader"] = {"inputs":["files"],"skill":"Choose one file.","parameters":{
            "name":{"question":"Which file?", "candidates":{"source":{"ref":"state.files"},
                    "where":{"op":"contains","args":[{"ref":"task.instruction"},{"ref":"item"}]},
                    "value":{"ref":"item"},"label":{"ref":"item"}}}}}
        events = [{"event":"parameter_state","card":"reader","available":["files"],
                   "values":{"files":["electricity","gas","water"]}},
                  {"event":"parameter_bindings","card":"reader","bound":{"name":"electricity"}}]
        row, = compare_wording(parent,events,"electricity gas water","power gas water")
        self.assertEqual(row["shared_values"],2)
        self.assertEqual(row["observed_binding_offered"],[True,False])

    def test_probe_service_interruption_retries_same_revision(self):
        from jev_harness.phase1_optimizer import propose_one
        from jev_harness.transport import WorkerUnavailable
        worker=Mock()
        study=Mock(workers=[worker],settings={"optimizer":{"batch_revision":True}})
        result={"keep_parent":True,"reason":"Measured"}
        with TemporaryDirectory() as temp, patch("jev_harness.batch_revision.BatchRevisionWorkbench") as editor:
            editor.return_value.work.side_effect=[WorkerUnavailable("lost service"), result]
            self.assertEqual(propose_one(study,library(),[],[],Path(temp),set()),result)
            self.assertEqual(editor.call_args_list[0].args,editor.call_args_list[1].args)
            self.assertEqual([c.args[0] for c in worker.close_worker.call_args_list],["model","environment"])
