from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from formal_run import read
from jev_harness.batch_revision import BatchRevisionWorkbench, task_summary
from jev_harness.batch_revision_trials import preservation_case, probe_cases
from jev_harness.contrast_probes import MeteredModel, observations
from jev_harness.history_impact import handoff_view, parameter_changes, trace_sites
from jev_harness.menu_preview import initial_change, initial_pages
from jev_harness.phase1_optimizer import ProbeComplete
from jev_harness.revision_evidence import paired_outcome_costs
from jev_harness.runtime import MAX_REAL_OPTIONS, NativeNone, Trace, Uncovered
from jev_harness.scoped_menu import scoped_library
from jev_harness.semantic_assembly import SemanticAssembly
from jev_harness.semantic_handoff import snapshot
from jev_harness.semantic_library import initial_library, patch_library


def library():
    return scoped_library(initial_library(semantic_handoffs=True))


def actor(lib,executor=SemanticAssembly):
    e = executor(None,Mock())
    e.prepare_options()
    e.context = {"task":{"instruction":"Look up the requested file.","functions":[],"supervisor":{},"datetime":"2025-01-01"},
                 "history":[],"vars":{},"generated":{}}
    e.trace,e.done = Trace(),False
    e.initialize(lib)
    return e


def card(name="result", **fields):
    return {"app":"file_system","inputs":[],"output":name,"compute":[1],"description":"Read a public value.",
            "skill":"Local slot guidance.","handoff_skill":"Use the result from this reader.",**fields}


class ScopedRuntimeTests(TestCase):
    def test_preferred_slot_uses_one_ballot_then_broadens_only_to_eligible_records(self):
        e = actor(library())
        e.active_card = "reader"
        e.public["files"] = ["electricity April","internet April","private draft"]
        reader = card(inputs=["files"],parameters={"file":{"question":"Select a statement",
            "candidates":{"source":{"ref":"state.files"},"where":{"op":"contains","args":[{"ref":"item"},"April"]}},
            "prefer":{"op":"contains","args":[{"ref":"item"},"electricity"]}}})
        before = deepcopy(reader)
        del before["parameters"]["file"]["prefer"]
        exposure, = parameter_changes(before,reader,{"values":e.public,"bound":{"file":"internet April"}},e.request)
        self.assertTrue(exposure["same_ordered_candidates"])
        self.assertTrue(exposure["bound_value_still_offered"])
        self.assertEqual(exposure["menu_sizes"],[[2],[1,1]])
        outcomes = [["electricity April"],[NativeNone({}),"internet April"],[NativeNone({}),NativeNone({})],
                    [Uncovered("Decision service returned an invalid index")]]
        for replies in outcomes:
            with self.subTest(replies=replies), patch.object(e,"choose_inline",side_effect=replies) as pick:
                if isinstance(replies[-1],Exception):
                    with self.assertRaises(Uncovered):
                        e.parameter_context(reader)
                else:
                    self.assertEqual(e.parameter_context(reader)["params"]["file"],replies[-1])
                seen = [[c["value"] for c in call.args[1]] for call in pick.call_args_list]
                self.assertEqual(seen,[["electricity April"],["internet April"]][:len(replies)])
        reader["parameters"]["file"]["prefer"] = False
        with patch.object(e,"choose_inline",return_value="internet April") as pick:
            e.parameter_context(reader)
            self.assertEqual([c["value"] for c in pick.call_args.args[1]],["electricity April","internet April"])

    def test_old_menu_comparison_holds_only_an_available_family_fixed(self):
        parent = library()
        updated = deepcopy(parent)
        updated["cards"]["read_paths"] = card()
        events = [{"event":"task","instruction":"Find the file.","datetime":"2025-01-01"},
                  {"event":"assembly_selection","family":"file_system"}]
        local = initial_change(parent,updated,events)
        self.assertEqual(local["changed_fields"]["fragment/assembly"],[])
        self.assertIn("options",local["changed_fields"]["fragment/step"])
        events[1]["family"] = "retired_family"
        unavailable = initial_change(parent,updated,events)
        self.assertIsNone(unavailable["changed_fields"]["fragment/step"])
        self.assertFalse(unavailable["action_menu_comparable"])
        projected = patch_library(parent,{"objects":[{"kind":"state","name":"clock","value":{
            "program":"return state['datetime']"}}]},set())[0]
        self.assertIn("state",initial_change(parent,projected,events)["changed_fields"]["fragment/assembly"])

    def test_family_stable_and_action_local_across_catalog_growth(self):
        parent = library()
        candidate = deepcopy(parent)
        candidate["cards"]["read_paths"] = card(scope="File paths, without content.")
        for request in ("Locate my receipt.","Open the document I mentioned.","Show me phone alarms."):
            before,after = [initial_pages(v,request,"phone") for v in (parent,candidate)]
            self.assertEqual(before,after)
            foreign = initial_pages(candidate,request,"file_system")
            self.assertEqual(before[0],foreign[0])
            action = next(s for s in foreign[1]["options"] if "File paths" in s)
            self.assertNotIn("Local slot guidance",action)
            self.assertNotIn("Use the result",action)
            self.assertIn("result",action)

    def test_specialists_share_a_slot_and_abstention_can_recover_a_primitive(self):
        lib = library()
        lib["cards"].update(a=card("a",applies_to="Read total payment."),b=card("b",applies_to="Read latest receipt."))
        e = actor(lib)
        ballots = []
        def pick(name, choices, **facts):
            ids = [c["value"] for c in choices]
            ballots.append(ids)
            if name=="assembly":
                return "file_system"
            if {"specialized":True} in ids:
                return {"specialized":True}
            if "a" in ids:
                raise Uncovered("Jev selected the uncovered option")
            return "directory"
        with patch.object(e,"choose",side_effect=pick):
            self.assertEqual(e.next_card(),"directory")
        self.assertNotIn("a",ballots[1])
        self.assertEqual(ballots[2],["a","b"])
        self.assertIn("directory",ballots[3])

    def test_growing_specialist_bank_keeps_common_menu_stable_and_last_card_reachable(self):
        lib = library()
        for i in range(48):
            lib["cards"][f"special_{i}"] = card(f"special_value_{i}",applies_to=f"Report statistic {i}.")
            if i == 1:
                small = initial_pages(lib,"Inspect my files.","file_system")
        self.assertEqual(initial_pages(lib,"Inspect my files.","file_system"),small)
        e = actor(lib)
        sizes = []
        def pick(question,evidence,candidates,**kwargs):
            sizes.append(len(candidates))
            for c in candidates:
                if c["value"] in ("file_system",{"specialized":True}) or "Report statistic 47." in c["label"]:
                    return c["value"]
            raise NativeNone({"probabilities":[0.0]*len(candidates)+[1.0]})
        with patch.object(e,"decision",side_effect=pick):
            self.assertEqual(e.next_card(),"special_47")
        self.assertLessEqual(max(sizes),MAX_REAL_OPTIONS)
        self.assertGreater(len(sizes),4)

    def test_no_result_after_failure_but_real_recovery_and_receipts_remain(self):
        lib = library()
        lib["cards"]["detail"] = card(compute={"ref":"params.note"},
            parameters={"note":{"question":"Which note?","candidates":[]}})
        e = actor(lib)
        with self.assertRaises(Uncovered):
            e.perform_card("detail")
        e.receipts.append({"item":7,"result":"done"})
        progress = e.handoff("Missing note ID")
        self.assertNotIn(lib["cards"]["detail"]["handoff_skill"],progress["fragment_guidance"])
        self.assertEqual(progress["unresolved"],"Missing note ID")
        self.assertIn("executed_items",progress["variables"])

    def test_live_writer_and_independent_guidance_survive_invalidation(self):
        lib = library()
        lib["cards"].update(old=card("items",handoff_skill="Old writer."),
            new=card("items",handoff_skill="New writer."),derived=card("rows",inputs=["items"],handoff_skill="Rows."),
            extra=card("fragment_usage_by_role",handoff_skill="Independent."))
        e = actor(lib)
        e.perform_card("old")
        e.perform_card("derived")
        e.perform_card("extra")
        e.forget_derived({"items"})
        e.perform_card("new")
        report = e.handoff("Continue")
        values = e.environment.request.call_args.kwargs["values"]
        self.assertEqual(values["fragment_usage_by_role"],[1])
        self.assertEqual(set(values[report["usage_lookup"]].values()),{"New writer.","Independent."})
        self.assertNotIn("rows",report["decision_evidence"]["sources"])
        self.assertEqual(report["decision_evidence"]["sources"]["items"]["producer"],"new")
        self.assertEqual(len(report["fragment_guidance"]),2)  # One live note and the existing default.
        self.assertEqual(snapshot(e)["producers"]["items"],"new")

    def test_legacy_trace_reconstructs_writers_without_promoting_failed_card(self):
        lib = library()
        lib["cards"].update(good=card(),bad=card(handoff_skill="No such result"))
        es = [{"event":"card_executed","card":"good","output":"result"},{"event":"card_failed","card":"bad"},
              {"event":"handoff_state","values":{"result":[]},"completed":["good"],"failed":"bad"}]
        state = trace_sites(es)["handoffs"][2]
        self.assertEqual(state["producers"],{"result":"good"})
        self.assertEqual(handoff_view(lib,state)[3]["good"],lib["cards"]["good"]["handoff_skill"])
        self.assertNotIn("bad",handoff_view(lib,state)[3])

    def test_installer_rejects_pilot_only_opening_skill(self):
        guide = dict(label="Use the bound list.", opening_skill="Inspect the list first.",
                     skill="Preserve all object identities.")
        with self.assertRaises(ValueError):
            patch_library(library(), {"objects": [dict(kind="handoff", name="use_list", value=guide)]}, set(), [])

    def test_partial_read_reports_its_actual_endpoint_and_coverage_and_stays_unfinished(self):
        lib = library()
        lib["cards"]["credited"] = {"app":"spotify","inputs":[],"output":"liked_rows","api":"spotify__show_liked_songs",
                "description":"Read liked songs.","scope":"Liked songs only.","skill":"Read song rows.","arguments":{}}
        e = actor(lib)
        with patch.object(e,"call_endpoint",return_value=[]):
            e.perform_card("credited")
        info = e.handoff("Additional population required")["decision_evidence"]
        self.assertEqual(info["sources"]["liked_rows"]["scope"],"Liked songs only.")
        self.assertEqual(info["read_scopes"]["liked_rows"]["api"],"spotify__show_liked_songs")
        self.assertFalse(e.done)
        self.assertNotIn("effect_receipt",e.public)

    def test_reentry_replaces_old_residual_and_keeps_live_read_fallback(self):
        e = actor(library())
        e.residual_progress={"unresolved":"old","stale_note":"must disappear"}
        with patch.object(e,"delegate_read"),patch.object(e,"assemble_available",side_effect=Uncovered("next")):
            e.code("yield_read('test__read', {})",node="generation_agent")
        self.assertEqual(e.residual_progress["unresolved"],"next")
        self.assertNotIn("stale_note",e.residual_progress)

    def test_retirement_can_update_references_in_same_patch(self):
        lib=library()
        lib["cards"]["duplicate"]=card()
        lib["assembly"]["custom"]={"cards":["duplicate"],"description":"Custom group"}
        def remove(kind,name):
            return {"kind":kind,"name":name,"value":None}
        with self.assertRaises(ValueError):
            patch_library(lib,{"objects":[remove("card","duplicate")]},set())
        new,_=patch_library(lib,{"objects":[remove("card","duplicate"),remove("assembly","custom")]},set())
        self.assertNotIn("duplicate",new["cards"])
        self.assertIn("duplicate",lib["cards"])
        for kind,name in (("card","submit_answer"),("handoff","default")):
            with self.assertRaises(ValueError):
                patch_library(lib,{"objects":[remove(kind,name)]},set())


class ScopedOptimizerTests(TestCase):
    def test_paired_cost_separates_early_failure_from_successful_savings(self):
        pairs = [((True, True), (20, 23)), ((True, True), (10, 12)),
                 ((False, False), (80, 9)), ((True, False), (15, 4)), ((False, True), (7, 14))]
        result = paired_outcome_costs(pairs)
        self.assertEqual({key: (row["tasks"], row["parent_output_tokens"], row["candidate_output_tokens"])
                          for key, row in result.items()},
                         {"both_solved": (2, 30, 35), "both_failed": (1, 80, 9),
                          "regressed": (1, 15, 4), "newly_solved": (1, 7, 14)})
        self.assertEqual(paired_outcome_costs([]), {})

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bench = object.__new__(BatchRevisionWorkbench)
        b = self.bench
        b.folder,b.lib,b.tasks,b.sources,b.study = Path(self.tmp.name)/"sol",library(),{"a_1","b_1","a_2"},set(),Mock(functions=[])
        b.current_events = {t:[{"event":"task"},{"event":"tool_batch"}] for t in b.tasks}
        b.index = [{"task":t,"success":t!="a_1"} for t in sorted(b.tasks)]
        self.draft = {"focus":{"task":"a_1","event":1,"observable":"Consumer opens the selected path."},
                 "preserve_task":"b_1","reason":"Public paths missing","prediction":"A file is read",
                 "batch_effects":[{"tasks":sorted(b.tasks),"effect":"Current batch expectation"}],
                 "objects":[{"kind":"card","name":"paths","value":card()}]}

    def test_focus_requires_an_execution_event_and_another_solved_task_when_available(self):
        b = self.bench
        self.assertEqual(b.validate_draft(self.draft),self.draft)
        self.assertEqual(b.validate_draft({**self.draft,"preserve_task":"a_2"})["preserve_task"],"a_2")
        for change in ({"preserve_task":"a_1"},{"preserve_task":"d_3"},{"focus":dict(task="d_1",event=1,observable="x")},
                       {"focus":dict(task="a_1",event=0,observable="x")},{"batch_effects":[]}):
            with self.assertRaises(ValueError):
                b.validate_draft({**self.draft,**change})
        self.assertTrue(b.validate_draft({"keep_parent":True,"reason":"No supported repair"})["keep_parent"])

    def test_measured_exposure_selects_a_solved_sibling_and_keeps_cases_paired(self):
        b = self.bench
        b.current_events["a_2"] = [{"event":"handoff_selection","guide":"pay"}]
        proposal = {**self.draft,"objects":[{"kind":"handoff","name":"pay"}]}
        current = {t:{"status":"changed"} for t in b.tasks}
        selection = preservation_case(b,proposal,current)
        self.assertEqual(selection["task"],"a_2")
        self.assertEqual(selection["exercised_edits"],["handoff/pay"])
        measured = {**proposal,"preserve_task":selection["task"]}
        current["a_2"]["status"] = "equal_on_record"
        self.assertEqual(preservation_case(b,measured,current,fixed=True)["task"],"a_2")
        cells = [case["cells"] for case in probe_cases(measured,protect_generation=True)]
        self.assertEqual(cells,[1,1])
        self.assertEqual(probe_cases(measured)[1]["cells"],0)

    def test_last_lookup_is_returned_before_the_decision_only_reply(self):
        b=self.bench
        account=b.study.solver.return_value
        end={"keep_parent":True,"reason":"No executable improvement"}
        account.turn.side_effect = [json.dumps({"read":["dsl"]})]*9+[json.dumps({"action_json":json.dumps(end)})]
        with patch.object(b,"lookup",return_value=["public evidence"]) as lookup:
            self.assertEqual(b.ask("draft",{},b.folder,b.validate_draft),end)
        self.assertEqual(lookup.call_count,9)
        last=account.turn.call_args
        self.assertIn("public evidence",last.args[1])
        self.assertIn("schema",last.kwargs)

    def test_candidate_evidence_cache_identifies_the_tested_objects_and_cases(self):
        from jev_harness.contrast_probes import compare_candidates
        from formal_run import save
        b = self.bench
        folder=b.folder/"trial"
        proposal={"cases":[{"trace":"current/a_1","anchor":"start"}],"alternatives":[{"id":"draft","objects":self.draft["objects"]}]}
        save(folder/"tested_patch.json",{"cases":[],"objects":[]})
        save(folder/"local_evidence.json",{"unused":True})
        with self.assertRaisesRegex(ValueError,"different objects or cases"):
            compare_candidates(b,proposal,Mock(),folder=folder)

    def test_reconnecting_sol_reuses_paid_trials_and_preserves_previous_dialogue(self):
        from formal_run import save
        from jev_harness.phase1_optimizer import propose_one
        from jev_harness.sol_account import AccountError
        b = self.bench
        b.study.settings = {"optimizer":{"batch_revision":True}}
        done = {"keep_parent":True,"reason":"Preserve solved tasks"}
        with patch("jev_harness.batch_revision.BatchRevisionWorkbench") as editor:
            editor.return_value.work.side_effect = [AccountError("temporary connection loss"),done]
            result = propose_one(b.study,b.lib,[],[],b.folder,set())
        self.assertEqual(result,done)
        self.assertEqual(editor.call_args_list[0].args[4],editor.call_args_list[1].args[4])
        review = b.folder/"review"
        save(review/"turn_01/usage.json",{"outputTokens":3})
        b.study.solver.return_value.turn.return_value=json.dumps(done)
        b.ask("Review",{},review,b.validate_draft)
        self.assertEqual(b.study.solver.return_value.start.call_args.args[0],review/"retry_02")
        self.assertEqual(read(review/"turn_01/usage.json"),{"outputTokens":3})

    def test_parse_failure_after_long_prefix_is_visible_to_sol(self):
        es=[{"event":"api_action","method":"GET","url":"/files"} for _ in range(70)]
        es += [{"event":"model","operation":"generate","request":{"purpose":"G"},"response":{"output_tokens":1024,"truncated":True}},
               {"event":"generation_parse_error","error":"An open code block was not closed"}]
        result=observations(es,[])["events"]
        self.assertTrue(any(e.get("event")=="open_generation" and e["truncated"] for e in result))
        self.assertTrue(any(e.get("event")=="generation_parse_error" for e in result))

    def test_fallback_counts_separate_unexecuted_code_adaptive_choices_and_tool_errors(self):
        item = dict(task="a_1",trace="current/a_1",instruction="Read the value.",success=False,
                    output_tokens=1200,executed_cards=[],handoffs=[],modes={})
        es = [{"event":"model","operation":"generate","request":{"purpose":"R_review"},"response":{"truncated":True}},
              {"event":"model","operation":"generate","request":{"purpose":"G"},"response":{"truncated":True}},
              {"event":"generation_parse_error","error":"An open code block was not closed"},
              {"event":"model","operation":"generate","request":{"purpose":"G"},"response":{"truncated":False}},
              {"event":"tool_batch","node":"generation_agent","response":{"output":"Execution failed. Unknown argument."}}]
        self.assertEqual(task_summary(item,es)["open_generation"],
                         dict(calls=2,truncated=1,parse_errors=1,executed_cells=1,cells_with_errors=1))

    def test_offline_costs_include_nested_reviews_and_charge_shared_meter_once(self):
        from formal_run import save
        from phase1_run import offline_optimizer_costs
        root = Path(self.tmp.name)
        editor = root/"batches/b03/b03r01/sol_scoped"
        for key in ("focus","review","final_review"):
            save(editor/key/"turn_01/usage.json",{"outputTokens":10,"cachedInputTokens":5})
        save(editor/"trial_draft/meter.json",{"open_generations":1,"output_tokens":64})
        save(editor/"trial_revised/meter.json",{"open_generations":2,"output_tokens":128})
        row = offline_optimizer_costs(root,historical=False)
        self.assertEqual(row["offline_sol_usage"]["outputTokens"],30)
        self.assertEqual(row["offline_sol_usage"]["cachedInputTokens"],15)
        self.assertEqual(row["offline_27b_probes"],{"open_generations":2,"output_tokens":128})

    def test_revised_patch_inherits_case_and_has_its_own_measurements(self):
        b = self.bench
        other = [{"kind":"decision","name":"step","value":{"skill":"Keep the scope."}}]
        def ask(prompt,payload,folder,parse):
            if folder.name=="review":
                return parse({"use":"revise","focus":{"task":"b_1"},"preserve_task":"a_1","objects":other,
                              "reason":"The data existed; the consumer missed it.","prediction":"Select correctly"})
            return parse({"use":"revised","reason":"Matched revised observation"})
        seen=[]
        def probe(bench,p,worker,key,prior):
            seen.append((key,probe_cases(p),prior,p["objects"]))
            return {"cases":[key],"batch":[],"historical_impact":{key:{"arm":key}},
                    "cost":{"fresh_decisions":3 if not prior else 6}, "stage_cost":{"fresh_decisions":3}}
        with patch.object(b,"packet",return_value={}),patch.object(b,"draft",return_value=self.draft),patch.object(b,"ask",side_effect=ask), \
             patch("jev_harness.batch_revision_trials.preview_batch",side_effect=probe):
            result=b.work(Mock())
        self.assertEqual(seen[0][1],seen[1][1])
        self.assertEqual([c["trace"] for c in seen[0][1]],["current/a_1","current/b_1"])
        self.assertEqual(seen[1][2],{"fresh_decisions":3})
        self.assertEqual(result["objects"],other)
        self.assertNotIn("batch_effects",result)
        self.assertEqual(read(b.folder.parent/"optimizer_impact.json")["arm"],"revised")
        self.assertEqual(result["measured_preview"],"revised")

    def test_revised_patch_can_update_scope_and_wording_without_reusing_draft_claims(self):
        b = self.bench
        edits = [{"kind":"decision","name":"step","value":{"skill":"Use current data."}}]
        effects = [{"tasks":sorted(b.tasks),"effect":"The existing selector sees a new decision skill."}]
        check = {"scope":"same_operation","request":"Open that document to verify its amount."}
        replies = [{"use":"revise","objects":edits,"reason":"The result already exists.",
                    "batch_effects":effects,"wording_check":check}, {"use":"revised","reason":"Observed"}]
        def preview(bench,proposal,worker,key,prior):
            if key == "revised":
                self.assertEqual(proposal["wording_check"],check)
            return {"cases":[],"batch":[],"historical_impact":{key:{}},"cost":{},"stage_cost":{}}
        with patch.object(b,"packet",return_value={}),patch.object(b,"draft",return_value=self.draft), \
             patch.object(b,"ask",side_effect=lambda prompt,payload,folder,parse:parse(replies.pop(0))), \
             patch("jev_harness.batch_revision_trials.preview_batch",side_effect=preview):
            result = b.work(Mock())
        self.assertEqual(result["batch_effects"],effects)
        self.assertNotIn("prediction",result)

    def test_observations_retain_late_consumer_and_attempt_budget(self):
        es=[{"event":"api_action","method":"GET","url":"/files"} for _ in range(130)]
        es.insert(95,{"event":"generated_read_observation","available":True,"roles":["paths"]})
        self.assertTrue(any(row.get("roles")==["paths"] for row in observations(es,[])["events"]))
        head=Mock()
        head.request.return_value={"output_tokens":100,"reasoning_tokens":20}
        total=Counter(open_generations=1)
        meter=MeteredModel(head,{},total,1,2)
        for _ in range(2):
            meter.request("generate",max_tokens=1024)
        with self.assertRaises(ProbeComplete):
            meter.request("generate",max_tokens=1024)
        self.assertEqual(total["open_generations"],3)
        self.assertEqual((total["generation_output_tokens"],total["reasoning_tokens"]),(200,40))

    def test_keep_parent_skips_full_candidate_and_dev(self):
        from phase1_run import run
        from jev_harness.semantic_library import PROTOCOL
        b = self.bench
        b.study.root = Path(self.tmp.name)
        b.study.settings = {"protocol":PROTOCOL,"semantic_handoffs":True,"menu_format":"scoped","optimizer":{"batch_revision":True}}
        rows = [{"task_id":"a_1","success":True,"output_tokens":10,"trace_path":"parent","model_calls":{"G":1},
                 "seconds":1,"api_calls":1,"input_tokens":1,"reasoning_tokens":0}]
        b.study.batch.return_value=rows
        with patch("jev_harness.phase1_optimizer.propose_one",return_value={"keep_parent":True,"reason":"Preserve correct action"}):
            run(b.study,1,2,True,evaluate_test=False,reference_dev=lambda: self.fail("Unnecessary Dev call"))
        self.assertEqual(b.study.batch.call_count,1)
        b.study.candidate_batch.assert_not_called()
        checkpoint=read(b.study.root/"checkpoints.json")[0]
        self.assertEqual(checkpoint["next_version"],"h00")
        self.assertEqual(checkpoint["revisions_scored"],0)
        self.assertEqual(read(b.study.root/"versions/h00.json")["menu_format"],"scoped")
