import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from jev_harness.batch_revision import BatchRevisionWorkbench, revised_patch
from jev_harness.batch_revision_trials import compare_train_paths
from jev_harness.history_impact import trace_sites
from jev_harness.operation_evidence import batch_operations
from jev_harness.train_precedents import TrainPrecedents
from phase1_run import completed_train_entries
from test_scoped_revision import card, library


def generation(code, tokens, actions):
    return [{"event":"model", "operation":"generate", "request":{}, "response":{"output_tokens":tokens}},
            {"event":"tool_batch", "node":"generation_agent", "code":code,
             "response":{"output":"ok", "actions":[{"method":"GET", "url":url} for url in actions]}}]


def trace(task, goal, middle, solved):
    return [{"event":"task", "task_id":task, "instruction":goal}, *middle,
            {"event":"summary", "success":solved, "output_tokens":sum(
                e["response"]["output_tokens"] for e in middle if e["event"] == "model")}]


def index(current, past):
    groups, _ = batch_operations(current)
    return TrainPrecedents(current, groups, past.items())


class HistoricalTrainTests(TestCase):
    def test_related_by_actual_operation_across_goals_and_compiled_or_generated_routes(self):
        now = trace("bill_1", "Request payment from housemates.",
                    [{"event":"api", "app":"phone", "api":"search_contacts", "response":{}}], False)
        code = "people = apis.phone.search_contacts()\nprint(people)"
        old = trace("sms_1", "Send a meeting reminder.", generation(code, 912, ["/phone/contacts"]), True)
        sibling = trace("sms_2", "Send a reminder.", generation(code, 450, ["/phone/contacts"]), True)
        distractor = trace("noise_1", now[0]["instruction"],
                           generation("apis.phone.login()", 1300, ["/phone/auth/token"]), True)
        history = {"prior/sms_1":old, "prior/sms_2":sibling, "prior/noise_1":distractor}
        match, = index({"bill_1":now}, history).related("bill_1", 1)["matches"]
        self.assertEqual((match["trace"], match["event"], match["relation"]), ("prior/sms_1",2,"different_scenario"))
        self.assertEqual(match["shared"], ["api phone__search_contacts"])
        self.assertEqual(match["past_observed_actions"], ["GET /phone/contacts"])
        self.assertEqual(match["past_cell_output_tokens"], 912)
        self.assertEqual(index({"bill_1":now},history).suggestions([{"task":"bill_1","success":True,"output_tokens":800}])
                         [0]["past_trace"], "prior/sms_1")

    def test_reported_source_call_and_observed_effect_are_distinct(self):
        code = "if False:\n    apis.venmo.create_payment()"
        focus = trace("p_1", "Pay.", generation(code, 61, []), False)
        past = {"old/p_1":trace("p_1", "Pay.", generation(code, 77, []), True)}
        match, = index({"p_1":focus}, past).related("p_1", 2)["matches"]
        self.assertEqual(match["relation"], "earlier_same_task")
        self.assertEqual(match["shared"], ["api venmo__create_payment"])
        self.assertEqual(match["past_observed_actions"], [])

    def test_same_parse_failure_finds_a_recorded_recovery_across_apps(self):
        error = {"event":"generation_parse_error", "error":"An open code block was not closed"}
        now = trace("calendar_1", "Reschedule.", [error], False)
        old = trace("notes_1", "Read notes.", [error,*generation("print(1)", 75, [])], True)
        result = index({"calendar_1":now}, {"past/notes_1":old}).related("calendar_1", 1)
        match, = result["matches"]
        self.assertEqual((match["event"], match["past_kind"], match["past_reported_error"]), (3,"tool_batch",False))
        self.assertEqual(result["current_kind"], "generation_parse_error")
        current = trace("calendar_1", "Reschedule.", [error,*generation("apis.calendar.read_calendar()",24,[])],False)
        later = index({"calendar_1":current}, {"past/notes_1":old}).related("calendar_1",3)
        self.assertEqual(later["status"], "no_shared_operation")

    def test_repeated_failure_and_no_match_are_distinct_from_success(self):
        focus = trace("a_1", "Count songs.", [{"event":"card_failed","card":"load_ids"}],False)
        same = trace("a_2", "Count albums.", [{"event":"card_failed","card":"load_ids"}],False)
        other = trace("b_1", "Count songs.", [{"event":"card_executed","card":"read_calendar"}],True)
        i = index({"a_1":focus}, {"p/a_2":same,"p/b_1":other})
        row, = i.related("a_1", 1)["matches"]
        self.assertEqual((row["relation"], row["past_task_success"], row["past_reported_error"]), ("same_scenario",False,True))
        self.assertEqual(i.related("a_1", 0)["nearby_events"], [1])
        self.assertEqual(index({"a_1":focus},{"p/b_1":other}).related("a_1",1)["status"], "no_shared_operation")
        self.assertEqual(index({"a_1":focus},{}).related("a_1",1)["status"], "no_prior_train")

    def test_shared_menu_identifies_exposure_without_implying_a_card_ran(self):
        menu = {"event":"card_menu","assembly":"payments","offered":["read_incoming"]}
        select = {"event":"decision","node":"fragment/step","selected":0,"candidates":[{"value":"read_incoming"}]}
        current = trace("a_1","Send a request.",[menu],False)
        history = trace("b_1","Read incoming requests.",[menu,select],True)
        row, = index({"a_1":current},{"earlier/b_1":history}).related("a_1",1)["matches"]
        self.assertEqual(row["shared"],["offered payments/read_incoming"])
        self.assertEqual(row["past_observed_actions"],[])

    def test_normal_editor_lookup_draft_and_revision_use_only_available_train(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def write(stage, task, goal, endpoint, solved):
                path = root/"runs"/stage/task/"attempt_1/trace.jsonl"
                path.parent.mkdir(parents=True)
                es = trace(task,goal,generation(f"apis.phone.{endpoint}()",234,["/phone/"+endpoint]),solved)
                path.write_text("".join(json.dumps(e)+"\n" for e in es),encoding="utf-8")
                return str(path.relative_to(root))
            present = write("entry","bill_1","Pay a contact.","search_contacts",False)
            past = write("completed","sms_1","Text a contact.","search_contacts",True)
            dev = write("dev_parent","dev_1","Pay a contact.","search_contacts",True)
            future = write("later","future_1","Pay a contact.","search_contacts",True)
            rows = [{"task_id":"bill_1","trace_path":present,"success":False,"output_tokens":234}]
            study = SimpleNamespace(root=root,functions=[],execution={},prior_train_entries=[])
            def bench(history):
                return BatchRevisionWorkbench(study,library(),rows,history,root/"batches/b02/r01/sol",set())
            proposal = {"focus":{"task":"bill_1","event":2,"observable":"Resolve identity"},
                        "batch_effects":[{"tasks":["bill_1"],"effect":"Resolve a contact"}],
                        "objects":[{"kind":"card","name":"resolve_person","value":card()}], "history_use":[]}
            cold = bench([])
            self.assertEqual(cold.packet()["historical_comparisons"]["recordings"],0)
            self.assertEqual(cold.validate_draft(proposal),proposal)
            # The old visit returned its parent and never scored a new version.
            study.prior_train_entries = [past]
            editor = bench([{"train_traces":[present],"dev_traces":[dev],"test_traces":[future]}])
            packet = editor.packet()["historical_comparisons"]
            self.assertEqual((packet["recordings"],packet["distinct_tasks"]),(1,1))
            self.assertEqual(set(editor.precedents.records),{"completed/sms_1"})
            self.assertEqual(packet["examples"][0]["past_event"],2)
            match, = editor.lookup({"related_train":{"task":"bill_1","event":2}})["matches"]
            raw = editor.lookup({"read":[{"doc":match["trace"],"start":match["event"],"count":1}]})
            self.assertIn("search_contacts",raw[0]["rows"][0]["text"])
            source = {"trace":match["trace"],"event":2,"shared_step":"Resolve identity","scope_difference":"Text vs pay"}
            applied = editor.validate_draft({**proposal,"history_use":[source]})
            self.assertEqual(revised_patch(applied, {"objects":proposal["objects"]})["history_use"],[source])
            self.assertEqual(revised_patch(applied, {"objects":proposal["objects"],"history_use":[]})["history_use"],[])
            self.assertEqual(editor.validate_draft(proposal),proposal)
            for doc, event in (("dev_parent/dev_1",2),("later/future_1",2),("current/bill_1",2),("completed/sms_1",99)):
                with self.subTest(doc=doc,event=event), self.assertRaises(ValueError):
                    editor.validate_draft({**proposal,"history_use":[{**source,"trace":doc,"event":event}]})

    def test_replay_uses_exact_cited_version_even_if_task_has_a_newer_reference(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)/"trace.jsonl"
            source.write_text('{"event":"task","label":"old"}\n',encoding="utf-8")
            recent = [{"event":"task","label":"recent"}]
            bench = SimpleNamespace(reference_paths={"a_1":("current/a_1", recent, object()),
                                    "b_1":("recent/b_1",recent,object())}, tasks={"a_1"}, lib=library(),
                                    study=SimpleNamespace(functions=[],execution={}),paths={"old/b_1":source},
                                    trace_data=lambda doc:trace_sites(recent),
                                    precedents=SimpleNamespace(records={"old/b_1":{"task_success":True,"trace_output_tokens":80}},
                                        related=Mock(return_value={"matches":[{"trace":"old/b_1"}]})))
            proposal = {"focus":{"task":"a_1","event":0},"objects":[{"kind":"card","name":"read"}],
                        "history_use":[{"trace":"old/b_1","event":0}]}
            with patch("jev_harness.batch_revision_trials.ParentPath"), patch(
                    "jev_harness.batch_revision_trials.compare_path",
                    side_effect=lambda es,*args: {"status":"equal_on_record","label":es[0]["label"]}):
                _, old, report = compare_train_paths(bench,proposal,bench.lib)
            self.assertEqual(old["b_1"]["label"],"recent")
            self.assertEqual(report["cited_train_paths"][0]["replay"]["label"],"old")
            self.assertEqual(report["focus_history"]["matches"][0]["replay"]["label"],"old")

    def test_resume_recovers_old_unmodified_batch_from_its_scored_attempt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            past = root/"runs/b01_h00_entry/82e2fac_1"
            past.mkdir(parents=True)
            trace_path = "runs/b01_h00_entry/82e2fac_1/attempt_2/trace.jsonl"
            (past/"record.json").write_text(json.dumps({"trace_path":trace_path}),encoding="utf-8")
            checkpoint = {"batch":1,"entry_version":"h00","revisions_scored":0}
            self.assertEqual(completed_train_entries(root,"b01",checkpoint,{}),[trace_path])
            entry = root/"batches/b01/entry.json"
            entry.parent.mkdir(parents=True)
            entry.write_text(json.dumps({"train_traces":["retained_attempt"]}),encoding="utf-8")
            self.assertEqual(completed_train_entries(root,"b01",checkpoint,{}),["retained_attempt"])
