from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from jev_harness.historical_choice import historical_choice, recorded_decision
from jev_harness.transport import WorkerUnavailable


class HistoricalChoiceTests(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.target = Path(folder.name) / "historical_choice.json"
        past = "b01r02/song_1"
        events = [{"event":"task","instruction":"Choose a playlist"}]
        self.bench = SimpleNamespace(archived=[past], tasks={"current_task"},
            precedents=SimpleNamespace(records={past:{"task":"song_1","task_success":True}}),
            reference_paths={"song_1":(past,events,object())},
            study=SimpleNamespace(functions=[],execution={}))
        self.proposal = {"history_use":[{"trace":"dev/held_out"},{"trace":past}]}
        self.paths = {"focus_history":{"matches":[]}}
        self.change = {"parent":"decide","candidate":"decide","event":6,"fields":["options"]}
        self.old = {"song_1":{"trace":past,"first_change":self.change}}
        self.prior = {"label":"Playlist", "value":"playlist"}
        self.choices = [{"label":"Catalog","value":"catalog"},self.prior,{"label":"NONE","value":None}]
        self.page = {"request":{"question":"Which source?","options":[c["label"] for c in self.choices]},
                     "node":"fragment/step", "native_widens":False, "minimum":0}
        self.worker = Mock()
        self.model = self.worker.worker.return_value
        self.model.request.return_value = dict(index=0,probabilities=[.8,.1,.1],input_tokens=30,
                                               output_tokens=0,backend_requests=1)
        captured = (self.change,{"candidates":[self.prior],"selected":0},self.page,self.choices)
        self.capture = patch("jev_harness.historical_choice.changed_ballot",return_value=captured).start()
        self.addCleanup(patch.stopall)

    def observe(self):
        return historical_choice(self.bench,self.proposal,{},self.old,self.paths,self.worker,self.target)

    def test_changed_option_reaches_report_without_rerunning_or_redispatching(self):
        result = self.observe()
        self.assertFalse(result["same_choice"])
        self.assertEqual(result["parent_choice"]["value"],"playlist")
        self.assertEqual(result["candidate_choice"]["value"],"catalog")
        self.assertEqual(result["added_options"],["Catalog","NONE"])
        self.assertEqual(result,self.observe())
        self.model.request.assert_called_once_with("decide",**self.page["request"])

    def test_native_none_widens_even_if_fused_index_differs(self):
        self.page["native_widens"] = True
        self.model.request.return_value.update(native_probabilities=[.27,.23,.50],thinking_used=True,
                                               output_tokens=256,backend_requests=3)
        row = self.observe()
        self.assertIsNone(row["candidate_choice"]["value"])
        self.assertEqual((row["confidence"],row["output_tokens"],row["model_calls"]),(.50,256,1))

    def test_no_solved_historical_case_or_lost_reply(self):
        self.bench.precedents.records["b01r02/song_1"]["task_success"] = False
        self.assertEqual(self.observe()["model_calls"],0)
        self.worker.worker.assert_not_called()
        self.target = self.target.with_name("another_round.json")
        self.bench.precedents.records["b01r02/song_1"]["task_success"] = True
        self.model.request.side_effect = WorkerUnavailable("lost reply")
        self.assertEqual(self.observe()["status"],"outcome_unconfirmed")
        self.assertEqual(self.observe()["model_calls"],1)
        self.model.request.assert_called_once()

    def test_read_final_archived_choice_after_native_and_adaptive_votes(self):
        req = {"options":["A","NONE"],"state":"facts","question":"Which?","skill":""}
        native = {"event":"model","operation":"decide","request":req}
        review = {"event":"model","operation":"decide","request":{**req,"thinking":"auto"}}
        final = {"event":"decision","selected":1,"candidates":[{"label":"A"},{"label":"NONE"}]}
        self.assertEqual(recorded_decision([native,review,{"event":"step_review"},final],0),final)
        self.assertIsNone(recorded_decision([native,{"event":"api"},final],0))
