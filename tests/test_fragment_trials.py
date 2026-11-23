import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from jev_harness.fragment_replay import LocalReplay, RecordedModel, ReplayDiverged
from jev_harness.fragment_trials import FragmentTrialWorkbench, natural_use
from jev_harness.operation_evidence import batch_operations


def cell(code):
    return {"event":"tool_batch","node":"generation_agent","code":code,
            "response":{"actions":[{"method":"GET","url":"/example/items?offset=0"}]}}


class FragmentTrialTests(unittest.TestCase):
    def test_groups_repeated_data_dependency_across_cells_even_after_local_variable_renaming(self):
        a = [cell("rows = apis.example.show_items()\nids = set()\nfor row in rows:\n    ids.add(row['id'])"),
             cell("result = [x for x in scoped if x not in ids]")]
        b = [cell("records = apis.example.show_items()\nselected = {r['id'] for r in records}\n"
                  "answer = scoped & selected"),cell("unfinished = [")]
        groups, counts = batch_operations({"a":a,"b":b})
        g = next(g for g in groups if g["operation"]=="membership")
        self.assertEqual(g["tasks"],["a","b"])
        self.assertEqual([g["sites"][task][0]["variant"] for task in g["tasks"]],["exclude","intersection"])
        self.assertEqual(g["sites"]["a"][0]["event"],1)
        self.assertEqual((counts["generated_cells"],counts["parsed_cells"]),(4,3))
        self.assertEqual(counts["unparsed"][0]["task"],"b")

    def test_count_selection_execution_and_reads_separately(self):
        events = [{"event":"card_menu","offered":["new","unused"]},
                  {"event":"decision","node":"fragment/step","selected":0,"candidates":[{"value":"new"}]},
                  {"event":"card_executed","card":"new","output":"ids","inputs":[]},
                  {"event":"card_executed","card":"consume","output":"objects","inputs":["ids"]},
                  {"event":"generated_read_observation","available":True,"roles":["ids"]},
                  {"event":"card_executed","card":"another","output":"ids","inputs":[]},
                  {"event":"card_executed","card":"later","output":"titles","inputs":["ids"]}]
        report = natural_use(events,["new","unused"])
        self.assertTrue(report["unused"]["offered"])
        self.assertFalse(report["unused"]["executed"])
        self.assertIsNone(report["unused"]["generated_read_observed"])
        self.assertEqual(report["new"]["consumers"],[{"card":"consume","event":3}])
        self.assertTrue(report["new"]["generated_read_observed"])

    def test_missing_generation_observation_is_different_from_an_observed_no_read(self):
        ran = [{"event":"card_executed","card":"lookup","output":"rows","inputs":[]}]
        self.assertIsNone(natural_use(ran,["lookup"])["lookup"]["generated_read_observed"])
        ran.append({"event":"generated_read_observation","available":True,"roles":[]})
        self.assertIs(natural_use(ran,["lookup"])["lookup"]["generated_read_observed"],False)

    def test_natural_report_shows_actual_bound_slot(self):
        for record in ({"event":"trial_card_bindings","card":"filter","parameters":{"keep":"unliked"}},
                       {"event":"card_menu","offered":[],"state":{"bindings":{
                           "songs":{"producer":"filter","parameters":{"keep":"unliked"}}}}}):
            self.assertEqual(natural_use([record],["filter"])["filter"]["bound_parameters"],{"keep":"unliked"})

    def test_data_replay_checks_parent_choices_and_reference_provenance(self):
        model = RecordedModel([{"event":"model","operation":"decide",
            "request":{"options":["old","NONE"],"question":"Next?"},"response":{"index":0}}])
        with self.assertRaises(ReplayDiverged):
            model.request("decide",options=["new","NONE"],question="Next?")
        replay = LocalReplay(None,Mock())
        replay.observed_names = set()
        replay.remember_names("rows = []\ntoken = 'simulated'")
        replay.environment.request.return_value = {"output":"[]"}
        self.assertEqual(replay.variable("rows"),[])
        for name in ("token","unrecorded","rows[0]"):
            with self.assertRaises(ValueError):
                replay.variable(name)

    def test_old_traces_are_readable_but_cannot_be_used_as_a_local_probe(self):
        work = object.__new__(FragmentTrialWorkbench)
        work.tasks = {"current"}
        work.paths = {"current/current":"now","earlier/outside":"previous"}
        work.study = SimpleNamespace()
        self.assertIn("earlier/outside",work.paths)
        with self.assertRaisesRegex(ValueError,"current Train"):
            work.cases({"objects":[],"local":{"cards":["test"],"cases":[{"task":"outside"}]}},
                       "current",{"cards":{"test":{}}})
        with self.assertRaisesRegex(ValueError,"current Train"):
            work.probe({},"outside")

    def test_automatic_local_trial_includes_cross_app_consumers_and_orders_by_inputs(self):
        work = object.__new__(FragmentTrialWorkbench)
        work.tasks = {"task"}
        work.groups = []
        work.current_events = {"task":[{"event":"task","instruction":"Like these Venmo payments."}]}
        defs = {"contacts":{"app":"phone","inputs":["feed"],"output":"people","api":"phone__search_contacts"},
                "feed":{"app":"venmo","inputs":[],"output":"feed","api":"venmo__show_social_feed"}}
        patch = {"objects":[{"kind":"card","name":name} for name in defs]}
        _,cases = work.cases(patch,"task",{"cards":defs})
        self.assertEqual(cases,[{"task":"task","cards":["feed","contacts"],"anchor":"start"}])
