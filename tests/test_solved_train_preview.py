from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from formal_run import read, save
from phase1_run import offline_optimizer_costs
from jev_harness.fragment_assembly import FragmentAssembly
from jev_harness.runtime import Trace
from jev_harness.semantic_assembly import SemanticAssembly
from jev_harness.solved_train_preview import SolvedReference, first_change, recorded_library, review_solved, solved_documents
from jev_harness.transport import WorkerUnavailable


def menu(*items):
    return [{"label":label, "value":value} for label,value in items]


def execute_recorded_steps(engine, steps):
    engine.request = "Look up the saved item"
    engine.registry = {"questions":{"step":"Which item?"}, "skills":{}}
    for op, data in steps:
        if op == "decide":
            engine.decision("Which item?", {}, data, node="fragment/step")
        elif op == "paged":
            FragmentAssembly.choose(engine, "step", data)
        else:
            engine._model("generate", messages=[{"role":"user","content":data}])


def first_option(op, **args):
    if op == "generate":
        return {"text":"done","mode":"G","output_tokens":1}
    return {"index":0,"probabilities":[1.0]+[0.0]*(len(args["options"])-1),
            "input_tokens":23,"output_tokens":0,"backend_requests":1}


class SolvedPreviewTests(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.folder = self.root/"batches/b02/b02r01/sol_scoped"
        self.worker = Mock()
        self.model = self.worker.worker.return_value
        self.model.request.side_effect = first_option
        self.bench = SimpleNamespace(archived=[], reference_paths={}, docs={},
            study=SimpleNamespace(execute_jobs=lambda jobs, f:[f(job,self.worker) for job in jobs]))
        self.old, self.ballots = {}, {}
        self.task_event = {"event":"task", "task_id":"a_1","instruction":"Look up the saved item",
                           "seed":7,"supervisor":{},"datetime":"2025-01-01"}
        replay = patch.object(SemanticAssembly, "execute_task", execute_recorded_steps)
        replay.start()
        self.addCleanup(replay.stop)

    def add_path(self, task, parent_steps, child_steps, *, recording=None, source_library=None):
        original = SemanticAssembly(Mock(request=first_option), None)
        original.trace = Trace()
        original.seed = 7
        original.prepare_options()
        original.execute_task(parent_steps if recording is None else recording)
        events = [{**self.task_event,"task_id":task}, *original.trace.events, {"event":"summary","success":True}]
        parent = SolvedReference(events,parent_steps,[],{})
        path, ballot = first_change(events,parent,child_steps,[],{},source_library=source_library)
        doc = "b01r02/"+task
        self.bench.archived.append(doc)
        self.bench.reference_paths[task] = (doc,events,parent)
        self.old[task] = {"trace":doc,**path}
        if ballot:
            self.ballots[task] = ballot
        return path

    def review(self, stage="draft"):
        return review_solved(self.bench,self.old,self.ballots,self.folder/("trial_"+stage))

    def test_one_solved_reference_per_task_including_success_before_later_failures(self):
        def trace(stage, task):
            return f"runs/{stage}/{task}/attempt_1/trace.jsonl"
        statuses = {"entry/a":True, "r1/a":True, "r2/a":False, "r1/b":False, "r2/c":True,
                    "confirm/c":True, "r1/current":True}
        def refs(stage, tasks):
            return [trace(stage,t) for t in tasks]
        records = {s:{"task":s.split("/")[1],"task_success":v} for s,v in statuses.items()}
        bench = SimpleNamespace(tasks={"current"}, archived=list(statuses), precedents=SimpleNamespace(records=records),
            study=SimpleNamespace(prior_train_entries=refs("entry",["a"])),
            previous=[{"parent_traces":refs("entry",["a"]),"train_traces":refs("r1",["a","b","current"])},
                      {"parent_traces":refs("entry",["a"]),"train_traces":refs("r2",["a","c"]),
                       "paired_selection":{"repeats":{"c":[{"trace_path":trace("confirm","c")}]}}}])
        self.assertEqual(solved_documents(bench), {"a":"r1/a","c":"confirm/c"})

    def test_a_later_successful_entry_survives_a_revisit_with_no_patch(self):
        stages = ("b03_h00_entry","b03r01","b06_revisit03_b06r02_entry")
        docs = [s+"/a_1" for s in stages]
        paths = ["runs/"+doc+"/attempt_1/trace.jsonl" for doc in docs]
        records = {doc:{"task":"a_1","task_success":True} for doc in docs}
        bench = SimpleNamespace(tasks=set(), archived=docs, precedents=SimpleNamespace(records=records),
            study=SimpleNamespace(prior_train_entries=[paths[0],paths[2]]),
            previous=[{"visit":"b03","train_traces":[paths[1]],"parent_traces":[paths[0]]}])
        self.assertEqual(solved_documents(bench),{"a_1":docs[2]})

    def test_same_first_choice_stops_before_later_change_and_shared_input_is_charged_once(self):
        one = ("decide",menu(("Read notes","notes")))
        more = ("decide",menu(("Read notes","notes"),("Read requests","requests")))
        for task in ("a_1","b_1"):
            self.add_path(task,[one,("generate","old prompt")],[more,("generate","changed prompt")])
        result = self.review()
        self.assertEqual(result["counts"],{"same_first_choice":2})
        self.assertEqual((result["model_calls"],result["distinct_requests"],result["exceptions"]),(1,1,[]))
        self.assertEqual(len(self.bench.docs[result["details"]]),2)
        self.assertEqual(self.review(),result)
        self.assertEqual(self.review("revised")["model_calls"],0)
        self.model.request.assert_called_once()
        cost = offline_optimizer_costs(self.root,historical=False)["offline_27b_historical_choice"]
        self.assertEqual((cost["model_calls"],cost["input_tokens"]),(1,23))

    def test_shared_response_is_bound_independently_on_each_task(self):
        before = ("decide",menu(("Read","old")))
        for task,value in (("a_1","old"),("b_1","another")):
            self.add_path(task,[before],[("decide",menu(("Read",value),("New","other")))])
        out = self.review()
        self.assertEqual(out["counts"],{"same_first_choice":1,"changed_choice":1})
        row, = out["exceptions"]
        self.assertEqual((row["task"],row["parent_offered"],row["candidate_choice"]["value"]),("b_1",False,"another"))
        self.assertEqual(out["model_calls"],1)

    def test_first_nondecision_and_unaligned_archives_never_dispatch_a_later_page(self):
        one = ("decide",menu(("Read","read")))
        more = ("decide",menu(("Other","other"),("Read","read")))
        self.add_path("a_1",[("generate","old"),one],[("generate","new"),more])
        self.add_path("b_1",[one],[more],recording=[("generate","earlier recording")])
        self.add_path("c_1",[one],[one])
        out = self.review()
        self.assertEqual(out["counts"],{"changed_request":1,"unaligned":1,"equal_on_record":1})
        self.assertEqual(out["exceptions"][0]["first_change"]["parent"],"generate")
        self.model.request.assert_not_called()

    def test_changed_menu_at_recorded_state_is_asked_even_if_inherited(self):
        old = ("decide",menu(("Read pending","pending")))
        extended = ("decide",menu(("Read pending","pending"),("Read history","history")))
        reordered = ("decide",menu(("Read history","history"),("Read pending","pending")))
        for task, page in (("inherited",extended),("edited",reordered),("restored",old)):
            self.add_path(task,[extended],[page,("generate","later change")],recording=[old])
        result = self.review()
        self.assertEqual(result["counts"],{"same_first_choice":2,"changed_choice":1})
        self.assertEqual((result["model_calls"],result["compared_to_recorded_menu"]),(2,3))
        rows = {r["task"]:r for r in read(self.folder/"trial_draft/historical_choice.json")["tasks"]}
        self.assertTrue(rows["inherited"]["recording_comparison"]["candidate_menu_matches_parent"])
        changed = rows["edited"]
        self.assertFalse(changed["recording_comparison"]["candidate_menu_matches_parent"])
        self.assertTrue(changed["recorded_choice_offered"])
        self.assertNotIn("parent_choice", changed)
        self.assertEqual((changed["recorded_choice"]["value"],changed["candidate_choice"]["value"]),("pending","history"))
        self.assertEqual(rows["restored"]["source"], "recorded_index")
        self.assertEqual(self.review("revised")["model_calls"],0)

    def test_drifted_paged_menu_uses_archived_actions_and_keeps_missing_bindings_unknown(self):
        archive = [("paged",menu(*[(f"Row {i}",100+i) for i in range(18)]))]
        new = [("paged",menu(("New",900), *[(f"Row {i}",100+i) for i in range(18)]))]
        self.add_path("with_snapshot",new,new,recording=archive,source_library=lambda:archive)
        self.add_path("without_snapshot",new,new,recording=archive)
        out = self.review()
        self.assertEqual(out["counts"], {"changed_choice":1,"unaligned":1})
        self.assertEqual(out["model_calls"],1)
        row = next(r for r in out["exceptions"] if r["task"] == "with_snapshot")
        self.assertEqual((row["recorded_choice"]["value"],row["candidate_choice"]["value"]),(100,900))

    def test_recorded_version_uses_confirmation_arm_and_scored_entry(self):
        self.bench.study.root = self.root
        self.bench.study.settings = {}
        self.bench.previous = [{"parent":"v1","candidate":"v2","paired_selection":{"repeats":{
            "b":[{"arm":"parent","trace_path":"runs/confirm_parent/b/attempt_1/trace.jsonl"},
                 {"arm":"candidate","trace_path":"runs/confirm_candidate/b/attempt_1/trace.jsonl"}]}}}]
        save(self.root/"checkpoints.json",[{"batch":1,"visit":"b01","entry_version":"v0"}])
        save(self.root/"batches/b01/entry.json",{"train_traces":["runs/entry/a/attempt_1/trace.jsonl"]})
        for version in ("v0","v1","v2"):
            save(self.root/"versions"/(version+".json"), {"version":version})
        for doc,version in (("entry/a","v0"),("confirm_parent/b","v1"),("confirm_candidate/b","v2")):
            self.assertEqual(recorded_library(self.bench,doc),{"version":version})

    def test_unchanged_model_input_with_a_new_grounded_value_reuses_the_recorded_index(self):
        for count,kind in ((2,"decide"),(18,"paged")):
            with self.subTest(kind=kind):
                options = menu(*[(f"Row {i}",100+i) for i in range(count)])
                changed = [{"label":o["label"],"value":999 if i==0 else o["value"]} for i,o in enumerate(options)]
                task = kind+"_1"
                path = self.add_path(task,[(kind,options)],[(kind,changed)])
                self.assertEqual(path["first_change"]["fields"],["option_values"])
        out = self.review()
        self.assertEqual(out["counts"],{"changed_choice":2})
        for row in out["exceptions"]:
            self.assertEqual((row["parent_choice"]["value"],row["candidate_choice"]["value"]),(100,999))
            self.assertEqual(row["source"],"recorded_index")
        self.model.request.assert_not_called()

    def test_lost_reply_stays_unknown_across_resume_and_revision(self):
        self.add_path("a",[("decide",menu(("Read","a")))],[("decide",menu(("Read","a"),("Ask","b")))])
        self.model.request.side_effect = WorkerUnavailable("lost reply")
        self.assertEqual(self.review()["counts"],{"outcome_unconfirmed":1})
        (self.folder/"trial_draft/historical_choice.json").unlink()
        self.assertEqual(self.review()["unconfirmed_calls"],1)
        self.assertEqual(self.review("revised")["unconfirmed_calls"],0)
        self.assertEqual(self.model.request.call_count,1)
        cost = offline_optimizer_costs(self.root,historical=False)["offline_27b_historical_choice"]
        self.assertEqual((cost["model_calls"],cost["unconfirmed_calls"]),(1,1))

    def test_recovery_with_a_later_concurrent_dispatch_already_on_disk(self):
        for task in ("a","b"):
            self.add_path(task,[("decide",menu(("Read","old")))],
                          [("decide",menu((task,task),("Read","old")))])
        journal = self.folder/"trial_draft/historical_requests/0002.json"
        pending = {"request":self.ballots["b"]["request"],"status":"outcome_unconfirmed","model_calls":1}
        save(journal,pending)
        out = self.review()
        self.assertEqual((out["model_calls"],out["unconfirmed_calls"]),(2,1))
        self.assertEqual(read(journal),pending)
        self.model.request.assert_called_once()

    def test_revision_with_another_prompt_is_separately_observed_and_metered(self):
        old = [("decide",menu(("A",1)))]
        self.add_path("a",old,[("decide",menu(("B",2)))])
        self.review()
        self.add_path("a",old,[("decide",menu(("C",3)))])
        out = self.review("revised")
        self.assertEqual(out["exceptions"][0]["candidate_choice"]["value"],3)
        self.assertEqual(offline_optimizer_costs(self.root,historical=False)["offline_27b_historical_choice"]["model_calls"],2)

    def test_old_configs_resume_with_the_saved_protocol(self):
        from jev_harness.cli import RESOURCES, initialize
        for mode in (True,False,None):
            with self.subTest(mode=mode):
                base = read(RESOURCES/"experiment.json")
                if mode is None:
                    del base["optimizer"]["historical_choice"]
                else:
                    base["optimizer"]["historical_choice"] = mode
                save(self.root/"configuration.json",base)
                initialize(self.root,read(RESOURCES/"experiment.json"))
                self.assertEqual(read(self.root/"configuration.json"),base)
