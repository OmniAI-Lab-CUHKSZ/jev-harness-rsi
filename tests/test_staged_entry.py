import unittest
from unittest.mock import Mock, patch

from jev_harness.fragment_library import new_library
from jev_harness.formal_runtime import FormalExecutor
from jev_harness.phase1 import patch_library
from jev_harness.phase1_optimizer import PrefixProbe
from jev_harness.runtime import Trace
from jev_harness.staged_entry import COMPLETE_Q, READ_Q, StagedExecutor
from phase1_run import PhaseOneStudy, design_manifest, paired_holdout, summary


def fragment(name, output, result, inputs=(), **fields):
    return {"kind":"fragment","name":name,"value":{"definition":{
        "app":"phone","output":output,"inputs":list(inputs),"effect":"read-only",
        "steps":[],"source_tasks":[],"result":result,**fields}}}


class Native:
    def __init__(self, choose):
        self.choose,self.requests=choose,[]

    def request(self,op,**kw):
        if op!="decide":
            raise AssertionError("A compilation step requested generation")
        self.requests.append(kw)
        options=kw["options"]
        label=self.choose(kw["question"],options)
        i=next(i for i,o in enumerate(options) if o.startswith(label))
        return {"index":i,"probabilities":[int(j==i) for j in range(len(options))],
                "input_tokens":11,"output_tokens":0,"seconds":0.01,"mode":"D"}


class Simulation(StagedExecutor):
    def agent(self, **kwargs):
        self.stage="generation"
        self.trace.add("example_handoff",progress=self.handoff_progress([],self.live))

    def call(self,app,api,arguments,node=None):
        if (app,api)==("supervisor","complete_task"):
            self.done=True
            self.trace.add("submitted",data=arguments)
            return
        raise AssertionError((app,api,arguments))


def simulate(objects,choose,instruction="Please do this on my phone."):
    lib=new_library() | {"menus":{}}
    for i in range(0,len(objects),3):
        lib,_=patch_library(lib,{"objects":objects[i:i+3]},[],[])
    engine=Simulation(Native(choose),Mock())
    engine.prepare_options()
    engine.done=False;engine.trace=Trace()
    engine.context={"task":{"instruction":instruction,"datetime":"2024-01-01","supervisor":{}},
                    "history":[],"vars":{},"generated":{}}
    engine.execute_task(lib)
    return engine


class StagedTests(unittest.TestCase):
    def test_new_complete_fragment_needs_no_menu_and_binds_a_real_value(self):
        op=fragment("rank_records","selected_record",{"ref":"vars.metric"},["metric"],finish="answer",
                    slots={"metric":{"question":"Which measure?","candidates":[{"label":"Count","value":"cnt"},
                                                                             {"label":"Length","value":"len"}]}})
        engine=simulate([op],lambda q,o:"rank records" if q==COMPLETE_Q else "Length")
        self.assertTrue(engine.done)
        self.assertEqual([e["data"] for e in engine.trace.events if e["event"]=="submitted"],[{"answer":"len"}])
        self.assertEqual(len(engine.model.requests),2)
        self.assertNotIn("stage_enter",[e["event"] for e in engine.trace.events])
        from jev_harness.semantic_assembly import SemanticAssembly
        self.assertIs(PhaseOneStudy.compiled_executor, SemanticAssembly)
        self.assertTrue(issubclass(PrefixProbe,StagedExecutor))

    def test_declining_even_an_exact_route_runs_scoped_cross_app_read_before_handoff(self):
        action=fragment("approve_order","changed","yes",app="venmo",finish="action",effect_kind="write")
        action["value"]["route"]={"patterns":["^On Venmo, find my friends$"],"goals":["changed"],"complete":True}
        source=fragment("get_contacts","contacts",["human_1"],effect_kind="read")
        menu={"kind":"menu","name":"people","value":{"cues":["friends"],"question":"Which contact data?",
              "options":[{"fragment":"get_contacts","label":"Actual contacts"}]}}
        e=simulate([action,source,menu],lambda q,o:"NONE" if q==COMPLETE_Q else "Actual contacts",
                   "On Venmo, find my friends")
        self.assertFalse(e.done)
        self.assertNotIn("changed",e.live)
        self.assertEqual(e.live["contacts"],["human_1"])
        self.assertEqual([t["progress"]["live_values"]["contacts"] for t in e.trace.events
                          if t["event"]=="example_handoff"],[["human_1"]])
        self.assertEqual(len(e.model.requests),2)

    def test_local_state_can_finish_pending_operation_without_reasking_the_root(self):
        data=fragment("get_people","people",[{"name":"P","id":23}],effect_kind="read")
        op=fragment("report_person","answer",{"ref":"inputs.person"},["person"],finish="answer",
                    slots={"person":{"question":"Which person?", "candidates":{"source":{"ref":"vars.people"},
                          "label":{"ref":"item.name"},"value":{"ref":"item.id"}}}})
        e=simulate([data,op],lambda q,o:{COMPLETE_Q:"report person",READ_Q:"get people","Which person?":"P"}[q])
        self.assertTrue(e.done)
        self.assertEqual([t["data"] for t in e.trace.events if t["event"]=="submitted"],[{"answer":23}])
        self.assertEqual([r["question"] for r in e.model.requests],[COMPLETE_Q,READ_Q,"Which person?"])

    def test_app_scope_opens_a_read_even_when_its_keyword_is_absent(self):
        source=fragment("get_household","household_members",["person_1"],effect_kind="read")
        menu={"kind":"menu","name":"household","value":{"cues":["roommates"],"question":"Which data?",
              "options":[{"fragment":"get_household","label":"Household members"}]}}
        task="On my phone, find the people who share my apartment."
        engine=simulate([source,menu],lambda q,o:"Household members",task)
        self.assertEqual(engine.live["household_members"],["person_1"])
        self.assertTrue(any(e.get("opened_by")=="app_prefilter" for e in engine.trace.events))
        menu["value"]["after_roles"]=["needed_observation"]
        waiting=simulate([source,menu],lambda q,o:"Household members",task)
        self.assertNotIn("household_members",waiting.live)
        self.assertEqual(waiting.model.requests,[])

    def test_none_on_a_parameter_does_not_guess_or_replay_a_write(self):
        op=fragment("change_flag","changed",True,["direction"],finish="action",effect_kind="write",
                    slots={"direction":{"candidates":[{"label":"Enable","value":"on"},{"label":"Disable","value":"off"}]}})
        e=simulate([op],lambda q,o:"change flag" if q==COMPLETE_Q else "NONE")
        self.assertFalse(e.done)
        self.assertEqual(e.entry_status,"deferred")
        self.assertNotIn("changed",e.live)
        self.assertTrue(any(t["event"]=="example_handoff" for t in e.trace.events))

    def test_read_dependency_cannot_execute_an_action_the_user_declined(self):
        write=fragment("write_object","mutated",1,finish="action",effect_kind="write")
        read=fragment("read_after_write","data",{"ref":"inputs.mutated"},["mutated"],effect_kind="read")
        e=simulate([write,read],lambda q,o:"NONE" if q==COMPLETE_Q else "read after write")
        self.assertFalse(e.done)
        self.assertNotIn("mutated",e.live)
        self.assertNotIn("data",e.live)

    def test_no_progress_keeps_the_original_generator_input(self):
        e=simulate([],lambda q,o:"NONE")
        with patch.object(FormalExecutor,"agent",autospec=True) as generate:
            StagedExecutor.agent(e)
            self.assertIsNone(generate.call_args.kwargs["progress"])
            e.entry_name="selected_op"
            e.entry_reason="Missing a result needed for this operation"
            StagedExecutor.agent(e)
            self.assertEqual(generate.call_args.kwargs["progress"]["complete_operation"]["selected"],"selected_op")

    def test_tool_docs_can_fix_omission_without_evicting_essential_tools(self):
        e=simulate([],lambda q,o:"NONE")
        e.context["task"]["functions"]=[{"name":"phone__record"+str(i),"parameters":{}} for i in range(30)]
        e.context["task"]["functions"] += [{"name":"supervisor__"+n,"parameters":{}}
                                           for n in ("complete_task","show_account_passwords")]
        self.assertNotIn("record29",[x["api"] for x in e._function_docs()])
        e.library,_=patch_library(e.library,{"objects":[{"kind":"tool_context","name":"records",
            "value":{"cues":["phone"],"apis":["phone__record29"]}}]},[],e.context["task"]["functions"])
        docs=[x["api"] for x in e._function_docs()]
        self.assertEqual(len(docs),18)
        self.assertIn("complete_task",docs)
        self.assertIn("record29",docs)

    def test_execution_path_counts_success_after_actual_business_read(self):
        stage={"complete":"none","cost_by_stage":{},"handoff":True,"local_fragments":[]}
        row={"task_id":"sample_1","success":False,"seconds":1,"output_tokens":23,"model_calls":{"D":1,"G":1},
             "fragments":["venmo_access_token"],"staged":stage}
        successful={**row,"task_id":"sample_2","success":True,"output_tokens":15,
                    "staged":{**stage,"local_fragments":["read_contacts"]}}
        paths=summary([row,successful])["staged"]["execution_paths"]
        self.assertEqual(paths["generation_without_selected_partial"],{"tasks":1,"successes":0,"output_tokens":23})
        self.assertEqual(paths["partial_then_generation"],{"tasks":1,"successes":1,"output_tokens":15})

    def test_split_and_holdout_pairing(self):
        m=design_manifest()
        a={t for b in m["train_batches"] for t in b["tasks"]}
        d=set(m["dev_tasks"]); h=set(m["normal_holdout_tasks"])
        scenario=lambda ids:{x.rsplit("_",1)[0] for x in ids}
        self.assertEqual((len(a),len(d),len(h)),(90,57,90))
        self.assertFalse((scenario(a)&scenario(h))|(scenario(d)&scenario(h))|(scenario(a)&scenario(d)))
        self.assertEqual(len(a&set(m["historical_seed_train_ids"])),30)
        seen=[]
        class Study:
            def execute_jobs(self,jobs,fn):return [fn(v,"gpu"+str(i)) for i,v in enumerate(jobs)]
            def run_task(self,t,s,arm,lib,*,workers):
                seen.append((t,arm,workers))
                return t+arm
        self.assertEqual(paired_holdout(Study(),["t","u"],"v",{}),(["tA","uA"],["tB","uB"]))
        self.assertEqual(seen,[("t","A","gpu0"),("t","B","gpu0"),("u","B","gpu1"),("u","A","gpu1")])
