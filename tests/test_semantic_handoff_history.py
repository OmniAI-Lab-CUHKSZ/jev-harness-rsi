import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from jev_harness.fragment_trials import FragmentTrialWorkbench
from jev_harness.history_impact import assess, historical_report, parameter_changes, trace_sites
from jev_harness.history_workbench import HistoryWorkbench, check_choice
from jev_harness.phase1_optimizer import ProbeComplete
from jev_harness.runtime import Trace
from jev_harness.semantic_library import initial_library, patch_library
from jev_harness.semantic_workbench import SemanticProbe
from jev_harness.transport import WorkerUnavailable


GUIDE = {'kind':'handoff', 'name':'send_request', 'value':{
    'requires':['bill'], 'label':'Collect a share', 'skill':'Check the live bill before requesting money.',
    'alternatives':['Report the amount without collecting it.']}}
BASE = 'Collect the shares.'
STATE = {'values':{'bill':{'total':10}}, 'completed':[], 'failed':None}


def native(operation, **kw):
    assert operation == 'decide'
    request = json.loads(kw['state'])['request'].lower()
    idx = (len(kw['options'])-1 if 'none' in request else 1 if 'report' in request else 0)
    return {'index':idx, 'probabilities':[int(i==idx) for i in range(len(kw['options']))],
            'mode':'D', 'input_tokens':17, 'output_tokens':0}


def instance(lib, request=BASE, max_calls=40):
    engine = SemanticProbe(Mock(request=Mock(side_effect=native)),Mock(),max_calls=max_calls,step_review_threshold=0.6)
    engine.prepare_options()
    engine.set_probe_target(None)
    engine.trace = Trace()
    engine.context = {'task':{'instruction':request,'supervisor':{},'datetime':'2026-01-01','functions':[]},'history':[]}
    engine.initialize(lib)
    return engine


def g_messages(engine, progress):
    with TestCase().assertRaises(ProbeComplete):
        engine.generate_message('Regular generation.',progress)
    return engine.trace.events[-1]['messages']


class HandoffTests(TestCase):
    def setUp(self):
        self.h0 = initial_library(semantic_handoffs=True)
        self.lib,_ = patch_library(self.h0,{'objects':[GUIDE]},set())

    def test_editing_surface_preserves_old_runs(self):
        legacy = initial_library()
        old = {'kind':'handoff','name':'special','value':{'when':'return True','skill':'Old guidance'}}
        installed,_ = patch_library(legacy,{'objects':[old]},set())
        self.assertIn('Old guidance',instance(installed).handoff('test')['fragment_guidance'])
        self.assertNotIn('handoff',legacy['questions'])
        self.assertEqual(list(self.h0['handoffs']),['default'])
        self.assertEqual(len(self.h0['cards']),15)
        q = {'kind':'decision','name':'handoff','value':{'question':'Which intent fits?','skill':'Check direction.'}}
        control = {'kind':'handoff','name':'control','value':{'requires':['rows'],'defer':'return False'}}
        installed,edits = patch_library(self.h0,{'objects':[GUIDE,q,control]},set())
        self.assertEqual(installed['questions']['handoff'],q['value']['question'])
        self.assertEqual(len(edits),3)
        for invalid in (old, {**GUIDE,'value':{**GUIDE['value'],'when':'return True'}},
                        {**q,'value':{'program':'return []'}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                patch_library(self.h0,{'objects':[invalid]},set())

    def test_only_true_handoff_pays_and_selection_reaches_g_and_clears_on_reentry(self):
        engine = instance(self.lib)
        engine.public.update(STATE['values'])
        engine.preferred_tool_docs()
        engine.read_offers()
        engine.read_offers()
        self.assertEqual(engine.requests,0)
        progress = engine.handoff('Remaining actions')
        self.assertEqual(engine.requests,1)
        engine.environment.request.assert_called_once_with('bind_progress',values=STATE['values'])
        sel = engine.trace.events[-1]
        self.assertEqual((sel['guide'],sel['calls'],sel['input_tokens'],sel['output_tokens']),('send_request',1,17,0))
        expected = {'name':GUIDE['name'],'skill':GUIDE['value']['skill']}
        messages = g_messages(engine,progress)
        self.assertEqual(json.loads(messages[1]['content'])['selected_handoff_guidance'],expected)
        self.assertIn('selected_handoff_guidance',messages[0]['content'])
        self.assertEqual(engine.generation_view(progress,True)['selected_handoff_guidance'],expected)
        engine.public.clear()
        progress.update(engine.handoff('Data changed'))
        self.assertEqual(engine.requests,1)
        cleared = g_messages(engine,progress)
        ordinary = instance(self.h0)
        self.assertEqual(cleared,g_messages(ordinary,ordinary.handoff('Data changed')))

    def test_contrast_and_native_none_deliver_the_same_ordinary_prompt_as_h0(self):
        for request, reason in [('Report the amount.','ordinary alternative'),('Choose none.','native NONE')]:
            h0 = instance(self.h0,request)
            new = instance(self.lib,request)
            for engine in (h0,new):
                engine.public.update(STATE['values'])
            p0,p1 = (e.handoff('open') for e in (h0,new))
            self.assertEqual((h0.requests,new.requests),(0,1))
            self.assertEqual(new.trace.events[-1]['reason'],reason)
            self.assertIsNone(p1['selected_handoff_guidance'])
            self.assertEqual(g_messages(h0,p0),g_messages(new,p1))

    def test_paged_menu_reserves_actual_generation_capacity(self):
        lib = deepcopy(self.h0)
        for i in range(16):
            lib['handoffs'][f'l{i}'] = {'label':f'Action {i}','skill':'Example'}
        short = instance(lib,max_calls=3)
        short.handoff('open')
        self.assertEqual(short.requests,0)
        self.assertIn('reserved',short.trace.events[-1]['reason'])
        sufficient = instance(lib,max_calls=4)
        sufficient.handoff('open')
        self.assertEqual((sufficient.requests,sufficient.trace.events[-1]['guide']),(3,'l0'))

    def test_archived_preview_honors_original_remaining_model_calls(self):
        engine = instance(self.lib,max_calls=3)
        engine.requests = 2
        engine.public.update(STATE['values'])
        engine.handoff('open')
        state = next(e for e in engine.trace.events if e['event']=='handoff_state')
        self.assertEqual(state['remaining_model_calls'],1)
        observed = check_choice(None,self.lib,BASE,state)
        self.assertEqual((observed['calls'],observed['eligible'],observed['guide']),(0,['send_request'],None))

    def test_historical_exposure_menu_and_conditional_text_are_different(self):
        archive = trace_sites([{'event':'task','instruction':BASE},{'event':'handoff_state',**STATE},
            {'event':'handoff_selection','guide':None},{'event':'card_menu','offered':['read_notes']},
            {'event':'tool_batch','node':'generation_agent','code':'rows = apis.simple_note.search_notes()'}])
        hist = {'earlier/a_1':archive,'other_version/a_1':archive}
        summary,details = historical_report(hist,self.h0,self.lib,[GUIDE])
        self.assertEqual((summary['historical_tasks'],summary['trace_entries'],summary['matched_tasks']),(1,2,1))
        self.assertEqual(summary['whole_task_outcome'],'unmeasured')
        self.assertTrue(details[0]['handoff_changes'][0]['menu_changed'])
        changed = deepcopy(self.lib)
        changed['handoffs']['send_request']['skill'] += ' More context.'
        diff = assess(archive,self.lib,changed,[GUIDE])['handoff_changes'][0]
        self.assertFalse(diff['menu_changed'])
        self.assertEqual(diff['changed_conditional_guidance'],['send_request'])
        changed = deepcopy(self.h0)
        changed['handoffs']['default']['skill'] += ' Extra text for every handoff.'
        plain = assess(archive,self.h0,changed,[{'kind':'handoff','name':'default'}])['handoff_changes'][0]
        self.assertIn('Extra text for every handoff.',plain['changed_direct_guidance']['handoff/default']['after'])
        op = {'kind':'card','name':'read_notes'}
        hits = assess(archive,self.h0,self.lib,[op])['exposures']['card/read_notes']
        self.assertIn('same_api_in_generated_code',hits)
        self.assertNotIn('card_executed',hits)

    def test_grounded_parameter_replay_and_unavailable_inputs(self):
        def lookup(choices):
            return {'inputs':['rows'],'skill':'Pick one','parameters':{'target':{
                'question':'Which record?','candidates':[{'label':s,'value':s} for s in choices]}}}
        old,new = lookup(['A','B']),lookup(['B','A'])
        state = {'values':{'rows':[]},'available':['rows'],'bound':{'target':'A'}}
        row = parameter_changes(old,new,state,BASE)[0]
        self.assertFalse(row['same_ordered_candidates'])
        self.assertTrue(row['bound_value_still_offered'])
        self.assertFalse(parameter_changes(old,lookup(['B']),state,BASE)[0]['bound_value_still_offered'])
        self.assertIn('unavailable_or_absent',parameter_changes(old,new,{**state,'values':{}},BASE)[0])

    def test_failed_offline_native_check_is_never_scored_as_an_ordinary_selection(self):
        work = object.__new__(HistoryWorkbench)
        work.lib = self.h0
        work.paths = {'prior/old_1':None}
        work.sites = {'prior/old_1':{'request':BASE,'handoffs':{1:STATE}}}
        worker = Mock()
        worker.worker.return_value.request.side_effect = WorkerUnavailable('Connection stopped')
        work.study = SimpleNamespace(workers=[worker])
        case = {'trace':'prior/old_1','relation':'different_goal','request':'Report only.','expected':None}
        row = work.wording_feedback([case],self.lib,{},[])[0]
        self.assertIsNone(row['candidate']['agrees'])
        self.assertIn('Connection stopped',row['candidate']['unavailable'])

    def test_real_editor_loop_returns_current_and_old_train_preview_before_accepting_patch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def trace(stage,task):
                path = root/'runs'/stage/task/'attempt_1'/'trace.jsonl'
                with_trace = Trace(path)
                with_trace.add('task',instruction=BASE)
                with_trace.add('handoff_state',**STATE)
                with_trace.close()
                return path
            earlier = trace('prior','old_1')
            now = trace('before','cur_1')
            account = Mock()
            account.json.return_value = {'same_goal':'Please collect these shares.','different_goal':'Report the amount.'}
            proposal = {'probe_task':'cur_1','objects':[GUIDE]}
            account.turn.return_value = json.dumps(proposal)
            study = SimpleNamespace(root=root,execution={},functions=[],solver=lambda:account,
                                    workers=[Mock(worker=Mock(return_value=Mock(request=Mock(side_effect=native))))])
            raw = [{'task_id':'cur_1','success':True,'output_tokens':20,'trace_path':str(now.relative_to(root))}]
            history = [{'batch':1,'candidate':'prior','train_traces':[str(earlier.relative_to(root))]}]
            folder = root/'batches'/'b02'/'b02r01'/'sol'
            folder.mkdir(parents=True)
            bench = HistoryWorkbench(study,self.h0,raw,history,folder,set())
            def current_prefix(_proposal,task):
                bench.calls += 1
                d = folder/f'probe_{bench.calls:02d}'
                d.mkdir()
                path = d/'trace.jsonl'
                path.write_bytes(now.read_bytes())
                doc = 'probe/'+d.name
                bench.paths[doc] = path
                return {'trace_document':doc,'current_batch_prefixes':[{'task':task,'candidate_trace':doc,'changes':{}}]}
            with patch.object(FragmentTrialWorkbench,'probe',side_effect=current_prefix):
                self.assertEqual(bench.work(),proposal)
            self.assertEqual((account.turn.call_count,account.json.call_count),(2,2))
            feedback = account.turn.call_args_list[-1].args[1]
            self.assertIn('historical_impact',feedback)
            details = json.loads((folder/'probe_01'/'historical_impact.json').read_text(encoding='utf-8'))
            s = details['summary']
            self.assertEqual((s['historical_tasks'],s['matched_tasks']),(1,1))
            self.assertEqual(s['local_native_cost'],{'calls':6,'input_tokens':102,'output_tokens':0})
            self.assertEqual(sum(x['relation']=='missing_role' for x in details['wording']),2)
            same = next(x for x in details['wording'] if x['relation']=='same_goal')
            self.assertIsNone(same['parent']['agrees'])
            self.assertTrue(same['candidate']['agrees'])
            resumed = HistoryWorkbench(study,self.h0,raw,history,folder,set())
            self.assertEqual(bench.lines('impact/probe_01'),resumed.lines('impact/probe_01'))
            self.assertIn(self.h0['questions']['handoff'],''.join(resumed.lines('library/questions')))
