"""Train-only editing with archived state comparisons and local semantic routing checks."""
from pathlib import Path

from .formal_optimizer import read_events
from .fragment_trials import FragmentTrialWorkbench
from .history_impact import historical_report, trace_sites
from .phase1_optimizer import leading_choices, serialized
from .runtime import Trace
from .semantic_assembly import SemanticAssembly
from .semantic_handoff import menu, select_guidance
from .sol_account import AccountError
from .transport import WorkerError, WorkerUnavailable


INSTRUCTIONS = '''The runner can perform native Jev selection at a real generation handoff. Program checks live
roles, then the decision head chooses a specialized guidance or ordinary assistance. Propose general action
semantics including actor, intended effect and direction. Think about paraphrases, opposite actions, read-only
requests and missing inputs. Sol can add or revise these objects and their question, skills and alternatives.
Use card.handoff_skill for factual instructions common to consumers of that card. Put specialized text in a
named semantic handoff. This lets the shared selector distinguish intent even when new wording is used.
Every ordinary trial returns historical_impact: recorded exposures, reproducible parameter/menu differences,
and optional local wording checks. Only current-batch formal reruns measure whole-task outcomes. Open the
impact document to inspect individual historical sites. Use wiring, menu and state evidence to localize errors.
Use optional wording_checks in a patch to test the guide on saved Train states. Example shape:
[{"trace":"current/<task> or trial/<task> or an indexed past Train trace","event":integer (optional),
"relation":"same_goal or different_goal","request":"a complete rewritten user request","expected":"guide name or null (optional for same_goal)"}].
For changed goals null means ordinary guidance. The workbench can supply an offline minimal pair if needed.
A successful local selection is evidence for routing. Use Train task results to assess success and cost.
If an observed failure needs an unavailable runtime operation, optionally include capability_gap in the patch:
describe the trace event, the missing capability and what the six editable objects could not express.
The runner keeps this note with the revision for later inspection.'''

SCHEMA = '''Semantic handoff / name: {requires:[live role names],label:"the supported action and who does what",
skill:"guidance supplied to generation only if chosen",alternatives?:["plausible different outcome leading to ordinary assistance"]}.
Use structural prerequisites for data. The request, projected values for the required roles, labels and an
ordinary alternative go to the native head. An alternative never authorizes an action. decision / handoff edits
{question?,skill?}. Existing default general guidance and deterministic control-only handoffs support their
original fields. Semantic guide objects carry text only. The host implements the model call and tool execution.'''


def check_choice(model, lib, request, state):
    engine = SemanticAssembly(model,None,max_calls=state.get('remaining_model_calls',40))
    engine.prepare_options()
    engine.trace = Trace()
    engine.context = {'task':{'instruction':request,'functions':[]}}
    engine.initialize(lib)
    try:
        select_guidance(engine,state)
    except (WorkerError,WorkerUnavailable) as exc:
        replies = [e['response'] for e in engine.trace.events if e['event']=='model' and not e['response'].get('error')]
        return {'guide':None,'choice':None,'eligible':menu(lib,{**state,'request':request})[0],'unavailable':str(exc)[:200],
                'calls':len(replies),'input_tokens':sum(r.get('input_tokens',0) for r in replies),
                'output_tokens':sum(r.get('output_tokens',0) for r in replies)}
    event = engine.trace.events[-1]
    visible = {k:event[k] for k in ('guide','choice','eligible','reason','calls','input_tokens','output_tokens')}
    ballots = [e for e in engine.trace.events if e['event']=='model' and e['operation']=='decide'
               and 'probabilities' in e['response']]
    if ballots:
        visible['last_ballot'] = leading_choices(ballots[-1])
    return visible


class HistoryWorkbench(FragmentTrialWorkbench):
    prompt = FragmentTrialWorkbench.prompt + '\n' + INSTRUCTIONS

    def __init__(self,*args):
        super().__init__(*args)
        # The legacy editor permits a text-triggered handoff. Publish the actual new-mode schema.
        prior = self.docs['dsl']
        start = next(i for i,line in enumerate(prior) if line.startswith('handoff / name,'))
        end = next(i for i,line in enumerate(prior[start:],start) if line.startswith('Any program gets'))
        self.docs['dsl'][start:end] = SCHEMA.splitlines() + [
            'Control-only handoff: {requires?,when?,defer?,apis?,reads?}. General handoff/default: {skill}.']
        for group in ('questions','skills'):
            self.docs['library/'+group].append(serialized({'handoff':self.lib[group]['handoff']}))
        self.docs['code/semantic_handoff'] = Path(__file__).with_name('semantic_handoff.py').read_text(encoding='utf-8').splitlines()
        self.sites = {}
        self.archived = list(dict.fromkeys(doc for doc in self.paths if doc.rsplit('/',1)[-1] not in self.tasks
                                           and not doc.startswith(('probe/','current/'))))
        from formal_run import read
        for file in self.folder.glob('probe_*/historical_impact.json'):
            prior = read(file)
            self.docs['impact/'+file.parent.name] = [serialized(row) for row in prior['details']+prior['wording']]

    def opening_evidence(self):
        info = super().opening_evidence()
        prior = self.previous[-1] if self.previous else {}
        info['historical_impact_feedback'] = {'past_train_trace_entries':len(self.archived),
            'editable_handoff_interface':SCHEMA,'last_preview':prior.get('historical_impact'),
            'last_reported_capability_gap':prior.get('capability_gap')}
        return info

    def probe_key(self,proposal,task):
        return (*super().probe_key(proposal,task),proposal.get('wording_checks',[]))

    def trace_data(self,doc):
        if doc not in self.sites:
            self.sites[doc] = trace_sites(read_events(self.paths[doc]))
        return self.sites[doc]

    def probe(self,proposal,task):
        from formal_run import save
        checks = proposal.get('wording_checks',[])
        if not isinstance(checks,list) or len(checks)>6 or any(not isinstance(c,dict) for c in checks):
            raise ValueError('wording_checks accepts up to six named Train cases')
        focus = super().probe(proposal,task)
        folder = self.folder/f'probe_{self.calls:02d}'
        candidate,_ = self.install(self.lib,proposal,self.sources,self.study.functions)
        archive = {doc:self.trace_data(doc) for doc in self.archived}
        summary,details = historical_report(archive,self.lib,candidate,proposal['objects'])
        docname = 'impact/'+folder.name
        self.docs[docname] = [serialized(row) for row in details]
        trial = {row['task']:row['candidate_trace'] for row in focus['current_batch_prefixes'] if 'candidate_trace' in row}
        # Cover a current execution and a previous Train task sharing prerequisites. Prefer an actual positive selection.
        target_names = {x['name'] for x in proposal['objects'] if x['kind']=='handoff' and 'label' in x['value']}
        selector_edit = any(x['kind']=='decision' and x['name']=='handoff' for x in proposal['objects'])
        anchors = []
        if target_names or selector_edit:
            current = [focus.get('trace_document'),*(trial[t] for t in dict.fromkeys([task,*sorted(trial)]) if t in trial),
                       *('current/'+t for t in dict.fromkeys([task,*sorted(self.tasks)]))]
            panels = [list(dict.fromkeys(doc for doc in current if doc in self.paths)),list(reversed(self.archived))]
            for docs in panels:
                available = [(doc,index,state) for doc in docs for index,state in self.trace_data(doc)['handoffs'].items()
                             if any(selector_edit or n in target_names for n in menu(candidate,state)[0])]
                if available:
                    anchors.append(next((item for item in available if item[2].get('observed_guide') and
                                         (selector_edit or item[2]['observed_guide'] in target_names)), available[0]))
        checks = list(checks)
        author_status = []
        for i,anchor in enumerate(anchors,1):
            doc,idx,_ = anchor
            aliases = {doc, *('trial/'+task_id for task_id,path in trial.items() if path == doc)}
            covered = {x.get('relation') for x in checks if isinstance(x.get('trace'),str) and x['trace'] in aliases and x.get('event',idx)==idx}
            if not {'same_goal','different_goal'}.issubset(covered):
                automatic,status = self.author_pair(folder/f'wording_author_{i}',candidate,anchor)
                author_status.append({'trace':doc,'event':idx,'status':status})
                checks.extend(x for x in automatic if x['relation'] not in covered)
        if (target_names or selector_edit) and not anchors:
            author_status.append({'status':'no recorded eligible state'})
        audits = self.wording_feedback(checks,candidate,trial,anchors)
        cost = {k:sum(obs[k] for row in audits for obs in row.get('evaluations',[]))
                for k in ('calls','input_tokens','output_tokens')}
        compact = [{k:v for k,v in row.items() if k!='evaluations'} for row in audits]
        summary.update(document=docname, wording_checks=compact, local_native_cost=cost, offline_author=author_status)
        self.docs[docname].extend(serialized(row) for row in audits)
        save(folder/'historical_impact.json',{'summary':summary,'details':details,'wording':audits})
        save(self.folder.parent/'optimizer_impact.json',summary)
        focus['historical_impact'] = summary
        save(folder/'result.json',{**focus,'comparison_objects':proposal['objects']})
        self.last_probed = self.probe_key(proposal,task)
        return focus

    def author_pair(self,folder,lib,anchor):
        doc,index,state = anchor
        names,choices,facts = menu(lib,state)
        payload = {'original_request':self.trace_data(doc)['request'], 'eligible_guides':names,
                   'labels':[c['label'] for c in choices], 'public_evidence':facts}
        prompt = ('Create two diagnostic user requests in the language of the original AppWorld request. '
                  'Preserve the constraints and required effect while paraphrasing the action for same_goal. '
                  'For different_goal, create a close alternative: keep the same data source, entities and topic, '
                  'but change the actual deliverable enough that every specialist would be inappropriate. Prefer a '
                  'natural request for an intermediate piece of information before the user decides whether to act. '
                  'When that is unsuitable, reverse the direction or change the intended actor or scope. Keep the '
                  'case plausible from the evidence; avoid unrelated requests or simply negating the original wording. '
                  'Return JSON with two nonempty distinct strings, same_goal and different_goal.\n'+serialized(payload))
        try:
            answer = self.study.solver().json(prompt,folder)
            texts = [answer[k] for k in ('same_goal','different_goal')]
            if not all(isinstance(s,str) and s.strip() for s in texts) or len(set([*texts,payload['original_request']]))!=3:
                raise ValueError('Expected distinct rewrites')
            return [dict(trace=doc,event=index,relation=k,request=s,**({'expected':None} if k=='different_goal' else {}))
                    for k,s in zip(('same_goal','different_goal'),texts)], 'offline Sol pair'
        except (AccountError,KeyError,ValueError,TypeError) as exc:
            return [], 'unavailable: '+str(exc)[:160]

    def wording_feedback(self,checks,candidate,trial,anchors):
        if not checks and not anchors:
            return []
        if getattr(self.study, 'lanes_per_gpu', 1) != 2:
            return self.wording_on_worker(checks, candidate, trial, anchors, self.study.workers[0])
        # Keep all rewrites of one recorded trace together so repeated original
        # ballots are reused and the parent and candidate share a physical GPU.
        groups = {}
        for position, case in enumerate(checks):
            doc = case.get('trace')
            if isinstance(doc, str) and doc.startswith('trial/'):
                doc = trial.get(doc.split('/', 1)[1], '')
            group = groups.setdefault(doc if isinstance(doc, str) else None, ([], []))
            group[0].append((position, case))
        for anchor in anchors:
            groups.setdefault(anchor[0], ([], []))[1].append(anchor)

        def compare(group, worker):
            items, sites = group
            rows = self.wording_on_worker([case for _, case in items], candidate, trial, sites, worker)
            return list(zip((i for i, _ in items), rows[:len(items)])), rows[len(items):]

        reports = self.study.execute_jobs(list(groups.values()), compare)
        primary = sorted((item for items, _ in reports for item in items), key=lambda item: item[0])
        order = {(doc, index): i for i, (doc, index, _) in enumerate(anchors)}
        extra = sorted((row for _, rows in reports for row in rows), key=lambda row: order[row['trace'], row['event']])
        return [row for _, row in primary] + extra

    def wording_on_worker(self,checks,candidate,trial,anchors,worker):
        rows,cache = [],{}
        def evaluate(doc,idx,arm,request,state,missing=None):
            key = (doc,idx,arm,request,missing)
            if key not in cache:
                lib = self.lib if arm=='parent' else candidate
                names = menu(lib,{**state,'request':request})[0]
                try:
                    model = worker.worker('model') if names else None
                    cache[key] = check_choice(model,lib,request,state)
                except (WorkerUnavailable,WorkerError) as exc:
                    cache[key] = {'guide':None,'choice':None,'eligible':names,'unavailable':str(exc)[:200],
                                  'calls':0,'input_tokens':0,'output_tokens':0}
                if 'unavailable' in cache[key]:
                    worker.close_worker('model')
                return cache[key]
            return {**cache[key],'calls':0,'input_tokens':0,'output_tokens':0,'reused':True}
        for case in checks:
            doc = case.get('trace','')
            if isinstance(doc,str) and doc.startswith('trial/'):
                doc = trial.get(doc.split('/',1)[1],'')
            relation,request = case.get('relation'),case.get('request')
            expected_name = case.get('expected')
            if (not isinstance(doc,str) or doc not in self.paths or relation not in ('same_goal','different_goal')
                    or not isinstance(request,str) or not request.strip() or relation=='different_goal' and 'expected' not in case
                    or expected_name is not None and (not isinstance(expected_name,str) or
                    not any('label' in lib['handoffs'].get(expected_name,{}) for lib in (self.lib,candidate)))):
                rows.append({'trace':doc,'status':'invalid case; supply Train trace, complete request, relation and expected changed goal'})
                continue
            sites = self.trace_data(doc)
            idx = case.get('event',next(iter(sites['handoffs']),None))
            if not isinstance(idx,int) or idx not in sites['handoffs'] or request==sites['request']:
                rows.append({'trace':doc,'status':'distinct request and a recorded handoff_state needed'})
                continue
            state = sites['handoffs'][idx]
            row = {'trace':doc,'event':idx,'relation':relation,'request':request,'evaluations':[]}
            for arm in ('parent','candidate'):
                old = evaluate(doc,idx,arm,sites['request'],state)
                new = evaluate(doc,idx,arm,request,state)
                expected = case.get('expected',old['guide'])
                row[arm] = {'original':old['guide'],'rewritten':new['guide'],'expected':expected,
                            'reference':'provided' if 'expected' in case else 'original selection',
                            'agrees':new['guide']==expected if expected is not None or 'expected' in case else None}
                if 'unavailable' in old or 'unavailable' in new:
                    row[arm].update(agrees=None,unavailable=old.get('unavailable') or new.get('unavailable'))
                row['evaluations'].extend([old,new])
            rows.append(row)
        for anchor in anchors:
            doc,idx,state = anchor
            names,_,_ = menu(candidate,state)
            chosen = evaluate(doc,idx,'candidate',self.trace_data(doc)['request'],state) if names else None
            preferred = ([chosen['guide']] if chosen and chosen['guide'] is not None else []) + names
            role = next((r for n in preferred for r in candidate['handoffs'][n].get('requires',[])),None)
            if role:
                reduced = {**state,'values':{k:v for k,v in state['values'].items() if k!=role}}
                obs = evaluate(doc,idx,'candidate',self.trace_data(doc)['request'],reduced,role)
                rows.append({'trace':doc,'event':idx,'relation':'missing_role','removed':role,'eligible':obs['eligible'],
                             'selected':obs['guide'],'evaluations':[chosen,obs],
                             **({'unavailable':obs['unavailable']} if 'unavailable' in obs else {})})
            elif chosen and chosen['calls']:
                rows.append({'trace':doc,'event':idx,'relation':'original','selected':chosen['guide'],'evaluations':[chosen]})
        return rows
