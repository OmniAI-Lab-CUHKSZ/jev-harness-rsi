"""Local effects of a patch at observed Train states. No task or API is replayed."""
import re

from .expressions import parameter_groups
from .semantic_handoff import guide_groups, menu
from .scoped_menu import consumer_tips, is_scoped


def trace_sites(events):
    sites = {'request':events[0].get('instruction','') if events else '', 'handoffs':{}, 'parameters':[],
             'cards':{}, 'offered':set(), 'apis':set(), 'generated_apis':set(), 'nodes':set(), 'families':set()}
    current = None
    writers = {}
    for i,event in enumerate(events):
        kind = event['event']
        if kind == 'handoff_state':
            sites['handoffs'][i] = {**event,'request':sites['request']}
            sites['handoffs'][i].setdefault('producers',{k:v for k,v in writers.items() if k in event['values']})
        elif kind == 'handoff_selection' and sites['handoffs']:
            sites['handoffs'][next(reversed(sites['handoffs']))]['observed_guide'] = event.get('guide')
        elif kind == 'parameter_state':
            current = {**event,'index':i,'bound':{}}
            sites['parameters'].append(current)
        elif kind == 'parameter_bindings' and current and event['card']==current['card']:
            current['bound'] = event['bound']
        elif kind in ('card_executed','card_failed','fragment_reentry') and event.get('card'):
            sites['cards'].setdefault(event['card'],set()).add(kind)
            role = event.get('output') or event.get('role')
            if role and kind!='card_failed':
                writers[role] = event['card']
        elif kind=="result_source":
            writers[event["role"]] = event["producer"]
        elif kind == 'card_menu':
            sites['offered'].update(event['offered'])
        elif kind == 'assembly_candidates':
            sites['families'].update(event['families'])
        elif kind == 'decision':
            sites['nodes'].add(event['node'])
        elif kind == 'api':
            sites['apis'].add(event['app']+'__'+event['api'])
        elif kind == 'tool_batch' and event.get('node')=='generation_agent':
            sites['generated_apis'].update(app+'__'+api for app,api in re.findall(
                r'\bapis\.(\w+)\.(\w+)\s*\(',event.get('code','')))
    return sites


def parameter_views(card, state, request, retrieve=False):
    views = {}
    ctx = {'state':state['values'],'task':{'instruction':request,'datetime':state.get('datetime')},'params':{}}
    # A trial may add inputs, or the original trace may have omitted a large public object.
    absent = set(card.get('inputs',[])) - state['values'].keys()
    uncaptured = set(state.get('available',[])) - state['values'].keys()
    def needs_uncaptured(value):
        if isinstance(value,list):
            return any(needs_uncaptured(item) for item in value)
        if isinstance(value,dict):
            ref = value.get('ref')
            if isinstance(ref,str) and (ref == 'state' or ref.startswith('state.') and ref.split('.')[1] in uncaptured):
                return bool(uncaptured)
            return any(needs_uncaptured(v) for v in value.values())
        return False
    for key,slot in card.get('parameters',{}).items():
        if absent or needs_uncaptured(slot['candidates']) or needs_uncaptured(slot.get('prefer')):
            views[key] = {'unavailable':'Required public inputs were absent or not recorded exactly'}
            break
        try:
            groups = parameter_groups(slot,ctx,retrieve=retrieve)
            choices = [choice for group in groups for choice in group]
        except (KeyError,TypeError,ValueError,IndexError,ArithmeticError,re.error) as exc:
            views[key] = {'unavailable':str(exc)[:180]}
            break
        views[key] = {'choices':choices,'question':slot['question'],'skill':card['skill']+' '+slot.get('skill','')}
        if 'prefer' in slot or retrieve:
            views[key]['menu_sizes'] = [len(group) for group in groups]
        if key not in state['bound'] or not any(c['value']==state['bound'][key] for c in choices):
            break
        ctx['params'][key] = state['bound'][key]
    return views


def parameter_changes(parent, candidate, state, request, retrieval_modes=(False,False)):
    views = [parameter_views(card,state,request,mode) for card,mode in zip((parent,candidate),retrieval_modes)]
    results = []
    for slot in sorted(set(views[0]) | set(views[1])):
        before,after = [v.get(slot,{}) for v in views]
        if 'choices' not in before or 'choices' not in after:
            results.append({'slot':slot,'unavailable_or_absent':[v.get('unavailable') or ('present' if v else 'absent') for v in (before,after)]})
            continue
        left,right = before['choices'],after['choices']
        results.append({'slot':slot,'counts':[len(left),len(right)],'same_ordered_candidates':left==right,
                        'question_changed':before['question']!=after['question'],'skill_changed':before['skill']!=after['skill'],
                        'bound_value_still_offered':(any(x['value']==state['bound'][slot] for x in right)
                                                     if slot in state['bound'] else None)})
        if 'menu_sizes' in before or 'menu_sizes' in after:
            results[-1]['menu_sizes'] = [v.get('menu_sizes',[len(v['choices'])]) for v in (before,after)]
    return results


def handoff_view(lib, state):
    names,choices,evidence = menu(lib,state)
    static = list(dict.fromkeys(state['completed'] + ([state['failed']] if state.get('failed') else [])))
    lookup = {}
    if is_scoped(lib):
        writers = state.get("producers",{})
        tips = consumer_tips(lib,state["values"],{role:{"producer":card} for role,card in writers.items()})
        text = {writers[role]:value for role,value in list(tips.items())[-1:]}
        if len(tips)>1:
            lookup = tips
    else:
        text = {n:lib['cards'][n]['handoff_skill'] for n in static if lib['cards'].get(n,{}).get('handoff_skill')}
    default = lib['handoffs'].get('default',{}).get('skill','')
    if default:
        text['handoff/default'] = default
    return names,choices,evidence,text,lookup


def compare_handoff(parent, candidate, state):
    old, new = (handoff_view(lib,state) for lib in (parent,candidate))
    changed_text = {k:[len(old[3].get(k,'')),len(new[3].get(k,''))] for k in old[3].keys()|new[3].keys()
                    if old[3].get(k)!=new[3].get(k)}
    batches = [[menu(lib,state,g) for g in guide_groups(lib,state)] for lib in (parent,candidate)]
    names = [[name for group,_,_ in pages for name in group] for pages in batches]
    common = set(names[0]).intersection(names[1])
    prompt_changed = any(names) and any(parent[k].get('handoff')!=candidate[k].get('handoff')
                                        for k in ('questions','skills'))
    return {'eligible_before':names[0], 'eligible_after':names[1], 'menu_changed':batches[0]!=batches[1],
            'model_evidence_changed':old[2]!=new[2], 'selector_prompt_changed':prompt_changed,
            'changed_conditional_guidance':sorted(n for n in common if parent['handoffs'][n]['skill']!=candidate['handoffs'][n]['skill']),
            'changed_lookup_roles':sorted(k for k in old[4].keys()|new[4].keys() if old[4].get(k)!=new[4].get(k)),
            'changed_direct_guidance_lengths':changed_text,
            'changed_direct_guidance':{k:{'before':old[3].get(k,'')[:400],'after':new[3].get(k,'')[:400]}
                                       for k in changed_text}}


def assess(sites, parent, candidate, edits):
    hits, bindings, handoffs = {}, [], []
    for obj in edits:
        kind,name = obj['kind'],obj['name']
        reasons = []
        if kind=='card':
            reasons.extend(sorted(sites['cards'].get(name,())))
            if name in sites['offered']:
                reasons.append('offered')
            old, new = (lib['cards'].get(name,{}) for lib in (parent,candidate))
            if any(c.get('api') in sites['generated_apis'] for c in (old,new)):
                reasons.append('same_api_in_generated_code')
            if any(c.get('api') in sites['apis'] for c in (old,new)):
                reasons.append('same_public_api_called')
            for st in sites['parameters']:
                if st['card']==name:
                    modes = [bool(lib.get("menu_retrieval")) for lib in (parent,candidate)]
                    bindings.append({'event':st['index'],'card':name,'slots':parameter_changes(old,new,st,sites['request'],modes)})
            if not old and any(set(new.get('inputs',[])).issubset(s['values']) for s in sites['handoffs'].values()):
                reasons.append('inputs_available_at_recorded_handoff')
        elif kind=='adapter':
            if any(api==name or api.startswith(name+'__') for api in sites['apis']):
                reasons.append('adapter_endpoint_in_trace')
        elif kind=='decision' and ('fragment/'+name in sites['nodes'] or name=='handoff' and sites['handoffs']):
            reasons.append('decision_site')
        elif kind=='assembly' and sites['families']:
            reasons.append('family_in_menu' if name in sites['families'] else 'assembly_ran')
        elif kind in ('state','handoff'):
            role_key = 'views' if kind=='state' else 'handoffs'
            rules = [lib[role_key][name] for lib in (parent,candidate) if name in lib[role_key]]
            if any(rule.get('at')=='parameter' and set(rule.get('requires',[])).issubset(s.get('available',s['values']))
                   for rule in rules for s in sites['parameters']):
                reasons.append('parameter_projection_site')
            if any(rule.get('at')!='parameter' and all(k in s['values'] for k in rule.get('requires',[]))
                   for rule in rules for s in sites['handoffs'].values()):
                reasons.append('required_roles_present')
        if reasons:
            hits[kind+'/'+name] = reasons
    for i,st in sites['handoffs'].items():
        diff = compare_handoff(parent,candidate,st)
        if any(diff[k] for k in ('menu_changed','model_evidence_changed','selector_prompt_changed','changed_conditional_guidance','changed_direct_guidance_lengths','changed_lookup_roles')):
            handoffs.append({'event':i,**diff})
    return {'exposures':hits,'parameter_checks':bindings,'handoff_changes':handoffs}


def historical_report(archive, parent, candidate, edits):
    details = []
    tasks = set()
    matched = set()
    for doc,sites in archive.items():
        task = doc.rsplit('/',1)[-1]
        tasks.add(task)
        effects = assess(sites,parent,candidate,edits)
        if any(effects.values()):
            matched.add(task)
            details.append({'task':task,'trace':doc,'request':sites['request'],**effects})
    # Return distinct representative tasks to Sol; all matched versions are available in the document.
    examples = {}
    for row in sorted(details,key=lambda x:not (x['handoff_changes'] or x['parameter_checks'])):
        examples.setdefault(row['task'],row)
    compact = [{**row,'request':row['request'][:450],'handoff_changes':row['handoff_changes'][:2],'parameter_checks':row['parameter_checks'][:2]}
               for row in list(examples.values())[:3]]
    return {'historical_tasks':len(tasks),'trace_entries':len(archive),'matched_tasks':len(matched),
            'tasks_with_handoff_changes':len({r['task'] for r in details if r['handoff_changes']}),
            'tasks_with_parameter_checks':len({r['task'] for r in details if r['parameter_checks']}),
            'examples':compact, 'whole_task_outcome':'unmeasured'},details
