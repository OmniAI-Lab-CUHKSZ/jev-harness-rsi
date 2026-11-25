"""The same program compiles live and archived public states into handoff choices."""
from .formal_runtime import ContextTooLong, bounded
from .fragment_assembly import FragmentAssembly, option
from .runtime import MAX_REAL_OPTIONS, Uncovered, abstained
from .retrieval import ranked, terms


QUESTION = 'Which specialized guidance fits both the requested outcome and the data already available?'
SKILL = ('Read the entire request for the intended effect, actor, recipient and direction. Use the live data '
         'as evidence. Distinguish acting from reporting, an action from its reverse, and completed work '
         'from work still requested. Select ordinary assistance when no specialized guide fits.')


def snapshot(executor):
    state = {'values':{name:bounded(value, 1200) for name,value in executor.public.items() if value is not None},
            'completed':list(executor.completed_cards), 'failed':executor.failed_card,
            'remaining_model_calls':max(0,executor.max_calls-executor.requests)}
    if executor.registry.get("menu_format") == "scoped":
        state["producers"] = {role:info["producer"] for role,info in executor.provenance.items() if role in state["values"]}
    return state


def guide_groups(library, state):
    guides = {name:rule for name,rule in library['handoffs'].items() if 'label' in rule and
              all(role in state['values'] for role in rule.get('requires', []))}
    if not library.get("menu_retrieval"):
        return [list(guides)]
    query = terms(state.get("request", ""))
    scores = {n: 3 * len(query & terms(n + " " + g["label"])) + len(query & terms(" ".join(g.get("requires", []))))
              for n,g in guides.items()}
    first = ranked(guides, scores, 3)
    remaining = [n for n in guides if n not in first]
    # The latest producer provides evidence for a paraphrase. Stale, unrelated roles do not.
    completed = state.get("completed") or []
    recent_roles = {role for role,writer in state.get("producers",{}).items() if completed and writer==completed[-1]}
    for n in remaining:
        scores[n] += 2 * len(recent_roles.intersection(guides[n].get("requires",[])))
    if not first and len(guides)==1 and not recent_roles:
        for n in remaining:
            scores[n] += bool(guides[n].get("requires"))
    second = ranked(remaining, scores, 3)
    return [group for group in (first, second) if group] or [[]]


def menu(library, state, group=None):
    if group is None:
        group = guide_groups(library, state)[0]
    choices = []
    for name in group:
        rule = library["handoffs"][name]
        choices.append(option(rule['label'],name))
        for contrast in rule.get('alternatives', []):
            choices.append(option(contrast,{'ordinary_for':name,'label':contrast}))
    roles = dict.fromkeys(role for n in group for role in library["handoffs"][n].get("requires",[]))
    evidence = {role:state['values'][role] for role in roles}
    return list(group), choices, bounded(evidence, 12000)


def select_guidance(executor, state):
    state = {**state, "request":executor.request}
    first = len(executor.trace.events)
    selected = None
    reason = 'no eligible guide'
    # Each full page can contribute a winner to the next pass. Preserve a call for G even then.
    names = []
    for i, group in enumerate(guide_groups(executor.registry, state)):
        _, choices, facts = menu(executor.registry, state, group)
        if not choices:
            break
        names.extend(group)
        size, needed = len(choices), 1
        while size > MAX_REAL_OPTIONS:
            size = (size + MAX_REAL_OPTIONS - 1) // MAX_REAL_OPTIONS
            needed += size
        if executor.requests + needed >= executor.max_calls:
            reason = 'last call reserved for open generation'
            break
        if executor.registry.get("menu_retrieval"):
            executor.trace.add("retrieval_menu", node="handoff", batch=i+1, offered=group)
        try:
            # Uses the same native head and paging as the action menu. No thinking/generation on this node.
            selected = FragmentAssembly.choose(executor,'handoff',choices,available_data=facts)
            reason = 'guide' if isinstance(selected,str) else 'ordinary alternative'
            break
        except (Uncovered,ContextTooLong) as exc:
            if abstained(exc):
                reason = 'native NONE'
                continue
            reason = str(exc)
            break
    guide = selected if isinstance(selected,str) else None
    calls = [e['response'] for e in executor.trace.events[first:] if e['event']=='model' and e['operation']=='decide']
    executor.trace.add('handoff_selection', eligible=names, choice=selected, guide=guide, reason=reason, calls=len(calls),
                       input_tokens=sum(r.get('input_tokens',0) for r in calls),
                       output_tokens=sum(r.get('output_tokens',0) for r in calls),
                       model_seconds=sum(r.get('seconds',0) for r in calls))
    return guide
