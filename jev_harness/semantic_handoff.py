"""The same program compiles live and archived public states into handoff choices."""
from .formal_runtime import ContextTooLong, bounded
from .fragment_assembly import FragmentAssembly, option
from .runtime import MAX_REAL_OPTIONS, NativeNone, Uncovered


QUESTION = 'Which specialized guidance fits both the requested outcome and the data already available?'
SKILL = ('Read the entire request for the intended effect, actor, recipient and direction. Use the live data '
         'as evidence. Distinguish acting from reporting, an action from its reverse, and completed work '
         'from work still requested. Select ordinary assistance when no specialized guide fits.')


def snapshot(executor):
    return {'values':{name:bounded(value, 1200) for name,value in executor.public.items() if value is not None},
            'completed':list(executor.completed_cards), 'failed':executor.failed_card,
            'remaining_model_calls':max(0,executor.max_calls-executor.requests)}


def menu(library, state):
    guides = {name:rule for name,rule in library['handoffs'].items() if 'label' in rule and
              all(role in state['values'] for role in rule.get('requires', []))}
    choices = []
    for name, rule in guides.items():
        choices.append(option(rule['label'],name))
        for contrast in rule.get('alternatives', []):
            choices.append(option(contrast,{'ordinary_for':name,'label':contrast}))
    roles = dict.fromkeys(role for rule in guides.values() for role in rule.get('requires',[]))
    evidence = {role:state['values'][role] for role in roles}
    return list(guides), choices, bounded(evidence, 12000)


def select_guidance(executor, state):
    names, choices, facts = menu(executor.registry, state)
    first = len(executor.trace.events)
    selected = None
    reason = 'no eligible guide'
    # Each full page can contribute a winner to the next pass. Preserve a call for G even then.
    size = len(choices)
    needed = int(size > 0)
    while size > MAX_REAL_OPTIONS:
        size = (size + MAX_REAL_OPTIONS - 1) // MAX_REAL_OPTIONS
        needed += size
    if choices and executor.requests + needed >= executor.max_calls:
        reason = 'last call reserved for open generation'
    elif choices:
        try:
            # Uses the same native head and paging as the action menu. No thinking/generation on this node.
            selected = FragmentAssembly.choose(executor,'handoff',choices,available_data=facts)
            reason = 'guide' if isinstance(selected,str) else 'ordinary alternative'
        except NativeNone:
            reason = 'native NONE'
        except (Uncovered,ContextTooLong) as exc:
            reason = str(exc)
    guide = selected if isinstance(selected,str) else None
    calls = [e['response'] for e in executor.trace.events[first:] if e['event']=='model' and e['operation']=='decide']
    executor.trace.add('handoff_selection', eligible=names, choice=selected, guide=guide, reason=reason, calls=len(calls),
                       input_tokens=sum(r.get('input_tokens',0) for r in calls),
                       output_tokens=sum(r.get('output_tokens',0) for r in calls),
                       model_seconds=sum(r.get('seconds',0) for r in calls))
    return guide
