"""Retrieve small menus from public task wording, live data and installed operation interfaces."""
from collections import Counter, defaultdict
from functools import lru_cache
import re

from .card_library import ports


MODE = "state"
FIRST = 5
SECOND = 7
STOP = set("""a an the and or of to in on by at as is are be for from with my me i all this that
              it which what who whose how please user current available account task requested request
              read get show list use data output record records public id ids operation step prepare""".split())


@lru_cache(maxsize=4096)
def terms(text):
    words = re.findall(r"[^\W_]+", text.casefold())
    def singular(word):
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            return word[:-1]
        return word
    return frozenset(singular(w) for w in words if w not in STOP)


def ranked(names, scores, limit):
    # Ties retain library order. An unrelated addition cannot change an existing item's score.
    return sorted((n for n in names if scores[n] > 0), key=scores.get, reverse=True)[:limit]


def card_scores(cards, request, live, apps):
    """Lexical goal matches and typed edges. A different app can supply a missing input."""
    query = terms(request)
    scores, providers = {}, defaultdict(list)
    for name, card in cards.items():
        app, inputs, output, _ = ports(card)
        description = " ".join(str(card.get(k, "")) for k in ("description", "applies_to", "scope", "api"))
        intent = len(query & terms(description + " " + name + " " + output))
        score = 3 * intent + (2 if app in apps else 0)
        if inputs and all(role in live for role in inputs):
            score += 3
            if output == "submitted":
                score += 6
        scores[name] = score
        providers[output].append(name)
    # Propagate through missing typed inputs only. Terminal answer/effect ports are shared across apps.
    pending = sorted(scores, key=scores.get)
    while pending:
        target = pending.pop()
        for role in ports(cards[target])[1]:
            if role in live or role in {"answer", "effect_receipt"}:
                continue
            for name in providers[role]:
                if scores[target] - 2 > scores[name]:
                    scores[name] = scores[target] - 2
                    pending.append(name)
    return scores


def family_groups(menus, scores, request, apps):
    query = terms(request)
    relevance = {name: max(scores[n] for n in members) + 3 * len(query & terms(name + " " + label))
                 + (3 if name in apps else 0) for name, (label, members) in menus.items()}
    first = ranked(menus, relevance, FIRST)
    if not first:
        return [list(menus)]
    # A dependency can put the same provider in multiple categories. Prefer the explicit scope initially.
    direct = [n for n in first if n in apps or query & terms(n + " " + menus[n][0])]
    first = direct or first
    other = ranked([n for n in menus if n not in first], relevance, FIRST)
    return [g for g in (first, other) if g]


def step_groups(focused, eligible, scores, cards, live, request):
    first = ranked(focused, scores, FIRST)
    if not first:
        first = list(focused) if len(focused) <= FIRST else ranked(eligible, scores, FIRST)
    if not first:
        return []
    # Reconsider consumers against the data that really exists, including cross-app consumers.
    query = terms(request)
    documents = {n:terms(n + " " + cards[n]["description"] + " " + ports(cards[n])[2]) for n in eligible}
    frequency = Counter(w for words in documents.values() for w in words)
    residual = {n: scores[n] + 3 * sum(role in live for role in ports(cards[n])[1]) +
                4 * sum(frequency[w] == 1 for w in query & documents[n]) for n in eligible}
    wider = ranked(eligible, residual, SECOND)
    if wider:
        # Re-query specifically for the missing outcome; keep close competing interpretations.
        cutoff = residual[wider[0]] * 0.65
        wider = [n for n in wider if residual[n] >= cutoff]
    alternatives = [n for n in wider if n not in first]
    consumers = [n for n in wider if ports(cards[n])[1]]
    if alternatives or consumers:
        return [first, wider]
    return [first]


def continuation_state(names, cards, public, state):
    """Render the data a retrieved operation can use now. A stale result is not shown as completed work."""
    inputs = {}
    for name in names:
        for role in ports(cards[name])[1]:
            value = public[role]
            if isinstance(value, list):
                inputs[role] = {"count":len(value)}
                if len(value) <= 3 and all(isinstance(v,(int,float,bool)) for v in value):
                    inputs[role]["values"] = value
            else:
                inputs[role] = state["available"].get(role, "available")
    facts = {"current_inputs":inputs, "scope":state.get("scope"),
             "applied_conditions":state.get("applied_conditions", {}),
             "next_results":{name:ports(cards[name])[2] for name in names}}
    if state.get("views"):
        facts["views"] = state["views"]
    sources = {k:v for k,v in state.get("sources",{}).items() if k in inputs}
    if sources:
        facts["sources"] = sources
    return facts


def row_text(choice):
    value = choice["value"]
    if isinstance(value, dict):
        fields = [str(v) for k,v in value.items() if isinstance(v, (str, int)) and not k.endswith("_id")]
        if fields:
            return " ".join(fields)
    return str(choice["label"])


def record_groups(choices, request, *, allow_unranked=True):
    """First show discriminating record matches. Keep ties together and preserve real option values."""
    if not choices:
        return []
    words = [terms(row_text(row)) | terms(str(row["label"])) for row in choices]
    frequency = Counter(w for doc in words for w in doc)
    query = terms(request)
    scores = [sum(2 * len(choices) - frequency[w] for w in query & doc
                  if frequency[w] < len(choices)) for doc in words]
    ordered = sorted((i for i, score in enumerate(scores) if score), key=scores.__getitem__, reverse=True)
    if not ordered:
        return [choices] if allow_unranked else []
    def take_ties(indices, size):
        if len(indices) <= size:
            return indices
        return [i for i in indices if scores[i] >= scores[indices[size-1]]]
    first = take_ties(ordered, FIRST)
    second = take_ties(ordered[len(first):], SECOND)
    return [[choices[i] for i in group] for group in (first, second) if group]


def refine_records(choices, request):
    """A refusal can concern a correct record. Show the terms that distinguish the strongest matches."""
    query = terms(request)
    matches = [query & terms(row_text(row)) for row in choices]
    if not matches:
        return []
    common = frozenset.intersection(*matches)
    distinct = [m - common for m in matches]
    best = max(map(len, distinct))
    if not best:
        return []
    return [{**row, "label":str(row["label"]) + " | Distinguishing request terms: " + ", ".join(sorted(words))}
            for row,words in zip(choices,distinct) if len(words) == best]
