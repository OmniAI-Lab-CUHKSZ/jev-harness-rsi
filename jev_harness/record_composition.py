"""Connect a contact-based filter to AppWorld record sources and consumers."""

from copy import deepcopy
from itertools import product
import re

from .policy_programs import calendar_windows, load_program
from .runtime import Uncovered
from .schema_tools import key


CONTACTS = "phone__search_contacts"
PAYMENTS = "venmo__show_transactions"
REQUESTS = "venmo__show_received_payment_requests"
CONTACT_ROLES = {
    "coworker": r"co[- ]?workers?|colleagues?|workmates?",
    "friend": r"friends?",
    "roommate": r"roommates?|housemates?",
    "manager": r"managers?",
    "sibling": r"siblings?",
    "parent": r"parents?",
    "brother": r"brothers?",
    "sister": r"sisters?",
    "father": r"fathers?",
    "mother": r"mothers?",
}
PAYMENT_WRITES = {
    "venmo__like_transaction": {"fields": {"transaction_id": "transaction_id"}},
    "venmo__create_transaction_comment": {
        "fields": {"transaction_id": "transaction_id"},
        "arguments": (
            """texts = captures(r'["“]([^"”]+)["”]', state["instruction"])\n"""
            'return {"comment": texts[0]} if len(texts) == 1 else None'
        ),
    },
}
REQUEST_WRITES = {
    endpoint: {"fields": {"payment_request_id": "payment_request_id"}}
    for endpoint in ("venmo__approve_payment_request", "venmo__deny_payment_request")
}
FILTER_INPUTS = {"rows": "counterparty_records", "people": "contact_identities"}


def validate_record_composition(spec):
    composition = spec["composition"]
    if (spec.keys() - {"label", "program", "composition", "goal_terms", "sources"}
            or spec.get("sources") or not isinstance(composition, dict)
            or composition.keys() - {"inputs", "output", "equivalent"}
            or composition.get("inputs") != FILTER_INPUTS or composition.get("output") != "matching_records"):
        raise ValueError("A record composition needs endpoint-independent roles and matching_records output")
    equivalent = composition.get("equivalent", [])
    if not isinstance(equivalent, list) or not all(isinstance(name, str) and name.isidentifier() for name in equivalent):
        raise ValueError("Name any equivalent filter by its fragment identifier")


def payment_directions(goal, tool):
    if not tool:
        return []
    directions = set(tool["parameters"]["properties"].get("direction", {}).get("enum", ()))
    sent = re.search(r"\b(?:sent|from me|to (?:and from )?my)\b", goal)
    received = re.search(r"\b(?:received|paid me|from my)\b", goal)
    if sent and received:
        wanted = [("sent", "received")]
    elif sent:
        wanted = [("sent",)]
    elif received:
        wanted = [("received",)]
    else:
        wanted = [("sent",), ("received",), ("sent", "received")]
    return [selection for selection in wanted if set(selection) <= directions]


def record_composition(engine, spec):
    """Build full read plans for the requested scope and at least one compatible use."""
    goal = engine.goal
    if (CONTACTS not in engine.tools or not re.search(r"\bvenmo\b", goal)
            or re.search(r"\b(?:social feed|simple note|except|unless|excluding)\b", goal)):
        return None, []
    roles = sorted(role for role, pattern in CONTACT_ROLES.items() if re.search(rf"\b(?:{pattern})\b", goal))
    try:
        windows = calendar_windows(goal, engine.context["task"]["datetime"])
    except (TypeError, ValueError, OverflowError):
        return None, []
    if not roles or not windows:
        return None, []
    pending = bool(re.search(r"\bpayment requests?\b", goal))
    requested = set(engine.required)
    if re.search(r"(?:^|[.!?]\s*|\band\s+)(?:please\s+)?like\s+(?:all\s+)?(?:the\s+)?venmo\s+transactions?\b",
                 goal):
        requested.add("venmo__like_transaction")
    scope = "incoming pending requests" if pending else "Venmo transactions"
    source = REQUESTS if pending else PAYMENTS
    if source not in engine.tools:
        return None, []
    choices = [("pending",)] if pending else payment_directions(goal, engine.tools.get(source))
    offered_writes = REQUEST_WRITES if pending else PAYMENT_WRITES
    writes = {name: rule for name, rule in offered_writes.items() if name in requested and name in engine.tools}
    calculations = {}
    if engine.answer_question and not requested:
        if not pending and re.search(r"\bhow much\b", goal):
            calculations["sum"] = "amount"
        elif re.search(r"\bhow many\s+(?:venmo\s+)?(?:transactions?|payments?)\b", goal):
            calculations["count"] = True
    if requested - writes.keys() or not (requested - engine.chosen_effects or calculations):
        return None, []
    fields = {"payment_request_id" if pending else "transaction_id": "integer"}
    if "sum" in calculations:
        fields["amount"] = "number"
    slots = {
        "records": {"question": "Which official records cover the requested direction?",
                    "skill": "Match the direction to the user's words. Both directions is the union."},
        "contacts": {"question": "Which contact relationships define the requested set?"},
        "date": {"question": "Which date interval, inclusive, matches the user's request?"},
    }
    result = {"fields": fields, "calculations": calculations, "writes": writes,
              "equivalent": spec["composition"].get("equivalent", [])}
    prepared = {**spec, "sources": [CONTACTS, source],
                "dynamic": {"slots": slots, "results": {scope: result}}}
    plans = []
    for directions, (start, end) in product(choices, windows):
        reads = {"people": {"source": CONTACTS, "arguments": [{"relationship": role} for role in roles]}}
        actors = {}
        for direction in directions:
            arguments = {"status": "pending"} if pending else {"direction": direction}
            if start and "min_created_at" in engine.tools[source]["parameters"]["properties"]:
                arguments["min_created_at"] = start
            reads[direction] = {"source": source, "arguments": [arguments]}
            actors[direction] = "receiver" if direction == "sent" else "sender"
        labels = {"records": scope if pending else " and ".join(directions) + " transactions",
                  "contacts": ", ".join(roles)}
        if start:
            labels["date"] = f"{start} through {end} inclusive"
        plans.append({"result": scope, "reads": reads, "choices": labels, "requested": sorted(requested),
                      "actors": actors, "window": [start, end]})
    return prepared, plans


def select_record_subset(program, plan, inputs, id_field):
    """Preserve the official rows and reject records outside the selected read scope."""
    people = inputs["people"]
    if len(plan["reads"]["people"]["arguments"]) > 1:
        if not isinstance(people, list) or any(not isinstance(group, list) for group in people):
            raise Uncovered("Incomplete contact groups")
        people = [person for group in people for person in group]
    if not isinstance(people, list):
        raise Uncovered("The selected contacts are not a record collection")
    start, end = plan["window"]
    selected = {}
    filter_rows = load_program(program)
    for port, actor in plan["actors"].items():
        rows = inputs[port]
        if not isinstance(rows, list) or any(not isinstance(row, dict) or id_field not in row for row in rows):
            raise Uncovered("The selected official read did not return the expected record type")
        if start:
            if any(not isinstance(row.get("created_at"), str) for row in rows):
                raise Uncovered("The selected records do not support the requested date range")
            rows = [row for row in rows if start <= row["created_at"][:10] <= end]
        originals = {key(row): row for row in rows}
        try:
            result = filter_rows({"rows": deepcopy(rows), "people": deepcopy(people),
                                  "record_key": [actor, "email"], "people_key": ["email"]})
        except (ArithmeticError, AttributeError, IndexError, KeyError, NameError, TypeError, ValueError) as exc:
            raise Uncovered(f"Cannot apply the filter to the selected records: {exc}") from exc
        if not isinstance(result, list) or any(not isinstance(row, dict) or key(row) not in originals for row in result):
            raise Uncovered("The filter must return a subset of the selected official records")
        for row in result:
            selected[row[id_field]] = originals[key(row)]
    return list(selected.values())
