"""Follow actually offered fragments through public observations to accepted effects."""

import json

from .composition import PROGRAM_ERRORS, run_card, validate_output
from .schema_tools import FINISH, Observation, is_read, public_value


def _brief(value, limit=185):
    safe = public_value(value)
    text = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False)
    return safe if len(text) <= limit else text[:limit] + "…"


def _fragment(choice):
    return isinstance(choice, (list, tuple)) and len(choice) >= 2 and choice[0] == "fragment"


def fragment_evidence(events, library):
    task = next((e for e in events if e["event"] == "task"), {})
    definitions = library.get("fragment", {})
    observed, lineage, fragments, writes = {}, {}, {}, []
    comparable = {}
    modern = any(e["event"] == "native_protocol" for e in events)

    def record(name):
        if name not in fragments:
            fragments[name] = {
                "eligible": 0 if modern else None,
                "recommended": 0 if modern else None,
                "offered": 0,
                "chosen": 0,
                "produced": [],
                "accepted_uses": [],
                "bypassed_for": [],
                "skipped_output_checks": [],
            }
        return fragments[name]

    for name in definitions:
        record(name)
    skipped, last = [], None
    for index, e in enumerate(events):
        kind = e["event"]
        if kind == "native_fragment_retrieval":
            for _, name, *refs in e["eligible"]:
                record(name)["eligible"] += 1
            for _, name, *refs in e.get("page", []):
                record(name)["recommended"] += 1
        elif kind == "decision" and e["node"] == "native/action":
            chosen = e["candidates"][e["selected"]]
            skipped = []
            for option in e["candidates"]:
                value = option.get("value")
                if not _fragment(value):
                    continue
                _, name, *refs = value
                entry = record(name)
                entry["offered"] += 1
                if option is chosen:
                    entry["chosen"] += 1
                else:
                    if len(entry["bypassed_for"]) < 6:
                        entry["bypassed_for"].append({"event": index, "choice": chosen["label"][:150]})
                    if refs:
                        skipped.append((name, tuple(refs)))
        elif kind == "native_fragment_choice":
            record(e["name"]).setdefault("internal_choices", []).append(
                {"event": index, **{k: e[k] for k in ("status", "role", "source", "ref", "row", "using", "reason") if k in e}}
            )
        elif kind == "native_observation":
            ref, source = e["ref"], e["source"]
            arguments = e.get("arguments")
            if arguments is None and last and last["tool"] == source and is_read(source):
                arguments = last.get("arguments")
            observed[ref] = Observation(ref, source, e["value"], e["scope"], e["complete"], arguments)
            lineage[ref] = set()
            for upstream in e.get("sources", {}):
                lineage[ref].update(lineage.get(upstream, ()))
            if source in fragments:
                fragments[source]["produced"].append({"event": index, "ref": ref, "count": e["count"]})
                lineage[ref].add((source, ref))
        elif kind == "native_operation":
            last = e
        elif kind == "native_effect" or (
            not modern
            and kind == "api"
            and last
            and last["tool"] == e.get("node")
            and not is_read(last["tool"])
            and not str(e["response"].get("value", "")).startswith("Execution failed.")
        ):
            endpoint = e.get("tool") or last["tool"]
            if not last or last["tool"] != endpoint:
                continue
            # Each effect is emitted only after its individual public call returned successfully.
            args = last["arguments"][e["batch_index"]] if kind == "native_effect" else e["arguments"]
            effect = {"event": index, "tool": endpoint, "submitted": _brief(args, 430)}
            if kind == "native_effect":
                effect["effect"] = _brief(e["effect"], 480)
            writes.append(effect)
            for ref, how in e.get("sources", {}).items():
                for name, produced in sorted(lineage.get(ref, ())):
                    if ref != produced:
                        role = "upstream_of_bound_source"
                    elif how.get("fields"):
                        role = "explicit_argument_fields"
                    elif how.get("scope_only"):
                        role = "row_scope_only"
                    else:
                        role = "input_to_binding_program"
                    fragments[name]["accepted_uses"].append(
                        {
                            "event": index,
                            "tool": endpoint,
                            "output": produced,
                            "via": ref,
                            "role": role,
                            **how,
                        }
                    )
            for name, refs in skipped:
                if name not in definitions or not all(ref in observed for ref in refs):
                    continue
                op = (name, refs)
                if op not in comparable:
                    card = definitions[name]
                    spec = card["composition"]
                    if len(spec["inputs"]) != len(refs):
                        continue
                    loaded = {port: observed[ref] for port, ref in zip(spec["inputs"], refs, strict=True)}
                    state = {"instruction": task.get("instruction", ""), "datetime": task.get("datetime")}
                    state["goal"] = state["instruction"].casefold()
                    try:
                        result = run_card(card, state, loaded)
                        if result is not None and not (isinstance(result, dict) and set(result) == {"$choose"}):
                            validate_output(spec["output"], result, loaded)
                        comparable[op] = result
                    except PROGRAM_ERRORS:
                        comparable[op] = None
                result = comparable[op]
                if isinstance(result, dict) and set(result) == {"$choose"}:
                    checks = fragments[name]["skipped_output_checks"]
                    if len(checks) < 6:
                        checks.append({"event": index, "tool": endpoint, "inputs": refs,
                                       "status": "needs_fragment_choice"})
                    continue
                row = result[0] if isinstance(result, list) and len(result) == 1 else result
                if not isinstance(row, dict):
                    row = {"answer": row} if row is not None and endpoint == FINISH else {}
                common = {
                    k: {"same": row[k] == v, "fragment_value": _brief(row[k]), "submitted": _brief(v)}
                    for k, v in args.items()
                    if k in row and k in public_value(args)
                }
                checks = fragments[name]["skipped_output_checks"]
                if len(checks) < 6 and (common or result is None):
                    checks.append(
                        {
                            "event": index,
                            "tool": endpoint,
                            "inputs": refs,
                            "returned_value": result is not None,
                            "comparable_fields": common,
                        }
                    )
    return {
        "has_explicit_dependency_events": modern,
        "fragments": fragments,
        "accepted_operation_count": len(writes),
        "last_accepted_operations": writes[-5:],
    }
