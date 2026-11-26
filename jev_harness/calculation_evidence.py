"""Retrieve earlier Train computations for a current native-tool editing round."""

import ast
from collections import defaultdict

from .formal_optimizer import cleaned_text, read_events
from .operation_evidence import batch_operations, operations
from .schema_tools import is_read


def field_paths(value, depth=3):
    if not depth:
        return set()
    if isinstance(value, list):
        return set().union(*(field_paths(row, depth - 1) for row in value[:3]))
    if isinstance(value, dict):
        paths = set(value)
        for name, child in value.items():
            paths.update(f"{name}.{path}" for path in field_paths(child, depth - 1))
        return paths
    return set()


def trace_observations(events):
    observations = {}
    for index, event in enumerate(events):
        if event["event"] != "native_observation" or not event.get("complete"):
            continue
        observations[event["ref"]] = {
            "event": index, "ref": event["ref"], "source": event["source"],
            "fields": sorted(field_paths(event["value"], depth=5)), "rows": event["count"],
            "arguments": sorted({key for args in (event.get("arguments") or []) for key in args}),
            "input_refs": list(event.get("sources", {})),
        }
    return observations


def read_inputs(observations):
    return [item for item in observations.values()
            if "__" in item["source"] and is_read(item["source"])]


def program_operations(program):
    try:
        return sorted({f"{kind}/{variant}" for kind, variant, _ in operations(ast.parse(program))})
    except (SyntaxError, ValueError, RecursionError):
        return []


def generated_operations(events):
    groups, _ = batch_operations({"trace": events})
    sites = []
    for group in groups:
        if group["operation"] == "API":
            continue
        for site in group["sites"]["trace"]:
            sites.append({"event": site["event"], "source": group["api"],
                          "operation": group["operation"] + "/" + site["variant"],
                          "line": site["line"], "code": site["code"],
                          "cell_reported_error": site["cell_reported_error"],
                          "observed_actions": site["observed_actions"][:4]})
    return sorted(sites, key=lambda row: (row["event"], row["line"], row["source"]))


def recorded_calculations(events, observations, library):
    consumers = defaultdict(dict)
    for index, event in enumerate(events):
        if event["event"] == "native_effect":
            for ref in event.get("sources", {}):
                consumers[ref].setdefault(event["tool"], index)
    for result in observations.values():
        spec = library.get("fragment", {}).get(result["source"])
        if (spec is None and result["source"] != "calculation") or not consumers[result["ref"]]:
            continue
        inputs = {ref: observations[ref] for ref in result["input_refs"] if ref in observations}
        if not (inputs := read_inputs(inputs)):
            continue
        program = (spec or {}).get("program", "")
        yield {
            "inputs": inputs, "output": result,
            "operations": program_operations(program),
            "program": cleaned_text(program)[:1200],
            "accepted_consumers": [{"tool": tool, "event": index}
                                   for tool, index in consumers[result["ref"]].items()],
            "evidence": "recorded output used by an accepted operation",
        }


def input_signals(inputs):
    return {row["source"] for row in inputs}, {
        (row["source"].split("__", 1)[0], field) for row in inputs for field in row["fields"]
    }


def scenario(task):
    return task.rsplit("_", 1)[0]


def related_calculations(samples, train):
    history = []
    for doc, sample in samples.items():
        if not doc.startswith("history") or not sample.row["success"]:
            continue
        events = read_events(sample.trace)
        observations = trace_observations(events)
        computations = list(recorded_calculations(events, observations, sample.library))
        if not computations:
            # Source inspection is available on successful generated paths as well. It does not
            # establish which branch produced the accepted value, so keep it distinct from receipts.
            generated = [row for row in generated_operations(events)
                         if row["observed_actions"] and not row["cell_reported_error"]]
            if generated and (inputs := read_inputs(observations)):
                computations.append({"inputs": inputs, "operations": sorted({r["operation"] for r in generated}),
                                     "generated_sites": generated[:4], "evidence": "generated code and observed actions"})
        history.extend((doc, sample.row["task_id"], events[0]["instruction"], item) for item in computations)

    reports = {}
    for task in train:
        doc = "h0_decision/" + task
        sample = samples[doc]
        if sample.row["success"] and not sample.row["open_calls"]:
            continue
        events = read_events(sample.trace)
        observations = trace_observations(events)
        reads = read_inputs(observations)
        generated = generated_operations(events)
        sites = [{"event": index, "reason": event.get("reason"), "endpoint": event.get("endpoint")}
                 for index, event in enumerate(events) if event["event"] in {"native_residual", "native_feedback"}]
        required = sorted({name for event in events if event["event"] == "native_residual"
                           for name in event.get("remaining", [])})
        current_sources, current_fields = input_signals(reads)
        current_ops = {row["operation"] for row in generated}
        related = []
        for old_doc, old_task, goal, prior in history:
            if scenario(task) == scenario(old_task):
                continue
            known_sources, known_fields = input_signals(prior["inputs"])
            endpoints = current_sources & known_sources
            fields = current_fields & known_fields
            common_ops = current_ops & set(prior["operations"])
            if not (endpoints or fields):
                continue
            common_consumers = set(required) & {c["tool"] for c in prior.get("accepted_consumers", [])}
            match = (len(common_ops), len(endpoints), len(fields), len(common_consumers),
                     "accepted_consumers" in prior)
            related.append((match, old_doc, old_task, {"doc": old_doc, "instruction": goal, **prior,
                            "matched_sources": sorted(endpoints), "matched_operations": sorted(common_ops),
                            "matched_fields": [f"{app}/{field}" for app, field in sorted(fields)[:10]]}))
        related.sort(key=lambda row: (row[0], row[1]), reverse=True)
        examples = {}
        for _, _, old_task, example in related:
            examples.setdefault(scenario(old_task), example)
            if len(examples) == 2:
                break
        rank = (bool(related), not sample.row["success"], bool(sites), sample.row.get("open_output_tokens", 0))
        if scenario(task) not in reports or rank > reports[scenario(task)][0]:
            reports[scenario(task)] = (rank, {"doc": doc, "instruction": events[0]["instruction"],
                                               "current_reads": reads[:5], "unresolved_steps": sites[:4],
                                               "remaining_operations": required, "generated_sites": generated[:4],
                                               "other_train_scenarios": list(examples.values())})
    return [report for _, report in reports.values()]
