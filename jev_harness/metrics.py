"""Task correctness first; actual executor output tokens break correctness ties.

Do not name this module evaluation.py: AppWorld imports task modules by that name.
"""
from collections import Counter, defaultdict
import random


def aggregate(records, reference_cost=1.0):
    if not records:
        raise ValueError("Cannot score an empty task batch")
    count = len(records)
    correct = sum(bool(row["success"]) for row in records)
    cost = sum(row["output_tokens"] for row in records) / count
    groups = defaultdict(list)
    for row in records:
        groups[row["task_id"].rsplit("_", 1)[0]].append(row)
    # SGC is reported only for complete three-variant scenarios in this batch.
    complete = [rows for rows in groups.values()
                if len({r["task_id"] for r in rows}) == 3]
    return {"tasks": count, "successes": correct, "accuracy": correct/count,
            "mean_output_tokens": cost,
            "total_output_tokens": sum(row["output_tokens"] for row in records),
            "mean_seconds": sum(row["seconds"] for row in records)/count,
            "score": correct + 0.5 * max(reference_cost, 1)/(max(reference_cost, 1)+cost),
            "sgc": sum(all(r["success"] for r in rows) for rows in complete)/len(complete) if complete else None,
            "complete_scenarios": len(complete)}


def comparison(parent, candidate):
    keys = lambda rows: Counter((r["task_id"], r["seed"]) for r in rows)
    if keys(parent) != keys(candidate) or len(parent) != len(candidate):
        raise ValueError("Compare the same complete task/seed batch")
    p = aggregate(parent)
    c = aggregate(candidate, p["mean_output_tokens"])
    p = aggregate(parent, p["mean_output_tokens"])
    return {"parent": p, "candidate": c, "beneficial": c["score"] > p["score"],
            "accuracy_delta": c["accuracy"]-p["accuracy"],
            "output_token_reduction": 1-c["mean_output_tokens"]/p["mean_output_tokens"] if p["mean_output_tokens"] else None}


def scenario(task):
    return task.rsplit("_", 1)[0]


def group_tasks(keys):
    groups = defaultdict(list)
    for task in keys:
        groups[scenario(task)].append(task)
    return dict(groups)


def final_report(a, b, seed):
    methods = (a, b)
    for rows in methods:
        if len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate task in formal report")
    score = comparison(a, b)
    group_a = group_tasks(r["task_id"] for r in a)
    lookup_a, lookup_b = ({r["task_id"]: r for r in rows} for rows in methods)
    pairs = []
    for ids in group_a.values():
        src, dst = ([lookup[t] for t in ids] for lookup in (lookup_a, lookup_b))
        pairs.append((sum(r["success"] for r in dst)-sum(r["success"] for r in src),
                      int(all(r["success"] for r in dst))-int(all(r["success"] for r in src)),
                      sum(r["output_tokens"] for r in dst)-sum(r["output_tokens"] for r in src)))
    n = len(pairs)
    rng = random.Random(seed)
    samples = [[] for _ in range(3)]
    for _ in range(10000):
        chosen = rng.choices(pairs, k=n)
        for k, vals in enumerate(zip(*chosen)):
            samples[k].append(sum(vals)/(n if k == 1 else len(a)))
    intervals = {key: [sorted(row)[249], sorted(row)[9749]] for key, row in zip(
        ("TGC_B_minus_A", "SGC_B_minus_A", "generated_tokens_per_task_B_minus_A"), samples)}
    def more(rows):
        return {"api_calls": sum(r["api_calls"] for r in rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "tasks_with_fragments": sum(bool(r.get("fragments")) for r in rows),
                "tasks_with_native_decision": sum(bool(r["model_calls"].get("D")) for r in rows),
                "tasks_with_generation": sum(bool(r["open_calls"]) if "open_calls" in r else
                                             any(n for m,n in r["model_calls"].items()
                                                 if m.startswith("G") or m=="R") for r in rows),
                "tasks_with_task_fallback": sum(bool(r.get("fallback")) for r in rows),
                "local_recoveries": sum(r.get("local_recoveries", 0) for r in rows),
                "multi_object_decisions": sum(r.get("decisions_with_multiple_objects", 0) for r in rows),
                "model_seconds": sum(r.get("model_seconds", 0) for r in rows),
                "mode_calls": {mode: sum(r["model_calls"].get(mode, 0) for r in rows)
                               for mode in sorted({m for r in rows for m in r["model_calls"]})}}
    report = {"A": {**score["parent"], **more(a)}, "B": {**score["candidate"], **more(b)},
            "differences": {"TGC_B_minus_A": score["accuracy_delta"],
                            "SGC_B_minus_A": score["candidate"]["sgc"]-score["parent"]["sgc"],
                            "generated_tokens_per_task_B_minus_A": score["candidate"]["mean_output_tokens"] - score["parent"]["mean_output_tokens"],
                            "relative_reduction": score["output_token_reduction"]},
            "paired_scenario_bootstrap_95_intervals": intervals,
            "failed_to_passed": [t for t in lookup_a if not lookup_a[t]["success"] and lookup_b[t]["success"]],
            "passed_to_failed": [t for t in lookup_a if lookup_a[t]["success"] and not lookup_b[t]["success"]]}
    return report
