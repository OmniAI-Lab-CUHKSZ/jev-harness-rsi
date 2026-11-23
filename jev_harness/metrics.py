"""Task correctness first; actual executor output tokens break correctness ties.

Do not name this module evaluation.py: AppWorld imports task modules by that name.
"""
from collections import Counter, defaultdict


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
