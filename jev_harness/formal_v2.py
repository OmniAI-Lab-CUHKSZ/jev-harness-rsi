"""Paired Train comparisons and token accounting for retried tasks."""
import json
from formal_run import quality, save
from .metrics import comparison
from .runtime import STEP_REVIEW


def gain(parent, candidate):
    verdict=comparison(parent,candidate)
    a={r["task_id"]:r["success"] for r in parent}
    verdict["gained"]=[r["task_id"] for r in candidate if r["success"] and not a[r["task_id"]]]
    verdict["lost"]=[r["task_id"] for r in candidate if not r["success"] and a[r["task_id"]]]
    verdict["tied"]=quality(parent)==quality(candidate)
    return verdict


def include_retries(root, row):
    trace=root/row["trace_path"]
    if "retry_output_tokens" in row or trace.parent.name=="attempt_1":
        return row
    known=[]
    for path in trace.parent.parent.glob("attempt_*/trace.jsonl"):
        if path==trace:
            continue
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    event=json.loads(line)
                except ValueError:
                    continue
                if event.get("event")=="model":
                    known.append(event["response"])
                    if "step_review" in row and event["request"].get("purpose") == STEP_REVIEW:
                        stats = row["step_review"]
                        stats["calls"] += 1
                        for key in ("output_tokens", "reasoning_tokens"):
                            stats[key] += event["response"].get(key, 0)
                elif "step_review" in row and event.get("event") == "step_review":
                    row["step_review"][event["outcome"]] += 1
                elif "semantic_handoff" in row and event.get("event") == "handoff_selection":
                    stats = row["semantic_handoff"]
                    stats["handoffs"] += 1
                    stats["eligible"] += int(bool(event["eligible"]))
                    stats["selected"] += int(event["guide"] is not None)
                    for key in ("calls","input_tokens","output_tokens","model_seconds"):
                        stats[key] = stats.get(key,0) + event.get(key,0)
                    if event["guide"] is not None:
                        stats["guides"] = sorted(set(stats.get("guides",())) | {event["guide"]})
    row["retry_output_tokens"]=sum(v.get("output_tokens",0) for v in known)
    for key in ("output_tokens","reasoning_tokens","input_tokens"):
        row[key]+=sum(v.get(key,0) for v in known)
    save(trace.parent.parent/"record.json",row)
    return row
