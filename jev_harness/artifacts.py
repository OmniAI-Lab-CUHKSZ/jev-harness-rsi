"""Read and write run records and public evidence."""

import json
from pathlib import Path
import re

from .runtime import STEP_REVIEW


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    temp.replace(path)


def read_events(path):
    with Path(path).open(encoding="utf-8") as fn:
        return [json.loads(line) for line in fn if line.strip()]


def cleaned_text(text):
    # Preserve ordinary app messages and code but do not copy credentials or raw headers into optimizer prompts.
    text = re.sub(r'(?i)(["\']?(?:password|access_token|Authorization)["\']?\s*[:=]\s*["\'])[^"\']+(["\'])',
                  r'\1<redacted>\2', str(text))
    text = re.sub(r"\beyJ[A-Za-z0-9_\-.]{15,}\b", "<redacted_token>", text)
    return text


def visible_event(event):
    if event["event"] == "summary":
        return {k:event[k] for k in ("event","success","output_tokens","seconds","error") if k in event}
    return event


def read_page(lines, start=0, count=6, offset=0, budget=16000):
    start = max(0, int(start))
    offset = max(0, int(offset))
    rows = []
    for i in range(start, min(len(lines), start+min(max(int(count),1),50))):
        body = lines[i]
        skip = offset if i == start else 0
        size = min(len(body)-min(skip,len(body)), budget)
        rows.append({"index":i,"offset":skip,"text":body[skip:skip+size]})
        budget -= size
        if skip+size < len(body):
            return {"rows":rows,"next":{"start":i,"offset":skip+size}}
        if budget == 0:
            break
    next_index = rows[-1]["index"]+1 if rows else len(lines)
    return {"rows":rows,"next":{"start":next_index,"offset":0} if next_index < len(lines) else None}


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
    row["retry_output_tokens"]=sum(v.get("output_tokens",0) for v in known)
    for key in ("output_tokens","reasoning_tokens","input_tokens"):
        row[key]+=sum(v.get(key,0) for v in known)
    save(trace.parent.parent/"record.json",row)
    return row
