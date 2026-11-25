"""Indexed public Train evidence shared by current and historical offline editors."""
import json
from pathlib import Path

from .expressions import OPS
from .formal_optimizer import LAYOUT, cleaned_text, read_events
from .runtime import STEP_REVIEW


def visible_event(event):
    if event["event"] == "summary":
        return {k:event[k] for k in ("event","success","output_tokens","seconds","error") if k in event}
    return event


def serialized(value):
    return cleaned_text(json.dumps(value, ensure_ascii=False, default=str))


def leading_choices(event):
    scores=event["response"].get("probabilities",[])
    return [(event["request"]["options"][j],round(scores[j],5))
            for j in sorted(range(len(scores)),key=scores.__getitem__,reverse=True)[:2]]


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


class TrainEvidence:
    def __init__(self, study, lib, rows, history, folder, sources):
        self.study, self.lib, self.folder, self.sources = study, lib, folder, sources
        self.previous = history
        self.tasks = {r["task_id"] for r in rows}
        self.docs, self.paths, self.index = {}, {}, []
        self.docs["dsl"] = (LAYOUT[LAYOUT.index("Step formats:"):]+"\nSupported operations: "+", ".join(OPS)).splitlines()
        for k,v in lib.items():
            self.docs["library/"+k] = [serialized({key:value}) for key,value in v.items()] if isinstance(v,dict) else [serialized(v)]
        definitions = lib["cards"] if "cards" in lib else lib["workflow"]["fragments"]
        for key,value in definitions.items():
            self.docs["fragment/"+key] = [serialized(value)]
        code_docs = ("card_assembly","card_library","fragment_assembly","expressions") if "cards" in lib else (
            "staged_entry","catalog_entry","phase1","composable_v2","fragment_library","policy_programs","expressions")
        for name in code_docs:
            self.docs["code/"+name] = (Path(__file__).parent/(name+".py")).read_text(encoding="utf-8").splitlines()
        fs = [f.get("function", f) for f in study.functions]
        self.docs["api_names"] = [f["name"] for f in fs]
        self.docs.update({"api/"+f["name"]:[serialized(f)] for f in fs})
        previous = sorted({p for rev in history for p in rev.get("train_traces",[])+rev.get("parent_traces",[])})
        previous.extend(r["trace_path"] for rev in history
                        for pair in rev.get("paired_selection", {}).get("repeats", {}).values() for r in pair)
        for p in previous:
            path = study.root/p
            self.paths[path.parts[-4]+"/"+path.parts[-3]] = path
        for row in rows:
            path = study.root/row["trace_path"]
            name = "current/"+row["task_id"]
            self.paths[name] = path
            es = read_events(path)
            details = [{"i":i,"event":e["event"],"detail":
                        e.get("reason") or e.get("fragment") or e.get("scope") or e.get("node") or e.get("error")}
                       for i,e in enumerate(es) if e["event"] in ("fragment_choice","fragment_choice_deferred",
                        "fragment_failure","composition_missing","fallback","execution_error",
                        "stage_prefilter","stage_choice","stage_binding","stage_deferred","stage_handoff",
                        "card_menu","card_executed","card_failed","fragment_handoff","fragment_reentry",
                        "assembly_candidates","assembly_selection","assembly_widen","adapter_login","program_error","step_review","handoff_state","handoff_selection","parameter_state")]
            executed = [(i,cleaned_text(e["code"])[:1100]) for i,e in enumerate(es) if e["event"]=="tool_batch"]
            positions = []
            for i,e in enumerate(es):
                item = {"i":i, "event":e["event"]}
                if e["event"] == "tool_batch":
                    result = e["response"]
                    item["calls"] = [a["method"]+" "+a["url"].split("?")[0] for a in result.get("actions",[])]
                    item["output_excerpt"] = cleaned_text(result.get("output",""))[-450:]
                elif e["event"] == "model":
                    item.update(op=e["operation"], output_tokens=e["response"].get("output_tokens"))
                    if e["request"].get("purpose") == STEP_REVIEW:
                        item["purpose"] = STEP_REVIEW
                    if "cards" in lib and e["operation"]=="decide":
                        item["top_choices"]=leading_choices(e)
                elif e["event"] == "api":
                    item.update(api=e["app"]+"."+e["api"],node=e.get("node"))
                else:
                    item.update({k:e[k] for k in ("menu","fragment","scope","node","stage","reason","error",
                                "apps","retained","complete","options","opened_by","role","source","completed",
                                "card","offered","excluded","output","inputs","eligible","families","family","mode",
                                "native_index","native_confidence","final_index","outcome","guide","choice") if k in e})
                positions.append(serialized(item))
            self.docs["events/"+row["task_id"]] = positions
            self.index.append({"task":row["task_id"],"instruction":es[0]["instruction"],"success":row["success"],
                "output_tokens":row["output_tokens"],"modes":row.get("model_calls"),"trace":name,"events":len(es),
                "staged":row.get("staged"),"locations":details[:15],"last_code":executed[-1:],
                **({"executed_cards":row.get("fragments",[]),"menus":row.get("card_menus",0),
                    "handoffs":row.get("fallback",[])} if "cards" in lib else {})})
        self.docs["trace_directory"] = sorted(self.paths)
        self.docs["history"] = [serialized({k:v for k,v in r.items() if k not in ("train_traces","parent_traces")}) for r in history]

    def lines(self, doc):
        if doc.startswith("api/"):
            doc = "api/" + doc[4:].removeprefix("apis.").replace(".", "__")
        if doc in self.docs:
            return self.docs[doc]
        if doc in self.paths:
            return [serialized(visible_event(e)) for e in read_events(self.paths[doc])]
        raise ValueError("Unknown document "+str(doc))

    def find_evidence(self, query):
        needle = query["text"].casefold()
        hits = []
        for i, line in enumerate(self.lines(query["doc"])):
            pos = line.casefold().find(needle)
            if i >= query.get("start", 0) and pos >= 0:
                hits.append({"index": i, "offset": pos, "snippet": line[max(0, pos-90):pos+220]})
                if len(hits) == 25:
                    break
        return hits

    def lookup(self, action):
        if "find" in action:
            return self.find_evidence(action["find"])
        output = []
        for query in action["read"][:6]:
            if isinstance(query, str):
                query = {"doc":query}
            doc = query["doc"]
            try:
                # Historic traces are indexed by stage/task, while current event maps use events/task.
                key = doc[7:] if doc.startswith("events/") and doc[7:] in self.paths and doc not in self.docs else doc
                page = read_page(self.lines(key), query.get("start",0), query.get("count",5),
                                 query.get("offset",0), budget=6500)
                output.append({"doc":doc, **page})
            except (ValueError, KeyError, TypeError) as exc:
                output.append({"doc":doc,"error":str(exc)})
        return output

    def add_last_round(self, folder, history):
        from .batch_feedback import prefix_changes, prefix_view
        # A revisit has the same task IDs as its first visit, whose results remain in history.
        current_visit = folder.parent.parent.name
        for revision in reversed(history):
            if "batch" in revision:
                visit = revision.get("visit", f"b{revision['batch']:02d}")
                if visit != current_visit:
                    continue
            candidates = self.paired_paths(revision.get("train_traces", []))
            parents = self.paired_paths(revision.get("parent_traces", []))
            if not self.tasks or candidates.keys() != self.tasks or parents.keys() != self.tasks:
                continue
            for item in self.index:
                task = item["task"]
                before, after = parents[task], candidates[task]
                left, right = (json.loads(p.parents[1].joinpath("record.json").read_text(encoding="utf-8"))
                               for p in (before, after))
                item["last_measured_round"] = {
                    "parent": revision["parent"], "candidate": revision["candidate"], "accepted": revision["accepted"],
                    "success_before_after": [left["success"], right["success"]],
                    "total_generated_tokens_before_after": [left["output_tokens"], right["output_tokens"]],
                    "parent_trace": before.parts[-4] + "/" + task, "candidate_trace": after.parts[-4] + "/" + task,
                    "prefix_changes": prefix_changes(prefix_view(read_events(before)), prefix_view(read_events(after)))}
                pair = revision.get("paired_selection", {}).get("repeats", {}).get(task)
                if pair:
                    item["last_measured_round"]["confirmation"] = [
                        {k:row[k] for k in ("arm","confirmation_pass","success","output_tokens","trace_path")
                         if k in row} for row in pair]
            break

    def paired_paths(self, traces):
        paths = [self.study.root / Path(t.replace("\\", "/")) for t in traces]
        return {p.parents[1].name: p for p in paths}
