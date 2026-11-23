"""Offline fragment induction and bounded patch proposals from Train experience."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .evolution import COMPONENTS
from .expressions import OPS
from .fragment_library import apply_fragment_patch, check_library


LAYOUT = r"""
The library has exactly these functional components: workflow, evidence, questions, candidates,
decision_skills, bindings, generation_skills, mode_policy. Every component is an object.
workflow holds objects `rules` and `fragments`, keyed by stable names. A rule is
  {"patterns":["regex",...], "exclude":["regex",...], "inputs":{"role":expression},
   "goals":["role",...], "complete":false, "source_tasks":[...]}
Patterns are independent regexes all matched against the current lowercased instruction. `exclude` is optional.
`inputs` and `exclude` may be omitted. The runtime assembles known goals from all matching rules and resolves
fragment data dependencies. Only use complete:true for a rule that recognizes the full requested effect/answer
and checks against incompatible additional constraints. Otherwise leave remaining work to 9B generation.
If the user asks for an answer, the terminal role must be `answer`; there is one provider per output role.
Each fragment is {"inputs":["role",...],"output":"role","app":"app_name","source_tasks":[...],
  "effect":"specific writes or read-only", "steps":[...], "result":expression}.
`inputs` get actual values from the current graph and are exposed as both `inputs.role` and initial `vars.role`.
`vars` is scoped to this fragment. Each successful output
is cached within this task and can be consumed by several fragments. Use separate fragments for reusable
business units with distinct outputs. Use precise roles e.g. `notes_full` vs `notes_summaries`.
The seed already publishes the `account_passwords` upstream role (list of account_name/password rows).
There is no source task ID in the runtime context. `task` has instruction, supervisor, datetime only.
Step formats:
 - {"kind":"call","binding":"key","store":"row","require":["field"]} (optionally `paginate`:true for APIs supporting
   page_index and page_limit; can use `expect`:{"message":"an exact observed success message"} for app writes).
   Bindings for calls are {"app":"app","api":"api","arguments":{"schema_arg":expression}}.
 - {"kind":"compute","binding":"key","store":"value"} with an expression in bindings.
 - {"kind":"if","binding":"key","then":[...],"else":[...]} where the binding is a boolean expression.
 - {"kind":"foreach","binding":"key","item":"row","body":[...],"collect":expression,"store":"values"}.
   `collect` and `store` are optional. During the body/collect the item is `vars.row` (if item is row).
 - {"kind":"decision","decision":"key","store":"item"} reads questions, evidence and candidates at key, with
   optional decision_skills and mode_policy. A candidate map is {"source":expression,"where":expression,
   "label":expression,"value":expression}; where/label/value are optional and can reference `item` (not vars.item).
   A literal list of {"label":...,"value":...} also works for stable finite operations, not for demo object IDs.
   The runtime adds NONE and supports 15 actual choices. Prefer deterministic filters with a valid basis and
   use native Jev decisions where data alone cannot select. Result is the candidate's live payload, not its label.
 - mode_policy decision keys: {"mode":"D" or "R","reason_on_none":true,"max_tokens":512}; use R sparingly.
 - For local fallback set generation_skills[fragment_name] to the prompt text for the narrow fragment only,
   and mode_policy[fragment_name] to {"recovery":"value" or "tool","local_type":"int"|"number"|"str"|"list"|"dict"|"bool",
   "local_tokens":384}. It returns one JSON value or one AppWorld operation cell printing a JSON value.
   The existing graph continues; do not use fallback to author a question, candidate list, or workflow.
Expressions: literals, `{"ref":"vars.foo.bar","default":...}`, `{"op":"name","args":[...]}`,
`{"template":"{name}","values":{"name":expression}}`, `{"map":{"source":expression,"item":"r","value":expression}}`,
`{"filter":{"source":expression,"item":"r","where":expression}}`. Use `r.field` (NOT vars.r) in map/filter.
API bindings do not return the wrapper: `call` stores actual JSON response. compute and if do not call any model.
Decision prompts consume current facts; prompt skills are text only. A fragment cannot call supervisor.complete_task.
Never copy a trace's answer, task IDs into runtime conditions, object IDs, password, access token, dates relative to an example,
or fixed names. Stable API/field names and genuine domain constants are fine. Only use public named AppWorld APIs.
For bulk writes verify API response and scope. Don't perform unrequested side effects or report unearned completion.
On a retry don't manufacture a new contract or give the model a larger budget. Return sole valid JSON as requested.
"""


def read_events(path):
    with Path(path).open(encoding="utf-8") as fn:
        return [json.loads(line) for line in fn if line.strip()]


def clipped(text, size):
    return text if len(text) <= size else text[:size] + " [truncated]"


def cleaned_text(text):
    # Preserve ordinary app messages and code but do not copy credentials or raw headers into optimizer prompts.
    text = re.sub(r'(?i)(["\']?(?:password|access_token|Authorization)["\']?\s*[:=]\s*["\'])[^"\']+(["\'])',
                  r'\1<redacted>\2', str(text))
    text = re.sub(r"\beyJ[A-Za-z0-9_\-.]{15,}\b", "<redacted_token>", text)
    return text


def compact_trace(path, maximum=6700):
    events = read_events(path)
    task = next(e for e in events if e["event"] == "task")
    summary = next(e for e in reversed(events) if e["event"] == "summary")
    rows = []
    dialect = {"show_account_passwords", "login"}
    for e in events:
        kind = e["event"]
        if kind == "tool_batch":
            rows.append({"kind": "cell", "code": clipped(cleaned_text(e["code"]), 1800),
                         "observation": clipped(cleaned_text(e["response"].get("output", "")), 700)})
        elif kind == "api":
            if e["api"] in dialect:
                value = e["response"].get("value", e["response"].get("output"))
                if isinstance(value, list):
                    observation = json.dumps([{"account_name": row.get("account_name")} for row in value if isinstance(row, dict)])
                elif isinstance(value, dict):
                    observation = json.dumps({k: ("<redacted>" if k in ("access_token", "password") else v) for k, v in value.items()})
                else:
                    observation = "<credential operation>"
            else:
                observation = cleaned_text(json.dumps(e["response"].get("value", e["response"].get("output")), ensure_ascii=False))
            rows.append({"kind": "api", "api": e["app"]+"."+e["api"],
                         "args": clipped(cleaned_text(json.dumps(e.get("arguments"), ensure_ascii=False)), 300),
                         "observation": clipped(observation, 390)})
        elif kind in ("fragment_failure", "composition_missing", "local_recovery", "fallback", "composition_goals", "decision"):
            rows.append({"kind": kind, **{key: e[key] for key in ("fragment", "goal", "goals", "reason", "node", "scope", "selected") if key in e}})
    action_urls = Counter(re.sub(r"\d+(?=/|$)", "{id}", urlsplit(e["url"]).path)
                          for e in events if e["event"] == "api_action")
    return {"task_id": task["task_id"], "instruction": task["instruction"],
            "status": {k: summary[k] for k in ("success", "error", "api_calls", "output_tokens", "model_calls", "fragments") if k in summary},
            "observed_request_paths": dict(action_urls.most_common(20)),
            "trace": clipped(json.dumps(rows if len(rows) <= 8 else rows[:4]+rows[-4:], ensure_ascii=False), maximum)}


def api_context(functions, scenarios, cap=32):
    terms = set(re.findall(r"[a-z_]{3,}", json.dumps(scenarios).casefold()))
    explicit = set(re.findall(r"\bapis\.([a-z_]+)\.([a-z_]+)\b", json.dumps(scenarios)))
    def score(entry):
        name = entry["name"]
        words = set(re.findall(r"[a-z]{3,}", (name+" "+entry.get("description", "")).casefold()))
        return (80*int(tuple(name.split("__", 1)) in explicit) + 12*int(name in terms)
                + 3*len(set(name.split("__")[-1].split("_")) & terms) + len(words & terms))
    docs = [row.get("function", row) for row in functions]
    return [{"name": f["name"], "description": f.get("description", "")[:210],
             "params": {key: spec.get("description", "")[:85] for key, spec in f.get("parameters", {}).get("properties", {}).items()},
             "required": f.get("parameters", {}).get("required", [])} for f in sorted(docs, key=score, reverse=True)[:cap]]


def induce(session, parent, paths, public_apis, folder, open_tasks):
    traces = [compact_trace(p, maximum=3400) for p in paths]
    traces = [p for p in traces if p["status"]["success"]]
    if not traces:
        return parent
    allowed = {p["task_id"] for p in traces}
    prompt = ("Act as the offline compiler. Produce an initial reusable LIBRARY ADDITION as JSON "
              "{\"additions\":{\"workflow\":{\"rules\":{},\"fragments\":{}},\"bindings\":{}, ...}}. "
              "Never replace unrelated entries from the current shared library. Each new fragment/rule must identify real "
              "successful source tasks from this group. Build reusable atomic pieces and conservative routes based on "
              "what was actually done and observed. A partial route is useful. You may touch all eight components for initial induction. "
              "Emit only definitions with enough source evidence. Produce at most six fragments, normally two to four. "
              "Keep output compact and omit absent optional fields. Start with the most reusable data dependencies. "
              "If a trace shows both an ambiguous object selection and tool usage, compile at least one concrete "
              "native decision with the observed public list/search response whenever scope is supported. "
              "If a whole scenario is complex, emit only a useful partial route.\n"
              + LAYOUT + "\nSupported ops: " + ", ".join(OPS)
              + "\nCURRENT LIBRARY:\n" + json.dumps(parent, ensure_ascii=False)
              + "\nTRAIN TRAJECTORIES:\n" + json.dumps(traces, ensure_ascii=False)
              + "\nPUBLIC API CATALOG:\n" + json.dumps(api_context(public_apis, traces), ensure_ascii=False))
    additions = session.json(prompt, folder)["additions"]
    if additions.keys()-set(COMPONENTS):
        raise ValueError("Induction returned an unknown component")
    candidate = json.loads(json.dumps(parent))
    for key, entries in additions.items():
        if key == "workflow":
            for collection, items in entries.items():
                for name, spec in items.items():
                    if name in parent[key][collection] and spec != parent[key][collection][name]:
                        raise ValueError(f"Initial induction attempted to replace {name}")
                    if name not in parent[key][collection] and (not spec.get("source_tasks")
                            or set(spec["source_tasks"])-allowed or set(spec["source_tasks"])-set(open_tasks)):
                        raise ValueError(f"Initial induction lacks a demonstrated source for {name}")
                    candidate[key][collection][name] = spec
        else:
            for name, spec in entries.items():
                if name in parent[key] and spec != parent[key][name]:
                    raise ValueError(f"Initial induction attempted to replace {key}/{name}")
                candidate[key][name] = spec
    return check_library(candidate, public_apis)


def propose(session, parent, paths, public_apis, history, folder, limit=2):
    traces = [compact_trace(p, maximum=4000) for p in paths]
    prompt = ("You are GPT-6.2 Sol Max, the offline optimizer for a frozen 9B Jev executor. "
              "Use only the Train traces below; no evaluator assertions or Test data. Diagnose one justified reusable "
              f"failure or high-output success. Return up to {limit} different complete candidates from the same parent. "
              "Each candidate is {\"reason\":\"observed failure and a useful fix\", \"operation\":\"edit\" or \"register\", "
              "\"changes\":{\"component\":{\"entry_key\":complete_entry}, ...}}. The reply is {\"candidates\":[...]}. "
              "Any candidate changes at most two functional components. A registration appends exactly one new reusable fragment, "
              "without editing current fragments. Prefer an existing chain of reusable slots over copying a whole task. "
              "Prefer a code/parameter fix when a choice has missing evidence; use reasoning when the evidence is already adequate. "
              "The first Train scenario in this prompt is the intended repair target; the next two are diversity/regression checks. "
              "Compare correctness first, then generated tokens, including failure and recovery. Record why a repair should transfer.\n"
              "A candidate that improves the Train replay may advance when both Dev success count and generated tokens stay equal.\n"
              + LAYOUT + "\nSupported ops: " + ", ".join(OPS)
              + "\nCURRENT LIBRARY:\n" + json.dumps(parent, ensure_ascii=False)
              + "\nRECENT EVALUATED ATTEMPTS (Dev is aggregates only):\n" + json.dumps(history[-5:], ensure_ascii=False)
              + "\nTRAIN TRAJECTORIES:\n" + json.dumps(traces, ensure_ascii=False)
              + "\nPUBLIC API CATALOG:\n" + json.dumps(api_context(public_apis, traces), ensure_ascii=False))
    candidates = session.json(prompt, folder)["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("Sol did not return a candidate list")
    return candidates[:limit]


def candidate_library(parent, patch, open_tasks, functions):
    candidate, touched = apply_fragment_patch(parent, patch, open_tasks)
    return check_library(candidate, functions), touched
