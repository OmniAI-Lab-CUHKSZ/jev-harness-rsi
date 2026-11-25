"""Describe the observed route and available interventions to the offline Train editor."""
from collections import Counter, defaultdict

from .formal_optimizer import cleaned_text
from .runtime import is_open_generation


# A generated Python cell calls AppWorld directly. An adapter only affects the compiled path.
# These descriptions follow the executor hooks and do not prescribe a repair for a symptom.
EDIT_SURFACES = {
    "assembly": ("Which eligible providers reach which menu, including dependencies across apps.",
                 "Compare eligible, family membership, final offered options and widening.",
                 ["library/assembly", "code/semantic_assembly"]),
    "decision": ("Which of the offered actions, parameters or guides Jev chooses.",
                 "Compare actual options, NONE, live evidence and both adaptive rankings. Can change a question, skill or supported option policy.",
                 ["library/questions", "library/skills", "library/decision_policies", "code/semantic_assembly"]),
    "card": ("What a selectable program reads, computes, binds or writes. One reusable intermediate result can serve several goals.",
             "Check installed interfaces, candidate values, actual output and consuming step before adding another operation.",
             ["library/cards", "batch/operations", "code/card_assembly", "code/card_library"]),
    "adapter": ("How a compiled public operation authenticates, maps arguments, pages and extracts a result.",
                "Inspect session and program API events. A traceback from direct generated Python bypasses these hooks.",
                ["library/adapters", "code/semantic_assembly"]),
    "state": ("What facts about live data, scope, bindings and completed work Jev and the handoff see.",
              "Compare the actual source role with the evidence shown to its next consumer. A view cannot fetch missing data.",
              ["library/views", "code/semantic_assembly", "code/scoped_menu"]),
    "handoff": ("Which unresolved work and live data reach open generation, and when it may return a public read to Jev.",
                "Inspect eligibility, selected guide, bound roles and the actual generated cell. A guide does not execute a tool.",
                ["library/handoffs", "code/semantic_handoff", "code/semantic_assembly"]),
}


def request_context(events, index, lib):
    """Show the active configuration at a trace location, without credentials or result payloads."""
    e = events[index]
    kind = e["event"]
    base = {"event":index,"kind":kind}
    if kind == "model":
        if is_open_generation(e["operation"],e["request"]):
            kind = "open_generation"
        else:
            for nxt in events[index+1:]:
                if nxt["event"] == "decision":
                    e,kind = nxt,"decision"
                    break
                if nxt["event"] == "model":
                    break
    if kind == "api":
        app,api = e["app"],e["api"]
        node = e.get("node")
        card = lib["cards"].get(node,{})
        endpoint = app+"__"+api
        base.update(route="compiled public call",node=node,endpoint=endpoint,
                    endpoint_adapter=lib["adapters"].get(endpoint),
                    argument_fields=list(e.get("arguments",{})))
        if node == "session" and api == "login":
            base["login_identity_field"] = lib["adapters"].get(app,{}).get("login_field","email")
        if card:
            base["operation"] = {k:card[k] for k in ("builtin","api","arguments","inputs","output") if k in card}
        return base
    if (kind == "tool_batch" and e.get("node")=="generation_agent") or kind in ("open_generation","generation_parse_error"):
        base.update(route="direct generated Python",compiled_endpoint_adapters_run=False)
        last = next((t for t in reversed(events[:index]) if t["event"]=="handoff_selection"),{})
        base["selected_guide"] = last.get("guide")
        return base
    if kind == "decision":
        node = e["node"].removeprefix("fragment/")
        base.update(route="native choice",node=node,
                    option_policy=lib.get("decision_policies",{}).get(node),
                    projected_views=list(e.get("evidence",{}).get("views",{})),
                    installed_question=node in lib["questions"])
        if node == "step":
            page = next((v for v in reversed(events[:index]) if v["event"]=="card_menu"),{})
            base["menu_family"] = page.get("assembly")
            base["recorded_offered"] = page.get("offered",[])
            base["installed_family"] = lib["assembly"].get(page.get("assembly"))
        elif node == "card_parameter":
            entry = next((v for v in reversed(events[:index]) if v["event"]=="parameter_state"),{})
            binding_card = lib["cards"].get(entry.get("card"),{})
            base["card"] = entry.get("card")
            base["slot"] = next((k for k,v in binding_card.get("parameters",{}).items() if v["question"]==e.get("question")),None)
            base["recorded_available_roles"] = entry.get("available",[])
            base["policy_inputs"] = {"datetime":entry.get("datetime"),"active_card":entry.get("card")}
            base["rendered_evidence_keys"] = list(e.get("evidence",{}))
            base["installed_parameter_views"] = [k for k,v in lib["views"].items() if v.get("at")=="parameter"]
        return base
    if kind in ("card_executed","card_failed","parameter_state","parameter_bindings","result_source"):
        name = e.get("card",e.get("producer"))
        spec = lib["cards"].get(name,{})
        return {**base,"route":"compiled operation","card":name,"source_api":spec.get("api"),
                "inputs":spec.get("inputs",[]),"parameter_names":list(spec.get("parameters",{})),"output":e.get("output",spec.get("output"))}
    if kind == "retrieval_menu":
        return {**base, **{k:e[k] for k in ("node","batch","offered","offered_count","slot","card") if k in e}}
    if kind in ("assembly_selection","assembly_candidates","assembly_widen","card_menu"):
        return {**base,"route":"menu assembly","family":e.get("family",e.get("assembly")),
                "offered":e.get("offered",[]),"eligible":e.get("eligible",[])}
    if kind in ("handoff_selection","handoff_state","fragment_handoff"):
        return {**base,"route":"handoff","guide":e.get("guide"),"eligible":e.get("eligible",[]),
                "live_roles":list(e.get("values",e.get("available",[])))}
    return base


def trace_incidents(events):
    """Locations for investigation. Counts are exposure, not labels of the optimal edit."""
    incidents = []
    last_open = 0
    live = []
    for i, e in enumerate(events):
        kind = e["event"]
        signal = None
        detail = {}
        if kind == "model" and is_open_generation(e["operation"], e["request"]):
            last_open = e["response"].get("output_tokens", 0)
        elif kind == "program_error":
            signal = "policy program raised an error"
            detail = {"policy": e.get("policy"), "error": cleaned_text(e.get("error", ""))[-180:]}
        elif kind == "adapter_login" and not e.get("success"):
            signal, detail = "compiled login returned no session", {"app": e.get("app"), "field": e.get("identity_field")}
        elif kind == "api" and "Execution failed." in str(e.get("response", {}).get("output", "")):
            signal, detail = "compiled public call failed", {"endpoint": e["app"]+"__"+e["api"], "node": e.get("node")}
        elif kind == "assembly_widen":
            signal, detail = "narrow menu widened", {"family": e.get("family")}
        elif kind == "decision":
            choices = e.get("candidates", [])
            selected = e.get("selected")
            if type(selected) is int and 0 <= selected < len(choices) and choices[selected].get("value") is None:
                signal, detail = "Jev abstained on an offered menu", {"node": e["node"], "options": len(choices)-1}
        elif kind == "step_review" and e.get("outcome") == "changed":
            signal, detail = "adaptive review changed choice", {"node": e.get("node"), "tokens": e.get("output_tokens", 0)}
        elif kind == "card_failed":
            signal, detail = "selected operation failed", {"card": e.get("card"), "error": cleaned_text(e.get("reason", ""))[-200:]}
        elif kind == "handoff_state":
            live = list(e.get("values", {}))
            signal, detail = "generation entry with observed live roles", {"roles": live, "failed_card": e.get("failed")}
        elif kind == "generation_parse_error":
            signal, detail = "generated output did not execute", {"output_tokens": last_open}
        elif kind == "tool_batch" and e.get("node") == "generation_agent":
            out = str(e.get("response", {}).get("output", ""))
            if "Execution failed." in out:
                signal, detail = "direct generated cell raised an error", {
                    "error": cleaned_text(out).strip().splitlines()[-1][:200], "live_roles_at_entry": live,
                    "output_tokens": last_open}
        if signal is not None:
            incidents.append({"event": i, "observation": signal, **detail})
    return incidents


def paired_outcome_costs(pairs):
    """Charge both versions' outputs to the observed whole-task transition."""
    names = {(True, True): "both_solved", (False, False): "both_failed",
             (False, True): "newly_solved", (True, False): "regressed"}
    grouped = {}
    for outcomes, (before, after) in pairs:
        group = grouped.setdefault(names[tuple(outcomes)], {
            "tasks": 0, "parent_output_tokens": 0, "candidate_output_tokens": 0,
        })
        group["tasks"] += 1
        group["parent_output_tokens"] += before
        group["candidate_output_tokens"] += after
    return grouped


def batch_patterns(items, incidents):
    groups = defaultdict(dict)
    for row in items:
        task = row["task"]
        for event in incidents[task]:
            # Keep the site: abstaining at the family level differs from abstaining at a slot.
            key = (event["observation"], event.get("policy") or event.get("node") or event.get("card") or event.get("endpoint"))
            groups[key].setdefault(task, []).append(event["event"])
    solved = {r["task"] for r in items if r["success"]}
    result = []
    for (label, site), occurrences in groups.items():
        affected = sorted(occurrences)
        result.append({"observation": label, "site": site, "failed_tasks": sorted(set(affected)-solved),
                       "also_on_successes": sorted(set(affected)&solved),
                       "distinct_scenarios": len({t.rsplit("_",1)[0] for t in affected}),
                       "first_events": {t: occurrences[t][0] for t in affected},
                       "occurrences": sum(map(len,occurrences.values()))})
    return sorted(result, key=lambda x: (-len(x["failed_tasks"]), -x["distinct_scenarios"], x["observation"], x["site"] or ""))


def edit_history(history):
    counts = Counter(edit.split("/",1)[0] for rev in history for edit in rev.get("edits", []))
    return {"previous_proposals": len(history), "object_edits": {k:counts[k] for k in EDIT_SURFACES}}
