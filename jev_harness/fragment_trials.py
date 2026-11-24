"""Train-only fragment substitution trials alongside the ordinary, unforced Jev run."""
from .batch_feedback import BatchFeedbackWorkbench
from .card_assembly import public_apps
from .card_library import ports
from .formal_optimizer import read_events
from .fragment_replay import run_local
from .operation_evidence import batch_operations
from .observed_bindings import ObserveBindings
from .phase1_optimizer import serialized
from .semantic_workbench import SemanticProbe
from .transport import WorkerUnavailable


class ObservedProbe(ObserveBindings, SemanticProbe):
    """A natural bounded prefix with evidence of actual downstream reads."""


def natural_use(events, cards):
    report = {name:{"offered":False,"selected":False,"executed":False,"consumers":[],
                    "generated_read_observed":None} for name in cards}
    producers = {}
    for i, e in enumerate(events):
        kind = e["event"]
        if kind == "card_menu":
            for name in report.keys() & set(e["offered"]):
                report[name]["offered"] = True
            for item in e.get("state",{}).get("bindings",{}).values():
                name = item.get("producer")
                if name in report and item.get("parameters"):
                    report[name]["bound_parameters"] = item["parameters"]
        elif kind == "decision" and e.get("node") == "fragment/step":
            picked = e["candidates"][e["selected"]]["value"]
            if isinstance(picked,str) and picked in report:
                report[picked].update(selected=True, selection_event=i)
        elif kind in {"card_executed","fragment_reentry"}:
            name = e.get("card")
            for role in e.get("inputs",[]):
                if role in producers:
                    report[producers[role]]["consumers"].append({"card":name,"event":i})
            role = e.get("output") or e.get("role")
            if name in report:
                report[name].update(executed=True, execution_event=i, output=role)
                producers[role] = name
            else:
                producers.pop(role,None)
        elif kind == "card_failed" and e["card"] in report:
            report[e["card"]]["failure"] = e["reason"]
        elif kind == "trial_card_bindings" and e["card"] in report:
            report[e["card"]]["bound_parameters"] = e["parameters"]
        elif kind == "generated_read_observation" and e.get("available"):
            for role, producer in producers.items():
                prior = report[producer]["generated_read_observed"]
                report[producer]["generated_read_observed"] = bool(prior or role in e["roles"])
    return report


class FragmentTrialWorkbench(BatchFeedbackWorkbench):
    probe_type = ObservedProbe
    prompt = BatchFeedbackWorkbench.prompt + """
The workbench has indexed repeated operations INSIDE actual generated programs of this batch. Historical
Train traces remain readable. Read batch/operations or operations/<current task> for source events,
line numbers, resulting variables and observed public calls. These are hypotheses, including failed programs.
Where a recurring operation is available, try replacing it with executable cards (or repair how an existing
card is used) before choosing the final candidate. Six edit types and the ordinary scoring rule still apply.
Separate the public input, local parameters and output from the final task action. Leave necessary open work
to generation, and allow unused cards to be skipped. A trial returns forced local execution separately from
the normal Jev menu, parameter choices, downstream card use and observed generated reads. Report the evidence.
generated_read_observed is null when no instrumented generated cell was observed with that output live.
A true value records a direct subscript or .get access; false means neither was observed in those cells.
consumers lists later cards that declare this role as an input. Inspect their arguments, parameters or
calculation to see whether they read the value or simply require its presence before appearing in the menu.
Optional patch field local: {"cards":["card_name",...],"cases":[{"task":"current id","anchor":"generation",
"parameters":{"card_name":{"slot":live_choice_value}},"reference":{"event":integer,"through_line":integer,
"variable":"recorded_variable","card":"card_name","order":"ordered"}}]}.
anchor is "start", "generation" (before the parent's first G), {"after_card":"parent_card"} or
{"event":parent_generated_tool_batch_index,"through_line":source_line}. At a code anchor, optional bindings
map card input roles to actual public Python variables. A reference is read in a separate fresh replay
through a complete top-level statement. Use "unordered" only when order is immaterial. Verify object identity,
scope and any writes. Agreement with a failed parent is diagnostic only. If unspecified, changed cards
are locally tried on the focus task and matching current-batch tasks. All scored work uses normal selection.
"""

    def __init__(self, *args):
        super().__init__(*args)
        self.current_events = {task:read_events(self.paths["current/"+task]) for task in sorted(self.tasks)}
        self.groups, self.mining_counts = batch_operations(self.current_events)
        self.docs["batch/operations"] = [serialized({k:v for k,v in g.items() if k!="sites"}) for g in self.groups]
        for task in self.current_events:
            self.docs["operations/"+task] = [serialized({"api":g["api"],"operation":g["operation"],**site})
                for g in self.groups for site in g["sites"].get(task,())]
        for item in self.index:
            last = item.get("last_measured_round",{})
            doc = last.get("candidate_trace")
            if doc in self.paths:
                old = next((r for r in reversed(self.previous) if r["candidate"] == last["candidate"]),{})
                cards = [s.split("/",1)[1] for s in old.get("edits",[]) if s.startswith("card/")]
                last["observed_card_use"] = natural_use(read_events(self.paths[doc]),cards)

    def opening_evidence(self):
        overview = []
        for g in self.groups[:10]:
            overview.append({"api":g["api"],"operation":g["operation"],"tasks":g["tasks"],
                "examples":[{"task":task,"event":sites[0]["event"],"code":sites[0]["code"]}
                            for task,sites in list(g["sites"].items())[:3]]})
        return {"current_train_operations":overview, "mining":self.mining_counts}

    def probe_key(self, proposal, task):
        return (*super().probe_key(proposal,task),proposal.get("local"))

    def cases(self, proposal, task, lib):
        local = proposal.get("local") or {}
        if not isinstance(local,dict):
            raise ValueError("local must describe cards and current Train cases")
        changed = [o["name"] for o in proposal["objects"] if o["kind"] == "card"]
        cards = local.get("cards",changed)
        if not isinstance(cards,list) or any(not isinstance(s,str) or s not in lib["cards"] for s in cards):
            raise ValueError("local.cards must be installed card names")
        if not cards:
            return [], []
        if "cases" in local:
            cases = local["cases"]
            if not isinstance(cases,list) or any(not isinstance(c,dict) or c.get("task") not in self.tasks for c in cases):
                raise ValueError("Local cases must be current Train tasks")
            return cards,[{**c,"cards":cards} for c in cases]
        defs = lib["cards"]
        focus_apis = {defs[s].get("api") for s in cards}
        related = {t for g in self.groups if task in g["tasks"] and g["api"] in focus_apis for t in g["tasks"]}
        cases = []
        for t in [task,*sorted(related-{task})]:
            apps = public_apps(self.current_events[t][0]["instruction"],defs)
            chosen = {s for s in cards if not apps or ports(defs[s])[0] in apps|{"general","supervisor"}
                      or s in (local.get("cards") or [])}
            # A cross-application provider or consumer can be part of the same replacement.
            while chosen:
                ins = {i for s in chosen for i in ports(defs[s])[1]}
                outs = {ports(defs[s])[2] for s in chosen} - {"effect_receipt","answer","submitted"}
                adjacent = {s for s in cards if ports(defs[s])[2] in ins or outs.intersection(ports(defs[s])[1])}
                if adjacent <= chosen:
                    break
                chosen |= adjacent
            if chosen:
                pending = [s for s in cards if s in chosen]
                ordered = []
                while pending:
                    produced_later = {ports(defs[s])[2] for s in pending}
                    ready = [s for s in pending if not produced_later.intersection(ports(defs[s])[1])]
                    if not ready:
                        ordered.extend(pending)
                        break
                    ordered.extend(ready)
                    pending = [s for s in pending if s not in ready]
                needed = {i for s in chosen for i in ports(defs[s])[1]} - {ports(defs[s])[2] for s in chosen}
                anchor = "generation" if needed else "start"
                for event in self.current_events[t]:
                    if not needed:
                        break
                    if event["event"] == "card_executed":
                        needed.discard(event["output"])
                        if not needed:
                            anchor = {"after_card":event["card"]}
                case = {"task":t,"cards":ordered,"anchor":anchor}
                last = defs[ordered[-1]]
                reads = {defs[s].get("api") for s in ordered}
                # Compare a concrete old variable only when the recorded operation has a unique result.
                # A single page is not a reference for a whole paginated collection.
                possibilities = [site for g in self.groups if t in g["sites"] and
                    ((last.get("compute") is not None and g["operation"]!="API" and g["api"] in reads) or
                     (g["operation"]=="API" and g["api"]==last.get("api") and not last.get("paginate")))
                    for site in g["sites"][t] if site.get("variable") and not site.get("cell_reported_error",True)]
                if possibilities and len({s["variable"] for s in possibilities}) == 1:
                    site = possibilities[0]
                    case["reference"] = {k:site[k] for k in ("event","through_line","variable")}
                cases.append(case)
        return cards,cases

    def probe(self, proposal, task):
        from formal_run import save
        if task not in self.tasks:
            raise ValueError("Probes require a current Train task")
        lib,_ = self.install(self.lib,proposal,self.sources,self.study.functions)
        cards, cases = self.cases(proposal,task,lib)
        folder = self.folder/f"probe_{self.calls+1:02d}"
        passed = {i["task"]:i["success"] for i in self.index}
        def run(entry, worker):
            n, case = entry
            t = case["task"]
            dest = folder/"local"/f"{n}_{t}"
            for attempt in range(2):
                try:
                    return run_local(self.study,worker,self.lib,lib,self.current_events[t],case,
                                     dest/str(attempt+1),t,passed[t])
                except WorkerUnavailable as exc:
                    worker.close_worker("environment")
                    if attempt:
                        return {"task":t,"forced_cards":case["cards"],"trial_unavailable":str(exc)}
        local_results = self.study.execute_jobs(list(enumerate(cases)),run) if cases else []
        focus = super().probe(proposal,task)
        focus["local_replacements"] = local_results
        focus["natural_focus_use"] = natural_use(read_events(folder/"trace.jsonl"),cards)
        for row in focus["current_batch_prefixes"]:
            if row.get("candidate_trace") in self.paths:
                # Keep cached prefix records independent of which replacement is now being investigated.
                focus.setdefault("natural_prefix_use",{})[row["task"]] = natural_use(
                    read_events(self.paths[row["candidate_trace"]]),cards)
        save(folder/"result.json",{**focus,"comparison_objects":proposal["objects"]})
        self.last_probed = self.probe_key(proposal,task)
        return focus
