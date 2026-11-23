"""Pilot: select a root from raw library definitions, without operation registrations."""
import math
import re

from .composable_v2 import route_inputs, routed
from .phase1 import PhaseOneExecutor
from .pilot_interfaces import discover_consumers
from .runtime import MAX_REAL_OPTIONS, Uncovered


QUESTION = ("Which installed operation should run for this request? Prefer one that delivers "
            "the whole requested result or exactly the requested changes. Its dependencies will "
            "run automatically. Otherwise choose a read-only operation that provides specifically "
            "needed records for unfinished work. Choose NONE if nothing fits.")
SKILL = ("Match the data source, affected objects, conditions and intended output. An operation "
         "for a different object or effect is not a match. Do not choose a mere login or "
         "credential retrieval; those are dependencies. Do not choose an unrelated write.")
ENUM = re.compile(r"(?<!\\)\((?!\?)([A-Za-z][\w-]*(?:\|[A-Za-z][\w-]*)+)\)")


def display(name):
    return " ".join(name.split("_"))


def describe(name, spec):
    return (f"{display(name)} | {spec.get('effect') or ''} | inputs: "
            f"{', '.join(map(display, spec['inputs'])) or 'none'} | output: {display(spec['output'])}")


def candidates(fragments, instruction, prefilter):
    all_names = sorted(fragments)
    apps = {f["app"] for f in fragments.values()}
    words = " " + " ".join(re.findall(r"[a-z0-9]+", instruction.casefold())) + " "
    mentioned = {app for app in apps if " " + display(app).casefold() + " " in words}
    if not prefilter or not mentioned:
        return all_names, sorted(mentioned)
    keep = {n for n in all_names if fragments[n]["app"] in mentioned}
    while True:
        required = {r for n in keep for r in fragments[n]["inputs"]}
        expanded = keep | {n for n in all_names if fragments[n]["output"] in required}
        if keep == expanded:
            return sorted(keep), sorted(mentioned)
        keep = expanded


class CatalogExecutor(PhaseOneExecutor):
    def __init__(self, model, environment, *, prefilter, **kwargs):
        super().__init__(model, environment, **kwargs)
        self.prefilter = prefilter

    def select_root(self, names, instruction):
        fs = self.library["workflow"]["fragments"]
        def choose(group, node):
            rows = [{"label": describe(n, fs[n]), "value": n} for n in group]
            try:
                return self.decision(QUESTION, {"request": instruction}, rows, skill=SKILL, node=node)
            except Uncovered as exc:
                if str(exc) != "Jev selected the uncovered option":
                    raise
                return None
        if not names:
            return None
        # Native head accepts 15 real options plus NONE. Every candidate is shown.
        pages = math.ceil(len(names) / MAX_REAL_OPTIONS)
        size = math.ceil(len(names) / pages)
        winners = []
        for i in range(0, len(names), size):
            chosen = choose(names[i:i+size], "catalog/page/" + str(i // size + 1))
            if chosen is not None:
                winners.append(chosen)
        if len(winners) < 2:
            return next(iter(winners), None)
        return choose(winners, "catalog/final")

    def bind_inputs(self, name, task):
        fs = self.library["workflow"]["fragments"]
        route_defs = routed(self.library)
        args, required, visited, extractors = {}, [], set(), {}
        def visit(n):
            if n in visited:
                return
            visited.add(n)
            extractors.update(route_defs.get(n, {}).get("captures", {}))
            for role in fs[n]["inputs"]:
                providers = self.by_role.get(role, [])
                if len(providers) == 1:
                    visit(providers[0])
                elif not providers and role not in required:
                    required.append(role)
        visit(name)
        # Existing readers can extract spans such as note titles or plural contact groups.
        observed = {}
        for source in discover_consumers(self.library, task["instruction"]).values():
            observed.update(route_inputs(source, task, {}))
        for role in required:
            cap = extractors.get(role, {})
            finite = ENUM.findall(cap.get("pattern", "")) if cap.get("mode", "one") == "one" else []
            if len(finite) == 1 and re.compile(cap["pattern"]).groups == 1:
                choices = list(dict.fromkeys(finite[0].split("|")))
                args[role] = self.decision(f"Which value of '{display(role)}' does the request specify?",
                    {"request": task["instruction"], "operation": describe(name, fs[name]),
                     "already_bound": args}, [{"label": display(k), "value": k} for k in choices],
                    node="catalog/slot/"+role)
                via = "native enum"
            elif role in observed:
                args[role], via = observed[role], "existing public span reader"
            elif cap:
                args.update(route_inputs({"captures": {role: cap}}, task, {}))
                via = "existing capture"
            else:
                raise Uncovered("No existing parameter domain or public span for " + role)
            self.trace.add("catalog_binding", role=role, value=args[role], source=via)
        return args

    def assemble(self):
        # This experiment changes the first entry decision only. Residual work goes
        # to the same generation agent; the old exact-match routes never pick a root.
        if getattr(self, "catalog_started", False):
            return
        self.catalog_started = True
        self.assembling = True
        try:
            task = self.context["task"]
            fragments = self.library["workflow"]["fragments"]
            names, apps = candidates(fragments, task["instruction"], self.prefilter)
            self.trace.add("catalog_candidates", names=names, mentioned_apps=apps, prefilter=self.prefilter)
            name = self.select_root(names, task["instruction"])
            self.trace.add("catalog_choice", fragment=name)
            if name is None:
                return
            role = fragments[name]["output"]
            args = self.bind_inputs(name, task)
            if not self.activate("catalog/entry", {"goals":[role]}, args, name, native=True):
                return
            existing_rule = routed(self.library).get(name, {})
            terminal = existing_rule.get("complete", False) and existing_rule["goals"] == [role]
            self.trace.add("catalog_executed", fragment=name, terminal=terminal)
            if terminal and not self.done:
                self.call("supervisor", "complete_task", {"answer": self.live[role]} if role=="answer" else {},
                          "composition_finish")
        except Uncovered as exc:
            self.trace.add("catalog_uncovered", reason=str(exc))
        finally:
            self.assembling = False
