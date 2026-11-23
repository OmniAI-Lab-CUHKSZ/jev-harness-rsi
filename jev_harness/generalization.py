"""Compare compiled structures before executing any held-out task."""
from collections import defaultdict

from .fragment_library import matching_routes


GROUPS = ("seen_structure", "new_composition", "partial_or_uncovered")


def planned_structure(workflow, instruction):
    routes = dict(matching_routes(workflow["rules"], instruction))
    goals = {goal for route in routes.values() for goal in route["goals"]}
    external = {role for route in routes.values() for role in route.get("inputs", {})}
    providers = {fragment["output"]: (name, fragment["inputs"])
                 for name, fragment in workflow["fragments"].items()}
    fragments, edges, missing, inputs = set(), set(), set(), set()

    def visit(role, visiting=()):
        if role in external:
            inputs.add(role)
            return
        if role in visiting or role not in providers:
            missing.add(role)
            return
        name, dependencies = providers[role]
        if name in fragments:
            return
        fragments.add(name)
        for dep in dependencies:
            visit(dep, (*visiting, role))
            if dep not in external and dep in providers:
                edges.add((providers[dep][0], name))

    for goal in sorted(goals):
        visit(goal)
    return {"rules": sorted(routes), "goals": sorted(goals), "fragments": sorted(fragments),
            "edges": sorted(edges), "input_roles": sorted(inputs), "missing_roles": sorted(missing),
            "complete_route": bool(fragments and not missing and any(r.get("complete") for r in routes.values()))}


def classify(library, train, test):
    workflow = library["workflow"]

    def signature(plan):
        return tuple(tuple(plan[key]) for key in ("goals", "fragments", "edges", "input_roles"))

    train_plans = defaultdict(list)
    seen_fragments = set()
    train_ids = {row["task_id"] for row in train}
    for row in train:
        plan = planned_structure(workflow, row["instruction"])
        if not plan["missing_roles"]:
            train_plans[signature(plan)].append(row["task_id"])
        seen_fragments.update(plan["fragments"])
    for name, spec in workflow["fragments"].items():
        if train_ids.intersection(spec.get("source_tasks", ())):
            seen_fragments.add(name)

    tasks = {}
    scenarios = defaultdict(list)
    for row in test:
        task = row["task_id"]
        plan = planned_structure(workflow, row["instruction"])
        matches = train_plans.get(signature(plan), [])
        group = "partial_or_uncovered"
        if plan["complete_route"]:
            if matches:
                group = "seen_structure"
            elif len(plan["fragments"]) >= 2 and seen_fragments.issuperset(plan["fragments"]):
                group = "new_composition"
        tasks[task] = {**plan, "group": group, "matching_train_tasks": matches}
        scenarios[task.rsplit("_", 1)[0]].append(group)

    return {"basis": "Final library and public Train/Test-N instructions, before any Test-N rollout",
            "scenario_rule": "Partial if any variant is partial, then new if any variant is new, otherwise seen",
            "tasks": tasks,
            "scenarios": {s: max(groups, key=GROUPS.index) for s, groups in scenarios.items()}}
