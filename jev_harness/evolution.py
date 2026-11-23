"""Local edits count functional components, and are tested before adoption."""
from copy import deepcopy
import json
from pathlib import Path

from .metrics import comparison

COMPONENTS = ("workflow", "evidence", "questions", "candidates", "decision_skills",
              "bindings", "generation_skills", "mode_policy")


def apply_patch(library, patch):
    candidate = deepcopy(library)
    matches = [h for h in candidate["workflows"] if h["id"] == patch["workflow_id"]]
    if "new_workflow" in patch:
        workflow = patch["new_workflow"]
        if "changes" in patch:
            raise ValueError("Evaluate a workflow registration separately from component edits")
        if matches or workflow["id"] != patch["workflow_id"]:
            raise ValueError("Registration requires one new workflow ID")
        candidate["workflows"].append(deepcopy(workflow))
        return candidate, ["workflow"]
    if len(matches) != 1:
        raise ValueError("Patch must identify one existing reusable workflow")
    workflow = matches[0]
    changes = patch["changes"]
    unknown = set(changes)-set(COMPONENTS)
    if unknown:
        raise ValueError("Unknown component: " + ", ".join(sorted(unknown)))
    changed = [name for name in changes if workflow.get(name) != changes[name]]
    if not 1 <= len(changed) <= 2:
        raise ValueError("Each modification must change one or two components")
    for name in changed:
        workflow[name] = deepcopy(changes[name])
    return candidate, changed


def evaluate_patch(library, patch, tasks, run, experience_path=None):
    candidate, components = apply_patch(library, patch)
    before, after = [], []
    for task in tasks:
        before.append(run(task, library, "parent"))
        after.append(run(task, candidate, "candidate"))
    result = {"workflow_id": patch["workflow_id"], "components": components,
              "operation": "register" if "new_workflow" in patch else "edit",
              "reason": patch.get("reason", ""), "evaluation": comparison(before, after),
              "parent_records": before, "candidate_records": after}
    if experience_path:
        path = Path(experience_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, default=str)+"\n")
    return (candidate if result["evaluation"]["beneficial"] else library), result
