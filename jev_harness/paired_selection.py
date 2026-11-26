"""Conditional, task-paired confirmation for Train and Dev selection."""
from itertools import zip_longest
from pathlib import Path
from threading import Lock

from formal_run import save
from .formal_optimizer import read_events


INITIAL_PLUS_ONE = "initial_plus_one"
TWO_REPEATS_ONLY = "two_repeats_only"


def _actions(events):
    native = False
    for event in events:
        kind = event["event"]
        if kind == "native_protocol":
            native = True
        elif kind == "model":
            request, response = event["request"], event["response"]
            if event["operation"] == "generate":
                yield "generate", request, response["text"]
            elif event["operation"] == "decide":
                index = response["index"]
                options = request["options"]
                yield "choose", request, options[index] if 0 <= index < len(options) else index
        elif kind == "decision":
            selected = event["candidates"][event["selected"]]
            value = selected.get("value", selected.get("label")) if isinstance(selected, dict) else selected
            yield "decision", (event["node"], value)
        elif kind == "card_executed":
            yield "card", (event["card"], event.get("inputs"), event.get("output"))
        elif kind == "tool_batch":
            response = event["response"]
            yield "tool", (event["node"], event["code"], response.get("output"), response.get("done"))
        elif kind == "api_action":
            yield "api", (event.get("node"), event["method"], event["url"], event.get("data"))
        elif native and kind == "api":
            response = event["response"]
            yield "public_api", (event["app"], event["api"], event["arguments"],
                                 response.get("value", response.get("output")), response.get("done"),
                                 response.get("api_total"), response.get("api_budget_hit"))
        elif native and kind in {"native_operation", "native_observation", "native_effect", "native_feedback",
                                 "native_discovery", "native_fragment_choice"}:
            yield kind, event


def first_fork(parent_events, candidate_events):
    """Locate a changed output on the same request before any changed action."""
    generation = 0
    for left, right in zip_longest(_actions(parent_events), _actions(candidate_events)):
        if left is None or right is None or left[0] != right[0]:
            break
        if left[0] == "choose":
            # A menu may change while Jev continues to pick the same action.
            if left[2] != right[2]:
                return "decision" if left[1] == right[1] else None
        elif left[0] == "generate":
            generation += 1
            if left[1] != right[1]:
                break
            if left[2] != right[2]:
                return f"generation_{generation}"
        elif left != right:
            break
    return None


def initial_comparison(root, parent, candidate):
    parent_by_id = {r["task_id"]: r for r in parent}
    candidate_by_id = {r["task_id"]: r for r in candidate}
    if (len(parent_by_id) != len(parent) or len(candidate_by_id) != len(candidate)
            or parent_by_id.keys() != candidate_by_id.keys() or not parent):
        raise ValueError("Selection requires both complete panels on the same distinct tasks")

    pairs = {}
    for task, left in parent_by_id.items():
        right = candidate_by_id[task]
        if left["seed"] != right["seed"]:
            raise ValueError(f"Selection seeds differ on {task}")
        scores = [int(r["success"]) for r in (left, right)]
        tokens = [r["output_tokens"] for r in (left, right)]
        fork = None
        if scores[0] != scores[1] or tokens[0] != tokens[1]:
            fork = first_fork(*(read_events(root / Path(r["trace_path"].replace("\\", "/")))
                                for r in (left, right)))
        pairs[task] = {"success": scores, "tokens": tokens, "same_request_fork": fork}
    return pairs


def review_scores(pairs, repeats, scoring=INITIAL_PLUS_ONE):
    scores = [[0, 0], [0, 0]]
    for task, row in pairs.items():
        for arm, score in enumerate(scores):
            success, tokens = row["success"][arm], row["tokens"][arm]
            if task in repeats:
                observations = [(int(r["success"]), r["output_tokens"]) for r in repeats[task][arm::2]]
                if scoring == INITIAL_PLUS_ONE:
                    observations.insert(0, (success, tokens))
                first, second = observations
                success = (first[0] + second[0]) / 2
                tokens = (first[1] + second[1]) / 2
            score[0] += success
            score[1] -= tokens
    return scores


def accuracy_review(pairs):
    flips = [task for task, r in pairs.items() if r["success"][0] != r["success"][1]]
    sensitive = [t for t in flips if pairs[t]["same_request_fork"]]
    fixed = sum(pairs[t]["success"][1] - pairs[t]["success"][0] for t in flips if t not in sensitive)
    bounds = [fixed - len(sensitive), fixed + len(sensitive)]
    tasks = flips if sensitive and bounds[0] <= 0 <= bounds[1] else []
    return bounds, tasks


def cost_review(pairs, repeats, scoring=INITIAL_PLUS_ONE):
    parent, candidate = review_scores(pairs, repeats, scoring)
    if parent[0] != candidate[0]:
        return []
    pending = [t for t,r in pairs.items() if t not in repeats and r["same_request_fork"]
               and r["tokens"][0] != r["tokens"][1]]
    saving = candidate[1] - parent[1]
    residual = saving - sum(pairs[t]["tokens"][0] - pairs[t]["tokens"][1] for t in pending)
    def sign(value):
        return (value > 0) - (value < 0)
    return pending if sign(saving) != sign(residual) else []


def select_with_review(study, stage, report_path, parent, candidate, paired_rerun, *, paired_confirmations=None):
    pairs = initial_comparison(study.root, parent, candidate)
    scoring = study.settings.get("confirmation_scoring", INITIAL_PLUS_ONE)
    passes = {INITIAL_PLUS_ONE: 1, TWO_REPEATS_ONLY: 2}[scoring]
    repeats = {}
    first_seed = {r["task_id"]: r["seed"] for r in parent}
    bounds, accuracy_tasks = accuracy_review(pairs)
    report = {"stage": stage, "confirmation_scoring": scoring, "initial_scores": review_scores(pairs, {}),
              "accuracy_difference_bounds": bounds, "initial_pairs": pairs,
              "accuracy_review_tasks": accuracy_tasks, "cost_review_tasks": [],
              "repeats": repeats}
    per_task = passes == 2 and paired_confirmations is not None
    if per_task:
        report["confirmation_schedule"] = "per_task"
    report_lock = Lock()

    def record_pair(task, pass_number, left, right):
        if (left["task_id"] != task or right["task_id"] != task or left["seed"] != right["seed"]
                or left["seed"] != first_seed[task] or left["gpu"] != right["gpu"]):
            raise ValueError(f"Unmatched confirmation for {task}")
        with report_lock:
            observations = repeats.setdefault(task, [])
            for arm, row in (("parent", left), ("candidate", right)):
                sample = {k:row[k] for k in ("success", "output_tokens", "gpu", "trace_path")}
                observations.append(dict(sample, arm=arm, confirmation_pass=pass_number))
            if per_task:
                save(report_path, report)

    def confirm(tasks):
        if not tasks:
            return
        if per_task:
            save(report_path, report)
            study.status("paired_selection_review", comparison=stage, tasks=tasks, confirmation_schedule="per_task")
            paired_confirmations(tasks, record_pair)
            return
        for pass_number in range(1, passes + 1):
            save(report_path, report)
            study.status("paired_selection_review", comparison=stage, tasks=tasks, confirmation_pass=pass_number)
            arms = paired_rerun(tasks, pass_number) if passes == 2 else paired_rerun(tasks)
            for task, left, right in zip(tasks, *arms, strict=True):
                record_pair(task, pass_number, left, right)

    confirm(accuracy_tasks)
    report["cost_review_tasks"] = cost_review(pairs, repeats, scoring)
    confirm(report["cost_review_tasks"])
    left, right = review_scores(pairs, repeats, scoring)
    report.update(parent_score=left, candidate_score=right, accepted=right >= left,
                  repeat_rollouts=sum(map(len, repeats.values())),
                  repeat_output_tokens=sum(r["output_tokens"] for pair in repeats.values() for r in pair))
    save(report_path, report)
    return {k:v for k,v in report.items() if k != "initial_pairs"}
