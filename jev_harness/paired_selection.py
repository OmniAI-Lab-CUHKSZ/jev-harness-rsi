"""Conditional, task-paired confirmation for Train and Dev selection."""
from itertools import zip_longest
from pathlib import Path
from threading import Lock

from .artifacts import read, read_events, save


TWO_REPEATS_ONLY = "two_repeats_only"


def _actions(events):
    for event in events:
        kind = event["event"]
        if kind == "model":
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
        elif kind == "tool_batch":
            response = event["response"]
            yield "tool", (event["node"], event["code"], response.get("output"), response.get("done"))
        elif kind == "api_action":
            yield "api", (event.get("node"), event["method"], event["url"], event.get("data"))
        elif kind == "api":
            response = event["response"]
            yield "public_api", (event["app"], event["api"], event["arguments"],
                                 response.get("value", response.get("output")), response.get("done"),
                                 response.get("api_total"), response.get("api_budget_hit"))
        elif kind in {"native_operation", "native_observation", "native_effect", "native_feedback",
                                 "native_discovery", "native_fragment_choice", "native_card_exit"}:
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


def review_scores(pairs, repeats):
    scores = [[0, 0], [0, 0]]
    for task, row in pairs.items():
        for arm, score in enumerate(scores):
            success, tokens = row["success"][arm], row["tokens"][arm]
            if task in repeats:
                observations = [(int(r["success"]), r["output_tokens"]) for r in repeats[task][arm::2]]
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


def cost_review(pairs, repeats):
    parent, candidate = review_scores(pairs, repeats)
    if parent[0] != candidate[0]:
        return []
    pending = [t for t,r in pairs.items() if t not in repeats and r["same_request_fork"]
               and r["tokens"][0] != r["tokens"][1]]
    saving = candidate[1] - parent[1]
    residual = saving - sum(pairs[t]["tokens"][0] - pairs[t]["tokens"][1] for t in pending)
    def sign(value):
        return (value > 0) - (value < 0)
    return pending if sign(saving) != sign(residual) else []


def select_with_review(study, stage, report_path, parent, candidate, paired_confirmations):
    pairs = initial_comparison(study.root, parent, candidate)
    repeats = {}
    first_seed = {r["task_id"]: r["seed"] for r in parent}
    bounds, accuracy_tasks = accuracy_review(pairs)
    report = {"stage": stage, "confirmation_scoring": TWO_REPEATS_ONLY, "initial_scores": review_scores(pairs, {}),
              "accuracy_difference_bounds": bounds, "initial_pairs": pairs,
              "accuracy_review_tasks": accuracy_tasks, "cost_review_tasks": [],
              "repeats": repeats}
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
            save(report_path, report)

    def confirm(tasks):
        if not tasks:
            return
        save(report_path, report)
        study.status("paired_selection_review", comparison=stage, tasks=tasks, confirmation_schedule="per_task")
        paired_confirmations(tasks, record_pair)

    confirm(accuracy_tasks)
    report["cost_review_tasks"] = cost_review(pairs, repeats)
    confirm(report["cost_review_tasks"])
    left, right = review_scores(pairs, repeats)
    report.update(parent_score=left, candidate_score=right, accepted=right >= left,
                  repeat_rollouts=sum(map(len, repeats.values())),
                  repeat_output_tokens=sum(r["output_tokens"] for pair in repeats.values() for r in pair))
    save(report_path, report)
    return {k:v for k,v in report.items() if k != "initial_pairs"}


def _prepare_pair(study, ids, stages, variants, libraries, *, reverse_order=False):
    """Prepare one same-GPU pair per task, including any complete or partial receipts."""
    cached = {}
    if hasattr(study, "root"):
        runs = study.root / "runs"
        for task in ids:
            cached[task] = {"A": {}, "B": {}}
            for arm, stage in stages.items():
                for folder in [runs / stage, *sorted(runs.glob(stage + "_paired_*"))]:
                    record = folder / task / "record.json"
                    if record.is_file():
                        cached[task][arm][read(record).get("gpu")] = folder.name

    def pair(job, worker):
        i,task=job
        out={}
        previous = cached.get(task, {"A": {}, "B": {}})
        device = getattr(worker, "gpu", None)
        completed = next((gpu for gpu in previous["A"] if gpu in previous["B"]), None)
        parent_first = (i % 2 == 0) != reverse_order
        for arm in (("A","B") if parent_first else ("B","A")):
            stage = previous[arm].get(completed or device, stages[arm])
            if device and not completed and device not in previous[arm] and stages[arm] in previous[arm].values():
                stage += "_paired_" + device.replace(":", "_")
            out[arm]=study.run_task(task,stage,variants[arm],libraries[arm],workers=worker)
        return out["A"],out["B"]
    preferred = [next((gpu for arm in cached.get(task, {}).values() for gpu in arm), None) for task in ids]
    return pair, preferred


def _dispatch_pairs(study, ids, operation, preferred):
    jobs = list(enumerate(ids))
    if hasattr(study, "execute_preferred"):
        return study.execute_preferred(jobs, operation, preferred if any(preferred) else None)
    return study.execute_jobs(jobs, operation)


def paired_rollouts(study, ids, stages, variants, libraries, *, preferred_gpus=None, reverse_order=False):
    """Alternate arm order and run each task twice on one GPU in freshly prepared worlds."""
    pair, preferred = _prepare_pair(study, ids, stages, variants, libraries, reverse_order=reverse_order)
    if preferred_gpus:
        preferred = [gpu or preferred_gpus.get(task) for task, gpu in zip(ids, preferred, strict=True)]
    pairs = _dispatch_pairs(study, ids, pair, preferred)
    return [a for a,b in pairs],[b for a,b in pairs]


def review_versions(study, stage, path, parents, candidates, parent_lib, candidate_lib):
    stages = {"A": stage+"_review_parent", "B": stage+"_review_candidate"}
    variants, libraries = {"A":"B","B":"B"}, {"A":parent_lib,"B":candidate_lib}
    def two_passes(tasks, record_pair):
        second_stages = {arm:name+"_2" for arm, name in stages.items()}
        first, first_gpus = _prepare_pair(study, tasks, stages, variants, libraries)
        second, second_gpus = _prepare_pair(study, tasks, second_stages, variants, libraries, reverse_order=True)
        # A resumed second pass may already have one arm on a different GPU.
        preferred = [later or earlier for earlier, later in zip(first_gpus, second_gpus, strict=True)]
        def run_both(job, worker):
            _, task = job
            record_pair(task, 1, *first(job, worker))
            record_pair(task, 2, *second(job, worker))
        _dispatch_pairs(study, tasks, run_both, preferred)

    return select_with_review(study, stage, path, parents, candidates, two_passes)
