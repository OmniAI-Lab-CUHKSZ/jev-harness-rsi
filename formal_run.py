"""Resumable, two-arm AppWorld experiment. Only Train traces reach Sol."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
from threading import Event, Lock
import time

from jev_harness.formal_runtime import FormalExecutor
from jev_harness.fragment_library import FragmentExecutor
from jev_harness.generalization import GROUPS
from jev_harness.metrics import aggregate, comparison
from jev_harness.sol_account import AccountError, SolAccount, SolDemonstrator
from jev_harness.transport import Worker, WorkerError, WorkerUnavailable


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    temp.replace(path)


def scenario(task):
    return task.rsplit("_", 1)[0]


def group_tasks(keys):
    groups = defaultdict(list)
    for task in keys:
        groups[scenario(task)].append(task)
    return dict(groups)


class ExecutionWorkers:
    """One model copy and one simulator, used by one task at a time."""

    def __init__(self, root, connections, gpu):
        self.root = root
        self.connections = dict(connections)
        self.gpu = gpu
        self.workers = {}
        if gpu is not None:
            node, separator, device = gpu.rpartition(":")
            model = dict(connections["model_nodes"][node] if separator else connections["model"])
            args = list(model.get("args", ()))
            if "--gpu" in args:
                args[args.index("--gpu")+1] = device
            else:
                args.extend(["--gpu", device])
            self.connections["model"] = {**model, "args": args}

    def worker(self, name):
        if name not in self.workers:
            suffix = f".gpu_{self.gpu.replace(':', '_')}" if self.gpu is not None else ""
            self.workers[name] = Worker(self.connections[name], self.root/f"{name}{suffix}.stderr.log")
        return self.workers[name]

    def close_worker(self, name):
        worker = self.workers.pop(name, None)
        if worker:
            try:
                worker.close()
            except (WorkerUnavailable, WorkerError, OSError):
                pass

    def close(self):
        for name in list(self.workers):
            self.close_worker(name)


class Study:
    compiled_executor = FragmentExecutor

    def __init__(self, root, settings, connections, gpus=()):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.workers = [ExecutionWorkers(self.root, connections, gpu) for gpu in (gpus or (None,))]
        self.log_lock = Lock()
        self.account = None
        self.execution = settings["execution"]
        self.seed = settings["seed"]
        self.functions = []
        self.test_started = False

    def status(self, stage, **values):
        data = {"stage": stage, **values, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self.log_lock:
            save(self.root/"status.json", data)
            print(json.dumps(data, ensure_ascii=False, default=str), flush=True)

    def solver(self):
        if self.test_started:
            raise RuntimeError("No optimizer call is allowed during the held-out test")
        if self.account is None:
            # Keep the Sol working area separate from Dev/Test artifacts.
            self.account = SolAccount(self.root/"offline", self.root/"offline")
        return self.account

    def close_account(self):
        if self.account:
            self.account.close()
            self.account = None

    def close(self):
        self.close_account()
        for workers in self.workers:
            workers.close()

    def task_rows(self, split, *, include_instructions=False):
        result = self.workers[0].worker("environment").request("list_tasks", split=split, include_instructions=include_instructions)
        ids = [row["task_id"] for row in result["tasks"]]
        if result["listed"] != len(ids) or len(set(ids)) != len(ids) or any(len(xs) != 3 for xs in group_tasks(ids).values()):
            raise RuntimeError(f"Incomplete AppWorld scenario groups in {split}")
        return result["tasks"]

    def split(self, split):
        return [row["task_id"] for row in self.task_rows(split)]

    def paired_model_info(self):
        f = self.root/"execution_model.json"
        if f.is_file():
            return read(f)
        data = self.workers[0].worker("model").request("info")
        save(f, data)
        return data

    def record_public_catalog(self, stage):
        return not self.test_started

    def run_task(self, task, stage, variant, library=None, *, workers=None, compiled_executor=None):
        # A may serve multiple stages but each result remains bound to one task/seed.
        folder = self.root/"runs"/stage/task
        record_file = folder/"record.json"
        if record_file.is_file():
            return read(record_file)
        if variant == "Sol" and self.test_started:
            raise RuntimeError("Cannot run a Sol demonstration on Test")
        workers = workers or self.workers[0]
        device = {"gpu": workers.gpu} if variant != "Sol" and workers.gpu is not None else {}
        # Preserve partial receipts and use a fresh AppWorld run name on resume.
        prior_attempts = [int(p.name.split("_")[-1]) for p in folder.glob("attempt_*")
                          if p.name.split("_")[-1].isdigit()]
        first_attempt = max(prior_attempts, default=0) + 1
        for attempt in range(first_attempt, first_attempt+2):
            part = folder/f"attempt_{attempt}"
            folder.mkdir(parents=True, exist_ok=True)
            self.status(stage, task=task, variant=variant, attempt=attempt, **device)
            try:
                env = workers.worker("environment")
                if variant == "Sol":
                    executor = SolDemonstrator(self.solver(), env, folder=part/"model", **self.execution)
                else:
                    model = workers.worker("model")
                    executor_type = (compiled_executor or self.compiled_executor) if variant == "B" else FormalExecutor
                    executor = executor_type(model, env, **self.execution)
                run_name = re.sub(r"[^A-Za-z0-9_]", "_", self.root.name+"_"+stage+f"_{attempt}")
                trace = part/"trace.jsonl"
                result = executor.run(task, library, trace, self.seed, run_name)
                with self.log_lock:
                    if not self.functions and self.record_public_catalog(stage):
                        self.functions = executor.context["task"]["functions"]
                        save(self.root/"offline"/"public_functions.json", self.functions)
                result["trace_path"] = str(trace.relative_to(self.root))
                result["variant"] = variant
                result.update(device)
                save(record_file, result)
                with self.log_lock:
                    print(json.dumps({"stage": stage, "task": task, "variant": variant, "success": result["success"],
                                      "generated_tokens": result.get("outputTokens", result["output_tokens"]),
                                      "api_calls": result["api_calls"], "seconds": round(result["seconds"], 2),
                                      **device}, ensure_ascii=False), flush=True)
                return result
            except (WorkerUnavailable, WorkerError) as exc:
                save(part/"infrastructure.json", {"error": str(exc), "attempt": attempt})
                if isinstance(exc, AccountError):
                    self.close_account()
                    # A quota outage is not an AppWorld failure and should not burn another quota attempt.
                    if re.search(r"rate.?limit|usage limit|quota|sign.?in|not authorized|insufficient", str(exc), re.I):
                        raise
                workers.close_worker("environment")
                if not isinstance(exc, AccountError):
                    workers.close_worker("model")
                if attempt == first_attempt+1:
                    raise

    def execute_jobs(self, jobs, run):
        if len(self.workers) == 1 or len(jobs) <= 1:
            return [run(job, self.workers[0]) for job in jobs]
        pending = iter(enumerate(jobs))
        dispatch = Lock()
        stopped = Event()
        results = [None] * len(jobs)

        def execute_on(workers):
            while True:
                with dispatch:
                    item = None if stopped.is_set() else next(pending, None)
                if item is None:
                    return
                index, job = item
                try:
                    results[index] = run(job, workers)
                except BaseException:
                    stopped.set()
                    raise

        with ThreadPoolExecutor(max_workers=min(len(jobs), len(self.workers))) as pool:
            futures = [pool.submit(execute_on, workers) for workers in self.workers[:len(jobs)]]
            for future in futures:
                future.result()
        return results

    def batch(self, tasks, stage, variant="B", library=None):
        if variant == "Sol":
            rows = [self.run_task(task, stage, variant, library) for task in tasks]
            summary = {"tasks": len(rows), "successes": sum(r["success"] for r in rows),
                       "offline_output_tokens": sum(r.get("outputTokens", 0) for r in rows)}
        else:
            rows = self.execute_jobs(tasks, lambda task, workers: self.run_task(task, stage, variant, library, workers=workers))
            summary = aggregate(rows)
        save(self.root/"runs"/stage/"aggregate.json", summary)
        return rows

    def candidate_batch(self, tasks, stage, proposed, parent, library):
        devices = {w.gpu: w for w in self.workers if w.gpu and ":" in w.gpu}
        if not devices:
            return parent, self.batch(tasks, stage, library=proposed)
        primary_node = next(iter(devices)).split(":", 1)[0]
        previous = {row["task_id"]: row for row in parent}
        reference_stage = stage+"_matched_parent"

        def wrong_node(row):
            original = previous[row["task_id"]]
            old_node = (original.get("gpu") or "").rpartition(":")[0] or primary_node
            device = row.get("gpu", "")
            return device in devices and device.split(":", 1)[0] != old_node

        def run_pair(task, workers):
            row = self.run_task(task, stage, "B", proposed, workers=workers)
            if row.get("gpu") == workers.gpu and wrong_node(row):
                previous[task] = self.run_task(task, reference_stage, "B", library, workers=workers)
            return row

        candidate = self.execute_jobs(tasks, run_pair)
        # A completed candidate can be resumed on a different worker than its original card.
        reruns = defaultdict(list)
        for row in candidate:
            if wrong_node(row):
                reruns[row["gpu"]].append(row["task_id"])

        def replay(item):
            device, tasks = item
            return [self.run_task(task, reference_stage, "B", library, workers=devices[device]) for task in tasks]

        if reruns:
            with ThreadPoolExecutor(max_workers=len(reruns)) as pool:
                for rows in pool.map(replay, reruns.items()):
                    previous.update((row["task_id"], row) for row in rows)
        save(self.root/"runs"/stage/"aggregate.json", aggregate(candidate))
        return [previous[row["task_id"]] for row in parent], candidate

    def paths(self, rows):
        return [self.root/r["trace_path"] for r in rows]

def quality(rows):
    return (sum(r["success"] for r in rows), -sum(r["output_tokens"] for r in rows))


def final_report(a, b, seed, labels=None):
    methods = (a, b)
    for rows in methods:
        if len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate task in formal report")
    score = comparison(a, b)
    group_a = group_tasks(r["task_id"] for r in a)
    lookup_a, lookup_b = ({r["task_id"]: r for r in rows} for rows in methods)
    pairs = []
    for ids in group_a.values():
        src, dst = ([lookup[t] for t in ids] for lookup in (lookup_a, lookup_b))
        pairs.append((sum(r["success"] for r in dst)-sum(r["success"] for r in src),
                      int(all(r["success"] for r in dst))-int(all(r["success"] for r in src)),
                      sum(r["output_tokens"] for r in dst)-sum(r["output_tokens"] for r in src)))
    n = len(pairs)
    rng = random.Random(seed)
    samples = [[] for _ in range(3)]
    for _ in range(10000):
        chosen = rng.choices(pairs, k=n)
        for k, vals in enumerate(zip(*chosen)):
            samples[k].append(sum(vals)/(n if k == 1 else len(a)))
    intervals = {key: [sorted(row)[249], sorted(row)[9749]] for key, row in zip(
        ("TGC_B_minus_A", "SGC_B_minus_A", "generated_tokens_per_task_B_minus_A"), samples)}
    def more(rows):
        return {"api_calls": sum(r["api_calls"] for r in rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "tasks_with_fragments": sum(bool(r.get("fragments")) for r in rows),
                "tasks_with_native_decision": sum(bool(r["model_calls"].get("D")) for r in rows),
                "tasks_with_generation": sum(any(n for m,n in r["model_calls"].items()
                                                 if m.startswith("G") or m=="R") for r in rows),
                "tasks_with_task_fallback": sum(bool(r.get("fallback")) for r in rows),
                "local_recoveries": sum(r.get("local_recoveries", 0) for r in rows),
                "multi_object_decisions": sum(r.get("decisions_with_multiple_objects", 0) for r in rows),
                "model_seconds": sum(r.get("model_seconds", 0) for r in rows),
                "mode_calls": {mode: sum(r["model_calls"].get(mode, 0) for r in rows)
                               for mode in sorted({m for r in rows for m in r["model_calls"]})}}
    report = {"A": {**score["parent"], **more(a)}, "B": {**score["candidate"], **more(b)},
            "differences": {"TGC_B_minus_A": score["accuracy_delta"],
                            "SGC_B_minus_A": score["candidate"]["sgc"]-score["parent"]["sgc"],
                            "generated_tokens_per_task_B_minus_A": score["candidate"]["mean_output_tokens"] - score["parent"]["mean_output_tokens"],
                            "relative_reduction": score["output_token_reduction"]},
            "paired_scenario_bootstrap_95_intervals": intervals,
            "failed_to_passed": [t for t in lookup_a if not lookup_a[t]["success"] and lookup_b[t]["success"]],
            "passed_to_failed": [t for t in lookup_a if lookup_a[t]["success"] and not lookup_b[t]["success"]]}
    if labels is not None:
        breakdown = {}
        for name in GROUPS:
            ids = [task for task in lookup_a if labels["scenarios"][scenario(task)] == name]
            row = {"tasks": len(ids), "scenarios": len(group_tasks(ids))}
            if ids:
                left, right = ([lookup[t] for t in ids] for lookup in (lookup_a, lookup_b))
                paired = comparison(left, right)
                row.update(A={**paired["parent"], **more(left)}, B={**paired["candidate"], **more(right)},
                           accuracy_delta=paired["accuracy_delta"], output_token_reduction=paired["output_token_reduction"])
            breakdown[name] = row
        report["by_generalization"] = breakdown
    return report
