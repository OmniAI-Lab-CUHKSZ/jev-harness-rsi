"""Schedule independent task sessions on shared GPUs with matched Train and Test pairs."""
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import os
import subprocess
import time

from pathlib import Path
import json
import re
from threading import Lock

from .artifacts import include_retries, read, save
from .composed_assembly import ComposedAssembly
from .inference_workers import DeviceWorkers
from .native_assembly import NativeGeneration
from .sol_account import SolAccount
from .transport import WorkerError, WorkerUnavailable


SCAN_SECONDS = 60
FREE_MIB = 58000
LOW_UTIL = 30


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def inspect_node(entry):
    node, spec = entry
    command = ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]
    if "host" in spec:
        command = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", spec["identity"],
                   "-p", str(spec["port"]), spec["user"] + "@" + spec["host"], *command]
    try:
        output = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                check=True, timeout=18).stdout
        rows = [{"gpu": f"{node}:{index.strip()}", "name": name.strip(), "free_mib": int(free),
                 "total_mib": int(total), "util_pct": int(util)}
                for index, name, free, total, util in csv.reader(output.splitlines())]
        return node, rows
    except subprocess.CalledProcessError as exc:
        return node, {"unavailable": (exc.stderr or str(exc)).strip()[-240:]}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return node, {"unavailable": str(exc)}


class SharedStudy:
    compiled_executor = ComposedAssembly
    generation_executor = NativeGeneration

    def __init__(self, root, settings, connections, *, devices=(), max_gpus=6, lanes_per_gpu=2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.execution = settings["execution"]
        self.seed = settings["seed"]
        self.functions = []
        self.test_started = False
        self.account = None
        self.log_lock = Lock()
        self.workers = []
        self.devices = tuple(devices)
        self.max_gpus = len(self.devices) or max_gpus
        self.groups = {}
        self.last_scan = float("-inf")
        self.started_utc = utcnow()
        self.lanes_per_gpu = lanes_per_gpu
        self.decode_mode = "cuda_graph"
        self.connections = {
            **connections,
            "model_nodes": {
                node: spec if "--cuda-graph" in spec["args"] else {**spec, "args": [*spec["args"], "--cuda-graph"]}
                for node, spec in connections["model_nodes"].items()
            },
        }

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

    def paired_model_info(self):
        f = self.root/"execution_model.json"
        if f.is_file():
            return read(f)
        data = self.workers[0].worker("model").request("info")
        save(f, data)
        return data

    def record_public_catalog(self, stage):
        return not self.test_started and stage.startswith("b")

    def run_task(self, task, stage, variant, library, *, workers=None):
        # Each result is bound to one task, version and seed.
        folder = self.root/"runs"/stage/task
        record_file = folder/"record.json"
        if record_file.is_file():
            return include_retries(self.root, read(record_file))
        workers = workers or self.workers[0]
        device = {"gpu": workers.gpu, "lanes_per_gpu": workers.lanes_per_gpu, "lane": workers.lane,
                  "decode_mode": workers.decode_mode}
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
                executor_type = self.compiled_executor if variant == "B" else self.generation_executor
                executor = executor_type(workers.worker("model"), env, **self.execution)
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
                                      "generated_tokens": result["output_tokens"],
                                      "api_calls": result["api_calls"], "seconds": round(result["seconds"], 2),
                                      **device}, ensure_ascii=False), flush=True)
                return include_retries(self.root, result)
            except (WorkerUnavailable, WorkerError) as exc:
                save(part/"infrastructure.json", {"error": str(exc), "attempt": attempt})
                workers.close_worker("environment")
                workers.close_worker("model")
                if attempt == first_attempt+1:
                    raise

    def metadata(self):
        save(self.root / "controller.json", {"pid": os.getpid(), "started_utc": self.started_utc,
             "devices": ",".join(self.groups), "max_devices": self.max_gpus,
             "lanes_per_gpu": self.lanes_per_gpu, "task_slots": len(self.workers), "decode_mode": self.decode_mode})

    def load_worker(self, row):
        device = DeviceWorkers(self.root, self.connections, row, self.lanes_per_gpu)
        try:
            device.start()
            return device, None
        except Exception as exc:
            device.close()
            return None, str(exc)

    def maybe_expand(self):
        if len(self.groups) >= self.max_gpus or time.monotonic() - self.last_scan < SCAN_SECONDS:
            return
        self.last_scan = time.monotonic()
        active = set(self.groups)
        inspection = {"observed_utc": utcnow(), "active": sorted(active), "lanes_per_gpu": self.lanes_per_gpu}
        if self.devices:
            candidates = [{"gpu": name} for name in self.devices if name not in active]
        else:
            hosts = self.connections["model_nodes"]
            with ThreadPoolExecutor(len(hosts)) as pool:
                nodes = dict(pool.map(inspect_node, hosts.items()))
            candidates = [row for rows in nodes.values() if isinstance(rows, list) for row in rows
                          if row["gpu"] not in active and "A800" in row["name"]
                          and row["free_mib"] >= FREE_MIB and row["util_pct"] <= LOW_UTIL]
            candidates.sort(key=lambda row: (row["util_pct"], -row["free_mib"], row["gpu"]))
            inspection.update(nodes=nodes, min_free_mib=FREE_MIB, max_util_pct=LOW_UTIL)
        inspection["candidates"] = [row["gpu"] for row in candidates]
        take = candidates[:self.max_gpus - len(active)]
        path = self.root / "gpu_allocation.json"
        save(path, inspection)
        if not take:
            if not self.groups:
                self.status("waiting_for_gpus", max_devices=self.max_gpus)
            return
        if not self.groups:
            self.status("loading_27b", gpus=[row["gpu"] for row in take], lanes_per_gpu=self.lanes_per_gpu)
        with ThreadPoolExecutor(len(take)) as pool:
            for row, (device, error) in zip(take, pool.map(self.load_worker, take)):
                if device is None:
                    inspection.setdefault("load_errors", {})[row["gpu"]] = error
                else:
                    self.groups[device.gpu] = device
        # Use distinct GPUs first when fewer tasks than sessions are ready.
        self.workers = [device.sessions[lane] for lane in range(self.lanes_per_gpu) for device in self.groups.values()]
        inspection["active"] = list(self.groups)
        save(path, inspection)
        self.metadata()
        if self.devices and inspection.get("load_errors"):
            raise RuntimeError(f"Could not load requested GPU workers: {inspection['load_errors']}")

    def start_workers(self):
        self.metadata()
        while not self.groups:
            self.maybe_expand()
            if not self.groups:
                time.sleep(SCAN_SECONDS)

    def execute_jobs(self, jobs, operation):
        return self.execute_preferred(jobs, operation, None)

    def execute_preferred(self, jobs, operation, preferred_gpus):
        if not jobs:
            return []
        self.maybe_expand()
        pending = list(range(len(jobs)))
        results = [None] * len(jobs)
        active = set(self.groups)
        affinity = []
        if preferred_gpus:
            for gpu in preferred_gpus:
                if gpu in active:
                    affinity.append({gpu})
                else:
                    node = (gpu or "").rpartition(":")[0]
                    affinity.append({key for key in active if key.rpartition(":")[0] == node})

        def next_job(worker):
            if affinity:
                for index in pending:
                    if worker.gpu in affinity[index]:
                        return index
                return next((index for index in pending if not affinity[index]), None)
            return pending[0] if pending else None

        with ThreadPoolExecutor(self.max_gpus * self.lanes_per_gpu) as pool:
            running = {}
            busy = set()
            while pending or running:
                if pending and not affinity and len(self.groups) < self.max_gpus:
                    self.maybe_expand()
                for worker in self.workers:
                    if not pending or worker in busy:
                        continue
                    index = next_job(worker)
                    if index is not None:
                        pending.remove(index)
                        busy.add(worker)
                        running[pool.submit(operation, jobs[index], worker)] = (index, worker)
                if not running:
                    raise RuntimeError("No active GPU session for the pending tasks")
                finished, _ = wait(running, timeout=SCAN_SECONDS, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, worker = running.pop(future)
                    busy.remove(worker)
                    results[index] = future.result()
        return results

    def close_devices(self):
        devices = list(self.groups.values())
        try:
            if devices:
                with ThreadPoolExecutor(len(devices)) as pool:
                    list(pool.map(lambda device: device.close(), devices))
        finally:
            self.groups.clear()
            self.workers = []

    def close(self):
        try:
            self.close_account()
        finally:
            self.close_devices()
