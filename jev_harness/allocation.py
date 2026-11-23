"""Run one model and one AppWorld simulator per GPU, with stable Train/Test pairs."""
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import os
import subprocess
import time

from formal_run import ExecutionWorkers, read, save
from phase1_run import PhaseOneStudy


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


class SharedStudy(PhaseOneStudy):
    def __init__(self, root, settings, connections, *, devices=(), max_gpus=4, gpu_name="A800"):
        super().__init__(root, settings, connections)
        self.workers = []
        self.connections = connections
        self.devices = tuple(devices)
        self.max_gpus = len(self.devices) or max_gpus
        self.gpu_name = gpu_name
        self.last_scan = float("-inf")
        self.pairing = False
        self.started_utc = utcnow()

    def metadata(self):
        save(self.root / "controller.json", {"pid": os.getpid(), "started_utc": self.started_utc,
             "devices": ",".join(w.gpu for w in self.workers), "max_devices": self.max_gpus})

    def load_worker(self, row):
        worker = ExecutionWorkers(self.root, self.connections, row["gpu"])
        try:
            model = worker.connections["model"]["args"]
            model_path = model[model.index("--model") + 1]
            info = worker.worker("model").request("info")
            if (info.get("backend") != "official_vllm" or info["model"] != model_path
                    or info["precision"] != "bf16" or info["merged"]):
                raise ValueError(f"Expected the official BF16 AutoTrust service on {worker.gpu}")
            save(self.root / "workers" / (worker.gpu.replace(":", "_") + ".json"),
                 {**info, "loaded_utc": utcnow(), "allocation": row})
            return worker, None
        except Exception as exc:
            worker.close()
            return None, str(exc)

    def maybe_expand(self):
        if len(self.workers) >= self.max_gpus or time.monotonic() - self.last_scan < SCAN_SECONDS:
            return
        self.last_scan = time.monotonic()
        active = {worker.gpu for worker in self.workers}
        inspection = {"observed_utc": utcnow(), "active": sorted(active)}
        if self.devices:
            candidates = [{"gpu": name} for name in self.devices if name not in active]
        else:
            hosts = self.connections["model_nodes"]
            with ThreadPoolExecutor(len(hosts)) as pool:
                nodes = dict(pool.map(inspect_node, hosts.items()))
            candidates = [row for rows in nodes.values() if isinstance(rows, list) for row in rows
                          if row["gpu"] not in active and self.gpu_name in row["name"]
                          and row["free_mib"] >= FREE_MIB and row["util_pct"] <= LOW_UTIL]
            candidates.sort(key=lambda row: (row["util_pct"], -row["free_mib"], row["gpu"]))
            inspection.update(nodes=nodes, min_free_mib=FREE_MIB, max_util_pct=LOW_UTIL)
        inspection["candidates"] = [row["gpu"] for row in candidates]
        take = candidates[:self.max_gpus - len(active)]
        path = self.root / "gpu_allocation.json"
        save(path, inspection)
        if not take:
            if not self.workers:
                self.status("waiting_for_gpus", max_devices=self.max_gpus)
            return
        if not self.workers:
            self.status("loading_27b", gpus=[row["gpu"] for row in take])
        with ThreadPoolExecutor(len(take)) as pool:
            for row, (worker, error) in zip(take, pool.map(self.load_worker, take)):
                if worker is None:
                    inspection.setdefault("load_errors", {})[row["gpu"]] = error
                else:
                    self.workers.append(worker)
        inspection["active"] = [w.gpu for w in self.workers]
        save(path, inspection)
        self.metadata()
        if self.devices and inspection.get("load_errors"):
            raise RuntimeError(f"Could not load requested GPU workers: {inspection['load_errors']}")

    def start_workers(self):
        self.metadata()
        while not self.workers:
            self.maybe_expand()
            if not self.workers:
                time.sleep(SCAN_SECONDS)

    def candidate_batch(self, *args, **kwargs):
        self.maybe_expand()
        self.pairing = True
        try:
            return super().candidate_batch(*args, **kwargs)
        finally:
            self.pairing = False

    def execute_jobs(self, jobs, operation):
        if self.pairing or len(self.workers) >= self.max_gpus or len(jobs) <= 1:
            return super().execute_jobs(jobs, operation)
        self.maybe_expand()
        iterator = iter(enumerate(jobs))
        results = [None] * len(jobs)
        remaining = len(jobs)
        with ThreadPoolExecutor(self.max_gpus) as pool:
            running = {}

            def assign(worker):
                nonlocal remaining
                item = next(iterator, None)
                if item is not None:
                    index, job = item
                    running[pool.submit(operation, job, worker)] = (index, worker)
                    remaining -= 1

            for worker in self.workers:
                assign(worker)
            while running:
                finished, _ = wait(running, timeout=SCAN_SECONDS, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, worker = running.pop(future)
                    results[index] = future.result()
                    assign(worker)
                if remaining and len(self.workers) < self.max_gpus:
                    before = len(self.workers)
                    self.maybe_expand()
                    for worker in self.workers[before:]:
                        assign(worker)
        return results

    def batch(self, tasks, stage, variant="B", library=None):
        rows = super().batch(tasks, stage, variant, library)
        if stage == "dev_h00":
            path = self.root / "dev_reference.json"
            reference = read(path)
            reference.update(mode="reuse_measured_parent", tasks=len(rows),
                             successes=sum(row["success"] for row in rows),
                             output_tokens=sum(row["output_tokens"] for row in rows))
            save(path, reference)
        return rows
