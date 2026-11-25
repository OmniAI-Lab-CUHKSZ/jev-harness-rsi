"""Schedule independent task sessions on shared GPUs with matched Train and Test pairs."""
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import os
import subprocess
import time

from formal_run import read, save
from phase1_run import PhaseOneStudy
from .inference_workers import DeviceWorkers


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
    def __init__(self, root, settings, connections, *, devices=(), max_gpus=4, gpu_name="A800", lanes_per_gpu=None):
        super().__init__(root, settings, connections)
        self.workers = []
        self.connections = connections
        self.devices = tuple(devices)
        self.max_gpus = len(self.devices) or max_gpus
        self.gpu_name = gpu_name
        self.groups = {}
        self.last_scan = float("-inf")
        self.started_utc = utcnow()
        self.fixed_width = lanes_per_gpu
        checkpoints = self.root / "checkpoints.json"
        completed = read(checkpoints) if checkpoints.is_file() else []
        self.lanes_per_gpu = self.width_for("train", len(completed) + 1)
        self.decode_mode = "eager"
        self.preferred_devices = ()
        if settings.get("cuda_graph_from_start"):
            self.enable_cuda_graph()
        after = settings.get("cuda_graph_after")
        if after is not None:
            previous = self.root / "controller.json"
            if previous.is_file():
                self.preferred_devices = tuple(read(previous).get("devices", "").split(","))
            if any(c.get("visit", f"b{c['batch']:02d}") == f"b{after:02d}" for c in completed):
                self.enable_cuda_graph()

    def enable_cuda_graph(self):
        if self.decode_mode == "cuda_graph":
            return
        warm = bool(self.groups)
        if warm:
            self.preferred_devices = tuple(self.groups)
            self.status("switching_decode_mode", decode_mode="cuda_graph")
            self.close_devices()
        self.connections = {**self.connections, "model_nodes": {
            node: {**spec, "args": [*spec.get("args", []), "--cuda-graph"]} if "--cuda-graph" not in spec.get("args", []) else spec
            for node, spec in self.connections["model_nodes"].items()}}
        self.decode_mode = "cuda_graph"
        self.last_scan = float("-inf")
        if warm:
            self.start_workers()

    def width_for(self, phase, batch):
        schedule = self.settings.get("inference_schedule", {})
        key = "test" if phase == "test" else "first_batch" if batch == 1 else "later_batches"
        width = self.fixed_width if self.fixed_width is not None else schedule.get(key, 1)
        if width not in (1, 2):
            raise ValueError(f"Expected one or two inference sessions per GPU, got {width}")
        return width

    def begin_inference_phase(self, phase, batch=None):
        width = self.width_for(phase, batch)
        if width != self.lanes_per_gpu:
            self.status("switching_inference_concurrency", phase=phase, batch=batch, lanes_per_gpu=width)
            self.close_devices()
            self.lanes_per_gpu = width
            self.last_scan = float("-inf")
            self.start_workers()

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
                          if row["gpu"] not in active and self.gpu_name in row["name"]
                          and row["free_mib"] >= FREE_MIB and row["util_pct"] <= LOW_UTIL]
            candidates.sort(key=lambda row: (row["gpu"] not in self.preferred_devices,
                                             row["util_pct"], -row["free_mib"], row["gpu"]))
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

    def candidate_batch(self, *args, **kwargs):
        self.maybe_expand()
        return super().candidate_batch(*args, **kwargs)

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
