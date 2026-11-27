"""Independent AppWorld sessions backed by one official model service on each GPU."""
from datetime import datetime, timezone
from threading import Lock

from .artifacts import save
from .transport import Worker, WorkerError, WorkerUnavailable


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
            self.decode_mode = "cuda_graph" if "--cuda-graph" in args else "eager"

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


class DeviceWorkers:
    def __init__(self, root, connections, allocation, width):
        self.root = root
        self.allocation = allocation
        self.gpu = allocation["gpu"]
        self.width = width
        self.name = self.gpu.replace(":", "_")
        self.lock = Lock()
        self.owner = None
        self.info = None
        if width == 1:
            session = ExecutionWorkers(root, connections, self.gpu)
            session.lane = 0
            session.lanes_per_gpu = 1
            self.sessions = [session]
        else:
            self.sessions = [ModelSession(self, connections, lane) for lane in range(width)]
        spec = self.sessions[0].connections["model"]
        self.spec = {**spec, "args": [*spec["args"], "--max-num-seqs", str(width)]}

    def record_service(self, info):
        model_args = self.spec["args"]
        model_path = model_args[model_args.index("--model") + 1]
        if (info["backend"] != "official_vllm" or info["precision"] != "bf16" or info["merged"]
                or info["model"] != model_path or info.get("max_num_seqs", 1) != self.width
                or "--cuda-graph" in model_args and info.get("decode_mode") != "cuda_graph"):
            raise ValueError(f"Unexpected model or sequence limit on {self.gpu}")
        save(self.root / "workers" / (self.name + ".json"),
             {**info, "loaded_utc": datetime.now(timezone.utc).isoformat(),
              "allocation": self.allocation, "lanes_per_gpu": self.width})

    def start(self):
        if self.width == 1:
            self.record_service(self.sessions[0].worker("model").request("info"))
        else:
            self.endpoint()

    def endpoint(self):
        with self.lock:
            if self.owner is None:
                owner = Worker(self.spec, self.root / f"model.owner_{self.name}.stderr.log")
                try:
                    info = owner.request("info")
                    self.record_service(info)
                except BaseException:
                    owner.close()
                    raise
                self.owner = owner
                self.info = info
            return self.info["port"]

    def recover(self, port):
        # A simulator error or a lost client pipe does not imply an engine failure.
        with self.lock:
            if self.owner is None or self.info["port"] != port:
                return
            try:
                self.owner.request("health")
            except (WorkerUnavailable, WorkerError, OSError):
                self.close_owner()

    def close_owner(self):
        owner, self.owner, self.info = self.owner, None, None
        if owner is not None:
            try:
                owner.close()
            except (WorkerUnavailable, WorkerError, OSError):
                pass

    def close(self):
        for session in self.sessions:
            session.close()
        self.close_owner()


class ModelSession(ExecutionWorkers):
    def __init__(self, device, connections, lane):
        super().__init__(device.root / "workers" / device.name / f"lane_{lane}", connections, device.gpu)
        self.device = device
        self.lane = lane
        self.lanes_per_gpu = device.width
        self.port = None

    def worker(self, name):
        if name == "model":
            port = self.device.endpoint()
            if self.port != port:
                super().close_worker("model")
                self.port = port
                spec = self.device.spec
                self.connections["model"] = {**spec, "args": [*spec["args"], "--attach-port", str(port)]}
        return super().worker(name)

    def close_worker(self, name):
        super().close_worker(name)
        if name == "model" and self.port is not None:
            self.device.recover(self.port)
            self.port = None

    def close(self):
        for name in list(self.workers):
            super().close_worker(name)
