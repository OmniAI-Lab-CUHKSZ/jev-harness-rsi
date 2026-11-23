from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from unittest import TestCase
from unittest.mock import Mock, patch

from formal_run import save
from jev_harness.allocation import SharedStudy
from jev_harness.history_workbench import HistoryWorkbench
from jev_harness.inference_workers import DeviceWorkers
from jev_harness.official_model_worker import OfficialModel
from jev_harness.transport import WorkerUnavailable
from phase1_run import paired_holdout


WORKER = Path(__file__).resolve().parents[1] / "jev_harness/official_model_worker.py"
SETTINGS = {"seed": 1, "execution": {}, "inference_schedule": {"first_batch": 1, "later_batches": 2, "test": 2}}
ENV = """import json,sys
task = None
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "prepare":
        task = request["task"]
    print(json.dumps({"id":request["id"], "result":{"task":task}}), flush=True)
    if request["op"] == "close":
        break
"""


def connections():
    return {"model_nodes": {"node": {"python": sys.executable, "script": str(WORKER),
            "args": ["--model", "fixture", "--service", "."]}},
            "environment": {"python": sys.executable, "script": "-c", "args": [ENV]}}


def owner(port):
    info = {"model": "fixture", "precision": "bf16", "merged": False, "backend": "official_vllm",
            "port": port, "max_num_seqs": 2, "service_pid": 7}
    return Mock(request=Mock(side_effect=lambda op: info if op == "info" else {"status": 200}))


class InferenceConcurrencyTests(TestCase):
    def test_model_owner_configures_the_engine_for_the_requested_width(self):
        with TemporaryDirectory() as folder:
            profile = {"model": "autotrust/JEV-27B", "defaults": {"mix": 0.5}}
            with (patch("jev_harness.official_model_worker.tempfile.gettempdir", return_value=folder),
                  patch("jev_harness.official_model_worker.tempfile.TemporaryDirectory", return_value=Mock(name=folder)),
                  patch.object(os, "getuid", return_value=1, create=True),
                  patch("jev_harness.official_model_worker.subprocess.check_output", return_value="80000,81920"),
                  patch("jev_harness.official_model_worker.subprocess.Popen") as launch,
                  patch.object(OfficialModel, "http", return_value=profile)):
                launch.return_value.poll.return_value = None
                for width in (1, 2):
                    model = OfficialModel("fixture", "0", folder, 16384, width)
                    cmd = launch.call_args.args[0]
                    self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], str(width))
                    self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "16384")
                    self.assertEqual(cmd[cmd.index("--kv-cache-memory-bytes") + 1], str(3 * 1024**3))
                    self.assertEqual(model.info()["max_num_seqs"], width)
                self.assertEqual(launch.call_count, 2)

    def test_two_real_client_pipes_share_decision_service_and_isolate_simulators(self):
        simultaneous = Barrier(2)
        class Handler(BaseHTTPRequestHandler):
            def send_json(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send_json({"model": "autotrust/JEV-27B", "defaults": {"mix": 0.5}})

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                simultaneous.wait(timeout=10)
                index = int(data["state"])
                self.send_json({"choice_index": index, "probabilities": [float(i == index) for i in range(2)],
                                "usage": {"completion_tokens": 1, "prompt_tokens": 5},
                                "elapsed_seconds": 0.01, "num_model_requests": 1})

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, TemporaryDirectory() as folder:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            service = owner(server.server_port)
            device = DeviceWorkers(Path(folder), connections(), {"gpu": "node:0"}, 2)
            try:
                with patch("jev_harness.inference_workers.Worker", return_value=service) as launch:
                    device.start()
                    def call(lane):
                        session = device.sessions[lane]
                        session.worker("environment").request("prepare", task=str(lane))
                        result = session.worker("model").request("decide", state=str(lane), question="Choose",
                                                                 options=["first", "second"])
                        return result["index"], session.worker("environment").request("get")["task"]
                    with ThreadPoolExecutor(2) as pool:
                        self.assertEqual(list(pool.map(call, (0, 1))), [(0, "0"), (1, "1")])
                    device.sessions[0].close_worker("model")
                    self.assertEqual(device.sessions[1].worker("model").request("info")["port"], server.server_port)
                    launch.assert_called_once()
                    service.close.assert_not_called()
            finally:
                device.close()
                server.shutdown()
                thread.join()
            service.close.assert_called_once()

    def test_two_retries_replace_a_dead_service_only_once(self):
        with TemporaryDirectory() as folder:
            device = DeviceWorkers(Path(folder), connections(), {"gpu": "node:0"}, 2)
            first, second = owner(5010), owner(5011)
            with patch("jev_harness.inference_workers.Worker", side_effect=[first, second]) as launch:
                device.start()
                first.request.side_effect = WorkerUnavailable("service stopped")
                def reconnect(_):
                    device.recover(5010)
                    return device.endpoint()
                with ThreadPoolExecutor(2) as pool:
                    self.assertEqual(list(pool.map(reconnect, (0, 1))), [5011, 5011])
                self.assertEqual(launch.call_count, 2)
                first.close.assert_called_once()
                device.close()

    def test_switch_width_at_batch_two_and_resume_with_retained_dev(self):
        with TemporaryDirectory() as folder:
            study = SharedStudy(folder, SETTINGS, connections())
            self.assertEqual(study.lanes_per_gpu, 1)
            with patch.object(study, "close_devices") as close, patch.object(study, "start_workers") as start:
                study.begin_inference_phase("train", 1)
                start.assert_not_called()
                study.begin_inference_phase("train", 2)
                self.assertEqual(study.lanes_per_gpu, 2)
                study.begin_inference_phase("train", 3)
                study.begin_inference_phase("test")
                close.assert_called_once()
                start.assert_called_once()
            save(Path(folder) / "checkpoints.json", [{"batch": 1}])
            self.assertEqual(SharedStudy(folder, SETTINGS, connections()).lanes_per_gpu, 2)

    def test_candidate_runs_on_parent_gpu_without_replaying_parent(self):
        with TemporaryDirectory() as folder:
            study = SharedStudy(folder, SETTINGS, connections(), devices=("a:0", "b:0"), lanes_per_gpu=2)
            study.groups = {gpu: object() for gpu in study.devices}
            study.workers = [Mock(gpu=gpu, lane=lane) for lane in range(2) for gpu in study.devices]
            parents = [{"task_id": task, "gpu": gpu, "success": True, "seconds": 1, "output_tokens": 4}
                       for task, gpu in zip(("t1", "t2", "t3", "t4"), ("b:0", "b:0", "a:0", "a:0"))]
            together = Barrier(4)
            def score(task, stage, arm, library, *, workers):
                self.assertEqual(stage, "b02r01")
                together.wait(timeout=5)
                return {"task_id": task, "gpu": workers.gpu, "success": True, "seconds": 1, "output_tokens": 3}
            with patch.object(study, "run_task", side_effect=score) as run:
                before, after = study.candidate_batch([r["task_id"] for r in parents], "b02r01", {}, parents, {})
            self.assertEqual(run.call_count, 4)
            self.assertEqual(before, parents)
            self.assertEqual([(r["task_id"], r["gpu"]) for r in before], [(r["task_id"], r["gpu"]) for r in after])

    def test_holdout_partial_pair_resumes_in_order_on_one_available_gpu(self):
        with TemporaryDirectory() as folder:
            study = SharedStudy(folder, SETTINGS, connections(), devices=("node:0",), lanes_per_gpu=2)
            study.groups = {"node:0": object()}
            study.workers = [Mock(gpu="node:0", lane=i) for i in range(2)]
            for task in ("t1", "t2"):
                save(Path(folder) / "runs/test_generation" / task / "record.json", {"gpu": "gone:1"})
            calls = {task: [] for task in ("t1", "t2")}
            barrier = Barrier(2)
            def run(task, stage, arm, lib, *, workers):
                if not calls[task]:
                    barrier.wait(timeout=5)
                calls[task].append((arm, stage, workers.lane))
                return {"gpu": workers.gpu}
            with patch.object(study, "run_task", side_effect=run):
                a, b = paired_holdout(study, ["t1", "t2"], "v", {})
            self.assertEqual(a, b)
            self.assertEqual([e[0] for e in calls["t1"]], ["A", "B"])
            self.assertEqual([e[0] for e in calls["t2"]], ["B", "A"])
            for arms in calls.values():
                self.assertEqual(arms[0][2], arms[1][2])
                self.assertEqual(next(stage for arm, stage, _ in arms if arm == "A"), "test_generation_paired_node_0")

    def test_wording_comparisons_share_original_ballots_and_keep_input_order(self):
        with TemporaryDirectory() as folder:
            study = SharedStudy(folder, SETTINGS, connections(), devices=("node:0",), lanes_per_gpu=2)
            study.groups = {"node:0": object()}
            study.workers = [Mock(gpu="node:0", lane=i) for i in range(2)]
            workbench = object.__new__(HistoryWorkbench)
            workbench.study = study
            arrival = Barrier(2)
            def score(cases, candidate, trial, anchors, worker):
                arrival.wait(timeout=5)
                return ([{"request": case["request"], "lane": worker.lane} for case in cases]
                        + [{"trace": doc, "event": event} for doc, event, _ in anchors])
            cases = [{"trace": doc, "request": str(i)} for i, doc in enumerate(("trial/a", "other", "archive/a"))]
            anchors = [("other", 4, {}), ("archive/a", 3, {})]
            with patch.object(workbench, "wording_on_worker", side_effect=score):
                rows = workbench.wording_feedback(cases, {}, {"a": "archive/a"}, anchors)
            self.assertEqual([row["request"] for row in rows[:3]], ["0", "1", "2"])
            self.assertEqual(rows[0]["lane"], rows[2]["lane"])
            self.assertNotEqual(rows[0]["lane"], rows[1]["lane"])
            self.assertEqual([row["trace"] for row in rows[3:]], ["other", "archive/a"])
