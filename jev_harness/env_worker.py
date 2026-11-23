"""AppWorld worker; agent-facing responses expose public observations only."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
import traceback
from unittest.mock import patch


class Environment:
    def __init__(self, root, compatibility_source=None):
        from appworld import update_root
        update_root(root)
        self.root = Path(root)
        self.world = None
        self.evaluate_fn = None
        if compatibility_source:
            sys.path.insert(0, compatibility_source)
            from appworld_compat import install_database_url_fix, evaluate_world
            install_database_url_fix()
            self.evaluate_fn = evaluate_world

    def list_tasks(self, split="train", include_instructions=True, **_):
        ids = (self.root / "data/datasets" / (split + ".txt")).read_text().splitlines()
        tasks = []
        for task_id in ids:
            path = self.root / "data/tasks" / task_id / "specs.json"
            if path.exists():
                task = {"task_id": task_id, "split": split}
                if include_instructions:
                    task["instruction"] = json.loads(path.read_text())["instruction"]
                tasks.append(task)
        return {"split": split, "listed": len(ids), "available": len(tasks), "tasks": tasks}

    def prepare(self, task_id, experiment_name="jev_compiled_pilot", seed=0, api_budget=None,
                max_interactions=100, **_):
        from appworld import AppWorld
        if self.world:
            self.world.close()
        self.world = AppWorld(task_id=task_id, experiment_name=experiment_name,
                              load_ground_truth=False, random_seed=seed, max_interactions=max_interactions,
                              raise_on_failure=False, timeout_seconds=60)
        self.api_budget = api_budget
        task = self.world.task
        self.docs = task.api_docs.function_calling()
        supervisor = dict(task.supervisor)
        return {"task_id": task_id, "instruction": task.instruction,
                "supervisor": supervisor, "datetime": str(task.datetime),
                "app_descriptions": task.app_descriptions, "functions": self.docs}

    def execute(self, code, permit_finish=True, trace_actions=False, **_):
        requester = self.world.requester
        before = len(requester.requests)
        tracker = requester.request_tracker
        saved_limit = tracker.max_num_requests
        if self.api_budget is not None:
            remaining = max(0, self.api_budget - before)
            if not remaining:
                return {"output": "Task API budget exhausted", "done": self.world.task_completed(),
                        "api_calls": 0, "api_total": before, "api_budget_hit": True}
            tracker.max_num_requests = min(remaining, saved_limit) if saved_limit is not None else remaining
        try:
            def limited_request(_app_name, _api_name, *args, **kwargs):
                if (_app_name, _api_name) == ("supervisor", "complete_task"):
                    raise ValueError("Only the outer executor can complete a locally recovered task")
                return original_request(_app_name, _api_name, *args, **kwargs)

            original_request = requester.request
            context = nullcontext() if permit_finish else patch.object(requester, "request", limited_request)
            with context:
                output = self.world.execute(code)
        finally:
            tracker.max_num_requests = saved_limit
        after = len(requester.requests)
        result = {"output": output, "done": self.world.task_completed(), "api_calls": after-before,
                "api_total": after, "api_budget_hit": self.api_budget is not None and after >= self.api_budget}
        if trace_actions:
            result["actions"] = [{"method": action["method"], "url": action["url"], "data": action["data"]}
                                 for action in requester.requests[before:after]]
        return result

    def call(self, app, api, arguments=None, **_):
        if not re.fullmatch(r"[a-zA-Z_]\w*", app) or not re.fullmatch(r"[a-zA-Z_]\w*", api):
            raise ValueError("Invalid API identifier")
        arguments = arguments or {}
        code = "import json\nprint(json.dumps(apis." + app + "." + api + "(**" + repr(arguments) + "), default=str))"
        response = self.execute(code)
        try:
            response["value"] = json.loads(response["output"])
        except (TypeError, json.JSONDecodeError):
            response["value"] = response["output"]
        return response

    def bind_progress(self, values, **_):
        # Values arrive through the experiment runner's RPC, never from executable model text.
        names = sorted(values)
        self.world.shell.user_ns.update(compiled=values, compiled_progress={"live_values": values, "variables": names})
        return {"names": names}

    def evaluate(self, **_):
        # Ground truth is loaded only after execution, within the evaluator worker.
        # It is never returned by prepare/call or added to an executor observation.
        from appworld.task import GroundTruth
        self.world.task.ground_truth = GroundTruth.load(task_id=self.world.task.id, mode="minimal")
        tracker = self.evaluate_fn(self.world) if self.evaluate_fn else self.world.evaluate(suppress_errors=False)
        return {"success": bool(tracker.success), "failures": tracker.failures}

    def close(self, **_):
        if self.world:
            self.world.close()
        return {"closed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--compatibility-source")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        env = Environment(args.root, args.compatibility_source)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                result = getattr(env, request["op"])(**{
                    k: v for k, v in request.items() if k not in ("id", "op")})
            print(json.dumps({"id": request["id"], "result": result}, default=str, ensure_ascii=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"id": request["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        if request["op"] == "close":
            break


if __name__ == "__main__":
    main()
