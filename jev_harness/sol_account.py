"""Tool-free Sol requests through the current Codex sign-in."""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import threading
import time

from .formal_runtime import FormalExecutor
from .runtime import parse_json
from .transport import WorkerUnavailable


class AccountError(WorkerUnavailable):
    pass


def codex_binary():
    binary = shutil.which("codex")
    if binary:
        return binary
    runtime_path = os.environ.get("CODEX_APP_RUNTIME_CLI_PATH")
    if runtime_path and Path(runtime_path).is_file():
        return runtime_path
    if os.name == "nt":
        # A long-running controller can outlive the versioned desktop path it inherited.
        powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        try:
            query = subprocess.run(
                [str(powershell), "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-AppxPackage -Name OpenAI.Codex).InstallLocation"],
                capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AccountError("Could not locate the installed Codex desktop CLI") from exc
        location = query.stdout.strip()
        if query.returncode == 0 and location:
            packaged = Path(location) / "app/resources/codex.exe"
            if packaged.is_file():
                return str(packaged)
    raise AccountError("Could not locate the Codex CLI on PATH or in the installed desktop app")


class SolAccount:
    model = "gpt-6.2-sol"
    effort = "max"

    def __init__(self, directory, cwd):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.cwd = str(Path(cwd).resolve())
        binary = codex_binary()
        self.log = (directory/"account_server.stderr.log").open("a", encoding="utf-8")
        cmd = [binary, "-a", "never", "-c", "mcp_servers={}", "-c", "hooks={}",
               "-c", 'analytics.enabled=false', "app-server", "--disable", "apps", "--disable", "plugins",
               "--disable", "shell_tool", "--disable", "shell_snapshot", "--listen", "stdio://"]
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                encoding="utf-8", text=True, cwd=self.cwd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.messages = Queue()
            def reader():
                try:
                    for line in self.process.stdout:
                        self.messages.put(line)
                finally:
                    self.messages.put(None)
            threading.Thread(target=reader, daemon=True).start()
            self.next_id = 0
            self.event_log = None
            self.rpc("initialize", {"clientInfo": {"name": "jev_appworld_experiment", "version": "1"},
                                    "capabilities": {"experimentalApi": True}}, timeout=150)
            self.send({"method": "initialized"})
        except Exception:
            self.close()
            raise

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        self.log.close()

    def send(self, data):
        try:
            self.process.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise AccountError("Codex account server stopped communicating") from exc

    def receive(self, deadline):
        while True:
            try:
                line = self.messages.get(timeout=max(deadline-time.monotonic(), 0))
            except Empty as exc:
                raise AccountError("Timed out waiting for the Codex account server") from exc
            if line is None:
                raise AccountError("Codex account server exited")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            method = message.get("method", "")
            if self.event_log is not None and ("id" in message or method in {
                "error", "turn/completed", "thread/tokenUsage/updated", "account/rateLimits/updated",
            }):
                self.event_log.write(line)
                self.event_log.flush()
            if "id" in message and "method" in message:
                self.send({"id": message["id"], "error": {"code": -32601, "message": "Offline calls have no tools or approvals"}})
                continue
            return message

    def rpc(self, method, params, timeout=120):
        self.next_id += 1
        idx = self.next_id
        self.send({"id": idx, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            message = self.receive(deadline)
            if message.get("id") == idx:
                if "error" in message:
                    raise AccountError(f"Codex {method}: {message['error'].get('message')}")
                return message["result"]

    def start(self, folder, instructions="Return only the requested JSON result. No tools are needed or available."):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.event_log = None
        provider = os.environ.get("JEV_SOL_PROVIDER")
        response = self.rpc("thread/start", {"model": self.model, **({"modelProvider": provider} if provider else {}),
            "allowProviderModelFallback": False, "config": {"model_reasoning_effort": self.effort},
            "approvalPolicy": "never", "sandbox": "read-only", "ephemeral": True, "environments": [],
            "cwd": str(Path(folder).resolve()), "baseInstructions": instructions})
        if response.get("model") != self.model or response.get("reasoningEffort") != self.effort:
            raise AccountError("Codex did not select GPT-6.2 Sol Max")
        Path(folder, "model.json").write_text(json.dumps({"model": response["model"],
            "model_provider": response.get("modelProvider"), "requested_effort": self.effort,
            "reported_effort": response.get("reasoningEffort")}, indent=2), encoding="utf-8")
        return response["thread"]["id"]

    def turn(self, thread, prompt, folder, *, timeout=1200, schema=None):
        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(folder, "request.txt").write_text(prompt, encoding="utf-8")
        params = {"threadId": thread, "effort": self.effort, "environments": [],
                  "input": [{"type": "text", "text": prompt, "text_elements": []}]}
        if schema:
            params["outputSchema"] = schema
        started = time.perf_counter()
        latest_usage = {}
        texts = []
        with Path(folder, "events.jsonl").open("w", encoding="utf-8") as events:
            self.event_log = events
            try:
                response = self.rpc("turn/start", params, timeout=min(180, timeout))
                turn_id = response["turn"]["id"]
                deadline = time.monotonic() + timeout
                while True:
                    message = self.receive(deadline)
                    method, data = message.get("method"), message.get("params", {})
                    if data.get("threadId", thread) != thread:
                        continue
                    if method == "thread/tokenUsage/updated":
                        latest_usage = data["tokenUsage"]["last"]
                    elif method == "item/completed" and data.get("item", {}).get("type") == "agentMessage":
                        texts.append(data["item"]["text"])
                    elif method == "error" and not data.get("willRetry", True):
                        problem = data.get("error", {})
                        if problem.get("message"):
                            raise AccountError(f"Sol request failed: {problem['message']}")
                    elif method == "turn/completed" and data["turn"]["id"] == turn_id:
                        turn = data["turn"]
                        if turn["status"] != "completed":
                            raise AccountError(f"Sol turn {turn['status']}: {turn.get('error')}")
                        break
            finally:
                self.event_log = None
        if not texts:
            raise AccountError("Sol completed without a final text response")
        result = texts[-1]
        Path(folder, "response.txt").write_text(result, encoding="utf-8")
        spent = {**latest_usage, "seconds": time.perf_counter()-started}
        Path(folder, "usage.json").write_text(json.dumps(spent, indent=2), encoding="utf-8")
        return result

    def json(self, prompt, folder):
        thread = self.start(folder)
        return parse_json(self.turn(thread, prompt, folder))


class SolDemonstrator(FormalExecutor):
    def __init__(self, session, environment, *, folder, **kwargs):
        super().__init__(None, environment, **kwargs)
        self.session = session
        self.folder = Path(folder)

    def execute_task(self, _):
        thread = self.session.start(self.folder, "You write Python cells for the AppWorld experiment. "
            "A separate experiment runner executes your returned cells in the isolated simulator. Do not use tools yourself.")
        schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}
        rules = (
            "Complete this simulated AppWorld task. Return JSON {\"code\": \"one Python cell\"}. Only the runner executes it. "
            "Use public apis.<app>.<method>, no filesystem, internal database or task data access. Python variables persist. "
            "Use loops and pagination when needed. Get credentials from apis.supervisor.show_account_passwords(), "
            "use the current supervisor phone/email as appropriate and call the app's login. Print small relevant "
            "observations; the runner returns stdout and errors in its next message. The password API returns a list "
            "of rows with `account_name` and `password`; select by `account_name`. "
            "Call apis.supervisor.complete_task(answer=...) once the task is done, with no answer for action tasks.\n")
        view = json.dumps(self.generation_view(compact=True), ensure_ascii=False, default=str)
        prompt = rules + view
        for index in range(self.max_calls):
            self.check_time()
            if self.api_total >= self.api_budget:
                break
            self.requests += 1
            turn_directory = self.folder/f"turn_{index+1:02d}"
            remaining = int(self.deadline - time.monotonic())
            if remaining <= 0:
                break
            text = self.session.turn(thread, prompt, turn_directory, timeout=min(remaining, 600), schema=schema)
            usage = json.loads((turn_directory/"usage.json").read_text(encoding="utf-8"))
            self.trace.add("offline_model", **usage)
            try:
                code = self.code_from(text)
            except (TypeError, ValueError) as exc:
                prompt = f"The previous response was invalid: {exc}. Respond with one JSON code cell."
                continue
            result = self.code(code, "sol_demonstration")
            if self.done:
                return
            observation = str(result["output"])
            prompt = f"Previous cell stdout/error:\n{observation[:11000]}\nRemaining API calls: {self.api_budget-self.api_total}. Continue the original task with one JSON code cell."

    def summarize(self, task_id, label, seed, started, error, evaluation):
        row = super().summarize(task_id, label, seed, started, error, evaluation)
        row["optimizer"] = f"{self.session.model}/{self.session.effort}"
        for key in ("inputTokens", "cachedInputTokens", "outputTokens", "reasoningOutputTokens"):
            row[key] = sum(e.get(key, 0) for e in self.trace.events if e["event"] == "offline_model")
        return row
