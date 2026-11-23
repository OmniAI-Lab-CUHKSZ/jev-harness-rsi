"""Common AppWorld execution core for the two formal experiment arms."""
from __future__ import annotations

import json
import re
import time

from .runtime import Executor, Uncovered, parse_json
from .transport import WorkerUnavailable


class BudgetExhausted(RuntimeError):
    pass


class ContextTooLong(ValueError):
    pass


def bounded(value, chars):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= chars else text[:chars] + " [truncated]"


class FormalExecutor(Executor):
    def __init__(self, model, environment, *, api_budget=300, timeout_seconds=900,
                 thinking_budget=256, **kwargs):
        super().__init__(model, environment, **kwargs)
        self.api_budget = api_budget
        self.timeout_seconds = timeout_seconds
        self.thinking_budget = thinking_budget

    def prepare_options(self):
        self.deadline = time.monotonic() + self.timeout_seconds
        self.api_total = 0
        self.requests = 0
        self.allowed_apis = None
        return {"api_budget": self.api_budget, "max_interactions": self.api_budget + self.max_calls + 1}

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("Task wall-clock budget exhausted")

    def _model(self, operation, **values):
        self.check_time()
        if self.requests >= self.max_calls:
            raise BudgetExhausted("Model call budget exhausted")
        self.requests += 1
        if operation == "generate":
            values.setdefault("thinking_budget", self.thinking_budget)
            values.setdefault("seed", self.seed)
        try:
            result = super()._model(operation, **values)
            if result.get("thinking_used") and not values.get("native_prechecked"):
                self.requests += 1  # Fused endpoint performed both the native choice and its deliberation.
            return result
        except WorkerUnavailable:
            raise
        except RuntimeError as exc:
            if "cuda out of memory" in str(exc).casefold():
                raise WorkerUnavailable(f"GPU allocation failed: {exc}") from exc
            if "exceeds" in str(exc) and "context" in str(exc).lower():
                raise ContextTooLong(str(exc)) from exc
            if "exceeds 16384" in str(exc):
                raise ContextTooLong(str(exc)) from exc
            if "model budget exhausted" in str(exc):
                raise BudgetExhausted(str(exc)) from exc
            raise

    def known_apis(self):
        if self.allowed_apis is None:
            self.allowed_apis = {x.get("function", x)["name"] for x in self.context["task"]["functions"]}
        return self.allowed_apis

    def tool_request(self, op, event, node, **params):
        self.check_time()
        if self.api_total >= self.api_budget:
            raise BudgetExhausted("Task API budget exhausted")
        try:
            response = self.environment.request(op, **params)
        except WorkerUnavailable:
            raise
        except RuntimeError as exc:
            # Transport failures are retried from a clean task; API errors are returned as observations.
            raise WorkerUnavailable(f"Environment {op} failed: {exc}") from exc
        self.api_total = response.get("api_total", self.api_total + response.get("api_calls", int(op == "call")))
        self.done = bool(response.get("done"))
        public_params = dict(params)
        self.trace.add(event, node=node, **public_params, response=response)
        if event == "tool_batch":
            # Keep called endpoint/argument provenance alongside the batch's compact stdout.
            start = self.api_total - response.get("api_calls", 0)
            for index, action in enumerate(response.get("actions", []), start=start):
                self.trace.add("api_action", node=node, request_index=index + 1, **action)
        observation = {**public_params, "response": {"output": bounded(response.get("value", response.get("output")), 8000)}}
        self.context["history"].append(observation)
        return response

    def call(self, app, api, arguments, node=None):
        if f"{app}__{api}" not in self.known_apis():
            raise Uncovered(f"Unknown public API: {app}.{api}")
        response = self.tool_request("call", "api", node, app=app, api=api, arguments=arguments)
        value = response.get("value", response.get("output"))
        if isinstance(value, str) and value.startswith("Execution failed."):
            if response.get("api_budget_hit"):
                raise BudgetExhausted("Task API budget exhausted during tool execution")
            raise Uncovered(value[-1500:])
        return value

    def code(self, code, node=None, permit_finish=True):
        if not permit_finish:
            # Local recovery returns a value to the outer executor; the existing environment
            # monitors public API requests and blocks completion in these calls.
            return self.tool_request("execute", "tool_batch", node, code=code, permit_finish=False, trace_actions=True)
        return self.tool_request("execute", "tool_batch", node, code=code, trace_actions=True)

    def prime_generated_session(self, live_values):
        """Transfer in-memory public tool results to the existing AppWorld Python session."""
        values = {key: value for key, value in live_values.items()
                  if re.fullmatch(r"[a-z][a-z0-9_]*", key) and key != "account_passwords"}
        if not values:
            return []
        # Python's literal repr of RPC JSON values contains no executable expressions.
        try:
            self.environment.request("bind_progress", values=values)
        except RuntimeError as exc:
            raise WorkerUnavailable(f"Could not transfer completed fragment results: {exc}") from exc
        self.trace.add("generation_resume_bindings", names=sorted(values))
        return sorted(values)

    def generation_view(self, progress=None, compact=False):
        task = self.context["task"]
        docs = self._function_docs()
        if compact:
            docs = docs[:10]
            history = self.context["history"][-2:]
            history = [bounded(row, 2400) for row in history]
        else:
            history = self.generation_history()
        visible = {"instruction": task["instruction"], "supervisor": task["supervisor"],
                   "datetime": task["datetime"], "api_schemas": docs, "recent_actions": history,
                   "remaining_api_calls": self.api_budget-self.api_total}
        if progress:
            visible["compiled_progress"] = bounded(progress, 1800 if compact else 4200)
        return visible

    def generate_message(self, system, progress=None, *, purpose="G", max_tokens=None):
        for compact in (False, True):
            messages = [{"role": "system", "content": system}, {"role": "user", "content":
                json.dumps(self.generation_view(progress, compact), ensure_ascii=False, default=str)}]
            try:
                return self._model("generate", messages=messages, thinking=self.generation_thinking,
                                   max_tokens=max_tokens or self.generation_tokens, purpose=purpose)
            except ContextTooLong:
                if compact:
                    raise

    @staticmethod
    def code_from(text):
        text = text.split("</think>")[-1].strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) < 3:
                raise ValueError("An open code block was not closed")
            text = parts[1].strip()
            if text.startswith(("python\n", "py\n")):
                text = text.split("\n", 1)[1]
            return text.strip()
        try:
            obj = parse_json(text) if text.startswith("{") else None
        except ValueError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("code"), str):
            return obj["code"]
        if not text:
            raise ValueError("No code was generated")
        return text

    def agent(self, skill="", progress=None):
        system = (
            "Execute the user's task in the simulated AppWorld. Return one executable Python block and no prose. "
            "Python state persists between turns. Tools are apis.<app>.<api>(keyword_args). The supplied schemas "
            "are a shortlist. If an operation is missing, first inspect the app with apis.api_docs.show_api_descriptions "
            "or show_api_doc before writing anything. Preserve object identity when asked to update an existing object. "
            "Get account passwords with "
            "apis.supervisor.show_account_passwords() (list of account_name/password rows), use the supervisor's current email or phone, then log in "
            "to the applicable app. Use loops, pagination and local calculations in Python when useful. "
            "Print short relevant observations, including errors and counts. Only use AppWorld tools; do not read "
            "filesystem, database internals, or task files directly. Do not repeat actions already completed. "
            "For a request to change app state, finish with exactly apis.supervisor.complete_task(). "
            "For a question, pass the requested value as the answer to apis.supervisor.complete_task. "
            "If the prompt lists compiled_progress.variables, those values already exist in Python as "
            "compiled_progress['live_values'] (also available as compiled). Use them directly instead of "
            "repeating successful reads or login. "
            + str(skill))
        while not self.done:
            self.check_time()
            if self.api_total >= self.api_budget:
                raise BudgetExhausted("Task API budget exhausted")
            response = self.generate_message(system, progress)
            try:
                code = self.code_from(response["text"])
            except ValueError as exc:
                self.trace.add("generation_parse_error", error=str(exc))
                self.context["history"].append({"format_error": str(exc)})
                continue
            result = self.code(code, "generation_agent")
            if result.get("api_budget_hit") and not self.done:
                raise BudgetExhausted("Task API budget exhausted")

    def execute_task(self, library):
        self.agent()

    def summarize(self, task_id, label, seed, started, error, evaluation):
        summary = super().summarize(task_id, label, seed, started, error, evaluation)
        summary["api_calls"] = self.api_total
        summary["model_requests"] = self.requests
        summary["model_seconds"] = sum(e["response"].get("seconds", 0) for e in self.trace.events if e["event"] == "model")
        summary["execution_settings"].update(api_budget=self.api_budget, timeout_seconds=self.timeout_seconds,
                                             thinking_budget=self.thinking_budget)
        return summary
