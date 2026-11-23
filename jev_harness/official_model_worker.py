"""JSON-line transport for AutoTrust's published /v1/decide and the same vLLM base model."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OfficialModel:
    def __init__(self, model, gpu, service, max_context):
        self.model = model
        self.max_context = max_context
        self.served = "autotrust/JEV-27B"
        self.tokenizer = None
        service = Path(service)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        memory = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free,memory.total",
                                          "--format=csv,noheader,nounits"], text=True)
        free, total = map(int, memory.strip().split(","))
        utilization = min(0.90, (free - 512) / total)
        cmd = [sys.executable, str(service / "serve_decide.py"), "--host", "127.0.0.1", "--port", str(port),
               "--model", model, "--served-model-name", self.served, "--dtype", "bfloat16",
               "--enable-lora", "--max-lora-rank", "32", "--lora-modules", "jev-decision="+str(service / "adapter_vllm"),
               "--trust-request-chat-template", "--logprobs-mode", "processed_logprobs", "--max-logprobs", "32",
               "--max-model-len", str(max_context), "--max-num-seqs", "1", "--enforce-eager",
               "--gpu-memory-utilization", f"{utilization:.4f}", "--kv-cache-memory-bytes", str(3 * 1024**3)]
        cache = Path(tempfile.gettempdir()) / f"jev_official_engine_{os.getuid()}"
        cache.mkdir(parents=True, exist_ok=True)
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        # ZeroMQ uses Unix sockets. Long shared TMPDIR paths on some nodes exceed the socket path limit.
        self.rpc_dir = tempfile.TemporaryDirectory(prefix="jev_rpc_", dir="/dev/shm")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "OMP_NUM_THREADS": "4",
               "VLLM_RPC_BASE_PATH": self.rpc_dir.name,
               "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": cuda_home,
               "PATH": os.pathsep.join([str(Path(sys.executable).parent), str(Path(cuda_home) / "bin"), os.environ.get("PATH", "")]),
               "TRITON_CACHE_DIR": str(cache / "triton"), "VLLM_CACHE_ROOT": str(cache / "vllm"),
               "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer")}
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                                     start_new_session=True)
        try:
            end = time.monotonic() + 780
            while self.proc.poll() is None and time.monotonic() < end:
                try:
                    self.profile = self.http("/v1/decide/info", timeout=3)
                    break
                except (URLError, TimeoutError):
                    time.sleep(2)
            else:
                raise RuntimeError(f"Published decision server did not start (exit {self.proc.poll()})")
            if self.profile["defaults"]["mix"] != 0.5 or self.profile["model"] != self.served:
                raise RuntimeError("Unexpected official decision service configuration")
        except BaseException:
            self.close()
            raise

    def http(self, path, values=None, timeout=620):
        data = None if values is None else json.dumps(values, ensure_ascii=False).encode("utf8")
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as result:
                return json.load(result)
        except HTTPError as exc:
            raise RuntimeError(f"Official server {path} {exc.code}: {exc.read().decode(errors='replace')[:1600]}") from exc

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)
        self.rpc_dir.cleanup()

    def info(self):
        return {"model": self.model, "precision": "bf16", "merged": False, "backend": "official_vllm",
                "service_pid": self.proc.pid, "profile": self.profile}

    def decide(self, state, question, options, skill="", thinking="off", threshold=None, think_budget=None, **_):
        prompt = (skill.strip() + "\n\n" + question).strip()
        payload = {"kind": "choice", "state": state, "question": prompt, "options": options, "thinking": thinking}
        if thinking == "auto":
            payload.update(threshold=threshold, think_budget=think_budget, debug=True, return_reasoning=True)
        r = self.http("/v1/decide", payload)
        t = r.get("thinking", {})
        used = t.get("used", False)
        n = t.get("think_tokens", 0)
        usage = r["usage"]
        return {"index": r["choice_index"], "probabilities": r["probabilities"],
                "native_probabilities": t.get("system1", r["probabilities"]),
                "reasoned_probabilities": t.get("system2"), "reasoning": t.get("reasoning"),
                "thinking_used": used, "finished_within_budget": t.get("finished_within_budget"),
                "seconds": r["elapsed_seconds"], "input_tokens": usage["prompt_tokens"],
                "output_tokens": n, "reasoning_tokens": n, "readout_tokens": usage["completion_tokens"] - n,
                "backend_requests": r["num_model_requests"], "service_usage": usage,
                "mode": "R_review" if used else "D"}

    def complete(self, ids, count, seed, stop_ids):
        r = self.http("/v1/completions", {"model": self.served, "prompt": ids, "max_tokens": count,
                      "temperature": 0.0, "seed": seed, "stop_token_ids": stop_ids,
                      "skip_special_tokens": False, "include_stop_str_in_output": True, "return_token_ids": True})
        choice = r["choices"][0]
        return choice["token_ids"], r["usage"]["prompt_tokens"]

    def generate(self, messages, max_tokens=1024, thinking=False, seed=0, thinking_budget=256, **_):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        tok = self.tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) + max_tokens + 8 > self.max_context:
            raise ValueError(f"Generation input {len(ids)} + budget {max_tokens} exceeds {self.max_context}")
        start = time.monotonic()
        reason, input_tokens, forced = [], 0, False
        end_think = tok.encode("</think>", add_special_tokens=False)
        if thinking and max_tokens > 1:
            reason, input_tokens = self.complete(ids, min(thinking_budget, max_tokens-1), seed,
                                                 [end_think[-1], tok.eos_token_id])
            forced = reason[-len(end_think):] != end_think
            ids = ids + reason
            if forced:
                ids += tok.encode("\n</think>\n\n", add_special_tokens=False)
        answer, final_inputs = self.complete(ids, max_tokens-len(reason), seed, [tok.eos_token_id])
        text = tok.decode(reason, skip_special_tokens=True) + ("\n</think>\n\n" if forced else "")
        count = len(reason) + len(answer)
        return {"text": text + tok.decode(answer, skip_special_tokens=True), "input_tokens": input_tokens + final_inputs,
                "output_tokens": count, "reasoning_tokens": len(reason), "forced_reasoning_closure": forced,
                "truncated": count >= max_tokens, "seconds": time.monotonic()-start, "mode": "R" if thinking else "G"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--max-context", default=16384, type=int)
    args = parser.parse_args()
    model = None
    try:
        for line in sys.stdin:
            req = json.loads(line)
            op = req.pop("op")
            request_id = req.pop("id")
            try:
                if op == "close":
                    print(json.dumps({"id": request_id, "result": {"closed": True}}), flush=True)
                    break
                if model is None:
                    model = OfficialModel(args.model, args.gpu, args.service, args.max_context)
                if op not in ("info", "decide", "generate"):
                    raise ValueError("Unknown model operation: "+op)
                result = getattr(model, op)(**req)
                print(json.dumps({"id": request_id, "result": result}, ensure_ascii=False), flush=True)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                print(json.dumps({"id": request_id, "error": str(exc)}), flush=True)
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
