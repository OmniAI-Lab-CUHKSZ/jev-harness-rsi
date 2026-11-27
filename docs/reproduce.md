# Reproduce the AppWorld experiment

Use Python 3.10 or newer for the controller. The model worker and AppWorld worker can run in separate Linux environments or on separate machines.

## Install and inspect

```bash
python -m pip install -e ".[test]"
python -m pytest -q
jev-appworld plan > plan.json
```

The plan runs on a CPU and prints all task IDs, H0, budgets and the schedule. [Starting harness](../jev_harness/resources/composed_h0.json) · [Task split](../jev_harness/resources/task_split.json)

## Configure the workers

The reference GPU environment uses an A800 80 GB, Python 3.10, vLLM 0.31.0, PyTorch 2.13.0 and Transformers 5.17.0. Install an NVIDIA driver, CUDA build tools and Python development headers suitable for that environment. The released model includes the decision service, calibration data and adapter.

```bash
python3.10 -m venv .venv-model
.venv-model/bin/python -m pip install "vllm==0.31.0" "transformers==5.17.0" "huggingface_hub==1.33.0"
export JEV_MODEL_PYTHON="$PWD/.venv-model/bin/python"
export JEV_MODEL_DIR="$PWD/models/jev-27b"
.venv-model/bin/hf download autotrust/AutoTrust-Decider-Pro-27B \
  --revision 14b425269ed24452e0a5ad5a8f3e51d139647a88 --local-dir "$JEV_MODEL_DIR"
```

The decision endpoint is `/v1/decide`. The same backbone serves open generation. [Model release](https://huggingface.co/autotrust/AutoTrust-Decider-Pro-27B/tree/14b425269ed24452e0a5ad5a8f3e51d139647a88)

Use a separate Python 3.12 environment for AppWorld. Set the data root before installing the dataset.

```bash
python3.12 -m venv .venv-appworld
export APPWORLD_PYTHON="$PWD/.venv-appworld/bin/python"
export APPWORLD_ROOT="$PWD/appworld-data"
export APPWORLD_ROOT_DIR="$APPWORLD_ROOT"
"$APPWORLD_PYTHON" -m pip install \
  "appworld @ git+https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5" \
  "sqlalchemy==2.1.6" "sqlmodel==0.0.48"
"$APPWORLD_PYTHON" -m appworld.cli install
"$APPWORLD_PYTHON" -m appworld.cli download data --version 0.2.0 --mode minimal
```

The environment worker uses a compatibility adapter for in-memory SQLite and scoring. The connection files resolve variables from the controller environment. For SSH, the Python, repository, model and AppWorld paths refer to the corresponding worker machines. `SSH_IDENTITY` refers to a local controller key. Add available hosts under `model_nodes`. The optional `env` mapping can set worker environment variables, such as `CPATH` when Triton needs Python headers outside the default locations. [Local connections](../configs/connections.local.example.json) · [SSH connections](../configs/connections.ssh.example.json)

## Run and resume

A full run requires a signed-in Codex CLI with access to `gpt-6.2-sol` at `max` effort. Use `JEV_SOL_PROVIDER` if the account has a configured provider alias. The controller verifies the model and effort reported by the account server. [Account integration](../jev_harness/sol_account.py)

```bash
jev-appworld run --connections configs/connections.local.example.json \
  --root runs/composition --max-gpus 6
```

Pass `--through react-dev`, `--through h0-dev` or `--through train` to finish those stages. Each command includes the preceding stages. Resume in the same directory with the same configuration. Use a fresh root for the composition protocol if moving from earlier experiment code.

Automatic allocation collects eligible A800s with at least 58,000 MiB free and at most 30% utilization, up to the specified limit. Each replica uses BF16 and decode-only CUDA Graphs. Two independent AppWorld sessions share one replica. Use `--max-gpus 1` for a single eligible GPU. Each task uses the same seed across arms. Matched requests can still yield different generations, which is why the runner checks for consequential same-request divergence.

## Read the outputs

Paths in this table are relative to the run directory. A recorded `trace_path` is also relative to that directory.

| File | Contents |
|---|---|
| `initial_dev.json` | Paired ReAct and H0 results for the 57 Dev tasks |
| `runs/<stage>/<task>/record.json` | Official success, token use, executed cards and trace location |
| `offline/<round>/sol/` | Editor instructions, requests, replies and offline token use |
| `offline/<round>/draft_preview.json` | First-change evidence on current and previously solved Train tasks, plus a focused draft trial when applicable |
| `offline/<round>/proposal.json` | Final named edits submitted for scoring |
| `batches/<batch>/<round>/paired_selection.json` | Initial comparisons, paired confirmations and scores |
| `versions/` and `checkpoints.json` | Harness versions and the parent retained for the next batch |
| `controller.json`, `gpu_allocation.json` and `workers/` | Allocation, worker settings and process records |
| `test_report.json` | Paired Test-N scores, scenario bootstrap intervals and online and offline costs |

Each output directory belongs to one controller. Treat traces and worker logs as experiment data. [Complete protocol](experiment.md)
