# Reproduce the AppWorld experiment

The controller runs on Python 3.10 or newer. Each task uses a model worker serving the 27B model and an environment worker running AppWorld. The commands below assume a local Linux deployment and a shell in the repository root. Separate machines can connect over SSH. [SSH setup](#use-separate-machines)

## Install the controller

```bash
python -m pip install -e ".[test]"
python -m pytest -q
jev-appworld plan > plan.json
```

The tests and plan run on a CPU. The plan includes the 15 initial fragments, their settings and the task IDs for all three splits. The split file also records the 30 historical Train tasks used when developing H0. [Starting harness](../jev_harness/resources/h00.json) · [Task split](../jev_harness/resources/task_split.json) · [Run configuration](../jev_harness/resources/experiment.json)

## Set up the 27B worker

The reference setup uses a Linux GPU worker with an NVIDIA A800 80 GB, Python 3.10, vLLM 0.31.0, PyTorch 2.13.0 and Transformers 5.17.0. Each GPU loads a full BF16 model replica. Install a compatible NVIDIA driver, CUDA build tools and Python development headers. The model download is about 54 GB and includes the official decision service, calibration data and adapter.

```bash
python3.10 -m venv .venv-model
.venv-model/bin/python -m pip install "vllm==0.31.0" "transformers==5.17.0" "huggingface_hub==1.33.0"
export JEV_MODEL_PYTHON="$PWD/.venv-model/bin/python"
export JEV_MODEL_DIR="$PWD/models/jev-27b"
.venv-model/bin/hf download autotrust/AutoTrust-Decider-Pro-27B \
  --revision 14b425269ed24452e0a5ad5a8f3e51d139647a88 --local-dir "$JEV_MODEL_DIR"
```

The decision service exposes `/v1/decide` with the released adapter. Open generation uses the same backbone. [Model revision and usage terms](https://huggingface.co/autotrust/AutoTrust-Decider-Pro-27B/tree/14b425269ed24452e0a5ad5a8f3e51d139647a88)

## Set up AppWorld

Use a separate Python 3.12 environment for AppWorld 0.2.0.dev0 and data version 0.2.0. The controller starts the worker using `APPWORLD_PYTHON` and passes it `APPWORLD_ROOT`. The included compatibility adapter handles the simulator's in-memory SQLite databases and scoring exceptions.

```bash
python3.12 -m venv .venv-appworld
export APPWORLD_PYTHON="$PWD/.venv-appworld/bin/python"
export APPWORLD_ROOT="$PWD/appworld-data"
"$APPWORLD_PYTHON" -m pip install \
  "appworld @ git+https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5" \
  "sqlalchemy==2.1.6" "sqlmodel==0.0.48"
"$APPWORLD_PYTHON" -m appworld.cli install
"$APPWORLD_PYTHON" -m appworld.cli download data --version 0.2.0 --mode minimal
```

[AppWorld source revision](https://github.com/StonyBrookNLP/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5) · [Scoring adapter](../jev_harness/appworld_compat.py)

## Run the experiment

**One task.** With the four worker variables above exported in the controller shell, run H0 on an official Train task. The controller starts the decision service and simulator. The example below runs the H0 side of the Spotify example in the README. A completed smoke run can have `success: false` when AppWorld marks the agent's answer wrong.

```bash
jev-appworld smoke --connections configs/connections.local.example.json \
  --root runs/spotify --gpu local:0 --task 82e2fac_1
```

**Ten batches and the paired test.** The full run also needs a signed-in Codex CLI on the controller with access to `gpt-6.2-sol` at `max` reasoning effort. Sol uses the configured account provider through `codex app-server`. If your account uses a specific provider alias, set `JEV_SOL_PROVIDER` to that configured alias. The controller checks the model and reasoning effort reported by the service. [Sol account integration](../jev_harness/sol_account.py)

```bash
jev-appworld run --connections configs/connections.local.example.json \
  --root runs/main --gpus local:0,local:1,local:2,local:3
```

The four workers process different tasks concurrently. For one GPU, pass `--gpus local:0`. To resume an interrupted run, use the same command and output directory. The controller reads the completed tasks, patches and batch selections before continuing. Use one controller per output directory.

### Use separate machines

Place the repository on the AppWorld and model machines. Start from the SSH template and set the variables it references in the controller's shell. The Python, repository, model and data paths are paths on the respective worker machine. `SSH_IDENTITY` points to the key on the controller. Add entries under `model_nodes` for additional nodes and set the SSH port for each as needed. The local template resolves `PACKAGE_DIR` from the installed controller package automatically. [Local template](../configs/connections.local.example.json) · [SSH template](../configs/connections.ssh.example.json)

```bash
jev-appworld run --connections configs/connections.ssh.example.json \
  --root runs/main --max-gpus 4
```

Automatic allocation checks at most once per minute while capacity remains. It selects A800 devices with at least 58,000 MiB free and utilization of 30% or less. Use `--gpu-name` for a different device name or `--gpus` for explicit devices.

## Read the outputs

All paths in this table are relative to the directory given to `--root`. A recorded `trace_path` is also relative to that directory.

| File | Contents |
|---|---|
| `smoke_result.json` | Single-task success, generated tokens, and path to the trace |
| `runs/<stage>/<task>/record.json` | Per-task score, token counts, decisions, fragment sequence and trace path |
| `batches/<batch>/<revision>/proposal.json` and `result.json` | Sol's proposed edits and the scored parent/candidate comparison for the batch |
| `batches/<batch>/<revision>/sol*/turn_*/` | Prompts Sol received, its responses and token usage |
| `versions/` and `checkpoints.json` | Installed harness versions and the version retained after each batch |
| `test_report.json` and `test_mechanisms.json` | Paired Test-N outcomes and confidence intervals, offline usage and a breakdown of how the evolved harness executed the tasks |

Task traces can contain AppWorld session values. The repository ignores run directories, model files, simulator data and local connection settings.

## Code map

| What to inspect | Implementation |
|---|---|
| Batch selection, Dev scoring and paired Test-N | [phase1_run.py](../phase1_run.py) · [formal_run.py](../formal_run.py) |
| Online menus, fragment execution and generation handoff | [semantic_assembly.py](../jev_harness/semantic_assembly.py) · [semantic_handoff.py](../jev_harness/semantic_handoff.py) · [formal_runtime.py](../jev_harness/formal_runtime.py) |
| Editable objects, fragment definitions and policy programs | [semantic_library.py](../jev_harness/semantic_library.py) · [card_library.py](../jev_harness/card_library.py) · [policy_programs.py](../jev_harness/policy_programs.py) |
| Sol's prompts, trace access, local trials and history previews | [phase1_optimizer.py](../jev_harness/phase1_optimizer.py) · [semantic_workbench.py](../jev_harness/semantic_workbench.py) · [fragment_trials.py](../jev_harness/fragment_trials.py) · [history_workbench.py](../jev_harness/history_workbench.py) |
| Model worker, AppWorld worker and GPU allocation | [official_model_worker.py](../jev_harness/official_model_worker.py) · [env_worker.py](../jev_harness/env_worker.py) · [allocation.py](../jev_harness/allocation.py) |

[Project overview](../README.md) · [Experiment design](experiment.md)
