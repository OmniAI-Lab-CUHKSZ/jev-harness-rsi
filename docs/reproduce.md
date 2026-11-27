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
  --root runs/retrieval-main --gpus local:0,local:1,local:2,local:3,local:4,local:5 \
  --cuda-graph --paired-selection-from-batch 1 --confirmation-scoring two_repeats_only
```

Use a new output directory for H0 with state-based retrieval. The six BF16 replicas use decode-only CUDA Graphs and serve up to twelve independent AppWorld sessions with vLLM's `--max-num-seqs 2`. Train candidates return to the parent's GPU when available. Sol uses the same workers for bounded previews. Each Test-N pair uses one worker with fresh simulator states for its two arms.

For one GPU, pass `--gpus local:0`. Resume with the same command and output directory. The runner reuses completed task records and the retained parent's Dev panel within that run. Existing experiment directories keep their saved configuration. Each Dev comparison records the session count for both sides in `checkpoints.json`. Generated tokens can vary with concurrent inference even with greedy decoding. Use one controller per output directory.

To enable conditional paired confirmation starting with Batch 2, add `--paired-selection-from-batch 2 --confirmation-scoring two_repeats_only`. For Train and Dev, the initial panel identifies tasks to repeat. If different outputs on identical model requests could change the accuracy ordering, the runner repeats every task whose outcome flipped. If accuracy is tied and the outstanding same-request token differences could change the cost ordering, it also repeats those cost-sensitive tasks.

For each flagged task, each version runs twice in fresh AppWorld environments using the task seed. Starting with 0.2.3.post2, a worker runs both versions on one GPU and starts that task's second pass as soon as its first pair finishes. The second pass reverses the order of the versions. Other tasks progress independently. The runner assesses cost-sensitive tasks after the accuracy confirmations have finished. Each pass has separate run directories. On resume, the runner reuses completed observations and matches the two arms of each pair by physical GPU.

For a flagged task, the score uses the two confirmation successes and token counts with equal weight. All other tasks contribute their initial measurements. Selection compares total success first, then lower output tokens, with exact ties going to the candidate. Reports include the initial panel, the scored observations, and the full additional cost of both passes.

New paired runs default to `two_repeats_only`. A run created under the original scoring rule keeps `initial_plus_one` on resume unless the command explicitly requests the new rule. The original rule averages the initial observation and one confirmation. Changing the rule applies it to pending and future selections. Earlier completed selections retain their recorded scores.

For the November 25 continuation, add `--revisit-after 4 3` to the run command. The controller saves this schedule in the run configuration. After Batch 4 Dev selection it uses the retained harness for a fresh visit to the nine Batch 3 Train tasks, stored under `b04_revisit03`. Its selected harness enters Batch 5. The original `b03` results remain available. From version 0.2.3.post3 the scoped optimizer attempts both configured rounds, including when it keeps the parent in round one. The per-visit `round_outcomes` field records each attempt and `revisions_scored` counts the candidates that received a full Train evaluation.

To reproduce the subsequent engine switch, also pass `--cuda-graph-after 4` and use the model worker from version 0.2.3.post4. The fourth regular visit completes Dev selection on the eager engine. Before the next Train visit, the controller starts the same BF16 model with decode-only CUDA graphs and evaluates the retained harness on all 57 Dev tasks. The resulting `cuda_graph_dev_reference.json` identifies the harness and its Dev records. Later comparisons reuse the retained parent's CUDA Graph panel. Graph Dev stages end in `_cuda_graph`, and a restart reuses each completed task in that stage. Both final Test-N arms use the same engine.

For the 24-judgment continuation that also repeats Batch 4, run with `--revisit-after 4 3 4 --restart-revisit 24j --cuda-graph-after 4`. A completed `b04_24j_revisit03` carries straight into `b04_revisit04`. Its nine Batch 4 entry tasks run under CUDA Graph. The Dev winner starts `b05_after_repeat04` with the updated Train history. The Batch 5 entry stage is keyed by its parent harness, so previously completed online tasks for that parent are reused. Work from the interrupted Batch 5 optimizer remains in its original directory for cost accounting.

### Use separate machines

Place the repository on the AppWorld and model machines. Start from the SSH template and set the variables it references in the controller's shell. The Python, repository, model and data paths are paths on the respective worker machine. `SSH_IDENTITY` points to the key on the controller. Add entries under `model_nodes` for additional nodes and set the SSH port for each as needed. The local template resolves `PACKAGE_DIR` from the installed controller package automatically. [Local template](../configs/connections.local.example.json) · [SSH template](../configs/connections.ssh.example.json)

A model worker can take an `env` mapping in its connection entry. For a Python installation with headers outside the usual include path, set `CPATH` there to the worker's Python and system include directories so Triton can compile its kernels.

```bash
jev-appworld run --connections configs/connections.ssh.example.json \
  --root runs/retrieval-main --max-gpus 6 --cuda-graph \
  --paired-selection-from-batch 1 --confirmation-scoring two_repeats_only
```

Automatic allocation checks at most once per minute while capacity remains. It selects A800 devices with at least 58,000 MiB free and utilization of 30% or less. Use `--gpu-name` for a different device name or `--gpus` for explicit devices.

## Read the outputs

All paths in this table are relative to the directory given to `--root`. A recorded `trace_path` is also relative to that directory.

| File | Contents |
|---|---|
| `smoke_result.json` | Single-task success, generated tokens, and path to the trace |
| `runs/<stage>/<task>/record.json` | Per-task score, token counts, decisions, fragment sequence and trace path |
| `batches/<batch>/<revision>/proposal.json` and `result.json` | Sol's proposed edits and the scored parent/candidate comparison for the batch |
| `batches/<batch>/<revision>/sol_scoped/` | The Train index, Sol's `focus`, `review` and optional `final_review` conversations, per-trial path comparisons and measured 27B preview costs |
| `controller.json`, `gpu_allocation.json` and `workers/` | Active GPUs, task slots, model sequence limits and per-session logs |
| `versions/` and `checkpoints.json` | Installed harness versions and the version retained after each batch |
| `test_report.json` and `test_mechanisms.json` | Paired Test-N outcomes and confidence intervals, offline usage and a breakdown of how the evolved harness executed the tasks |

Task traces can contain AppWorld session values. The repository ignores run directories, model files, simulator data and local connection settings.

## Public tool pilot

`native-pilot` runs independent AppWorld episodes on official Train. Give Sol optimization tasks with `--train` and reserve other Train IDs with `--check`. Each task runs with the generation-only interface, the entry decision harness, and the candidate produced by one offline Sol revision. A check from the same scenario tests a change of instance. Use a different scenario to test transfer.

```bash
jev-appworld native-pilot --connections configs/connections.local.example.json \
  --root runs/native-pilot --max-gpus 2 \
  --train 60d0b5b_1 07b42fd_1 29caf6f_1 --check 60d0b5b_2 e85d92a_1
```

The allocator uses two independent simulator sessions per GPU and CUDA Graph decoding. Task pairs prefer the same GPU. The execution settings come from `jev_harness/resources/experiment.json`, including the 40-request and 8,192-output-token budgets. One open JSON response is capped at 640 tokens. Official adaptive choice uses a 0.60 threshold and a 256-token reasoning limit when the remaining request budget permits it.

The default `--phase all` runs the full pilot. `initial` measures the two entry arms, `revise` builds the candidate from optimization Train traces, and `evaluate` scores that candidate. To continue with an earlier native pilot, pass `--history-run <earlier-root>`. The last named candidate supplies the entry harness. Sol can inspect the specified runs' optimization tasks. On previously solved optimization tasks the editor can request a real Jev decision at the first altered ballot, up to 24 historical calls in total. A changed open request is reported without regenerating a task. The pilot reports whole-task scores and leaves version selection to the main experiment.

Protocol v2 shows the target and proposed field changes for public update operations. The editor can inspect whether a recalled fragment reached an accepted write or answer. It can run one complete affected current Train task on the draft before final review. The three scored arms run in fresh environments. To evaluate saved libraries without an editor, use `--phase score --entry-library entry.json --candidate-library candidate.json` with a new run directory. The full protocol, worked examples, and commands for the measured task panel are in the [Chinese reproduction guide](native_protocol.zh.md).

| File | Contents |
|---|---|
| `pilot_plan.json` | Optimization and check IDs, novel check scenarios, seed and limits |
| `entry_harness.json` | Starting retrieval, choice guidance, bindings, transforms, state views and local generation guidance |
| `runs/{h0_generation,h0_decision,candidate}/<task>/` | Official score, costs and a trace of visible menus, selected sources, calls and receipts |
| `offline/sol/` | Editor requests, responses and offline usage |
| `offline/proposal.json`, `offline/candidate.json` | Named edits against the parent and the executable library |
| `offline/draft_preview.json` | The draft and the first changed request on each optimization trace. The editor can inspect one fresh 27B choice on the current panel before the whole-task rerun. |
| `pilot_results.json` | One row per measured task and arm, including open output and decision-reasoning output |

Sol can change a search cue, a choice cue, an argument binding, a read-only fragment, an evidence view, or guidance for a selected tool's open binding. A fragment may name companion sources for a join. The program enumerates combinations of observed sources and Jev selects the one to use. Both arms can request a specific public page, inspect an observed field, or compute an aggregate before the next tool call. A write receipt is recorded after the simulator accepts the request.

## Composed card pilot

`native-pilot --composed` uses the composed executor in both arms. It starts with [the composed H0 library](../jev_harness/resources/composed_h0.json) and collects one entry-harness rollout per optimization task. Sol can edit up to three named objects, check changed decisions on Train recordings and try one affected task before the candidate is scored. The four-task follow-up also used [this saved library](../configs/composed_pilot_library.json), which adds a learned contact-cohort fragment and its comment and like bindings to H0.

Start a new Sol revision with the following command. Replace the connection template with your configured file if the workers run on another machine.

```bash
jev-appworld native-pilot --composed \
  --connections configs/connections.local.example.json \
  --root runs/composed-evolve --max-gpus 2 \
  --train cf6abd2_1 82e2fac_1 afc0fce_1 \
  --check cf6abd2_2 82e2fac_2 afc0fce_2
```

To evaluate the supplied libraries on the four-task follow-up panel, use fixed-library scoring. This starts two independent episodes per task and requires only the model and AppWorld workers.

```bash
jev-appworld native-pilot --composed --phase score \
  --connections configs/connections.local.example.json \
  --root runs/composed-score --max-gpus 2 \
  --check b7a9ee9_1 6ea6792_2 d0b1f43_2 afc0fce_1 \
  --entry-library jev_harness/resources/composed_h0.json \
  --candidate-library configs/composed_pilot_library.json
```

The measured saved library passed the playlist-follow, request-denial and comment-and-like tasks with zero open-generation calls. The date-based amount query still required generation and failed. Read `pilot_results.json` and `comparison.json` for the new run. Per-task `fragment_flow.json` records whether a card was offered, was selected and supplied arguments to an accepted effect.

The composed executor checks known read parameters before building the menu. When a read requires upstream data, it checks those parameters again after that data is available. Its field projections consume complete object sets and use the same official API validation and receipts as direct actions. When several sets can supply a selected write, Jev chooses the set. [Composed executor](../jev_harness/composed_assembly.py) · [Sol prompt and patch format](../jev_harness/native_revision.py)

### Parameterized fragments and Train evidence

A composed patch can declare an endpoint-independent contact filter with `label`, `program` and `composition`. The declaration gives the input roles `counterparty_records` and `contact_identities` and the output `matching_records`. The policy receives the two record lists and the identity field paths, and returns a subset of the original records. The AppWorld adapter builds the official reads and compatible consumers from the current goal. The executor binds known values directly and lets Jev choose among remaining alternatives. A declined fragment leaves a diagnostic event outside the standard model context. [AppWorld adapter](../jev_harness/record_composition.py)

Earlier composed patches can still carry a dynamic planner with named slots and typed results. The planner returns concrete official read arguments or references to complete observations.

The Venmo example reuses one contact-email filter for received payments, sent payments and pending requests. Different result types connect to the relevant official operations. Payment results retain the amount so the same filter can support an aggregate answer. The example is an editor patch for the mechanism comparison. Apply it with `apply_edits(parent, proposal, catalog(public_functions), composed=True)` to score it as a candidate. [Example patch](../configs/dynamic_venmo_example.json) · [Dynamic executor](../jev_harness/dynamic_cards.py)

The composed editor uses the current Train evidence and can open available historical Train traces on request. The historical impact check can reuse an old trace past a changed menu after one actual Jev reselection and records the next request that diverges. To reproduce the earlier calculation-evidence comparison, set `calculation_reuse_evidence` in the pilot settings. This opt-in reports reads, field paths, calculation sites and recorded consumers in `calculation_evidence.json`. [Train replay](../jev_harness/native_replay.py) · [Evidence extractor](../jev_harness/calculation_evidence.py)

### Formal composed experiment

Inspect the composed task split, initial library and model settings with `jev-appworld native-plan --composed`. Start the complete experiment in a fresh run directory.

```bash
jev-appworld native-run --composed \
  --connections configs/connections.ssh.example.json \
  --root runs/composed-formal --max-gpus 6 --through test
```

The runner measures ReAct and composed H0 on Dev in that order. It then runs two Sol revisions per Train batch, selects the version to carry forward on Dev, and evaluates the retained version against ReAct on Test-N. Sol can give up to 12 draft-stage replies per round, including evidence reads and the final proposal. Draft review has a separate allowance of up to two replies. Resume with the same command and run directory.

## Code map

| What to inspect | Implementation |
|---|---|
| Batch selection, Dev scoring and paired Test-N | [phase1_run.py](../phase1_run.py) · [formal_run.py](../formal_run.py) |
| Online menus, fragment execution and generation handoff | [semantic_assembly.py](../jev_harness/semantic_assembly.py) · [semantic_handoff.py](../jev_harness/semantic_handoff.py) · [formal_runtime.py](../jev_harness/formal_runtime.py) |
| Editable objects, fragment definitions and policy programs | [semantic_library.py](../jev_harness/semantic_library.py) · [card_library.py](../jev_harness/card_library.py) · [policy_programs.py](../jev_harness/policy_programs.py) |
| Sol's prompts, component diagnosis and earlier Train evidence | [batch_revision.py](../jev_harness/batch_revision.py) · [revision_evidence.py](../jev_harness/revision_evidence.py) · [train_precedents.py](../jev_harness/train_precedents.py) |
| Recorded paths and model previews | [trace_comparison.py](../jev_harness/trace_comparison.py) · [batch_revision_trials.py](../jev_harness/batch_revision_trials.py) · [contrast_probes.py](../jev_harness/contrast_probes.py) |
| Model worker, AppWorld worker and GPU allocation | [official_model_worker.py](../jev_harness/official_model_worker.py) · [env_worker.py](../jev_harness/env_worker.py) · [allocation.py](../jev_harness/allocation.py) |
| Composed pilot, card validation and Train replay | [native_pilot.py](../jev_harness/native_pilot.py) · [composed_assembly.py](../jev_harness/composed_assembly.py) · [native_revision.py](../jev_harness/native_revision.py) · [native_replay.py](../jev_harness/native_replay.py) |

[Project overview](../README.md) · [Experiment design](experiment.md)
