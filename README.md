<h1 align="center">Jev Harness RSI</h1>

<p align="center"><strong>Learn reusable operations from AppWorld traces. Let one 27B model select and execute them.</strong></p>

<p align="center">
  <a href="#example">Example</a> · <a href="#how-it-works">How it works</a> ·
  <a href="#quickstart">Quickstart</a> · <a href="#evaluation">Evaluation</a> ·
  <a href="docs/experiment.md">Protocol</a> · <a href="docs/reproduce.md">Reproduce</a>
</p>

<p align="center">
  <img src="assets/jev-harness-teaser.png" width="960" alt="The harness builds choices from task evidence. A shared 27B model selects or generates actions, while Sol revises the harness offline from training traces.">
</p>

Jev Harness RSI evolves a shared agent harness for AppWorld. Offline, Sol reads official Train trajectories and edits the harness. Online, the program retrieves applicable operations, binds public data and asks AutoTrust-Decider-Pro-27B to choose a step where needed. Uncovered work goes to the same model in generation mode. The model weights stay fixed. [AppWorld](https://github.com/StonyBrookNLP/appworld) · [AutoTrust-Decider-Pro-27B](https://huggingface.co/autotrust/AutoTrust-Decider-Pro-27B)

## Example

For an official Spotify question about the most played song in the user's playlists, the harness can present a card that reads the playlists and returns distinct song IDs. A second card hydrates those IDs through the official song API and ranks the records. Jev chooses the relevant operations and submits the answer from the observed result. The ranking card also accepts IDs from the album library and rows from the song library. [H0 card definitions](jev_harness/resources/composed_h0.json)

## How it works

1. **Retrieve and assemble.** The program combines the public instruction, available observations and card declarations. A card has named inputs, a pure calculation and a typed output. Inputs can be existing results or plans for official reads. The first page can offer zero cards.
2. **Choose and execute.** Jev sees a scoped menu of cards and official tools. The program binds known arguments, acquires the selected inputs, and validates the returned data. A card can ask Jev a question over observed records and use that selection in a later question. Writes require a separate, grounded tool selection.
3. **Continue from the result.** A result can feed another card, an official operation or the final answer. Generation can fill missing arguments or continue in AppWorld's Python session with completed observations and receipts. A delegated public read can return control to the choice loop.
4. **Revise offline.** Sol can edit retrieval, decision cues, bindings, cards, state and handoff guidance. It can change up to three named items per candidate. Before final submission, it sees a draft trial and evidence from the first change on previously solved Train tasks when available. The experiment then scores the candidate on full tasks. [Execution and optimizer protocol](docs/experiment.md)

## Quickstart

```bash
python -m pip install -e ".[test]"
python -m pytest -q
jev-appworld plan
```

After configuring the model and AppWorld workers, run the composition experiment from a fresh output directory.

```bash
jev-appworld run --connections configs/connections.local.example.json \
  --root runs/composition --max-gpus 6
```

Use the same directory and configuration to resume. Use `--through react-dev` or `--through h0-dev` to stop after the respective Dev measurement. [Worker setup and output files](docs/reproduce.md)

## Evaluation

| Split | Scenarios / tasks | Role |
|---|---:|---|
| Train | 30 / 90 | Ten batches of nine tasks, with two Sol rounds per batch. The entry harness supplies the Train trajectories. |
| Dev | 19 / 57 | Measure ReAct and H0 first. Then compare the Train winner of each batch with its retained parent. |
| Test-N | 30 / 90 | Pair generation-only ReAct and the final retained harness on the same tasks and GPUs. |

Selection compares whole-task success first and output tokens second. Exact ties keep the candidate. When same-request variation can change the result, the affected tasks contribute the mean of two paired confirmations. The default deployment uses up to six A800s, BF16 CUDA Graph decoding and two independent task sessions per GPU. [Task split](jev_harness/resources/task_split.json) · [Evaluation protocol](docs/experiment.md)
