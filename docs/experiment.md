# AppWorld experiment design

**Phase 1 protocol, November 24, 2026.** Starting from H0, a fixed 27B agent runs the first nine Train tasks. GPT-6.2 Sol studies the execution traces and edits the harness shared by all tasks. Semantic handoffs, feedback from the rest of the current batch, and previews on earlier Train states are available from the first round. The controller writes records under the directory given to `--root`. [Run configuration](../jev_harness/resources/experiment.json) · [Reproduction guide](reproduce.md)

## Research question

Can Sol improve the shared harness so that the same 27B agent solves more tasks in new scenarios, or uses fewer online generated tokens at the same success rate? The harness governs what data enters the system, how it is represented, which choices reach Jev, what a fragment executes, and what the generator sees when it takes over.

The unit of evaluation is one AppWorld task with its own initial simulator state. Within a task, the program assembles the next steps from the current shared library. Between tasks, offline Sol edits that library and its policies. Model weights are fixed. Executor-authored fragments are a separate, later phase.

## The executor and H0

The model is `autotrust/AutoTrust-Decider-Pro-27B`, served in BF16. Its released adapter and native decision head score the options supplied by the program. In open generation, the same backbone runs with the adapter disabled and writes Python against the public AppWorld APIs. Both modes share one session and its actual observations.

| Part of H0 | Starting state |
|---|---|
| Fifteen fragments | `library`, `playlist`, `album`, `directory`, `filter_genre`, `sort_songs`, `take_first`, `count_songs`, `titles`, `play`, `submit_answer`, `submit_effect`, `read_alarms`, `read_pending_requests`, `read_notes` |
| Assembly | The program checks installed operations, required inputs and values already obtained. It can group candidates by app or subgoal and find available data providers. H0 describes Spotify, Phone, Venmo, File System and Simple Note. New fragments can extend these menus. |
| Decision and state | Shared questions and guidance cover menu selection, next steps and parameter choices. Per-task state contains actual results, their sources, selected parameters and effects already executed. H0 starts with zero custom state views. |
| Execution and handoff | Generic API execution, initial Phone login fields, default generation guidance, and an editable Jev handoff question and selection skill. H0 starts with zero task-specific handoff guides. |

Scoped menus show the short purpose, applicability, inputs and output of an eligible fragment. Parameter guidance appears during binding. Handoff guidance follows the fragment that produced a value still present in the session. The initial fragments and parameter guidance come from earlier Train work. The split manifest identifies the 30 historical source tasks. [Full H0](../jev_harness/resources/h00.json) · [Task manifest](../jev_harness/resources/task_split.json)

### One task, step by step

1. **Build the menu.** The program reads the public instruction, fragment declarations and live values. If there are multiple distinct menus, Jev can choose the most promising one. Selecting `NONE` in a focused menu can open the full eligible menu. App names influence hints and ordering.
2. **Select and bind.** Jev chooses a fragment and, where needed, its arguments. For example, after listing notes, the next choice may use a real title or ID returned by the app. Installed templates and live values supply both the question and its options. Generative calls to author online questions or options total zero.
3. **Execute and reassess.** The program writes the actual tool result to the fragment's declared output. This can satisfy an input for a downstream fragment. For the next-fragment choice, the official service scores the options to obtain `p1`. At a top probability of **0.60** or above, the initial choice stands. Below 0.60, the same 27B can reason for up to **256 tokens** over the same state and choices, then score them again to obtain `p2`. The final choice maximizes `(p1 + p2) / 2`. The service reads option scores even when thinking reaches its cap and records whether thinking ended naturally. An initial `NONE` on a narrow menu first triggers expansion. Menu selection, argument binding and handoff guidance use native judgment. The threshold and reasoning budget apply from H0 through Test-N.
4. **Choose guidance for generation.** Each specialist guide declares which real data it requires. With eligible guides, Jev sees the task, a summary of those data, descriptions of the possible next actions, and `NONE`. Sol may also name an easily confused alternative such as “only report the amount.” Selecting that alternative or `NONE` gives the generator the ordinary handoff. Selecting a specialist adds that guide's skill. With zero eligible guides, the ordinary handoff is immediate. Menus hold up to 15 substantive choices plus `NONE` per page. The program combines page selections while reserving a call for generation.
5. **Continue in the same session.** The generator receives the live variables, available tools and handoff. It can continue writing Python or use `yield_read` to return a registered public read to the executor. After the read, the program builds a fresh menu. The next handoff selects guidance using the updated state. Selecting guidance only determines what the generator sees. App actions such as paying or playing music take place separately.

A newly registered guide reaches the generation prompt only when selected. Two unrelated tasks with the same kind of available data may both trigger a handoff decision. Historical previews can reveal when Jev sends the wrong guide to one of them.

The handoff judge receives about 1,200 characters per data role and up to about 12,000 characters in total. Full session variables remain available to the generator. The chosen skill has its own prompt field of up to about 3,200 characters in regular context or 1,600 in compact context. With too little call budget for both a judgment and a generation opportunity, the executor uses the ordinary handoff.

## What Sol can edit

One patch changes **one to three named objects**. Editing a shared handoff question and two distinct guides uses all three. Each Train batch has at most two rounds. Dev selects the harness that starts the next batch.

| Component | Editable content | Example diagnosis |
|---|---|---|
| Execution adapter (`adapter`) | Login fields, paths to session values, public API argument mappings, response paths and page size | Jev chooses the correct contacts read, but the login sends the wrong field. |
| Fragment (`card`) | Local reads, computations or actions, inputs and outputs, candidate arguments, descriptions and skills for execution or handoff | Several tasks generate code to find one note and read its contents. |
| State view (`state`) | A view computed from existing public data for Jev or the generator | The objects were retrieved, but the next step cannot tell which collection they belong to. |
| Menu assembly (`assembly`) | Menu descriptions, included apps and fragments, and restricted programs that add candidates based on state | A suitable fragment exists but is missing from the menu Jev sees. |
| Decision (`decision`) | Menu, step and parameter questions and skills, candidate ordering at supported nodes, and the handoff question and selection skill | Jev confuses “collect the money” with “tell me how much is owed” after reading the same note. |
| Generation handoff (`handoff`) | Required live data, semantic labels, ordinary alternatives, specialist and default guidance, tool hints and public-read reentry rules | Wording that requests immediate playback should select a playback guide. A recommendation request should receive ordinary guidance. |

The workbench supplies the source of each editable component and shows which component supplied the arguments, candidates, guidance or visible state at a recorded event. For example, a login issued by generated Python and one issued by a compiled fragment take different paths. A compiled-adapter edit can affect the latter.

## How Sol revises a batch

1. **Locate one costly or failed step.** The incoming harness runs all nine tasks. Sol receives their whole-task outcomes, costs, repeated operations, component usage and earlier measured changes. It can open complete public Train events, including the task, Jev candidates and probabilities, generated code, actual tool calls and observations. Sol identifies a task, event and downstream effect, then compares two explanations for that same fault.
2. **Find relevant earlier Train evidence.** A lookup for a current event retrieves up to three distinct scenarios from earlier Train traces in this run. Sol can inspect the exact event and its consumers, see whether the task was solved, and record which part of the operation transfers. The history includes entry traces from earlier batches that produced no patch. When the archive is empty or the earlier operation has the wrong scope, Sol can repair the current issue directly. The editor receives Train evidence only.
3. **Draft a patch.** Sol chooses up to three named objects from the six components and predicts the change for each of the nine tasks. The program installs the edits together, reconstructs parent and candidate requests on current and earlier Train paths, and reports the first difference or the point where a path stops being comparable. Each individual edit is also checked at the focal state. A historical citation is replayed against its cited version even if a newer recording of the task exists.
4. **Test the change locally.** The workbench runs bounded 27B previews on the focal case and a preservation case. It prioritizes a previously solved task whose request changes and which used an edited component. When the draft is revised, the comparison stays on the same cases. Matching recorded requests can reuse their responses. The draft and revision together may spend up to **12 fresh judgments, three open generations and 4,096 offline 27B output tokens**. Optional wording checks regenerate the candidate arguments at a recorded state. They measure eligibility and ordering. Actual task success is measured in AppWorld.
5. **Submit or retain.** After inspecting the preview Sol can submit the draft, revise once and inspect the second preview, or retain the parent. A submitted version is scored on the nine complete tasks. The next round can use the per-task flips and costs. Retaining the parent during review ends that batch's editing. The drafting, review and optional final-review conversations allow up to ten replies each, with the last reserved for a decision. [Prompts and dialogue](../jev_harness/batch_revision.py) · [Recorded path comparisons](../jev_harness/trace_comparison.py) · [Local previews](../jev_harness/batch_revision_trials.py)

The draft review also receives one separately metered 27B choice from a successful earlier Train task. The editor first considers historical examples cited by Sol and related examples from the Train index. The program replays the recorded prefix and renders the first changed decision with the draft. It can query Jev when the parent matches the recorded request at that point and a comparable decision is available. Sol sees the goal, old and new selections, candidate differences and confidence before deciding whether to revise. The budget is one logical choice per round, including retries and the revised draft. An official adaptive choice can use multiple backend requests. A focused menu can still abstain and widen. This observation measures a single decision page. The final report accounts for its cost under `offline_27b_historical_choice`.

## Task splits, selection and held-out evaluation

The task seed is **20261119**. Train uses 30 official scenarios (90 tasks), Dev uses 19 (57 tasks), and Test-N uses 30 preselected Test-Normal scenarios (90 tasks). The sets are scenario-disjoint, with all three official variants kept together. Each task starts with a fresh simulator and model conversation. [Every task ID](../jev_harness/resources/task_split.json)

| Batch | Train scenario IDs | Batch | Train scenario IDs |
|---:|---|---:|---|
| 1 | 6ea6792, 82e2fac, ccb4494 | 6 | 29caf6f, 692c77d, 2a163ab |
| 2 | cf6abd2, afc0fce, 287e338 | 7 | d0b1f43, 229360a, e3d6c94 |
| 3 | 302c169, 3c13f5a, c901732 | 8 | 27e1026, 60d0b5b, 6104387 |
| 4 | e85d92a, 771d8fc, 22cc237 | 9 | 76f2c72, b7a9ee9, ce359b5 |
| 5 | 34d9492, aa8502b, b0a8eae | 10 | 07b42fd, 7d7fbf6, e7a10f8 |

1. **Train.** H0 starts Batch 1. Visit the ten batches once in the order shown, with at most two revisions each. A batch can finish early when it reaches 9/9 with zero online generated tokens.
2. **Compare revisions.** Rank versions first by tasks solved, then by fewer total online output tokens. An exact tie retains the candidate. If the candidate loses, the next round starts from its parent. Rejected traces remain available for diagnosis.
3. **Select on Dev.** When Train yields a new version, run it on all 57 Dev tasks and compare with the version that entered the batch. Evaluate H0 once under the same 0.60 threshold, task list and per-task limits. For later parents, reuse their 57 recorded outcomes under this setting. The Train ranking and tie rule also apply on Dev.
4. **Test the whole system.** After Batch 10, freeze the retained harness and close Sol editing. For each of the 90 Test-N tasks, compare **A**, the ordinary 27B generation agent, with **B**, the same 27B using the final harness and its native decisions. Run each pair on the same GPU with separate initial simulator states. Alternate A/B order across tasks. H0 is the Jev-capable starting point for B. The paired comparison estimates the effect of the evolved system as a whole.

Both arms allow 40 logical model calls, 8,192 total generated tokens, 1,024 tokens per open-generation call, 300 public API requests and 900 seconds per task. Open generation uses greedy decoding on the same backbone, with thinking enabled and a 256-token thinking budget. Online output includes all open-generation tokens, actual low-confidence reasoning tokens, and recorded output from failed retries. Native decision readouts, including post-reasoning scores, are tracked under `decision_readout_tokens`. Per-task records also include backend request counts, input tokens and timing.

The report gives task goal completion (TGC), the proportion of scenarios with all three variants solved (SGC), mean and total online output tokens, and tasks that changed from failure to success or vice versa. A paired bootstrap resamples scenarios 10,000 times to give 95% intervals for differences. Other recorded measures include native judgments, handoff selections and their scenario coverage, input tokens, model time, tasks using both fragments and generation, tasks completed entirely with fragments, and offline Sol and local judgment usage.

From H0 onward, every GPU runs one BF16 model replica with two inference slots. Each slot has an independent AppWorld session. Train, both sides of Dev selection and both Test-N arms share this concurrency setting. The runner reuses the retained parent's Dev records within this run. Paired Test arms run in sequence on the same GPU and session slot, resetting AppWorld between arms. The number of concurrent tasks can fall below capacity at a batch boundary or during a dependent optimizer preview.

## Worked Train scenario

In Batch 4, task `22cc237_1` asks the agent to request outstanding dinner payments. The user recorded each person's share in Simple Note, and some people have already repaid the user in Venmo. H0 can list notes. Reading one selected note's body and reconciling payments are possible targets for later patches.

Suppose Sol sees several traces generate the same search and note-read code. It could propose a fragment that lets Jev select among the retrieved notes and places the actual body in `selected_note`. If the remaining calculations or payment requests need generation, the generator receives that live result, its bound object and the guidance Jev chose. AppWorld then scores the final simulator state. A subsequent editing round can inspect the first changed step, its candidates, the input the generator received, and the resulting actions. This example illustrates the design. The README also includes a measured Train case with two recorded paths. [Spotify trace comparison](spotify-example.json) · [Run commands and outputs](reproduce.md)
