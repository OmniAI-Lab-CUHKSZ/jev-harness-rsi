# AppWorld experiment design

**Phase 1 protocol, November 23, 2026.** Starting from H0, a fixed 27B agent runs the first nine Train tasks. GPT-6.2 Sol studies the execution traces and edits the harness shared by all tasks. Semantic handoffs, feedback from the rest of the current batch, and previews on earlier Train states are available from the first round. The controller writes records under the directory given to `--root`. [Run configuration](../jev_harness/resources/experiment.json) · [Reproduction guide](reproduce.md)

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

The initial fragments and parameter guidance come from earlier Train work. The split manifest identifies the 30 historical source tasks. [Full H0](../jev_harness/resources/h00.json) · [Task manifest](../jev_harness/resources/task_split.json)

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

An earlier handoff implementation evaluated Sol-authored `requires` and `when` conditions. The current executor also supplies a native Jev call at handoff. The question, options and skill are editable. Call scheduling, accounting, evaluation and the public tool calls themselves belong to the host. Sol can read the host source and include a `capability_gap` in a proposal identifying the trace and the missing host capability.

## How Sol revises a batch

**Collect the traces.** The incoming harness runs all nine Train tasks. Sol receives per-task success, online token counts, failure locations, executed fragments, the first open-generation step and a searchable trace index. It can open the full instruction, Jev's menu, the adaptive call's actual reasoning usage, initial and post-reasoning probabilities, the fused result and final selection. It can inspect public API calls and returns, generated code, errors, handoffs and the generator's next actual input. Events support text search and paginated reads.

**Locate a change.** The workbench identifies repeated API calls and data operations in the batch's generated code, with source tasks, line numbers and variables. Sol uses that evidence to localize the issue to assembly, selection, execution or handoff and proposes an installable patch with a predicted effect. The current batch supplies the optimization target. Sol may open earlier Train traces from the same experiment. Dev and Test-N are reserved for selection and evaluation outside the optimizer.

**Try the patch.** The workbench can return three kinds of evidence.

| Evidence | What runs | What Sol can learn |
|---|---|---|
| Natural execution on the current batch | Run each task under the candidate until completion or first open generation, and compare with the parent trace. A focus task can continue to a later event with a limited number of generation calls. | Was the new fragment selected in a normal run? Where did the paths first diverge? Did the generator's input or another task change? |
| Local fragment execution | Try a relevant fragment at a recorded public state in the batch. Inspect bound arguments, returned values and downstream use. | Does the fragment compute the intended result when selected? Natural execution separately measures whether Jev chooses it. |
| Historical Train preview | At the same old state, compare parent and candidate menus, ordering, availability of old bindings, handoff options and visible guidance. Each location has an inspectable record. | Which earlier tasks and steps might change? Missing inputs are marked unknown. An API appearing in old generated code identifies a possible reuse opportunity. The preview measures state-level effects. |

For a specialist guide or handoff judgment, Sol may specify up to six wording checks in a patch. When possible, the workbench selects a state from this batch and one from earlier Train data, preferring actual guide hits. If comparison prompts are missing, offline Sol drafts a paraphrase and a closely matched request with a different intended effect. The latter keeps the objects, data and context as similar as possible.

The same 27B makes native decisions under both harnesses at the same recorded state. The workbench also tests removing one required input. It logs the choices, leading option scores and token counts. If the original used the ordinary handoff and there is no reference for the specialist guide, the paraphrase check is marked inconclusive. AppWorld's end-to-end score determines task success.

**Score one candidate.** Each editing round allows up to 12 interactions. A proposal submitted before the last turn can be revised or confirmed after the preview. A proposal submitted on the last turn is still previewed, and its feedback is recorded for later rounds. Only the final candidate is scored on the full nine tasks. Offline Sol usage, fragment trials and wording checks are accounted for separately from online task execution.

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

Batch 1 uses one active task per GPU for Train and for both Dev panels. Batch 2 switches Train, independent Sol trials and Dev to two task sessions per GPU. Test-N uses the same two-session setting across different task pairs. A change in concurrent inference can change the generated code and token count. The runner reuses the retained parent's Dev score and records the parent and candidate session counts alongside their comparison.

## Worked Train scenario

In Batch 4, task `22cc237_1` asks the agent to request outstanding dinner payments. The user recorded each person's share in Simple Note, and some people have already repaid the user in Venmo. H0 can list notes. Reading one selected note's body and reconciling payments are possible targets for later patches.

Suppose Sol sees several traces generate the same search and note-read code. It could propose a fragment that lets Jev select among the retrieved notes and places the actual body in `selected_note`. If the remaining calculations or payment requests need generation, the generator receives that live result, its bound object and the guidance Jev chose. AppWorld then scores the final simulator state. A subsequent editing round can inspect the first changed step, its candidates, the input the generator received, and the resulting actions. This example illustrates the design. The README also includes a measured Train case with two recorded paths. [Spotify trace comparison](spotify-example.json) · [Run commands and outputs](reproduce.md)
