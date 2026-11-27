# AppWorld composition experiment

The experiment measures how offline edits to one shared harness affect official AppWorld success and online generation cost while keeping the executor model fixed. The protocol identifier is `appworld_composition_v1`. Inspect the exact settings with `jev-appworld plan`. [Task split](../jev_harness/resources/task_split.json) · [Execution settings](../jev_harness/resources/experiment.json)

## Initial harness

All five H0 cards use the same declaration. Each card has a `label`, a pure Python `program`, a `composition` object and optional `goal_terms`. An output becomes an observation under the card's identifier.

| Card | Inputs | Output |
|---|---|---|
| `playlist_song_ids` | The user's Spotify playlists | Distinct song IDs |
| `album_song_ids` | The user's Spotify album library | Distinct song IDs |
| `rank_scoped_songs` | Playlist IDs, album IDs or song-library rows, plus official song details | The uniquely ranked song ID and title |
| `playlist_genre_follow_targets` | Playlist song IDs, official song details and current followed artists | IDs of matching artists who still need to be followed |
| `pending_requests_by_contact_group` | Phone contact groups and incoming pending payment requests | Scoped request IDs for official approval or denial |

H0 also has local retrieval and decision cues for official tools, argument binding rules, effect tracking for explicit multi-action goals, and handoff guidance. [Packaged H0](../jev_harness/resources/composed_h0.json)

## Online execution

**Plan.** The program ranks official tools and cards using the public goal, current observations and outstanding effects. It checks input availability and validates known arguments before display. The menu may contain zero cards. Browsing can reach executable cards with weaker lexical matches. A card's output fields can support another card, typed API binding or verified answer.

Each named input has source alternatives. A source can refer to a complete observation or acquire an official read using explicit public arguments, a pure argument planner or a projection from an earlier port. A dependent read is checked after its input is available. Authentication and pagination are handled by the runtime. Calls with several argument objects retain one response for each object.

Some applications have registered role bindings. For example, the contact filter's `counterparty_records` and `contact_identities` ports can assemble different Venmo directions, contact groups and dates. The filter sees the resolved records and identity paths. The adapter checks compatibility with the requested calculation or official follow-up before offering the card. Scoped duplicate offers are collapsed only when the concrete reads and outstanding write effects agree. [Composition implementation](../jev_harness/composition.py) · [Contact adapter](../jev_harness/record_composition.py)

**Select.** Programmatic bindings handle values that the public evidence determines. Jev can choose between eligible actions, scopes and observed records. An internal `composition.choices` question cites a named input and public fields. The card program receives the selected row, can ask a follow-up with relevant reference records, and shares the normal task budget. For uncertain native ballots, the official adaptive endpoint uses a 0.60 threshold and up to 256 reasoning tokens.

**Execute.** Cards acquire public inputs and compute typed records, scalars or a verified subset. Lineage connects result fields to the reads and records that supplied them. Projected writes run after selection and add a receipt only on acceptance. A completion choice requires the necessary effects or a verified answer. If a card cannot establish its scope or the model chooses NONE, the executor returns to the available tool path with the observed facts and accepted receipts. Exit diagnostics stay in the trace.

An unresolved card can return when a complete observation matches one of its input sources and read arguments. The executor remembers a card exit against the inputs available at that point, including any reads it acquired before exiting. After an accepted write, acquiring current input data refreshes earlier reads in that application and checks the lineage of derived inputs. Previously selected object sets and accepted receipts remain available to bind subsequent operations.

**Generate where needed.** A selected official tool can use local generation for missing arguments. An uncovered step can use Python with the already compiled observations and receipts. A generated `yield_read(endpoint, arguments)` delegates a public read and can return control to the action loop. After possible writes in the Python session, the generator continues in that session so it can observe the result. The ReAct arm generates its tool and argument choices on the same official interface. [Execution loop](../jev_harness/composed_assembly.py) · [Tool execution](../jev_harness/native_assembly.py)

## Offline optimization

Sol starts each round from the current batch's parent trajectories. The next round also sees the previous round's candidate and parent comparison where available. The editor can open prior official Train traces through an index. It can inspect tool signatures, runtime events, accepted writes, fragment use and failed generation sites.

| Component | Editable content |
|---|---|
| Retrieval | A cue for a public tool or card |
| Decision | A local tool or card cue, or guidance for the action menu |
| Binding | A pure public argument planner or field projection |
| Card | Named inputs, program, optional questions and typed output |
| State | Evidence summaries and outstanding effect rules |
| Handoff | Local guidance for missing arguments or residual generation |

Each candidate changes up to three named items. Sol gets up to twelve replies for the draft, including evidence requests, followed by draft review and one optional evidence read. If the draft window ends without a valid proposal, the controller records the reason and keeps the parent for that round. The next scheduled round still runs. The validator accepts the single card declaration described above. Programs are restricted to local computation and public data.

The preview identifies the first changed request on each recorded prefix. It replays previously solved Train prefixes until the first difference. A changed ballot goes to real 27B within the 24-successful-case reselection budget. If a replay can proceed, it identifies the next affected request or proposed inputs. An additional current failed case can receive a real reselection. One affected current task can receive a fresh draft episode. Sol sees the draft feedback before returning its final proposal. [Editor and prompt](../jev_harness/native_revision.py) · [Replay and reselection](../jev_harness/native_replay.py)

## Train, Dev and Test

1. Evaluate the fixed 27B ReAct arm on all 57 Dev tasks, then evaluate H0 on the same panel.
2. Start each of ten Train batches from the retained harness. Run its nine tasks and use these traces for Sol. Each batch has two rounds, even after a no-change or a rejection. Evaluate a candidate on the full batch.
3. Compare the Train winner with the batch entry on Dev if the version changed. Use the prior Dev measurement for the parent. Carry the winning harness into the next batch.
4. Pair ReAct with the final harness on 90 Test-N tasks. Each task has two fresh episodes, matched by GPU, with alternating arm order. Report task success, scenario success across all three variants, output tokens and paired scenario bootstrap intervals.

Train and Dev first compare whole-task successes and then online output tokens. A complete tie keeps the candidate. If same-request output variation could change the ranking, the runner confirms the affected tasks twice per version. Each paired pass uses one GPU, and the second reverses the order of the first. The two confirmation results replace the initial score for those tasks. Other tasks retain their measured results.

Both online arms use up to 40 model calls, 300 API calls, 8,192 output tokens and 900 seconds per task. The 27B service uses BF16 weights, CUDA Graph decode and two sessions per GPU. The allocator requests up to six eligible GPUs. Report online decision and generation costs and offline Sol, preview and confirmation costs separately. [Experiment controller](../jev_harness/native_formal.py) · [Paired selection](../jev_harness/paired_selection.py)
