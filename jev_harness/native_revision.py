"""Offline revision of retrieval, evidence, bindings and pure fragments using Train receipts."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from formal_run import read, save
from phase1_run import design_manifest
from .formal_optimizer import cleaned_text, read_events
from .native_assembly import FINISH, blank_library
from .native_evidence import fragment_evidence
from .policy_programs import load_program
from .schema_tools import MANAGED, available, catalog, is_read
from .train_evidence import read_page, visible_event


EDITOR = """Improve this shared AppWorld harness for the fixed 27B. Preserve or raise whole-task success, then
reduce online output including reasoning. Only official Train trajectories have been opened. Start with the
current parent harness's own runs. Follow Jev choices and abstentions, local generation, public tool calls and
receipts. On failed tasks, locate the first step where the action, bound values or result leave the user's goal
unmet. On successful tasks, identify costly generated work that public data and a reusable decision could cover.
When a prior round measured two harness versions on the same Train task, compare their first actionable
difference and follow it through the tool results.
Use fragment_flow to follow eligible -> recommended on an action page -> offered to Jev (including browsing)
-> chosen -> produced -> arguments of an accepted write or answer. An eligible fragment may have zero recommendations.
An API receipt confirms acceptance. Inspect the submitted payload against source evidence even on official passes.
If a skipped fragment was evaluated offline, its output comparison is a diagnostic, not an executed action or ground truth.
Distinguish direct field projection, upstream data, a program input, a scope, and an unreferenced generated literal.
For a partial update identify the target and actual change. Ground entity roles in the available public fields.
Check whether a different wording with the same goal and sources would still reach the proposed operation.
In the parent traces, G_tool generates a local action when the action menu leaves work uncovered. G_binding
supplies missing arguments after an endpoint has been selected. A handoff cue reaches only that endpoint's
G_binding. Use the recorded request purpose to locate the component to edit. Preserve filter/aggregation order and match the population
used by a superlative; checking a selected record's status may require a wider source than prefiltering it.
Prefer a public API's existing filters. Compile a repeated or expensive open step into a middle operation
usable for another downstream goal. Match public sources, distinguish actor/direction/scope, and never
embed a task ID, person, answer, specific amount or a whole-task sequence. A write needs explicit selection.
Consider all six locations and modify only the ones supported by evidence. A binding/fragment is a possible
choice for Jev, never an automatic write. Current and past solved Train cases matter. No model weights change.
Return either {"read":{"doc":"stage/task","start":0,"count":6,"offset":0}} or one JSON proposal
{"reason":"evidence and expected behavior","objects":[{"kind":"fragment","name":"...","value":...}]}.
Use at most three distinct named changes. Use [] only when a justified no-change is best.
For kind retrieval, decision or handoff: name = exact public tool name, value = a short string. Retrieval
adds search words; decision adds a cue only next to that option; handoff guides only that tool's local generation.
For fragment/state: give a new identifier. For binding: use the receiving official tool's exact name.
For these three kinds value={"sources":["upstream public tool or installed fragment name"],"label":"clear meaning",
"program":"body of a pure Python function receiving state"}. state contains instruction, datetime, source, scope,
complete, rows (list, with nested real fields). A fragment returns selected records or a scalar as a NEW observation,
a binding returns a list of official kwargs tied to the SAME row per call, a state rule returns compact evidence.
For a cross-source fragment or binding, add "inputs":["another public source"] to its definition. The runtime
offers combinations only after both inputs were observed. state["inputs"][that source] contains rows, ref, scope
and complete. The built-in observation sources "calculation" and "inspection" can also be used.
A fragment can set "goal_terms":["one goal cue","another wording"]. Each phrase is an alternative;
use short, distinctive words for the user's goal, including paraphrases supported by Train evidence. The
action page may recommend zero fragments. Sources alone do not establish relevance. Without this field
the runtime uses the label. All eligible fragments remain reachable by browsing.
A fragment may declare "choices":{"request":{"source":"one of its declared sources","fields":["public.field"],
"question":"Which observed request does the user mean?"},"target":{...}}. Compute direct bindings first. At each
ambiguity return {"$choose":{"role":"request","rows":[0,3]}} with 1-15 distinct zero-based row indices in that
role's source. The pure program resumes with state["selections"] (ordered role,source,ref,row,value records)
and state["selected"] (last bound source,ref,row,value). On the initial call use state.get("selections",[]).
Return a result, None for missing evidence, or a further ballot as needed. A dependent ballot can add
"using":{"request":3} inside "$choose" to show that observed row's declared fields, whether code or Jev found
the row. Keep questions neutral. Use a previous result to narrow the rows before asking. The same role can recur
with different candidates or reference evidence. Repeating an answered question without new evidence stops
the fragment. All ballots share the task's normal model and token budgets. NONE leaves the fragment unresolved.
The earlier single "choice":{source,fields,question} with {"$choose":[indices]} is also supported.
Return None when unsupported; handle ambiguity, empty inputs and ties. No imports/external calls. Builtins allowed:
len any all sorted min max sum enumerate zip str int float list dict set isinstance regex captures. regex checks
case-insensitive match, captures returns findall. Allowed methods: get keys items values startswith endswith lower
casefold strip split replace join append extend add count. For sorting use tuples or sorted values (no lambda).
Match each option's wording to its actual effect and check the menus and generation requests that your change
could reach on other tasks. Prefer the smallest change supported by this panel. Can request full events."""


COMPOSED_EDITOR = """
These runs use composed cards. G_binding can still fill a selected API's missing fields. G is a stateful
Python fallback for uncovered work. The code already has compiled['observations'][ref] and compiled['receipts'].
Use the native_residual, tool_batch, and api_action events to locate which read, join, arithmetic or write collection
generation attempted. Direct official tool calls share the card runtime's authentication, validation, pagination
and receipts. Keep the concrete task's data out of the patch.

A fragment is an independently reusable data operation. Use optional "produces":{"field":"integer"}
with JSON types to declare the output row fields. The identifier is the downstream source. A fragment may add
"reads":{"app__public_read":"pure planner code returning a list of keyword objects"} for a declared
primary source or companion in inputs. Materialization proceeds in that order, only after Jev selects the
card. The program checks known read arguments before offering a card and checks dependent reads again after
their upstream data arrives. A read can instead use {"source":"a previously loaded fragment","fields":
{"api_parameter":"declared_output_field"}} to project one argument set from every record. This is useful
for hydrating IDs already produced by a card. The pure read planner can see the full task text, datetime and previously loaded inputs; the first read
must be plannable from the task itself. For acquired reads return [{}] for public defaults, or None when
the goal leaves the scope ambiguous. The runtime supplies authentication and reads all pages when pagination
is omitted. When choosing a set, use public date filters and join stable identity fields instead of text
matching names. first_calendar_day(state["datetime"], n) gives the start of the last n calendar days including today.
The pure card receives observed rows after those reads. A public read with multiple argument objects yields
one result per argument. For a whole group use each correctly scoped row exactly once. Read observations have
an "arguments" list containing the actual public argument objects, both at state["arguments"] for the primary
source and within state["inputs"][source] for companion reads. Check scope against these structured arguments.
The "scope" string is only a display label.

The binding rule for a public tool may use "sources":[] when it only needs task literals. Add "covers":true
when the rule yields a deterministically grounded, complete binding for the proposed action, returning None
otherwise. For a complete typed object set, the binding can use "fields":{"api_parameter":"declared_output_field"}
in place of "program". The runtime validates the values, deduplicates argument sets and executes the batch
after Jev selects the effect. If different observed sets can supply the same operation, Jev chooses which
set to use. When a task clearly requires this effect, an intent rule should name the endpoint so the menu
waits for its complete projected scope. Its effect and scope are shown before Jev authorizes the write. A matched contact or a write receipt
alone is insufficient for all-of scope. Multiple requested actions should reuse the same selected cohort.
A state rule with "intent":true and "sources":[] can return a list of remaining write endpoint requirements from
the unquoted goal in state["goal"]. Endpoint success is tracked by actual receipts. Use this to prevent premature
completion when the user clearly requests multiple effects. Update "decision"/"action" to guide the action
ballot, or a tool-specific cue to clarify that tool only. "handoff"/"residual" applies when remaining work must
run in Python; tool-specific handoff cues are selected by the unresolved operation. Do not accumulate multiple
ways to do the same work. Reuse a card when its public inputs and output already fit.
"""


def validate_projection(fields, tool, sources, fragments):
    schema = tool["parameters"]
    if (
        not isinstance(fields, dict) or not fields
        or not all(isinstance(field, str) and field for field in fields.values())
        or fields.keys() - (schema["properties"].keys() - MANAGED)
        or (set(schema.get("required", [])) - MANAGED) - fields.keys()
    ):
        raise ValueError("A projection must supply the official tool's required public arguments")
    if not sources or any(set(fields.values()) - fragments.get(source, {}).get("produces", {}).keys()
                          for source in sources):
        raise ValueError("A projection requires fields declared by each source fragment")


def apply_edits(parent, proposal, tools, *, composed=False):
    objects = proposal.get("objects")
    if not isinstance(objects, list) or len(objects) > 3:
        raise ValueError("A patch can change up to three named items")
    lib = deepcopy(parent)
    touched = set()
    for edit in objects:
        kind, name, value = edit["kind"], edit["name"], edit["value"]
        if kind not in lib or not isinstance(name, str) or not name.isidentifier() or (kind, name) in touched:
            raise ValueError("Unknown, duplicate or invalid component")
        touched.add((kind, name))
        special = composed and (kind, name) in {("decision", "action"), ("handoff", "residual")}
        if kind in {"retrieval", "decision", "binding", "handoff"} and not special and (
            name not in tools or not (available(name) or name == FINISH)
        ):
            raise ValueError("This edit needs a published endpoint")
        if value is None and name in lib[kind]:
            del lib[kind][name]
            continue
        if kind in {"retrieval", "decision", "handoff"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Expected a short textual cue")
        else:
            fields = {"sources", "label"}
            optional = {"inputs"} if kind in ("fragment", "binding") else set()
            if kind == "fragment":
                optional.update(("choice", "choices", "goal_terms"))
            if composed:
                optional.update({
                    "fragment": {"reads", "produces"},
                    "binding": {"covers", "fields"},
                    "state": {"intent"},
                }.get(kind, set()))
            if (
                not isinstance(value, dict)
                or not fields <= value.keys()
                or value.keys() - fields - optional - {"program"}
                or not isinstance(value["sources"], list)
                or (not value["sources"] and not (composed and (kind == "binding" or value.get("intent"))))
            ):
                raise ValueError("A program needs sources, label and program")
            if not isinstance(value["label"], str) or not value["label"].strip():
                raise ValueError("A program needs a useful choice label")
            if "covers" in value and type(value["covers"]) is not bool:
                raise ValueError("covers must be a boolean")
            if "intent" in value and (value["intent"] is not True or value["sources"]):
                raise ValueError("An intent rule runs before tool observations and has empty sources")
            if ("program" in value) == ("fields" in value):
                raise ValueError("Use a pure program or a field projection")
            if "program" in value:
                load_program(value["program"])
        lib[kind][name] = value
    sources = set(tools) | set(lib["fragment"]) | {"calculation", "inspection"}
    for group in ("fragment", "binding", "state"):
        for name, spec in lib[group].items():
            companions = spec.get("inputs", [])
            if not isinstance(companions, list) or any(
                not isinstance(s, str) or s not in sources for s in [*spec["sources"], *companions]
            ):
                raise ValueError("Unknown input source in proposed harness")
            if len(set(companions)) != len(companions) or set(companions) & set(spec["sources"]):
                raise ValueError("Use distinct public sources for a joint operation")
            if group == "binding" and "fields" in spec:
                if not spec.get("covers") or companions or name not in tools or is_read(name) or name == FINISH:
                    raise ValueError("A write projection needs a complete source and a public write endpoint")
                validate_projection(spec["fields"], tools[name], spec["sources"], lib["fragment"])
            if group != "fragment":
                continue
            if "reads" in spec:
                if not isinstance(spec["reads"], dict):
                    raise ValueError("Card reads map public endpoints to pure planners")
                for endpoint, body in spec["reads"].items():
                    if (endpoint not in [*spec["sources"], *companions] or endpoint not in tools
                            or not available(endpoint) or not is_read(endpoint)):
                        raise ValueError("Card acquisition must use a declared public read source")
                    if isinstance(body, dict):
                        before = [
                            [primary, *companions[:companions.index(endpoint)]]
                            for primary in spec["sources"]
                        ] if endpoint in companions else []
                        if (
                            body.keys() != {"source", "fields"} or not before
                            or any(body["source"] not in path for path in before)
                        ):
                            raise ValueError("A read projection needs a fragment in an earlier input position")
                        validate_projection(body["fields"], tools[endpoint], [body["source"]], lib["fragment"])
                    else:
                        load_program(body)
            if "produces" in spec and (
                not isinstance(spec["produces"], dict) or not spec["produces"]
                or not all(isinstance(field, str) and field.strip() and kind in
                           ("integer", "number", "string", "object", "array", "boolean")
                           for field, kind in spec["produces"].items())
            ):
                raise ValueError("Card produces must map public field paths to JSON types")
            if "goal_terms" in spec and (
                not isinstance(spec["goal_terms"], list) or not spec["goal_terms"]
                or not all(isinstance(term, str) and term.strip() for term in spec["goal_terms"])
            ):
                raise ValueError("Fragment goal_terms must be a nonempty list of goal phrases")
            slots = spec.get("choices", {})
            if "choices" in spec and (
                "choice" in spec or not isinstance(slots, dict) or not slots
                or not all(isinstance(role, str) and role.isidentifier() for role in slots)
            ):
                raise ValueError("Use either one choice or a mapping of named choice roles")
            for slot in ([spec["choice"]] if "choice" in spec else slots.values()):
                if (
                    not isinstance(slot, dict) or slot.keys() != {"source", "fields", "question"}
                    or slot["source"] not in [*spec["sources"], *companions]
                    or not isinstance(slot["question"], str) or not slot["question"].strip()
                    or not isinstance(slot["fields"], list) or not slot["fields"]
                    or not all(isinstance(f, str) and f.strip() for f in slot["fields"])
                ):
                    raise ValueError("A fragment choice needs one of its sources, public field paths and a question")
    return lib


@dataclass
class TrainSample:
    trace: Path
    row: dict
    library: dict


@dataclass
class EditRound:
    folder: Path
    parent: dict
    samples: dict
    solved: dict
    stage_prefix: str = ""
    previous_round: dict = field(default_factory=dict)


def pilot_round(root, train, history, *, entry_only=False):
    folder = root / "offline"
    parent = read(root / "entry_harness.json")
    samples, solved = {}, {}

    def add(directory, stage, task, library, prefix=""):
        row = read(directory / "runs" / stage / task / "record.json")
        sample = TrainSample(directory / row["trace_path"], row, library)
        samples[f"{prefix}{stage}/{task}"] = sample
        return sample

    for task in train:
        if not entry_only:
            add(root, "h0_generation", task, blank_library())
        add(root, "h0_decision", task, parent)
    for i, directory in enumerate(history):
        plan = read(directory / "pilot_plan.json")
        for task in plan["train"]:
            for stage, path in (("h0_decision", "entry_harness.json"), ("candidate", "offline/candidate.json")):
                record = directory / "runs" / stage / task / "record.json"
                if record.is_file():
                    sample = add(directory, stage, task, read(directory / path), f"history{i}/")
                    if sample.row["success"]:
                        solved[task] = (sample.trace, directory / "offline/public_functions.json", plan["execution"])
    return EditRound(folder, parent, samples, solved)


def evidence(samples):
    permitted = {t for batch in design_manifest()["train_batches"] for t in batch["tasks"]}
    docs, panel = {}, {}

    for label, sample in samples.items():
        record = sample.row
        if record["task_id"] not in permitted:
            raise ValueError("Only official Train tasks can enter the editor")
        docs[label] = sample.trace
        if label.startswith(("history", "confirmation/")):
            # Keep old trajectories addressable without putting all old menus into the first Sol request.
            with sample.trace.open(encoding="utf-8") as stream:
                task_event = json.loads(stream.readline())
            panel[label] = {"instruction": task_event["instruction"], "success": record["success"],
                            "output_tokens": record["output_tokens"], "open_calls": record["open_calls"]}
            continue
        events = read_events(sample.trace)
        steps = []
        open_sites = []
        for i, e in enumerate(events):
            kind = e["event"]
            if kind == "model" and e["operation"] == "generate":
                request = e["request"]
                view = json.loads(request["messages"][-1]["content"])
                open_sites.append(
                    [
                        i,
                        request.get("purpose"),
                        view.get("selected_tool", {}).get("name"),
                        e["response"].get("output_tokens"),
                    ]
                )
            if kind == "decision":
                chosen = e["candidates"][e["selected"]]["label"]
                steps.append([i, e["node"], chosen[:175]])
            if kind in ("native_operation", "native_feedback", "native_observation", "native_discovery",
                        "native_card", "native_goal_progress", "native_residual", "native_residual_reentry"):
                detail = next((e[k] for k in ("tool", "source", "reason", "query", "name", "completed")
                               if k in e), None)
                steps.append([i, kind, detail, e.get("count", e.get("remaining"))])
        panel[label] = {
            "instruction": events[0]["instruction"],
            "success": record["success"],
            "output_tokens": record["output_tokens"],
            "open_calls": record["open_calls"],
            "steps": steps[:70],
            "open_sites": open_sites,
            "events": len(events),
            "observed_sources": sorted({e["source"] for e in events if e["event"] == "native_observation"}),
            "fragment_flow": fragment_evidence(events, sample.library),
        }
    return docs, panel


def evidence_page(docs, req):
    content = docs[req["doc"]]
    if isinstance(content, Path):
        content = [cleaned_text(json.dumps(visible_event(e), ensure_ascii=False)) for e in read_events(content)]
    return read_page(content, req.get("start", 0), req.get("count", 6), req.get("offset", 0))


def trial_task(train, impact, panel, parent, lib):
    changed = [
        (group, name)
        for group in lib
        for name in sorted(parent[group].keys() | lib[group].keys())
        if lib[group].get(name) != parent[group].get(name)
    ]
    ranked = []
    for order, task in enumerate(train):
        row = panel["h0_decision/" + task]
        seen = set(row["observed_sources"]) | {s[2] for s in row["steps"] if s[1] == "native_operation"}
        touched = 0
        for group, name in changed:
            obj = lib[group].get(name) or parent[group].get(name)
            if isinstance(obj, dict):
                touched += int(
                    bool(set(obj.get("sources", [])) & seen) and set(obj.get("inputs", [])) <= seen
                )
            else:
                touched += int(name in seen)
        different = impact[task]["status"] not in {"unchanged", "unresolved"}
        if touched or different:
            ranked.append(((touched, different, not row["success"], -order), task))
    return max(ranked)[1] if ranked else None


def full_train_trial(study, task, lib, current):
    folder = current.folder / "draft_trials"
    folder.mkdir(exist_ok=True)
    spec = {"task": task, "library": lib}
    existing = sorted(folder.glob("*.json"))
    path = next((p for p in existing if read(p) == spec), None)
    if path is None:
        path = folder / f"draft_train_{len(existing) + 1:02d}.json"
        save(path, spec)
    study.start_workers()
    gpu = current.samples["h0_decision/" + task].row["gpu"]
    stage = current.stage_prefix + path.stem
    (row,) = study.execute_preferred(
        [task], lambda t, w: study.run_task(t, stage, "B", lib, workers=w), [gpu]
    )
    events = read_events(study.root / row["trace_path"])
    return {
        "task": task,
        "stage": stage,
        "success": row["success"],
        "output_tokens": row["output_tokens"],
        "open_calls": row["open_calls"],
        "events": len(events),
        "trace_path": row["trace_path"],
        "fragment_flow": fragment_evidence(events, lib),
    }


def revise(study, train, history=(), *, current=None, composed=False):
    current = current or pilot_round(study.root, train, history)
    folder = current.folder
    candidate = folder / "candidate.json"
    if candidate.is_file():
        return read(candidate)
    docs, panel = evidence(current.samples)
    tools = catalog(read(study.root / "offline/public_functions.json"))
    parent = current.parent
    account = study.solver()
    (folder / "sol").mkdir(parents=True, exist_ok=True)
    instructions = EDITOR + (COMPOSED_EDITOR if composed else "")
    (folder / "sol/system.txt").write_text(instructions, encoding="utf-8")
    thread = account.start(folder / "sol", instructions)
    roles = {"h0_decision": "This round's parent harness, including its Jev choices and local generation."}
    if any(label.startswith("h0_generation/") for label in panel):
        roles["h0_generation"] = "Pilot comparison that generates each action. Use it to identify work to compile into the harness."
    prompt = json.dumps(
        {
            "current_library": parent,
            "Train evidence": panel,
            "trace roles": roles,
            "previous round on this batch": current.previous_round,
            "catalogue": (
                "Request doc=tool/<exact endpoint name> for one public signature, or doc=tools to browse."
                if composed else "Request doc=tools to inspect official signatures"
            ),
        },
        ensure_ascii=False,
    )
    docs["tools"] = [json.dumps(t, ensure_ascii=False) for t in tools.values()]
    if composed:
        docs.update({f"tool/{name}": [json.dumps(tool, ensure_ascii=False)] for name, tool in tools.items()})
    for attempt in range(8):
        if attempt == 7:
            prompt += "\nFinal response: submit a proposal now using the verified evidence."
        answer = account.turn(thread, prompt, folder / "sol" / f"turn_{attempt + 1}")
        try:
            from .runtime import parse_json

            proposal = parse_json(answer)
            if "read" in proposal and attempt < 7:
                req = proposal["read"]
                response = evidence_page(docs, req)
                prompt = json.dumps(response, ensure_ascii=False)
                continue
            lib = apply_edits(parent, proposal, tools, composed=composed)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            prompt = f"Could not apply this response: {exc}. Return corrected JSON."
            continue
        if lib != parent:
            from .native_replay import first_change, solved_preview

            draft_library = lib
            impact = {}
            used_current = False
            checks_used = 0
            runtime = {"executor_type": study.compiled_executor} if composed else {}
            for task in train:
                events = read_events(current.samples["h0_decision/" + task].trace)
                result = first_change(events, lib, list(tools.values()), study.execution, **runtime)
                solved = current.samples["h0_decision/" + task].row["success"]
                check_solved = composed and solved and checks_used < 24
                if (check_solved or not used_current) and result["status"] == "needs_real_choice":
                    study.start_workers()
                    result = first_change(
                        events, lib, list(tools.values()), study.execution, study.workers[0].worker("model"),
                        **runtime,
                    )
                    if check_solved:
                        checks_used += 1
                    else:
                        used_current = True
                impact[task] = result
            preview = {
                "draft": proposal,
                "current_train": impact,
                "previously_solved_train": solved_preview(
                    study, lib, current.solved, **({"budget": 24 - checks_used, **runtime} if composed else {})
                ),
            }
            task = trial_task(train, impact, panel, parent, lib)
            preview["whole_current_train_trial"] = full_train_trial(study, task, lib, current) if task else None
            if task:
                trial = preview["whole_current_train_trial"]
                docs["draft/" + task] = study.root / trial["trace_path"]
                trial["doc"] = "draft/" + task
            save(folder / "draft_preview.json", preview)
            review = account.turn(
                thread,
                "These first-change checks replay the draft on actual Train prefixes. "
                "The reselected ballot used the real 27B. A changed generation request was not executed. "
                "The whole_current_train_trial, when present, is an actual fresh episode on the draft. Read its "
                "accepted payloads and fragment_flow, including unused output, before attributing benefit. "
                "A cheaper pass may send the wrong content. Check against evidence and the public instruction. "
                "Check that the modified component is reachable, select a repair if needed, and return the complete "
                "proposal against the parent. You may request one further evidence page with read before your final proposal.\n"
                + json.dumps(preview, ensure_ascii=False),
                folder / "sol" / "review",
            )
            try:
                reviewed = parse_json(review)
                if "read" in reviewed:
                    req = reviewed["read"]
                    page = evidence_page(docs, req)
                    reviewed = parse_json(
                        account.turn(
                            thread,
                            json.dumps(page, ensure_ascii=False) + "\nSubmit your complete proposal now.",
                            folder / "sol" / "review_final",
                        )
                    )
                lib = apply_edits(parent, reviewed, tools, composed=composed)
                proposal = reviewed
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                save(folder / "review_error.json", {"error": str(exc), "used": "last valid draft"})
            preview["final_matches_draft"] = lib == draft_library
            save(folder / "draft_preview.json", preview)
        save(folder / "proposal.json", proposal)
        save(candidate, lib)
        return lib
    raise ValueError("Sol did not return an applicable native-tool revision; see the saved turns")
