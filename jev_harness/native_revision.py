"""Offline revision of retrieval, evidence, bindings and pure fragments using Train receipts."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from .artifacts import cleaned_text, read, read_events, read_page, save, visible_event
from .composition import declared_sources, output_fields, validate_card, validate_projection
from .native_assembly import FINISH
from .native_evidence import fragment_evidence
from .policy_programs import load_program
from .protocol import design_manifest
from .schema_tools import available, catalog, is_read


DRAFT_REPLY_LIMIT = 12


EDITOR = """Improve the shared AppWorld harness for the fixed 27B. First preserve or improve whole-task success;
then reduce online output tokens, including reasoning. You have only official Train evidence. Use the parent
runs first. Trace the first wrong choice, binding, computation or uncovered action on failed tasks. On successful
tasks, look for repeated or expensive generated work that the program can assemble from official tools. Use the
previous round's matched Train results when present. Past Train traces are available through the evidence index.

Follow fragment_flow from eligibility and recommendation to actual selection, output and accepted uses. Inspect
public inputs and exact accepted payloads, even on a pass. An accepted write only confirms what the API accepted.
Distinguish a directly bound field, an upstream dependency and a generated literal. Preserve the requested
population, ordering, range and object identity. Prefer native API filters. Factor out a computation that can take
different sources and feed different later operations. Do not copy a task ID, person, answer, specific amount, or
whole-task recipe. Check which other tasks could see your new menu or guidance. A write always needs Jev selection.

Use the exact key under "Train evidence" as the doc name in {"read":{"doc":"stage/task","start":0,"count":6,
"offset":0}}. Use doc=index to list available document names or doc=tool/<exact endpoint> for a signature.
The draft has twelve replies, including reads. Reserve the final reply for a complete proposal. Submit {"reason":"evidence and expected behavior",
"objects":[{"kind":"fragment","name":"identifier","value":{...}}]}. Change at most three distinct named items.
Use a justified empty list if no change is supported. An existing item can be deleted with value:null.

Consider all six components:
- retrieval: a short search cue for an endpoint or installed card. Use distinctive source and goal words.
- decision: a short instruction next to an endpoint or card, or name="action" for the action ballot.
- binding: name an official endpoint. Give label, sources, and either a pure program returning a list of public
  kwargs or fields mapping required API parameters to typed output fields. Set covers:true for a complete
  binding. Optional inputs name companion observations. Use sources:[] for a task-only argument program.
  Set projections are grounded in complete observed rows; Jev still selects the operation and applicable set.
- fragment: a reusable composition card described below.
- state: label, sources and pure program for compact public evidence. An intent rule uses intent:true and
  sources:[], and returns requested write endpoints based on state["goal"]. Actual writes require receipts.
- handoff: short guidance for local generation at an endpoint or card. name="residual" addresses uncovered
  work. G_binding fills missing arguments; residual Python can use compiled['observations'] and receipts.
  Use model request purposes, native_residual, tool_batch and api_action events to locate the right edit.

Every fragment uses {label, program, composition, optional goal_terms}. goal_terms are alternative short phrases
grounded in the public task. A page may recommend zero cards. composition.inputs is an ordered mapping of
named ports. A port has {"sources":[{"source":"public_endpoint_or_card"}]} to use a complete observation.
You can add "read" to a public read alternative to acquire it after selection. Use public argument objects,
a pure planner returning the list of keyword objects, or {"input":"earlier_port","fields":{"api_parameter":
"card_field"}} to hydrate every row from a typed earlier input. Use [{}] for a public default, and None for an unsupported
scope. The runtime validates the known arguments, provides authentication and paginates. Multiple argument
objects produce one result per argument. Do not infer scope from the display label.

The pure card and read planners see state["inputs"][port] with rows, source, ref, scope, complete and actual
public arguments as inputs become available, plus instruction, goal and datetime. Output must declare
{"fields":{"field":"integer"}} for typed records, {"value":"number"} for a scalar answer or
{"subset":"port"} for original rows. A subset may also declare fields. Return None when the evidence is
insufficient. Output fields make the result available by card identifier to downstream ports and bindings.

Use composition.choices only when a pure binding cannot resolve an observed record. Each named
question has {"input":"port","fields":["public.field"],"question":"Which observed record matches?"}.
Return {"$choose":{"role":"request","rows":[0,3]}} with up to 15 distinct indices into that port.
The program resumes with state["selected"] and state["selections"] containing source, ref, row, value
and role. It can ask again if a new ambiguity remains. For a dependent question, add
"using":{"request":3} to show the referenced public row. Do not repeat an answered question with the
same rows and reference evidence. NONE leaves the card unresolved. All choices share the task budget.

A contact-based filter can use independent input roles:
"composition":{"inputs":{"rows":{"role":"counterparty_records"},"people":{"role":"contact_identities"}},
"output":{"subset":"rows"}}. The runtime finds the official record direction and contact group, validates
the scope and exposes compatible write or aggregation choices. The program receives each named input's
rows and the field paths state["record_key"] and state["people_key"]. It returns the original matching
records with all fields intact. Keep task text, APIs and consumer selection out of the filter calculation.
An optional composition.equivalent list names existing cards with the same computation. The runtime
merges offers only when the concrete reads and the requested write effects agree.

Pure programs have no imports or external calls. Read actual fields, handle empty sets and ambiguous
values, and keep whole-set operations complete. Supported helpers include regex, captures, sorted,
enumerate, zip, sum and first_calendar_day(state["datetime"], n). Use the smallest evidence-backed edit.
"""


def apply_edits(parent, proposal, tools):
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
        if value is None and name in lib[kind]:
            del lib[kind][name]
        else:
            lib[kind][name] = value

    cues = {"retrieval", "decision", "handoff"}
    for kind, name in touched:
        if name not in lib[kind]:
            continue
        value = lib[kind][name]
        special = (kind, name) in {("decision", "action"), ("handoff", "residual")}
        if kind in cues:
            if not (special or name in lib["fragment"] or name in tools and (available(name) or name == FINISH)):
                raise ValueError("A cue must name a published endpoint or installed card")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Expected a short textual cue")

    for card in lib["fragment"].values():
        validate_card(card, tools, lib["fragment"])
    known = tools.keys() | lib["fragment"].keys() | {"calculation", "inspection"}
    for kind in ("binding", "state"):
        for name, spec in lib[kind].items():
            required = {"sources", "label"}
            optional = {"inputs", "covers", "fields"} if kind == "binding" else {"intent"}
            if (not isinstance(spec, dict) or not required <= spec.keys()
                    or spec.keys() - required - optional - {"program"}
                    or not isinstance(spec["label"], str) or not spec["label"].strip()
                    or not isinstance(spec["sources"], list)):
                raise ValueError("A binding or state rule needs public sources and a useful label")
            sources, companions = spec["sources"], spec.get("inputs", [])
            if not isinstance(companions, list) or any(
                not isinstance(source, str) or source not in known for source in [*sources, *companions]
            ):
                raise ValueError("Unknown input source in proposed harness")
            if len(set(companions)) != len(companions) or set(companions) & set(sources):
                raise ValueError("Use distinct public sources for a joint operation")
            if "intent" in spec and (spec["intent"] is not True or sources):
                raise ValueError("An intent rule runs before tool observations and has empty sources")
            if kind == "state" and not (sources or spec.get("intent")):
                raise ValueError("A state rule needs observations or intent")
            if kind == "binding" and (name not in tools or not (available(name) or name == FINISH)):
                raise ValueError("Bind a published endpoint")
            if "covers" in spec and type(spec["covers"]) is not bool:
                raise ValueError("covers must be a boolean")
            if ("program" in spec) == ("fields" in spec):
                raise ValueError("Use a pure program or a field projection")
            if "program" in spec:
                load_program(spec["program"])
            else:
                if not spec.get("covers") or not sources or companions or is_read(name) or name == FINISH:
                    raise ValueError("A write projection needs a complete source and a public write endpoint")
                for source in sources:
                    validate_projection(spec["fields"], tools[name], output_fields(lib["fragment"].get(source, {})))
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
                        "native_card", "native_card_exit", "native_goal_progress", "native_residual",
                        "native_residual_reentry"):
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
    name = req["doc"]
    if name == "index":
        content = sorted(docs)
    elif name in docs:
        content = docs[name]
    else:
        raise ValueError(f"Unknown document {name!r}. Use doc=index or copy an exact Train evidence key.")
    if isinstance(content, Path):
        content = [cleaned_text(json.dumps(visible_event(e), ensure_ascii=False)) for e in read_events(content)]
    return {"doc": name, **read_page(content, req.get("start", 0), req.get("count", 6), req.get("offset", 0))}


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
            if group == "fragment":
                ports = obj["composition"]["inputs"].values()
                supported = all(any("read" in source or source["source"] in seen
                                    for source in port.get("sources", ())) for port in ports)
                touched += int(supported and bool(declared_sources(obj) & seen))
            elif isinstance(obj, dict):
                touched += int(bool(set(obj.get("sources", [])) & seen) and set(obj.get("inputs", [])) <= seen)
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


def revise(study, train, *, current):
    folder = current.folder
    candidate = folder / "candidate.json"
    if candidate.is_file():
        return read(candidate)
    docs, panel = evidence(current.samples)
    tools = catalog(read(study.root / "offline/public_functions.json"))
    parent = current.parent
    account = study.solver()
    (folder / "sol").mkdir(parents=True, exist_ok=True)
    (folder / "sol/system.txt").write_text(EDITOR, encoding="utf-8")
    thread = account.start(folder / "sol", EDITOR)
    prompt = json.dumps(
        {
            "current_library": parent,
            "Train evidence": panel,
            "trace roles": {"h0_decision": "This round's parent harness, including its Jev choices and generation."},
            "previous round on this batch": current.previous_round,
            "documents": "Use doc=index or copy an exact Train evidence key. Use doc=tool/<endpoint> or doc=tools.",
        },
        ensure_ascii=False,
    )
    docs["tools"] = [json.dumps(t, ensure_ascii=False) for t in tools.values()]
    docs.update({f"tool/{name}": [json.dumps(tool, ensure_ascii=False)] for name, tool in tools.items()})
    for turn in range(1, DRAFT_REPLY_LIMIT + 1):
        final_draft_turn = turn == DRAFT_REPLY_LIMIT
        if final_draft_turn:
            prompt += "\nFinal response: submit a proposal now using the verified evidence."
        elif turn > 1 and prompt.startswith("{"):
            response = json.loads(prompt)
            response["draft_replies_remaining"] = DRAFT_REPLY_LIMIT - turn + 1
            prompt = json.dumps(response, ensure_ascii=False)
        answer = account.turn(thread, prompt, folder / "sol" / f"turn_{turn}")
        try:
            from .runtime import parse_json

            proposal = parse_json(answer)
            if "read" in proposal and not final_draft_turn:
                req = proposal["read"]
                response = evidence_page(docs, req)
                prompt = json.dumps(response, ensure_ascii=False)
                continue
            lib = apply_edits(parent, proposal, tools)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            prompt = f"Could not apply this response: {exc}. Return corrected JSON."
            continue
        if lib != parent:
            from .native_replay import first_change, solved_preview

            draft_library = lib
            impact = {}
            used_current = False
            checks_used = 0
            for task in train:
                events = read_events(current.samples["h0_decision/" + task].trace)
                result = first_change(events, lib, list(tools.values()), study.execution)
                solved = current.samples["h0_decision/" + task].row["success"]
                check_solved = solved and checks_used < 24
                if (check_solved or not solved and not used_current) and result["status"] == "needs_real_choice":
                    study.start_workers()
                    result = first_change(
                        events, lib, list(tools.values()), study.execution, study.workers[0].worker("model"),
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
                    study, lib, current.solved, budget=24 - checks_used
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
                "Continuation describes what the harness reached after a changed decision. Proposed reads are "
                "executable plans, and accepted uses in a fresh trace are the evidence of execution. "
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
                lib = apply_edits(parent, reviewed, tools)
                proposal = reviewed
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                save(folder / "review_error.json", {"error": str(exc), "used": "last valid draft"})
            preview["final_matches_draft"] = lib == draft_library
            save(folder / "draft_preview.json", preview)
        break
    else:
        lib = parent
        proposal = {
            "reason": f"Skipped after {DRAFT_REPLY_LIMIT} draft replies without a valid Sol proposal.",
            "objects": [],
        }
    save(folder / "proposal.json", proposal)
    save(candidate, lib)
    return lib
