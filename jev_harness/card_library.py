"""Public interfaces and offline edits for the fragment-card AppWorld executor."""
from copy import deepcopy
import json
from pathlib import Path

from .expressions import OPS


DESIGN = Path(__file__).resolve().parent / "resources"
PROTOCOL = "phase1_fragment_cards_v1"
ACTIVE_QUESTIONS = ("step","quantity","libraries","directory","entity","query","genre","order")
# The actual data requirements of installed Python operations are shared by the menu and dispatcher.
# (application, required live roles, new role, observed fields fetched by this operation)
BUILTINS = {
    "library": ("spotify", (), "song_ids", "membership of the selected libraries"),
    "playlist": ("spotify", (), "song_ids", "members of the named playlist"),
    "album": ("spotify", (), "song_ids", "members of the named catalog album"),
    "directory": ("file_system", (), "directory_entries", "recursive entries of the source path"),
    "filter_genre": ("spotify", ("song_ids",), "filtered_songs", "genre for each current song ID"),
    "sort_songs": ("spotify", ("song_ids",), "ranked_songs", "the selected ranking field for each current song ID"),
    "take_first": ("spotify", ("ranked_songs",), "limited_songs", ""),
    "titles": ("spotify", ("song_ids",), "answer", "title for each current song ID"),
    "play": ("spotify", ("song_ids",), "effect_receipt", ""),
    "submit_answer": ("supervisor", ("answer",), "submitted", ""),
    "submit_effect": ("supervisor", ("effect_receipt",), "submitted", ""),
}


def ports(card):
    if "builtin" in card:
        return BUILTINS[card["builtin"]]
    return card["app"], tuple(card["inputs"]), card["output"], card.get("loads", "")


def initial_cards():
    library = json.loads((DESIGN / "fragment_cards_h0.json").read_text(encoding="utf-8"))
    pilot = json.loads((DESIGN / "fragment_pilot_library.json").read_text(encoding="utf-8"))
    library["questions"] = {**pilot["questions"], **library["questions"]}
    library["skills"] = {**pilot["skills"], **library["skills"]}
    return library


def validate_card(name, card, functions):
    if not isinstance(card, dict) or not all(isinstance(card.get(k), str) and card[k].strip()
                                             for k in ("description", "skill")):
        raise ValueError(f"{name} needs a description and a decision skill")
    if not isinstance(card.get("handoff_skill",""),str):
        raise ValueError("A fragment handoff skill is text")
    modes = [key for key in ("builtin", "api", "compute") if key in card]
    if len(modes) != 1:
        raise ValueError(f"{name} needs one executable operation")
    if modes[0] == "builtin":
        if name != card["builtin"] or name not in BUILTINS or set(card) - {"builtin","description","skill","handoff_skill"}:
            raise ValueError(f"Unknown builtin or unused properties for {name}")
        return
    allowed = {"app","inputs","output","description","skill","handoff_skill","loads","parameters","compute","api",
               "arguments","paginate","each","public_read"}
    if set(card) - allowed or not isinstance(card.get("inputs"), list) or not all(
        isinstance(k, str) and k.isidentifier() for k in [*card["inputs"], card.get("output"),card.get("app")]
    ):
        raise ValueError(f"Invalid data roles or properties for {name}")
    if card["output"] == "submitted" or card["output"] in card["inputs"]:
        raise ValueError("Produce a distinct intermediate role; submission belongs to the executor")
    if modes[0] == "api":
        api = card["api"]
        if not isinstance(api,str) or not api.startswith(card["app"]+"__") or api=="supervisor__complete_task" or (
            api.split("__",1)[1] in {"login","show_account_passwords"}):
            raise ValueError(f"Use one public endpoint of {card['app']}")
        if functions and api not in functions:
            raise ValueError(f"Unknown public API {api}")
        if card.get("public_read") and (card.get("each") is not None or not api.split("__",1)[1].startswith(
                ("show_", "search_", "list_", "get_"))):
            raise ValueError("A generation reentry must use a single public read")
    elif any(k in card for k in ("arguments","paginate","each","public_read")):
        raise ValueError("A calculation uses a data expression, not API controls")
    if card["output"]=="effect_receipt" and ("api" not in card or card["api"].split("__",1)[1].startswith(
            ("show_","search_","list_","get_"))):
        raise ValueError("An effect receipt must come from the endpoint performing the change")
    for slot in card.get("parameters", {}).values():
        if not isinstance(slot,dict) or not slot.get("question") or "candidates" not in slot:
            raise ValueError("Parameter choices need a question and a candidate builder")


def patch_cards(parent, proposal, sources, functions=()):
    objects = proposal["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 3:
        raise ValueError("One candidate changes one to three named components")
    result = deepcopy(parent)
    edits = []
    schemas = {f.get("function", f)["name"] for f in functions}
    for obj in objects:
        kind, name, value = obj["kind"], obj["name"], obj["value"]
        if not isinstance(name,str) or not name.isidentifier() or (kind,name) in edits:
            raise ValueError("Use a distinct named component for each edit")
        if kind == "card":
            validate_card(name,value,schemas)
            if name in BUILTINS and value.get("builtin") != name:
                raise ValueError("Edit the builtin description and skill, or add a new executable card")
            result["cards"][name] = deepcopy(value)
        elif kind in ("question","skill","collection"):
            group = {"question":"questions","skill":"skills","collection":"collections"}[kind]
            active = set(parent["collections"]) if kind=="collection" else set(ACTIVE_QUESTIONS)
            if kind=="skill":
                active.add("residual")
            if name not in active or not isinstance(value,str) or not value.strip():
                raise ValueError(f"Expected an installed {kind} and nonempty text")
            result[group][name] = value
        else:
            raise ValueError("Supported components are card, question, skill and collection")
        edits.append((kind,name))
    return result, [f"{kind}/{name}" for kind,name in edits]


CARD_PROMPT = """You are GPT-6.2 Sol Max, the offline editor of the shared AppWorld execution harness for one frozen 27B.
Use supplied Train traces. Maximize official whole tasks solved first, then minimize total output tokens including
thinking and unsuccessful attempts. Improve how the harness selects operations, binds parameters, and hands state
to generation. Choose edits to cards, shared questions, skills or collection labels according to the observed
failure or cost. When a capability is missing, decompose the observed code into independently useful operations.
Register a reusable intermediate input/output, not a task instruction, a scenario special case,
a stored sequence or a complete-task workflow. A separate submission card owns completion.
For each missed opportunity distinguish absent candidate, unavailable inputs, useful candidate ranked below NONE,
incorrect binding, incorrect API result, or failed generation handoff. Read the actual menu, state, choice scores,
API calls and downstream result. A NONE can also be correct. The decision skill should explain when this one
step advances a multi-step task, which fields it fetches, and which step can consume its output. Do not claim a
capability the card does not implement. Keep both data scope and object identity. Do not put task IDs, fixed
answers, private data or known test wording in components. The online program supplies all questions and options.
Use read, find or probe, then return {reason,prediction,probe_task,objects:[{kind,name,value}]}; at most three
distinct named components. read=[{doc,start,count,offset}]; find={doc,text,start}; probe={task,patch:{objects:[...]}}.
Read fragment/<name> for one definition, library/cards, dsl, code/card_assembly, or the raw current/<task>
trace. One card includes description, skill, executable operation and its local parameters. Shared prompts count
separately. Compare the selected and runner-up alternatives and look for tasks where NONE was appropriate."""

CARD_DSL = """A card has a description and skill. Builtin cards keep their installed builtin name.
An added card supplies app, inputs (live role names), output (one fresh role), and exactly one of:
compute: an expression, or api: an app__API name with arguments (an expression). Each API card is one
endpoint; paginate:true reads all pages. each:expr applies that endpoint to each item. Use the same
scope and explicitly echoed receipts for writes. public_read:true on a single read allows G to propose its
arguments and the program to consume the real API result. Credentials and pagination are supplied by the program.
Use output=answer for an answer value or effect_receipt for a completed requested write. Other outputs are
ordinary intermediate roles. Read cards can supply an unrelated future goal.
parameters:{name:{question,skill?,candidates}} lets Jev bind a finite choice before execution.
Optional handoff_skill gives guidance to G when this card has executed or failed in the current task.
Expressions use {ref:'state.role'}, {ref:'task.instruction'}, {ref:'params.name'}, {ref:'item.field'},
{op:name,args:[...]}, {filter:{source,where,item?}}, {map:{source,value,item?}} or literal JSON.
Candidates are [{label,value},...] or {source,where?,label,value}. Names of roles and API schemas are
the dispatch interface. Cards are offered from live inputs; no manual menu entry is required.
question and skill edit one installed named global prompt; collection edits one installed source label.
Supported expression operations: """ + ", ".join(OPS)
