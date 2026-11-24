"""Editable components of the Jev-assisted AppWorld menu assembler."""
from copy import deepcopy

from .card_library import ACTIVE_QUESTIONS, BUILTINS, initial_cards, validate_card
from .policy_programs import load_program


PROTOCOL = "phase1_semantic_menus_v2"
COMPONENTS = ("adapter", "card", "state", "assembly", "decision", "handoff")
READ_PREFIXES = ("show_", "search_", "list_", "get_")


def initial_library(*, semantic_handoffs=False):
    lib = initial_cards()
    lib["adapters"] = {"phone": {"login_field": "phone_number"}}
    lib["views"] = {}
    # Groups are also inferred from installed cards, including cards added by Sol.
    lib["assembly"] = {
        "spotify": {"apps": ["spotify"], "description": "Music collections, recorded song attributes and playback."},
        "phone": {"apps": ["phone"], "description": "Phone data and actions, including alarms."},
        "venmo": {"apps": ["venmo"], "description": "Payments and payment requests."},
        "file_system": {"apps": ["file_system"], "description": "Files and directories in the user's account."},
        "simple_note": {"apps": ["simple_note"], "description": "Personal notes."},
    }
    lib["questions"]["assembly"] = "Which collection of available operations should the program use to assemble the next action menu?"
    lib["skills"]["assembly"] = (
        "Use the wording of the request, completed work and live inputs. Choose a menu whose "
        "operations can advance the next missing step, including an intermediate source or a different app. "
        "Choosing a menu does not execute anything. NONE asks the program to show all executable operations."
    )
    lib["decision_policies"] = {}
    lib["handoffs"] = {"default": {"skill": lib["skills"]["residual"]}}
    if semantic_handoffs:
        from .semantic_handoff import QUESTION, SKILL
        lib['questions']['handoff'] = QUESTION
        lib['skills']['handoff'] = SKILL
    return lib


def string_list(value):
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value)


def check_fields(value, fields):
    if not isinstance(value, dict) or not value or set(value) - set(fields):
        raise ValueError("Expected a nonempty definition with fields from " + ", ".join(fields))


def read_card(card):
    endpoint = card.get("api", "")
    return "__" in endpoint and endpoint.split("__", 1)[1].startswith(READ_PREFIXES) and "each" not in card


def patch_library(parent, proposal, sources, functions=()):
    objects = proposal["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 3:
        raise ValueError("One revision edits one to three named components")
    installed = deepcopy(parent)
    known = {f.get("function", f)["name"] for f in functions}
    edits = []
    for obj in objects:
        kind, name, value = obj["kind"], obj["name"], obj["value"]
        if kind not in COMPONENTS or not isinstance(name, str) or not name.isidentifier() or (kind, name) in edits:
            raise ValueError("Use distinct named components from " + ", ".join(COMPONENTS))
        edits.append((kind, name))
        if value is None and installed.get("menu_format")=="scoped":
            group = {"card":"cards","state":"views","assembly":"assembly","handoff":"handoffs"}.get(kind)
            if (not group or name not in installed[group] or kind=="card" and name in BUILTINS
                    or kind=="handoff" and name=="default"):
                raise ValueError("Retire an installed, non-builtin card, view, menu, or specialized handoff")
            del installed[group][name]
            continue
        if kind == "card":
            validate_card(name, value, known)
            if name in ("submit_answer", "submit_effect") and value.get("builtin") != name:
                raise ValueError("The executor owns final submission")
            installed["cards"][name] = deepcopy(value)
        elif kind == "adapter":
            if "__" in name:
                check_fields(value, ("argument_map", "result_path", "page_size"))
                if known and name not in known:
                    raise ValueError("Expected an installed public endpoint")
                mapping = value.get("argument_map", {})
                if not isinstance(mapping, dict) or not all(isinstance(v, str) and v.isidentifier() for v in mapping.values()):
                    raise ValueError("argument_map maps argument names to API field names")
                if "access_token" in mapping or "access_token" in mapping.values():
                    raise ValueError("The executor provides the session")
                if "result_path" in value and not isinstance(value["result_path"], str):
                    raise ValueError("result_path is a dot path")
                if type(value.get("page_size", 20)) is not int or value.get("page_size", 20) < 1:
                    raise ValueError("page_size must be positive")
            else:
                check_fields(value, ("login_field", "token_path"))
                if value.get("login_field", "email") not in ("phone_number", "email") or not isinstance(value.get("token_path", ""), str):
                    raise ValueError("Use a public supervisor login field and response path")
            installed["adapters"][name] = deepcopy(value)
        elif kind == "decision":
            handoff_decision = name == 'handoff' and 'handoff' in parent['questions']
            if not handoff_decision and name not in (*ACTIVE_QUESTIONS, "assembly","card_parameter"):
                raise ValueError("Use an installed decision or define parameters on a card")
            allowed = ("question","skill") if handoff_decision else ("question","skill","program","labels")
            check_fields(value, ("skill","program") if name=="card_parameter" else allowed)
            for field in ("question", "skill"):
                if field in value:
                    if not isinstance(value[field], str) or not value[field].strip():
                        raise ValueError(field + " needs text")
                    installed[field + "s"][name] = value[field]
            if "labels" in value:
                if name != "libraries" or not isinstance(value["labels"], dict) or set(value["labels"]) - set(installed["collections"]) or not all(isinstance(v, str) and v for v in value["labels"].values()):
                    raise ValueError("labels can relabel installed collection options")
                installed["collections"].update(value["labels"])
            if "program" in value:
                load_program(value["program"])
                installed["decision_policies"][name] = value["program"]
        else:
            fields = {"assembly": ("description", "apps", "cards", "program"),
                      "state": ("requires", "program","at"),
                      "handoff": ("requires", "skill", "apis", "reads", "when", "defer")}
            if kind == 'handoff' and 'handoff' in parent['questions']:
                if 'label' in value:
                    fields['handoff'] = ('requires','label','skill','alternatives')
                    if (name == 'default' or not isinstance(value['label'],str) or not value['label'].strip()
                            or not isinstance(value.get('skill'),str) or not value['skill'].strip()
                            or not string_list(value.get('alternatives',[]))):
                        raise ValueError('A specialized handoff needs a label, guidance text and optional ordinary-action alternatives')
                elif value.get('skill') and (name != 'default' or any(k in value for k in ('when','requires','defer'))):
                    raise ValueError('Use a semantic label and live requires for specialized guidance, or default for general guidance')
            check_fields(value, fields[kind])
            for field in ("apps", "cards", "requires", "apis", "reads"):
                if field in value and not string_list(value[field]):
                    raise ValueError(field + " must contain names")
            for field in ("program", "when", "defer"):
                if field in value:
                    load_program(value[field])
            if kind == "state" and "program" not in value:
                raise ValueError("A view needs a projection program")
            if kind == "state" and value.get("at","state") not in ("state","parameter"):
                raise ValueError("A state view is rendered at state or parameter")
            if kind == "assembly" and (not isinstance(value.get("description"), str) or not any(value.get(k) for k in ("apps", "cards", "program"))):
                raise ValueError("A menu family needs a description and a source of installed cards")
            if kind == "handoff":
                if "skill" in value and not isinstance(value["skill"], str):
                    raise ValueError("handoff skill must be text")
                if known and set(value.get("apis", [])) - known:
                    raise ValueError("Tool hints use installed public API names")
            group = {"assembly": "assembly", "state": "views", "handoff": "handoffs"}[kind]
            installed[group][name] = deepcopy(value)
    for rule in installed["assembly"].values():
        if set(rule.get("cards", [])) - set(installed["cards"]):
            raise ValueError("Menu references an uninstalled card")
    for rule in installed["handoffs"].values():
        for n in rule.get("reads", []):
            if n not in installed["cards"] or not read_card(installed["cards"][n]):
                raise ValueError("A reentry uses an installed single public read")
    return installed, [kind + "/" + name for kind, name in edits]
