"""Materialize a whole decision interface and a read-only consumer for the Train pilot."""
from copy import deepcopy
import re

from .fragment_library import FragmentExecutor, check_library, matching_routes, new_library


def seed_primitives(seed):
    base = new_library()
    for name in ("phone_access_token", "phone_contact_relationship_labels", "simple_note_access_token",
                 "simple_note_search_summaries"):
        frag = deepcopy(seed["workflow"]["fragments"][name])
        base["workflow"]["fragments"][name] = frag
        for op in frag["steps"]:
            key = op["binding"]
            base["bindings"][key] = deepcopy(seed["bindings"][key])
    return base


def usable_pattern(pattern):
    if not isinstance(pattern, str) or not 1 <= len(pattern) <= 300:
        raise ValueError("Expected a bounded regular expression")
    return re.compile(pattern, re.I)


def install_interface(base, patch, public_apis):
    """A single interface owns its question/evidence/options. Its consumer owns discovery and bindings."""
    if set(patch) != {"interface", "consumer"}:
        raise ValueError("One decision interface and one consumer are the two editable objects")
    interface, consumer = patch["interface"], patch["consumer"]
    kind = consumer["primitive"]
    if kind not in ("relationship_contacts", "note_by_title"):
        raise ValueError("Unknown read primitive")
    name = interface["name"]
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,64}", name) or name != consumer["interface"]:
        raise ValueError("The consumer must reference its named interface")
    template = interface["question"]
    if "{query}" not in template or len(template) > 800 or not isinstance(interface["skill"], str):
        raise ValueError("Question must interpolate the live query")
    template.format(query="current query")
    if len(interface["skill"]) > 1200:
        raise ValueError("Decision skill is too long")
    usable_pattern(consumer["context_pattern"])
    matcher = usable_pattern(consumer["query_pattern"])
    if matcher.groups != 1 or matcher.search(""):
        raise ValueError("Subgoal matcher needs one nonempty capture group")
    library = deepcopy(base)
    if name in library["questions"] or kind in library.get("pilot_consumers", {}):
        raise ValueError("Refusing to overwrite an unrelated registered pilot interface")
    library.setdefault("pilot_consumers", {})[kind] = deepcopy(consumer)
    ref = lambda k: {"ref": "vars."+k}
    if kind == "relationship_contacts":
        data, role = "phone_contact_relationship_labels", "contacts_by_requested_group"
        options = {"source": ref(data), "label": {"ref":"item"}, "value":{"ref":"item"}}
        token = "phone_access_token"
        bindings = {
            "pilot_group_queries":ref("group_queries"),
            "pilot_search_selected_contacts":{"app":"phone","api":"search_contacts","arguments":{
                "access_token":ref(token),"relationship":ref("selection"),"page_limit":20}},
        }
        contact_fields = {k:{"ref":"row."+k,"default":None} for k in
                          ("first_name","last_name","email","phone_number","contact_id")}
        progress = {"requested_group":ref("query"),"relationship":ref("selection"),
                    "contacts":{"map":{"source":ref("contacts"),"item":"row","value":contact_fields}}}
        steps = [{"kind":"foreach","binding":"pilot_group_queries","item":"query","body":[
            {"kind":"decision","decision":name,"store":"selection"},
            {"kind":"call","binding":"pilot_search_selected_contacts","store":"contacts","paginate":True}],
            "collect":progress,"store":"groups"}]
        inputs, result = [token,data,"group_queries"], ref("groups")
    else:
        data, role = "simple_note_search_summaries", "requested_note"
        options = {"source":ref(data),"label":{"ref":"item.title"},"value":{"ref":"item.note_id"}}
        strategy = interface.get("candidate_selection", "all_results")
        if strategy == "title_contains_query":
            lower = lambda value: {"op":"lower","args":[value]}
            options["where"] = {"op":"contains","args":[lower({"ref":"item.title"}),lower(ref("query"))]}
        elif strategy != "all_results":
            raise ValueError("Unknown title-candidate builder")
        token = "simple_note_access_token"
        bindings = {"pilot_show_selected_note":{"app":"simple_note","api":"show_note","arguments":{
            "access_token":ref(token),"note_id":ref("selection")}}}
        steps = [{"kind":"decision","decision":name,"store":"selection"},
                 {"kind":"call","binding":"pilot_show_selected_note","store":"note","require":["content"]}]
        inputs, result = [token,data,"query"], ref("note")
    library["questions"][name] = {"template":template,"values":{"query":ref("query")}}
    library["evidence"][name] = {"requested":ref("query"),"instruction":{"ref":"task.instruction"}}
    library["candidates"][name] = options
    library["decision_skills"][name] = interface["skill"]
    library["bindings"].update(bindings)
    library["workflow"]["fragments"][role] = {"app":"phone" if kind=="relationship_contacts" else "simple_note",
        "output":role,"inputs":inputs,"steps":steps,"result":result,"effect":"read-only","source_tasks":[]}
    return check_library(library, public_apis)


def discover_consumers(library, instruction):
    """Match local query spans, never compile sample IDs, account IDs, or ready-made option lists."""
    routes = {}
    for kind, spec in library.get("pilot_consumers", {}).items():
        if not re.search(spec["context_pattern"], instruction, re.I):
            continue
        matches = [v.strip() for v in re.findall(spec["query_pattern"], instruction, re.I) if v.strip()]
        values = list(dict.fromkeys(matches))
        if kind == "relationship_contacts" and values:
            routes[kind] = {"inputs":{"group_queries":values},"goals":["contacts_by_requested_group"]}
        elif kind == "note_by_title" and len(values) == 1:
            routes[kind] = {"inputs":{"query":values[0],"simple_note_title_query":values[0]},
                            "goals":["requested_note"]}
    return routes


class PilotExecutor(FragmentExecutor):
    def execute_task(self, library):
        self.pilot_handoff_skills = library.get("handoff_skills", {})
        original = library["workflow"]
        instruction = self.context["task"]["instruction"]
        rules = dict(matching_routes(library.get("composition_rules", {}), instruction))
        for name, route in discover_consumers(library, instruction).items():
            rules[name] = {**route,"patterns":["."],"complete":False}
        self.trace.add("pilot_discovery", registered=list(library.get("pilot_consumers", {})),
                       matches=list(rules))
        super().execute_task({**library,"workflow":{**original,"rules":rules}})

    def agent(self, skill="", progress=None):
        if progress and "contacts_by_requested_group" in progress["completed_goals"]:
            skill += (" The complete contacts_by_requested_group is available in compiled. Each entry gives "
                      "requested_group, relationship and contacts with email and phone_number. Match the relevant "
                      "Venmo account using live identifiers and public API schemas. Use the request's direction, "
                      "dates and status. Read these existing results from compiled instead of repeating contact lookup.")
        if progress and "requested_note" in progress["completed_goals"]:
            skill += (" The full requested note is in compiled['requested_note']. Use its note ID and content. "
                      "For a checklist edit keep unrelated lines and the note identity unchanged, then finish.")
        if progress:
            for goal in progress["completed_goals"]:
                if goal in self.pilot_handoff_skills:
                    skill += "\n"+self.pilot_handoff_skills[goal]
        return super().agent(skill, progress)
