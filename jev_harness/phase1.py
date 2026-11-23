"""Phase-one public library: local native choices, deterministic wiring, independent tasks."""
from collections import defaultdict
from copy import deepcopy
import json
import re

from .composable_v2 import ComposableExecutor, apply_objects, route_inputs, routed
from .expressions import build_candidates, resolve
from .fragment_library import check_library
from .formal_runtime import ContextTooLong
from .pilot_interfaces import discover_consumers
from .runtime import Uncovered


def initial_library(seed):
    lib = deepcopy(seed)
    lib.pop("source_compiler", None)
    lib.get("programs", {}).pop("compiler", None)
    def option(fragment, label, parameters=None):
        return {"fragment": fragment, "label": label, **({"parameters": parameters} if parameters else {})}
    def menu(cues, question, options):
        return {"cues": [cues], "question": question, "options": options}
    lib["menus"] = {
        "venmo_records": menu(r"\b(venmo|payments?|transactions?|money|transfers?)\b",
            "Which available financial record source matches the requested operation, direction and status?",
            [option("venmo_pending_received_payment_requests", "Pending payment requests received by the user")]),
        "contact_source": menu(r"\b(contacts?|people|family|relationships?|venmo|roommates?|flatmates?|coworkers?|friends?|relatives?|colleagues?)\b",
            "Which contact information is needed to carry out the instruction?",
            [option("contacts_by_requested_group", "The actual people belonging to the relationship group(s) described in the request", "consumer_relationship_contacts"),
             option("phone_contact_relationship_labels", "Only the inventory of possible contact relationship labels, without the people")]),
        "note_source": menu(r"\b(notes?|notebook|checklist)\b",
            "What information must be retrieved from the notes application?",
            [option("requested_note", "The full text and live ID of a specific note", "consumer_note_by_title"),
             option("simple_note_search_summaries", "Only a search result list of note summaries or titles", "consumer_note_by_title")]),
        "spotify_source": menu(r"\b(spotify|artists?|songs?|playlists?|albums?|music)\b",
            "Which available music lookup can supply records required by this request?",
            [option("spotify_artists_by_genre_min_followers", "Artists of a given genre with at least a given number of followers", "spotify_artists_by_genre_min_followers"),
             option("spotify_song_play_count_page", "A page of songs matching an artist query ordered by ascending play count", "spotify_song_play_count_page")]),
        "alarm_source": menu(r"\b(alarms?|wake|bedtime)\b", "Does the task require this collection of alarm records?",
            [option("phone_alarms", "All of the current account's Phone alarms")]),
        "directory_source": menu(r"file[ _-]?system|director(y|ies)|folders?|~/",
            "Does the task require the contents of this directory?",
            [option("file_system_directory_session", "A recursive listing of the explicit source directory", "file_system_directory_session")]),
    }
    return validate_library(lib)


def execution_menus(lib):
    """Group registered operations under the broad public cues of their source menu."""
    menus = lib["menus"]
    goals = {}
    for fragment, spec in lib["workflow"]["fragments"].items():
        selection = spec.get("selection")
        if selection is None:
            continue
        source = selection["menu"]
        if source not in menus or selection.get("finish") not in ("answer", "action"):
            raise ValueError(f"Unknown menu or completion mode for {fragment}")
        key = "goals/" + source
        if key not in goals:
            goals[key] = {field: menus[source][field] for field in ("cues", "after_roles", "after_paths")
                          if field in menus[source]}
            goals[key].update(question="Which available operation fulfills the entire request, including its objects, conditions and requested result?",
                              skill="Select NONE if every option leaves part of the request undone or performs an unrequested action.",
                              options=[])
        goals[key]["options"].append({**selection, "fragment": fragment})
    return {**goals, **menus}


def validate_library(lib, functions=()):
    check_library(lib, functions, allow_alternatives=True)
    fragments = lib["workflow"]["fragments"]
    for name, spec in fragments.items():
        if "finish" in spec and spec["finish"] not in ("answer","action"):
            raise ValueError(f"Unknown completion type for {name}")
        if "effect_kind" in spec and spec["effect_kind"] not in ("read","write","internal"):
            raise ValueError(f"Unknown effect type for {name}")
        for role, slot in spec.get("slots",{}).items():
            if role not in spec["inputs"] or not public_role(role) or "candidates" not in slot:
                raise ValueError(f"Invalid input slot {name}.{role}")
            if any(not public_role(r) for r in slot.get("requires",[])):
                raise ValueError(f"Slot {name}.{role} requires a public result")
    for name, spec in lib.get("catalog_selectors",{}).items():
        if name not in ("complete","read") or not isinstance(spec,dict) or not spec or (
            spec.keys()-{"question","skill"} or any(not isinstance(x,str) or not x.strip() for x in spec.values())):
            raise ValueError("A catalog selector supplies a question or skill for complete or read")
    known={f.get("function",f)["name"] for f in functions}
    roles={f["output"] for f in fragments.values()}
    for name,r in lib.get("tool_contexts",{}).items():
        if (not r.get("apis") or not (r.get("cues") or r.get("after_roles")) or
                r.keys()-{"apis","cues","after_roles"} or set(r.get("after_roles",[]))-roles or
                (known and set(r["apis"])-known)):
            raise ValueError(f"Tool context {name} must reference public APIs and a task or state cue")
        for p in r.get("cues",[]): re.compile(p,re.I)
    for name, menu in execution_menus(lib).items():
        if not isinstance(menu, dict) or not isinstance(menu.get("question"), str) or not menu["question"].strip():
            raise ValueError(f"Menu {name} needs a public-state question")
        if not (menu.get("cues") or menu.get("after_roles") or menu.get("after_paths")):
            raise ValueError(f"Menu {name} needs a public cue")
        for pattern in [*menu.get("cues", []), *menu.get("after_paths", [])]:
            re.compile(pattern, re.I)
        options = menu["options"]
        if not isinstance(options, list) or not options:
            raise ValueError(f"Menu {name} needs installed options")
        seen = set()
        for opt in options:
            if opt["fragment"] not in fragments or not isinstance(opt.get("label"), str) or not opt["label"].strip():
                raise ValueError(f"Invalid option in {name}: {opt}")
            key = (opt["fragment"], opt.get("parameters"), json.dumps(opt.get("inputs", {}), sort_keys=True))
            if key in seen:
                raise ValueError(f"Duplicate option in {name}")
            seen.add(key)
            for cap in opt.get("captures", {}).values():
                if re.compile(cap["pattern"], re.I).groups != 1 or cap.get("mode","one") not in ("one","many"):
                    raise ValueError("Each capture must bind a single group")
            for role, slot in opt.get("slots", {}).items():
                if not public_role(role) or not slot.get("question") or "candidates" not in slot:
                    raise ValueError(f"Invalid semantic parameter {role} in {name}")
    for name, policy in lib.get("selector_policies", {}).items():
        if name not in lib["menus"] or policy.keys()-{"direct","defer","minimum_probability"}:
            raise ValueError("A selector policy belongs to one menu")
        if not 0 <= policy.get("minimum_probability",0) <= 1:
            raise ValueError("Invalid native probability threshold")
        options = {o["fragment"] for o in lib["menus"][name]["options"]}
        for fragment, pattern in policy.get("direct",{}).items():
            if fragment not in options:
                raise ValueError("Direct selection needs an option in the same menu")
            re.compile(pattern,re.I)
        for pattern in policy.get("defer",[]):
            re.compile(pattern,re.I)
    for role, view in lib.get("state_views",{}).items():
        if role not in {spec["output"] for spec in fragments.values()} or view.keys()-{"project","description"}:
            raise ValueError("A state projection belongs to a produced role")
        if not public_role(role) or "project" not in view:
            raise ValueError("Expected a public result projection")
    return lib


def patch_library(lib, patch, sources, functions):
    def extend(result, kind, name, value):
        if kind == "compiler" or (kind == "program" and name == "compiler"):
            raise ValueError("Phase one has no post-task compiler")
        if kind == "menu":
            result["menus"][name] = value
        elif kind == "selector":
            result.setdefault("catalog_selectors",{})[name] = value
        elif kind == "tool_context":
            result.setdefault("tool_contexts",{})[name] = value
        elif kind == "routing":
            result.setdefault("selector_policies",{})[name] = value
        elif kind == "state":
            result.setdefault("state_views",{})[name] = value
        elif kind in ("decision_skill", "generation_skill"):
            if not isinstance(value, str):
                raise ValueError("Expected a scoped prompt")
            result["decision_skills" if kind == "decision_skill" else "generation_skills"][name] = value
        else:
            return False
        return True
    return apply_objects(lib, patch, sources, functions, max_objects=3, extension=extend, validate=validate_library)


def public_role(role):
    return not any(word in role for word in ("password", "token", "session"))


class PhaseOneExecutor(ComposableExecutor):
    direct_menu_choices = True
    def execute_task(self, library):
        self.by_role = defaultdict(list)
        for name, spec in library["workflow"]["fragments"].items():
            self.by_role[spec["output"]].append(name)
        self.managed = {library["workflow"]["fragments"][o["fragment"]]["output"]
                        for m in library["menus"].values() for o in m["options"]}
        self.execution_menus = execution_menus(library)
        self.seen_menus = set()
        self.cache = {}
        self.failed_fragments = {}
        self.scopes = defaultdict(list)
        self.parameter_sources = {**routed(library), **{
            "consumer_" + k: v for k, v in discover_consumers(library, self.context["task"]["instruction"]).items()}}
        super().execute_task(library)

    def provider(self, role, chosen=None):
        names = self.by_role.get(role, [])
        if chosen in names:
            return chosen
        if len(names) == 1:
            return names[0]
        raise Uncovered(f"{role} has {len(names)} providers; select a named provider or supply its value")

    def plan(self, role, args, preferred=None, trail=()):
        if role in args:
            return ("argument", role, json.dumps(args[role], sort_keys=True, default=str))
        if role in trail:
            raise Uncovered("Dependency cycle at " + role)
        if preferred is None and role in self.live and len(self.scopes.get(role, ())) == 1 and not args:
            return self.scopes[role][0]
        name = self.provider(role, preferred)
        spec = self.library["workflow"]["fragments"][name]
        return ("fragment", name, tuple(self.plan(k, args, trail=(*trail, role)) for k in spec["inputs"]))

    def materialize(self, plan, args):
        kind, name, dependencies = plan
        if kind == "argument":
            return args[name]
        if plan in self.failed_fragments:
            raise Uncovered(self.failed_fragments[plan])
        if plan in self.cache:
            return self.cache[plan]
        spec = self.library["workflow"]["fragments"][name]
        values = [self.materialize(child, args) for child in dependencies]
        self.trace.add("fragment_start", fragment=name, input_roles=spec["inputs"], goal=spec["output"])
        try:
            value = self.run_fragment(name, spec, values, self.library)
        except (Uncovered, ValueError, KeyError, IndexError, TypeError) as exc:
            self.failed_fragments[plan] = str(exc)
            self.trace.add("fragment_failure", fragment=name, reason=str(exc))
            raise Uncovered(str(exc)) from exc
        role = spec["output"]
        self.cache[plan] = value
        self.scopes[role].append(plan)
        self.live[role] = value
        if name in self.context["generated"] or any(k in self.generated_inputs for k in spec["inputs"]):
            self.generated_inputs.add(role)
        if len(self.scopes[role]) > 1 and public_role(role) and role not in self.generated_inputs:
            self.live.setdefault("scoped_results", {})[role] = [
                {"bindings": self.public_bindings(key), "value": self.cache[key]} for key in self.scopes[role]]
        self.visible_progress = None
        self.trace.add("fragment_complete", fragment=name, input_roles=spec["inputs"], output_role=role,
                       source_tasks=spec.get("source_tasks", []))
        return value

    def run_fragment(self, name, spec, values, library):
        def has_decision(steps):
            return any(s["kind"]=="decision" or any(has_decision(s.get(branch,[]))
                       for branch in ("body","then","else")) for s in steps)
        if self.generated_inputs.intersection(spec["inputs"]) and has_decision(spec["steps"]):
            raise Uncovered("A native interface needs observed public inputs; this dependency came from local generation")
        return super().run_fragment(name,spec,values,library)

    @staticmethod
    def public_bindings(plan):
        if plan[0] == "argument":
            return {plan[1]: json.loads(plan[2])} if public_role(plan[1]) else {}
        return {k: v for dep in plan[2] for k, v in PhaseOneExecutor.public_bindings(dep).items()}

    def activate(self, name, route, args, preferred=None, native=False):
        self.used_routes.add(name)
        self.trace.add("route_active", route=name, goals=route["goals"], after_roles=route.get("after_roles", []))
        try:
            for goal in route["goals"]:
                plan = self.plan(goal, args, preferred)
                already = plan in self.cache
                self.materialize(plan, args)
                if goal not in self.completed_roles:
                    self.completed_roles.append(goal)
                if native:
                    self.trace.add("fragment_choice_used", menu=name, fragment=preferred, output=goal,
                                   reused=already, bindings=self.public_bindings(plan))
            self.solved_routes.append(name)
            self.unresolved.pop(name, None)
            self.trace.add("route_satisfied", route=name, goals=route["goals"])
            return True
        except (Uncovered, ValueError, KeyError, TypeError) as exc:
            self.unresolved[name] = str(exc)
            self.trace.add("composition_missing", goal=name, reason=str(exc))
            return False

    def option_args(self, opt, task, live):
        source = opt.get("parameters")
        bindings = {}
        if source:
            if source in self.parameter_sources:
                bindings.update(route_inputs(self.parameter_sources[source], task, live))
            elif source == "consumer_relationship_contacts":
                bindings["group_queries"] = [task["instruction"]]
            else:
                raise Uncovered("Missing public bindings for " + source)
        bindings.update(route_inputs(opt, task, {**live, **bindings}))
        return bindings

    def slot_domains(self, opt, task, live, args):
        if not opt.get("slots"):
            return {}
        public = {key: value for key, value in {**live, **args}.items() if public_role(key)}
        context = {"vars": public, "inputs": public,
                   "task": {key: task[key] for key in ("instruction", "datetime", "supervisor") if key in task}}
        domains = {}
        for role, slot in opt.get("slots", {}).items():
            if role not in args:
                choices = build_candidates(slot["candidates"], context)
                if not choices:
                    raise Uncovered("No public candidates for " + role)
                domains[role] = choices
        return domains

    def execute_option(self, menu, opt, output, args, domains, task, native):
        name = opt["fragment"]
        values = dict(args)
        for role, choices in domains.items():
            slot = opt["slots"][role]
            evidence = {"instruction":task["instruction"],"operation":opt["label"],
                        "bound_parameters":{k:v for k,v in values.items() if public_role(k)}}
            try:
                values[role] = self.decision(slot["question"], evidence, choices, skill=slot.get("skill",""),
                                             node=f"slot/{name}/{role}")
            except (Uncovered, ContextTooLong) as exc:
                self.trace.add("slot_deferred", menu=menu, fragment=name, role=role, reason=str(exc))
                return
            self.trace.add("slot_bound", menu=menu, fragment=name, role=role, value=values[role])
        if not self.activate("menu/" + menu, {"goals": [output]}, values, name, native=native):
            return
        if opt.get("finish"):
            direct = self.routes.get(name, {})
            if direct.get("complete") and direct["goals"] == [output] and name not in self.used_routes:
                self.unresolved.pop(name, None)
            if not self.done and not self.unresolved:
                arguments = {"answer": self.live[output]} if opt["finish"] == "answer" else {}
                self.call("supervisor", "complete_task", arguments, "composition_finish")

    def consider(self, name, menu, task, live, *, loose_reads=()):
        cue = any(re.search(p, task["instruction"], re.I) for p in menu.get("cues", []))
        state_required = bool(menu.get("after_roles") or menu.get("after_paths"))
        state_match = (state_required
                       and set(menu.get("after_roles", [])) <= live.keys()
                       and all(any(re.search(p, obs, re.I) for obs in self.observations) for p in menu.get("after_paths", [])))
        options, excluded = [], []
        for opt in menu["options"]:
            frag = opt["fragment"]
            role = self.library["workflow"]["fragments"][frag]["output"]
            try:
                if not (cue or state_match or (not state_required and frag in loose_reads)):
                    raise Uncovered("No public instruction or state cue")
                args = self.option_args(opt, task, live)
                domains = self.slot_domains(opt, task, live, args)
                # Only check that dependencies can be wired. The selected values bind before execution.
                reachable = {**args, **{k:choices[0]["value"] for k,choices in domains.items()}}
                plan = self.plan(role, reachable, frag)
                if not domains and plan in self.failed_fragments:
                    raise Uncovered(self.failed_fragments[plan])
                options.append((opt, role, args, domains))
            except (Uncovered, KeyError, ValueError, TypeError) as exc:
                excluded.append({"fragment": frag, "reason": str(exc)})
        signature = (name, json.dumps([(o["fragment"], a, d) for o, _, a, d in options], sort_keys=True), str(excluded))
        if signature in self.seen_menus:
            return
        self.seen_menus.add(signature)
        self.trace.add("fragment_recall", menu=name, options=[{"fragment":o["fragment"], "label":o["label"],
                       "bindings":{k:v for k,v in a.items() if public_role(k)},
                       **({"semantic_parameters":list(d)} if d else {})} for o,_,a,d in options], excluded=excluded,
                       **({"opened_by":"app_prefilter"} if options and not cue and not state_match else {}))
        if not options:
            return
        policy = self.library.get("selector_policies",{}).get(name,{})
        if any(re.search(p,task["instruction"],re.I) for p in policy.get("defer",[])):
            self.trace.add("fragment_choice_deferred",menu=name,reason="Public instruction matched the local generation policy")
            return
        direct = [(o, role, a, d) for o,role,a,d in options if o["fragment"] in policy.get("direct",{})
                  and re.fullmatch(policy["direct"][o["fragment"]], task["instruction"], re.I)]
        if self.direct_menu_choices and len(direct) == 1:
            opt,role,args,domains=direct[0]
            self.trace.add("program_fragment_selection",menu=name,fragment=opt["fragment"])
            self.execute_option(name,opt,role,args,domains,task,native=False)
            return
        candidates = [{"label": o["label"], "value": i} for i, (o, _, _, _) in enumerate(options)]
        evidence = {"instruction":task["instruction"], "completed":self.completed_roles,
                    "observed_public_calls":sorted(self.observations)}
        if any(o.get("finish") for o,_,_,_ in options):
            evidence["available_bindings"] = [{k:v for k,v in a.items() if public_role(k)} for _,_,a,_ in options]
        try:
            index = self.decision(menu["question"], evidence,
                candidates, skill=self.library["decision_skills"].get("menu_"+name.replace("/", "_"), menu.get("skill","")),
                policy={"mode":"D","minimum_probability":policy.get("minimum_probability",0)}, node="menu/"+name)
        except (Uncovered, ContextTooLong) as exc:
            self.trace.add("fragment_choice_deferred", menu=name, reason=str(exc))
            return
        opt, role, args, domains = options[index]
        self.trace.add("fragment_choice", menu=name, fragment=opt["fragment"], real_options=len(options))
        self.execute_option(name,opt,role,args,domains,task,native=True)

    def finish_if_complete(self):
        complete = any(self.routes.get(n, {}).get("complete") for n in self.solved_routes)
        if self.done or self.unresolved or not complete:
            return False
        args = {"answer":self.live["answer"]} if "answer" in self.live else {}
        self.call("supervisor","complete_task",args,"composition_finish")
        return True

    def handoff_progress(self, names, public):
        progress = super().handoff_progress(names, public)
        for role, spec in self.library.get("state_views",{}).items():
            if role not in public:
                continue
            try:
                view = resolve(spec["project"],{"item":public[role],"vars":public,
                    "task":{"instruction":self.context["task"]["instruction"]}})
                progress.setdefault("scoped_evidence",{})[role] = {"description":spec.get("description",""),"value":view}
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                self.trace.add("handoff_projection_error",role=role,error=str(exc))
        return progress

    def assemble(self):
        self.assembling = True
        try:
            while not self.done:
                before = (len(self.used_routes), len(self.cache))
                task = self.context["task"]
                live = {k:v for k,v in self.live.items() if k not in self.generated_inputs}
                if "discovery" in self.library.get("programs", {}):
                    self.discover_from_state()
                for name, route in self.routes.items():
                    if name in self.used_routes or (not route.get("complete") and set(route["goals"]) <= self.managed):
                        continue
                    if set(route.get("after_roles", [])) - live.keys() or not all(
                        any(re.search(p, obs, re.I) for obs in self.observations) for p in route.get("after_paths", [])):
                        continue
                    try:
                        args = route_inputs(route, task, live)
                        for goal in route["goals"]:
                            self.plan(goal, args)
                    except (Uncovered, KeyError, TypeError, ValueError) as exc:
                        self.unresolved[name] = str(exc)
                        continue
                    self.activate(name, route, args)
                if self.finish_if_complete():
                    return
                for name, menu in self.execution_menus.items():
                    self.consider(name, menu, task, live)
                    if self.done:
                        return
                if before == (len(self.used_routes), len(self.cache)):
                    break
            self.finish_if_complete()
        finally:
            self.assembling = False

    def summarize(self, *args):
        row = super().summarize(*args)
        row["fragment_selections"] = [e for e in self.trace.events if e["event"] == "fragment_choice_used"]
        row["fragment_decisions"] = sum(e["event"] == "decision" and str(e.get("node","")).startswith("menu/")
                                        for e in self.trace.events)
        row["semantic_slots"] = [e for e in self.trace.events if e["event"]=="slot_bound"]
        return row
