"""Editable behavioral objects and state-driven AppWorld fragment assembly."""
from copy import deepcopy
import json
import re

from .expressions import resolve
from .formal_runtime import FormalExecutor
from .fragment_library import FragmentExecutor, check_library, matching_routes
from .pilot_interfaces import discover_consumers
from .policy_programs import load_program, run_program
from .runtime import Uncovered


def routed(library):
    return {**library["workflow"]["rules"], **library.get("composition_rules", {})}


def apply_objects(parent, patch, sources, functions, *, max_objects=2, extension=None, validate=check_library):
    edits = patch["objects"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= max_objects:
        raise ValueError(f"A candidate edits between one and {max_objects} named behavioral objects")
    result = deepcopy(parent)
    touched = []
    for edit in edits:
        kind, name, value = edit["kind"], edit["name"], edit["value"]
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,90}", name) or [kind, name] in touched:
            raise ValueError("Expected a unique named object")
        touched.append([kind, name])
        if extension and extension(result, kind, name, value):
            continue
        if kind == "interface":
            required = {"question", "evidence", "candidates"}
            if not required <= value.keys() or value.keys() - required - {"skill", "policy"}:
                raise ValueError("An interface owns its question, evidence, candidates, skill and policy")
            for field, key in (("question","questions"),("evidence","evidence"),("candidates","candidates"),
                               ("skill","decision_skills"),("policy","mode_policy")):
                if field in value:
                    result[key][name] = value[field]
        elif kind == "fragment":
            if value.keys() - {"definition", "bindings", "route"}:
                raise ValueError("A fragment contains a definition, its own bindings and optional discovery")
            frag = value["definition"]
            if set(frag.get("source_tasks", [])) - set(sources):
                raise ValueError("Unknown source for the fragment")
            # Each fragment can replace only its own private binding definitions.
            bindings = value.get("bindings", {})
            def lower(steps):
                for step in steps:
                    key = step.get("binding")
                    if key in bindings:
                        owned = name + "__" + key
                        result["bindings"][owned] = bindings[key]
                        step["binding"] = owned
                    for branch in ("body", "then", "else"):
                        lower(step.get(branch, []))
            frag = deepcopy(frag)
            lower(frag["steps"])
            result["workflow"]["fragments"][name] = frag
            if "route" in value:
                if value["route"].get("goals") != [frag["output"]]:
                    raise ValueError("A fragment's attached route dispatches only its own output")
                result.setdefault("composition_rules", {})[name] = value["route"]
        elif kind == "route":
            result.setdefault("composition_rules", {})[name] = value
        elif kind == "handoff":
            if not isinstance(value, str) or len(value) > 3200:
                raise ValueError("Expected a scoped handoff skill")
            result.setdefault("handoff_skills", {})[name] = value
        elif kind == "recovery":
            if name not in result["workflow"]["fragments"] or value.keys() - {"skill", "policy"}:
                raise ValueError("Recovery belongs to one existing fragment")
            result["generation_skills"][name] = value["skill"]
            result["mode_policy"][name] = value.get("policy", {})
        elif kind == "compiler":
            if name != "source" or value.keys() - {"selection_skill", "composition_skill", "allow_generation", "extractors"}:
                raise ValueError("Unknown source compiler setting")
            for ex in value.get("extractors", {}).values():
                if re.compile(ex["pattern"], re.I).groups != 1 or ex["mode"] not in ("one", "many"):
                    raise ValueError("A parameter extractor supplies one capture")
            result.setdefault("source_compiler", {}).update(value)
        elif kind == "program":
            if name not in ("discovery","handoff","compiler"):
                raise ValueError("A policy program owns one of discovery, handoff or compiler")
            load_program(value)
            result.setdefault("programs",{})[name]=value
        else:
            raise ValueError("Unknown behavioral object " + str(kind))
    if result == parent:
        raise ValueError("Patch has no effect")
    providers = {f["output"] for f in result["workflow"]["fragments"].values()}
    for route in routed(result).values():
        if (not route.get("goals") or set(route["goals"]) - providers
                or not any(route.get(k) for k in ("patterns", "after_roles", "after_paths"))
                or set(route.get("source_tasks", [])) - set(sources)):
            raise ValueError("Route needs an observation trigger and installed output(s)")
        for field in ("patterns", "exclude", "after_paths"):
            for pattern in route.get(field, []):
                re.compile(pattern, re.I)
        for extractor in route.get("captures", {}).values():
            if re.compile(extractor["pattern"], re.I).groups != 1 or extractor.get("mode", "one") not in ("one", "many"):
                raise ValueError("Each live parameter capture requires one group")
    return validate(result, functions), touched


def route_inputs(route, task, live):
    instruction = task["instruction"]
    context = {"task": {k:task[k] for k in ("instruction","datetime","supervisor") if k in task},
               "vars": live, "inputs": live}
    arguments = resolve(route.get("inputs", {}), context)
    for name, extractor in route.get("captures", {}).items():
        values = list(dict.fromkeys(x.strip() for x in re.findall(extractor["pattern"], instruction, re.I) if x.strip()))
        if not values or (extractor.get("mode", "one") == "one" and len(values) != 1):
            raise Uncovered("No unambiguous live parameter for " + name)
        arguments[name] = values if extractor.get("mode") == "many" else values[0]
    return arguments


class ObservedEnvironment:
    """Use the existing public simulator cell to collect HTTP receipts for compiled API calls too."""
    def __init__(self, environment):
        self.environment = environment

    def request(self, op, **params):
        if op != "call":
            return self.environment.request(op, **params)
        app, api = params["app"], params["api"]
        if not all(re.fullmatch(r"[a-zA-Z_]\w*", s) for s in (app,api)):
            raise ValueError("Invalid public API name")
        code = "import json\nprint(json.dumps(apis."+app+"."+api+"(**"+repr(params["arguments"])+"), default=str))"
        result = self.environment.request("execute",code=code,trace_actions=True)
        try:
            result["value"] = json.loads(result["output"])
        except (TypeError,ValueError):
            result["value"] = result["output"]
        return result


class ComposableExecutor(FragmentExecutor):
    def __init__(self, model, environment, **settings):
        super().__init__(model, ObservedEnvironment(environment), **settings)

    def execute_task(self, library):
        self.library = library
        self.fragment_inputs = {}
        self.generated_inputs = set()
        self.write_progress = []
        self.live = {}
        self.used_routes = set()
        self.completed_roles = []
        self.broken = {}
        self.observations = set()
        self.choice_vars = {}
        self.bound_choices = {}
        self.last_choice = None
        self.unresolved = {}
        self.solved_routes = []
        self.visible_progress = None
        self.assembling = False
        self.providers = {spec["output"]: (name, spec) for name, spec in library["workflow"]["fragments"].items()}
        instruction = self.context["task"]["instruction"]
        self.routes = dict(matching_routes({k: {**v, "patterns": v.get("patterns", [])}
                                            for k, v in routed(library).items()}, instruction))
        for key, spec in discover_consumers(library, instruction).items():
            self.routes["consumer_" + key] = spec
        self.trace.add("composition_goals", matched_rules=list(self.routes),
                       goals=list(dict.fromkeys(g for route in self.routes.values() for g in route["goals"])))
        self.assemble()
        if not self.done:
            self.trace.add("fallback", scope="unfinished_goals" if self.completed_roles else "generation_only",
                           completed_goals=list(self.completed_roles))
            self.agent()

    def unresolved_inputs(self, role, available, chain=()):
        if role in available:
            return set()
        if role in chain or role not in self.providers:
            return {role}
        return set().union(*(self.unresolved_inputs(k,available,chain+(role,))
                             for k in self.providers[role][1]["inputs"]))

    def need(self, role, chain=()):
        if role in self.live:
            return self.live[role]
        if role in self.broken or role in chain or role not in self.providers:
            raise Uncovered(self.broken.get(role, "Missing input " + role))
        name, spec = self.providers[role]
        values = [self.need(k, chain + (role,)) for k in spec["inputs"]]
        self.trace.add("fragment_start", fragment=name, input_roles=spec["inputs"], goal=chain[0] if chain else role)
        try:
            value = self.run_fragment(name, spec, values, self.library)
        except (Uncovered, KeyError, ValueError, TypeError, IndexError) as exc:
            self.broken[role] = str(exc)[:300]
            self.trace.add("fragment_failure", fragment=name, reason=self.broken[role])
            raise Uncovered(self.broken[role]) from exc
        self.live[role] = value
        if name in self.context["generated"]:
            self.generated_inputs.add(role)
        self.trace.add("fragment_complete", fragment=name, input_roles=spec["inputs"], output_role=role,
                       source_tasks=spec.get("source_tasks", []))
        return value

    def assemble(self):
        self.assembling = True
        try:
            while not self.done:
                before = (len(self.used_routes), len(self.live))
                if "discovery" in self.library.get("programs",{}):
                    self.discover_from_state()
                for name, route in self.routes.items():
                    if name in self.used_routes or set(route.get("after_roles", [])) - self.live.keys():
                        continue
                    if any(not any(re.search(p, observed, re.I) for observed in self.observations)
                           for p in route.get("after_paths", [])):
                        continue
                    try:
                        args = route_inputs(route, self.context["task"],
                                            {k:v for k,v in self.live.items() if k not in self.generated_inputs})
                    except (KeyError, ValueError, TypeError, Uncovered) as exc:
                        self.unresolved[name] = str(exc)[:200]
                        continue
                    missing = set().union(*(self.unresolved_inputs(g, self.live.keys() | args.keys()) for g in route["goals"]))
                    if missing:
                        self.unresolved[name] = "Missing " + ", ".join(sorted(missing))
                        continue
                    self.unresolved.pop(name, None)
                    self.used_routes.add(name)
                    if any(k in self.live and self.live[k] != v for k, v in args.items()):
                        self.unresolved[name] = "Incompatible parameter values"
                        self.trace.add("composition_missing", goal=name, reason="Incompatible parameter values")
                        continue
                    self.live.update(args)
                    self.trace.add("route_active", route=name, goals=route["goals"], after_roles=route.get("after_roles", []))
                    try:
                        for goal in route["goals"]:
                            self.need(goal)
                            if goal not in self.completed_roles:
                                self.completed_roles.append(goal)
                    except (Uncovered, KeyError, ValueError, TypeError) as exc:
                        self.unresolved[name] = str(exc)[:300]
                        self.trace.add("composition_missing", goal=goal, reason=str(exc)[:300])
                    else:
                        self.solved_routes.append(name)
                        self.trace.add("route_satisfied",route=name,goals=route["goals"])
                if before == (len(self.used_routes), len(self.live)):
                    break
            if (not self.done and not self.unresolved and not self.broken
                    and any(self.routes[name].get("complete") for name in self.solved_routes)):
                self.call("supervisor","complete_task",
                          {"answer":self.live["answer"]} if "answer" in self.live else {}, "composition_finish")
        finally:
            self.assembling = False

    def discover_from_state(self):
        available = {k:v for k,v in self.live.items() if k not in self.generated_inputs
                     and not any(word in k for word in ("password","token","session"))}
        additions=run_program(self.library,"discovery",{
            "instruction":self.context["task"]["instruction"],"live":available,"observations":sorted(self.observations),
            "completed":self.completed_roles,"available_outputs":list(self.providers)},self.trace)
        for name in list(self.routes):
            if name.startswith("program/") and name not in self.used_routes:
                self.routes.pop(name)
        if not isinstance(additions,list):
            return
        for item in additions[:12]:
            if not isinstance(item,dict):
                continue
            key,goals,inputs=item.get("name"),item.get("goals"),item.get("inputs",{})
            if (not isinstance(key,str) or not re.fullmatch(r"[a-z][a-z0-9_]*",key)
                    or not isinstance(goals,list) or not goals or any(not isinstance(g,str) or g not in self.providers for g in goals)
                    or not isinstance(inputs,dict)):
                self.trace.add("program_error",policy="discovery",error="Expected a named route to installed output roles")
                continue
            self.routes["program/"+key]={"goals":goals,"inputs":inputs}

    def call(self, app, api, arguments, node=None):
        value = super().call(app, api, arguments, node)
        self.observations.add(app + "." + api)
        api_event=self.trace.events[-1]
        actions=[{"method":a["method"],"url":a["url"].split("?")[0]} for a in api_event["response"].get("actions",[])]
        for argument, choice in self.bound_choices.items():
            if actions and not choice["counted"]:
                self.trace.add("choice_used",node=choice["node"],mode=choice["mode"],real_choices=choice["count"],
                               app=app,api=api,argument_names=[argument],actions=actions,
                               public_arguments={argument:arguments[argument]} if argument not in ("password","access_token") else {})
                choice["counted"]=True
        return value

    def run_fragment(self,name,spec,values,library):
        parent=self.choice_vars
        self.choice_vars={}
        try:
            return super().run_fragment(name,spec,values,library)
        finally:
            self.choice_vars=parent

    def fragment_steps(self,steps,library,name):
        for step in steps:
            previous=self.bound_choices
            self.bound_choices={}
            try:
                if step["kind"]=="call":
                    binding=library["bindings"][step["binding"]]
                    for arg, expression in binding.get("arguments",{}).items():
                        if isinstance(expression,dict):
                            ref=expression.get("ref","").split(".")
                            if len(ref)>=2 and ref[0]=="vars" and ref[1] in self.choice_vars:
                                self.bound_choices[arg]=self.choice_vars[ref[1]]
                super().fragment_steps([step],library,name)
                if "store" in step:
                    self.choice_vars.pop(step["store"],None)
                    if step["kind"]=="decision" and self.last_choice:
                        self.choice_vars[step["store"]]=self.last_choice
            finally:
                self.bound_choices=previous

    def decision(self, question, evidence, candidates, skill="", policy=None, node=None):
        value = super().decision(question, evidence, candidates, skill, policy, node)
        self.last_choice={"node":node,"count":len(candidates),"mode":(policy or {}).get("mode","D"),"counted":False}
        return value

    def code(self, code, node=None, permit_finish=True):
        response = super().code(code, node, permit_finish)
        if not self.done and permit_finish and not self.assembling:
            for action in response.get("actions", []):
                self.observations.add(action["method"].upper() + " " + action["url"].split("?")[0])
            self.assemble()
        return response

    def handoff_progress(self, names, public):
        return {"completed_goals":self.completed_roles, "variables":[k for k in names if k!="account_passwords"],
                "goal_descriptions":{k:self.providers[k][1].get("effect",k) for k in self.completed_roles if k in self.providers},
                "live_values":public, "recent_writes":self.write_progress[-15:], "unfinished":self.unresolved}

    def generate_message(self, system, progress=None, *, purpose="G", max_tokens=None):
        if purpose == "G":
            skills = self.library.get("handoff_skills", {})
            system += "\n" + skills.get("general", "")
            if self.live:
                names = tuple(sorted(self.live))
                if self.visible_progress != names:
                    self.prime_generated_session(self.live)
                    self.visible_progress = names
                public = {k:v for k,v in self.live.items() if not any(word in k for word in ("password","token","session"))}
                progress = self.handoff_progress(names, public)
                system += "\n" + "\n".join(skills[k] for k in self.completed_roles if k in skills)
            if "handoff" in self.library.get("programs",{}):
                extra=run_program(self.library,"handoff",{"instruction":self.context["task"]["instruction"],
                    "progress":progress,"remaining_calls":self.max_calls-self.requests,
                    "recent":self.context["history"][-2:]},self.trace)
                if isinstance(extra,str):
                    system+="\n"+extra[:3500]
        return FormalExecutor.generate_message(self, system, progress, purpose=purpose, max_tokens=max_tokens)

    def summarize(self, *args):
        result = super().summarize(*args)
        events = self.trace.events
        result["consumed_native_choices"] = [dict(e) for e in events if e["event"] == "choice_used" and e["mode"] == "D"]
        result["active_routes"] = [e["route"] for e in events if e["event"] == "route_active"]
        result["completed_routes"] = [e["route"] for e in events if e["event"] == "route_satisfied"]
        result["none_selections"] = sum(e["event"] == "decision" and e["candidates"][e["selected"]]["value"] is None for e in events)
        return result
