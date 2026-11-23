"""Shared phase-one executor: complete operation, useful partial reads, residual generation."""
import math
import re

from .catalog_entry import ENUM, candidates, describe, display
from .composable_v2 import route_inputs, routed
from .expressions import build_candidates
from .phase1 import PhaseOneExecutor, public_role
from .pilot_interfaces import discover_consumers
from .runtime import MAX_REAL_OPTIONS, Uncovered


COMPLETE_Q = "Which operation fulfills the entire request, including every object, condition and requested result? Choose NONE if none does."
COMPLETE_SKILL = "Match data source, direction, affected objects and output. Do not select an operation that only completes part of the request or makes an unrequested change."
READ_Q = "Which read would provide specifically needed data for the unfinished part of this request? Choose NONE if no listed source supplies it."
READ_SKILL = "Match the actual entity, direction and status in the requested work. Consider already completed reads. Do not select an unrelated record or a mere login."


def completion(name, spec, routes):
    value = spec.get("finish") or spec.get("selection",{}).get("finish")
    if value:
        return value
    rule = routes.get(name,{})
    if rule.get("complete") and rule.get("goals")==[spec["output"]]:
        return "answer" if spec["output"]=="answer" else "action"
    return None


def standalone_read(spec):
    return public_role(spec["output"]) and (spec.get("effect_kind")=="read" or
            ("effect_kind" not in spec and str(spec.get("effect","")).lower().startswith(("read-only","read only"))))


class WaitingForData(Uncovered):
    """A selected complete operation can be retried if partial reads produce new data."""


class StagedExecutor(PhaseOneExecutor):
    direct_menu_choices = False

    def execute_task(self, library):
        self.started_entry = False
        self.stage = "complete"
        self.entry_status = "not_offered"
        self.entry_name = None
        self.entry_reason = ""
        self.route_definitions = routed(library)
        self.finishes = {n:completion(n,f,self.route_definitions) for n,f in library["workflow"]["fragments"].items()}
        self.read_attempts = set()
        self.empty_read_menus = set()
        self.resolved_args = {}
        self.fetching_slot_data = False
        super().execute_task(library)

    def materialize(self,plan,args):
        if (self.stage=="partial" or self.fetching_slot_data) and plan[0]=="fragment":
            name=plan[1]
            spec=self.library["workflow"]["fragments"][name]
            if self.finishes[name]=="action" or spec.get("effect_kind")=="write":
                raise Uncovered("A read cannot execute a state-changing dependency: "+name)
        return super().materialize(plan,args)

    def _model(self, operation, **values):
        if operation=="generate" and self.assembling:
            raise Uncovered("Open generation is deferred until after partial reads")
        response = super()._model(operation, **values)
        self.trace.add("stage_model",stage=self.stage,mode=response["mode"],
                       input_tokens=response.get("input_tokens",0),output_tokens=response.get("output_tokens",0))
        return response

    def recover(self, name, skill, policy, reason):
        if self.assembling:
            self.trace.add("recovery_deferred",fragment=name,reason=reason)
            raise Uncovered(reason)
        return super().recover(name,skill,policy,reason)

    def _choose(self, question, evidence, rows, skill, node, policy=None):
        """Keep NONE on every page; never silently truncate, even at a second reduction."""
        choices=list(rows)
        if not choices:
            return None
        level=0
        while True:
            n_pages=math.ceil(len(choices)/MAX_REAL_OPTIONS)
            size=math.ceil(len(choices)/n_pages)
            survivors=[]
            for i in range(0,len(choices),size):
                page=choices[i:i+size]
                try:
                    picked=super().decision(question,evidence,
                        [{"label":p["label"],"value":j} for j,p in enumerate(page)],
                        skill=skill,policy=policy or {"mode":"D"},node=f"{node}/{level}/{i//size+1}")
                except Uncovered as exc:
                    if str(exc) not in ("Jev selected the uncovered option","Decision policy requested generation for this uncertain state"):
                        raise
                    continue
                survivors.append(page[picked])
            if not survivors:
                return None
            if len(survivors)==1:
                return survivors[0]["value"]
            choices=survivors
            level+=1

    def decision(self, question, evidence, candidates, skill="", policy=None, node=None):
        if self.assembling and (policy or {}).get("mode")=="R":
            raise Uncovered("This interface needs reasoning; defer until after partial reads")
        if self.assembling and len(candidates)>MAX_REAL_OPTIONS:
            picked=self._choose(question,evidence,candidates,skill,node or "local",policy)
            if picked is None:
                raise Uncovered("Jev selected the uncovered option")
            return picked
        return super().decision(question,evidence,candidates,skill,policy,node)

    def select(self, which, names):
        fs=self.library["workflow"]["fragments"]
        self.trace.add("stage_candidates",stage=which,options=names,
                       completed=list(self.completed_roles))
        opt=self.library.get("catalog_selectors",{}).get(which,{})
        defaults=(COMPLETE_Q,COMPLETE_SKILL) if which=="complete" else (READ_Q,READ_SKILL)
        rows=[{"label":describe(n,fs[n]),"value":n} for n in names]
        context={"request":self.context["task"]["instruction"],"completed":self.completed_roles}
        picked=self._choose(opt.get("question",defaults[0]),context,rows,opt.get("skill",defaults[1]),"stage/"+which)
        self.trace.add("stage_choice",stage=which,fragment=picked)
        return picked

    def arguments(self, name, remembered, suggested=None):
        fs=self.library["workflow"]["fragments"]
        task=self.context["task"]
        slots, captures, required, visited = {}, {}, [], set()
        def collect(n):
            if n in visited:
                return
            visited.add(n)
            spec=fs[n]
            slots.update(spec.get("selection",{}).get("slots",{}))
            slots.update(spec.get("slots",{}))
            captures.update(self.route_definitions.get(n,{}).get("captures",{}))
            for role in spec["inputs"]:
                if role in slots:
                    if role not in required: required.append(role)
                    continue
                providers=self.by_role.get(role,[])
                if len(providers)==1:
                    collect(providers[0])
                elif not providers and role not in required:
                    required.append(role)
        collect(name)
        spans={}
        for n in sorted(visited):
            inputs=self.route_definitions.get(n,{}).get("inputs",{})
            if inputs:
                try:
                    spans.update(route_inputs({"inputs":inputs},task,self.live))
                except (Uncovered,KeyError,IndexError,ValueError,TypeError):
                    pass  # State-dependent bindings can become available after a partial read.
        for rule in discover_consumers(self.library,task["instruction"]).values():
            spans.update(route_inputs(rule,task,{}))
        if suggested:
            spans.update(suggested)
        for role in required:
            if role in remembered:
                continue
            slot=slots.get(role)
            cap=captures.get(role,{})
            if slot is None and cap.get("mode","one")=="one" and cap.get("pattern"):
                enums=ENUM.findall(cap["pattern"])
                if len(enums)==1 and re.compile(cap["pattern"]).groups==1:
                    slot={"candidates":[{"label":display(k),"value":k} for k in dict.fromkeys(enums[0].split("|"))]}
            if slot is not None:
                for dep in slot.get("requires",[]):
                    if dep not in self.live:
                        providers=self.by_role.get(dep,[])
                        if len(providers)!=1 or not standalone_read(fs[providers[0]]):
                            raise WaitingForData("Missing public read "+dep)
                        self.fetching_slot_data=True
                        try:
                            if not self.activate("slot_read/"+dep,{"goals":[dep]},remembered,providers[0]):
                                raise WaitingForData("Cannot yet materialize "+dep)
                        finally:
                            self.fetching_slot_data=False
                live={k:v for k,v in {**self.live,**remembered}.items()
                      if public_role(k) and k not in self.generated_inputs}
                safe_task={k:task[k] for k in ("instruction","datetime","supervisor") if k in task}
                try:
                    opts=build_candidates(slot["candidates"],{"vars":live,"inputs":live,"task":safe_task})
                except (KeyError,IndexError,TypeError,ValueError) as exc:
                    raise WaitingForData(f"Missing public candidates for {role}: {exc}") from exc
                if not opts:
                    raise WaitingForData("No public candidates for "+role)
                value=self._choose(slot.get("question",f"Which value of '{display(role)}' does the request specify?"),
                    {"request":task["instruction"],"operation":describe(name,fs[name]),
                     "already_bound":{k:v for k,v in remembered.items() if public_role(k)}},
                    opts,slot.get("skill",""),"stage/slot/"+role)
                if value is None:
                    raise Uncovered("Jev declined parameter "+role)
                via="native"
            elif role in spans:
                value,via=spans[role],"public span or discovery"
            elif cap:
                value,via=route_inputs({"captures":{role:cap}},task,self.live)[role],"public capture"
            else:
                raise WaitingForData("Missing parameter source for "+role)
            remembered[role]=value
            self.trace.add("stage_binding",fragment=name,role=role,source=via,
                           **({"value":value} if public_role(role) else {}))
        return remembered

    def attempt_complete(self):
        self.stage="binding"
        name=self.entry_name
        spec=self.library["workflow"]["fragments"][name]
        try:
            args=self.arguments(name,self.resolved_args)
            self.stage="complete"
            if not self.activate("catalog/"+name,{"goals":[spec["output"]]},args,name,native=True):
                raise Uncovered(self.unresolved.get("catalog/"+name,"Operation failed"))
        except (Uncovered,ValueError,KeyError,TypeError,IndexError) as exc:
            self.entry_status="waiting_for_data" if isinstance(exc,WaitingForData) else "deferred"
            self.entry_reason=str(exc)
            self.trace.add("stage_deferred",stage=self.stage,fragment=name,reason=str(exc))
            return
        self.call("supervisor","complete_task",
                  {"answer":self.live[spec["output"]]} if self.finishes[name]=="answer" else {},"composition_finish")
        self.entry_status="completed"
        self.trace.add("stage_completed",fragment=name,finish=self.finishes[name])

    def partials(self, pool):
        self.stage="partial"
        fs=self.library["workflow"]["fragments"]
        loose_reads={name for name in pool if standalone_read(fs[name]) and not self.finishes[name]}
        # Existing scoped menus are independent of the app named by the user (e.g. Phone contacts for Venmo).
        scoped={n:{**m,"options":[o for o in m["options"] if not self.finishes[o["fragment"]]
                    and not o.get("finish") and fs[o["fragment"]].get("effect_kind") not in ("write","internal")]}
                for n,m in self.library.get("menus",{}).items()}
        owned={o["fragment"] for m in scoped.values() for o in m["options"]}
        self.trace.add("stage_enter",stage="partial",from_complete=self.entry_status)
        while not self.done:
            before=len(self.cache)
            live={k:v for k,v in self.live.items() if k not in self.generated_inputs}
            for n,m in scoped.items():
                if m["options"]:
                    self.consider(n,m,self.context["task"],live,loose_reads=loose_reads)
            if "discovery" in self.library.get("programs",{}):
                self.discover_from_state()
            hints={}
            for route in self.routes.values():
                if route.get("complete") or set(route.get("after_roles",[]))-self.live.keys() or not all(
                    any(re.search(p,obs,re.I) for obs in self.observations) for p in route.get("after_paths",[])):
                    continue
                try:
                    supplied=route_inputs(route,self.context["task"],self.live)
                except (Uncovered,KeyError,ValueError,TypeError):
                    continue
                for role in route["goals"]:
                    for n in self.by_role.get(role,[]):
                        hints.setdefault(n,supplied)
            eligible=[n for n in sorted(set(pool)|hints.keys()) if n not in owned and not self.finishes[n]
                      and standalone_read(fs[n]) and n not in self.read_attempts and fs[n]["output"] not in self.live]
            key=(tuple(eligible),tuple(self.completed_roles))
            picked=None
            if eligible and key not in self.empty_read_menus:
                picked=self.select("read",eligible)
                if picked is None:
                    self.empty_read_menus.add(key)
                else:
                    self.read_attempts.add(picked)
                    try:
                        args=self.arguments(picked,{},hints.get(picked))
                        self.activate("read/"+picked,{"goals":[fs[picked]["output"]]},args,picked,native=True)
                    except (Uncovered,KeyError,TypeError,ValueError,IndexError) as exc:
                        self.trace.add("stage_deferred",stage="partial",fragment=picked,reason=str(exc))
            if self.entry_status=="waiting_for_data" and len(self.cache)>before:
                self.attempt_complete()
                self.stage="partial"
            if len(self.cache)==before and not picked:
                break

    def assemble(self):
        if self.started_entry:
            return
        self.started_entry=True
        self.assembling=True
        try:
            f=self.library["workflow"]["fragments"]
            pool,apps=candidates(f,self.context["task"]["instruction"],True)
            roots=[n for n in pool if self.finishes[n]]
            self.trace.add("stage_prefilter",apps=apps,installed=len(f),retained=pool,complete=roots)
            self.entry_name=self.select("complete",roots) if roots else None
            if self.entry_name is not None:
                self.attempt_complete()
            elif roots:
                self.entry_status="none"
            if not self.done:
                self.partials(pool)
        finally:
            self.assembling=False

    def handoff_progress(self, names, public):
        data=super().handoff_progress(names,public)
        if self.entry_name and self.entry_reason:
            data["complete_operation"]={"selected":self.entry_name,"outcome":self.entry_status,"reason":self.entry_reason}
        return data

    def preferred_tool_docs(self):
        instruction=self.context["task"]["instruction"]
        rules=self.library.get("tool_contexts",{})
        selected=[]
        for name,rule in rules.items():
            if set(rule.get("after_roles",[]))-set(self.completed_roles):
                continue
            if rule.get("cues") and not any(re.search(p,instruction,re.I) for p in rule["cues"]):
                continue
            selected.extend(rule["apis"])
        return set(selected)

    def agent(self, skill="", progress=None):
        self.stage="generation"
        self.trace.add("stage_handoff",complete=self.entry_status,completed=list(self.completed_roles),
                       reason=self.entry_reason,unresolved=self.unresolved)
        failures=[e["fragment"] for e in self.trace.events if e["event"]=="recovery_deferred"]
        hints=[self.library.get("generation_skills",{}).get(n,"") for n in dict.fromkeys(failures)]
        if progress is None and (self.live or self.unresolved or self.write_progress or self.entry_reason):
            progress=self.handoff_progress([],{})
        return super().agent(skill="\n".join([skill,*hints]),progress=progress)

    def summarize(self,*args):
        row=super().summarize(*args)
        events=self.trace.events
        calls={}; root_cost={"calls":0,"input_tokens":0}; last_native={}
        for e in events:
            if e["event"]=="stage_model":
                c=calls.setdefault(e["stage"],{"calls":0,"input_tokens":0,"output_tokens":0})
                c["calls"]+=1
                for k in ("input_tokens","output_tokens"): c[k]+=e[k]
            elif e["event"]=="model" and e["operation"]=="decide":
                last_native=e["response"]
            elif e["event"]=="decision":
                if str(e.get("node","")).startswith("stage/complete/"):
                    root_cost["calls"]+=1
                    root_cost["input_tokens"]+=last_native.get("input_tokens",0)
                last_native={}
        row["fragment_decisions"]+=sum(e["event"]=="decision" and str(e.get("node","")).startswith(
            ("stage/complete/","stage/read/")) for e in events)
        local_names={e["fragment"] for e in events if e["event"]=="fragment_choice_used" and
                     not e["reused"] and public_role(e["output"]) and not str(e["menu"]).startswith("catalog/")}
        selected={e["fragment"] for e in events if e["event"]=="fragment_choice_used" and not e["reused"]}
        fragments=self.library["workflow"]["fragments"]
        origins={n:{t.rsplit("_",1)[0] for t in fragments[n].get("source_tasks",[])}
                 for n in selected if public_role(fragments[n]["output"])}
        scenario=row["task_id"].rsplit("_",1)[0]
        row["staged"]={"complete":self.entry_status,"selected":self.entry_name,
            "partial_entered":any(e["event"]=="stage_enter" for e in events),"local_fragments":sorted(local_names),
            "handoff":any(e["event"]=="stage_handoff" for e in events),"cost_by_stage":calls,"entry_cost":root_cost,
            "cross_scenario_choices":sorted(n for n,s in origins.items() if s and scenario not in s),
            "choices_without_source":sorted(n for n,s in origins.items() if not s)}
        return row
