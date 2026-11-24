"""Program-enumerated menus, with a native Jev choice at assembly and at execution."""
from copy import deepcopy
import json

from .card_assembly import CardAssembly, public_apps
from .card_library import ports
from .expressions import path_get
from .formal_runtime import bounded
from .fragment_assembly import option
from .policy_programs import load_program
from .runtime import REVIEW_MAX_TOKENS, REVIEW_THINKING_BUDGET, STEP_REVIEW, Uncovered, abstained
from .scoped_menu import is_scoped, result_sources, specific_use


class SemanticAssembly(CardAssembly):
    def initialize(self, library):
        super().initialize(library)
        self.app_hints = public_apps(self.request, library["cards"])
        self.pool = set(library["cards"])
        self.provenance = {}
        self.step_bindings = {}

    def policy_context(self):
        return {"request": self.request, "values": self.public, "bindings": self.provenance,
                "completed": self.completed_cards, "conditions": self.criteria,
                "receipts": self.receipts, "read_scopes": self.read_scopes,
                "datetime":self.context["task"].get("datetime"),"active_card":self.active_card}

    def policy(self, name, source, context, default=None):
        try:
            return load_program(source)(deepcopy(context))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
            self.trace.add("program_error", policy=name, error=str(exc)[:300])
            return default

    def token(self, app):
        if app in self.tokens:
            return self.tokens[app]
        adapter = self.registry["adapters"].get(app, {})
        field = adapter.get("login_field", "email")
        account = next((a for a in self.call("supervisor", "show_account_passwords", {}, "session")
                        if a["account_name"] == app), None)
        identity = self.context["task"]["supervisor"].get(field)
        if not identity or not account:
            raise Uncovered(f"Missing {app} account or supervisor {field}")
        result = self.call(app, "login", {"username": identity, "password": account["password"]}, "session")
        token = path_get(result, adapter.get("token_path", "access_token"))
        self.trace.add("adapter_login", app=app, identity_field=field, success=bool(token))
        if not token:
            raise Uncovered(f"{app} login with supervisor {field} produced no session")
        self.tokens[app] = token
        return token

    def call(self, app, api, arguments, node=None):
        adapter = self.registry["adapters"].get(app + "__" + api, {})
        field_map = adapter.get("argument_map", {})
        actual = {field_map.get(k, k): v for k, v in arguments.items()}
        result = super().call(app, api, actual, node)
        if adapter.get("result_path"):
            result = path_get(result, adapter["result_path"])
            if result is None:
                raise Uncovered(f"Missing declared result field for {app}.{api}")
        return result

    def call_endpoint(self, endpoint, arguments, paginate=False):
        adapter = self.registry["adapters"].get(endpoint)
        if not adapter:
            return super().call_endpoint(endpoint, arguments, paginate)
        app, api = endpoint.split("__", 1)
        if endpoint not in self.schemas or endpoint == "supervisor__complete_task" or api in ("login", "show_account_passwords"):
            raise Uncovered("Expected an installed public data operation")
        props = self.schemas[endpoint].get("properties", {})
        mapped = adapter.get("argument_map", {})
        if not isinstance(arguments, dict) or "access_token" in arguments or any(mapped.get(k,k) not in props for k in arguments):
            raise Uncovered("Arguments do not match the public operation")
        args = dict(arguments)
        if "access_token" in props:
            args["access_token"] = self.token(app)
        if not paginate:
            return self.call(app, api, args, self.active_card or "generated_read")
        size = adapter.get("page_size", 20)
        rows, page = [], 0
        while True:
            batch = self.call(app, api, {**args, "page_index":page, "page_limit":size}, self.active_card or "generated_read")
            if not isinstance(batch, list):
                raise Uncovered("Expected a page of records")
            rows.extend(batch)
            if len(batch) < size:
                return rows
            page += 1

    def choose(self, name, choices, *, widen_on_none=False, **facts):
        if name == "card_parameter":
            views = self.project_views("parameter")
            if views:
                facts["views"] = views
        program = self.registry["decision_policies"].get(name)
        if program:
            indexes = self.policy("decision/" + name, program,
                                  {**self.policy_context(), "options":choices, "evidence":facts})
            if isinstance(indexes, list) and all(type(i) is int and 0 <= i < len(choices) for i in indexes):
                choices = [choices[i] for i in dict.fromkeys(indexes)]
        selected = super().choose(name, choices, widen_on_none=widen_on_none, **facts)
        if self.active_card and name != "card_parameter":
            self.step_bindings[name] = selected
        return selected

    def parameter_context(self, card):
        if 'handoff' in self.registry['questions'] and card.get('parameters'):
            # The candidate can use any declared input. Retain exact small values for offline re-binding.
            values = {k:self.public[k] for k in card.get('inputs',[]) if k in self.public and
                      len(json.dumps(self.public[k],ensure_ascii=False,default=str)) <= 65536}
            self.trace.add('parameter_state', card=self.active_card, values=values,
                           available=list(self.public), datetime=self.context['task'].get('datetime'))
        ctx = super().parameter_context(card)
        self.step_bindings.update(ctx["params"])
        if 'handoff' in self.registry['questions'] and card.get('parameters'):
            self.trace.add('parameter_bindings', card=self.active_card, bound=dict(ctx['params']))
        return ctx

    def perform_card(self, name):
        self.step_bindings = {}
        self.active_read_scope = None
        super().perform_card(name)
        card = self.registry["cards"][name]
        _, inputs, output, _ = ports(card)
        self.provenance = {key: value for key, value in self.provenance.items() if key in self.public}
        if is_scoped(self.registry):
            self.provenance.pop(output,None)
        self.provenance[output] = {"producer":name, "inputs":list(inputs), "parameters":dict(self.step_bindings)}
        if is_scoped(self.registry):
            if self.public.get("scope") and self.registry["cards"][name].get("builtin") in ("library","playlist","album"):
                self.provenance[output]["source_scope"] = self.public["scope"]
            if self.active_read_scope is not None:
                self.read_scopes[output] = self.active_read_scope
            self.trace.add("result_source", role=output, **self.provenance[output])

    def forget_derived(self, changed):
        super().forget_derived(changed)
        if is_scoped(self.registry):
            self.provenance = {k:v for k,v in self.provenance.items() if k in self.public}
            self.read_scopes = {k:v for k,v in self.read_scopes.items() if k in self.public}

    def set_subset(self, songs):
        super().set_subset(songs)
        if is_scoped(self.registry) and self.active_card:
            for role in ("song_ids","song_records"):
                self.provenance[role] = {"producer":self.active_card, "inputs":["song_ids"],
                                         "parameters":dict(self.step_bindings), "source_scope":self.public.get("scope")}
                self.read_scopes.pop(role,None)

    def project_views(self, site):
        views = {}
        for name, rule in self.registry["views"].items():
            if rule.get("at","state") == site and all(role in self.public for role in rule.get("requires", [])):
                value = self.policy("state/" + name, rule["program"], self.policy_context())
                if value is not None:
                    views[name] = value
        return views

    def state_view(self):
        state = super().state_view()
        state["bindings"] = {role: info for role, info in self.provenance.items() if info["parameters"]}
        if is_scoped(self.registry) and self.provenance:
            state["sources"] = result_sources(self.registry,self.public,self.provenance)
        views = self.project_views("state")
        if views:
            state["views"] = views
        return state

    def families(self, eligible):
        cards = self.registry["cards"]
        rules = dict(self.registry["assembly"])
        by_app = {}
        for name, card in cards.items():
            by_app.setdefault(ports(card)[0], []).append(name)
        for app in by_app:
            rules.setdefault(app, {"description":app + " operations", "apps":[app]})
        rules = sorted(rules.items(), key=lambda item: not bool(self.app_hints.intersection(item[1].get("apps", []))))
        result, seen = {}, set()
        for name, rule in rules:
            seeds = set(rule.get("cards", []))
            for app in rule.get("apps", []):
                seeds.update(by_app.get(app, []))
            if "program" in rule:
                context = {**self.policy_context(), "eligible":eligible, "cards":list(cards)}
                extra = self.policy("assembly/" + name, rule["program"], context, [])
                if isinstance(extra, list):
                    seeds.update(s for s in extra if isinstance(s, str) and s in cards)
            # A downstream operation can bring its data providers into the same menu.
            pending = list(seeds)
            while pending:
                target = pending.pop()
                needed = set(ports(cards[target])[1]) - self.public.keys() - {"answer", "effect_receipt"}
                for provider, card in cards.items():
                    if provider not in seeds and ports(card)[2] in needed:
                        pending.append(provider)
                        seeds.add(provider)
            members = tuple(s for s in eligible if s in seeds)
            if members and members not in seen:
                result[name] = (rule["description"], list(members))
                seen.add(members)
        return result

    def matching_handoffs(self):
        for name, rule in self.registry["handoffs"].items():
            if 'label' in rule:
                continue  # Text-only guides are selected once at an actual generation handoff.
            if not all(role in self.public for role in rule.get("requires", [])):
                continue
            if "when" not in rule or self.policy("handoff/" + name, rule["when"], self.policy_context(), False) is True:
                yield name, rule

    def next_card(self):
        cards = self.registry["cards"]
        excluded = {}
        eligible = []
        for name, card in cards.items():
            reason = self.unavailable(name, card)
            if reason:
                excluded[name] = reason
            else:
                eligible.append(name)
        for name, rule in self.matching_handoffs():
            if "defer" in rule and self.policy("handoff/" + name, rule["defer"], self.policy_context(), False) is True:
                raise Uncovered("Open generation requested by handoff/" + name)
        state = self.state_view()
        menus = self.families(eligible)
        self.trace.add("assembly_candidates", app_hints=sorted(self.app_hints), eligible=eligible,
                       families={n:members for n, (_, members) in menus.items()}, excluded=excluded)
        chosen, mode = None, "all"
        if len(menus) > 1:
            choices = [option(desc if is_scoped(self.registry) else desc + " Available steps: " + " | ".join(cards[s]["description"] for s in members[:6]), name)
                       for name, (desc, members) in menus.items()]
            try:
                chosen = self.choose("assembly", choices, **state)
                mode = "jev"
            except Uncovered as exc:
                if not abstained(exc):
                    raise
                mode = "jev_widen"
        focused = menus[chosen][1] if chosen else eligible
        self.trace.add("assembly_selection", mode=mode, family=chosen, offered=focused, full_count=len(eligible))

        def pick(names, menu):
            if is_scoped(self.registry):
                return self.pick_scoped(names, menu, eligible, excluded, state)
            self.trace.add("card_menu", offered=names, excluded=excluded, state=state, assembly=menu)
            return self.choose("step", [self.card_option(n, cards[n]) for n in names],
                               widen_on_none=names != eligible, **state)

        try:
            return pick(focused, chosen)
        except Uncovered as exc:
            if focused == eligible or not abstained(exc):
                raise
        self.trace.add("assembly_widen", rejected=chosen, offered=eligible)
        return pick(eligible, None)

    def pick_scoped(self, names, menu, eligible, excluded, state):
        cards = self.registry["cards"]
        common = [n for n in names if not specific_use(cards[n])]
        specific = [n for n in names if specific_use(cards[n])]
        def select(group, choices, can_widen):
            self.trace.add("card_menu", offered=group, excluded=excluded, state=state, assembly=menu)
            return self.choose("step",choices,widen_on_none=can_widen,**state)
        # A growing collection of specialists occupies one slot in the ordinary menu.
        # A single specialist is already one slot and can be picked directly.
        if common and len(specific) > 1:
            choices = [self.card_option(n,cards[n]) for n in common]
            choices.append(option("Choose a specialized operation for a particular requested outcome.", {"specialized":True}))
            choice = select(common,choices,names != eligible)
            if choice != {"specialized":True}:
                return choice
            try:
                return select(specific,[self.card_option(n,cards[n]) for n in specific],True)
            except Uncovered as exc:
                if not abstained(exc):
                    raise
                names = common
        return select(names,[self.card_option(n,cards[n]) for n in names],names != eligible)

    def read_offers(self):
        offers = super().read_offers()
        for _, rule in self.matching_handoffs():
            for name in rule.get("reads", []):
                card = self.registry["cards"][name]
                if card["api"] in self.schemas and self.unavailable(name, card) is None:
                    offers[name] = card["api"]
        return offers

    def preferred_tool_docs(self):
        return {api for _, rule in self.matching_handoffs() for api in rule.get("apis", [])}

    def delegate_read(self, identifier, arguments):
        self.step_bindings = {}
        super().delegate_read(identifier, arguments)
        name = self.completed_cards[-1]
        _, inputs, output, _ = ports(self.registry["cards"][name])
        self.provenance[output] = {"producer":name, "inputs":list(inputs),
                                   "parameters":{**arguments, **self.step_bindings}}

    def handoff(self, reason):
        data = super().handoff(reason)
        data["decision_evidence"] = self.state_view()
        for name, rule in self.matching_handoffs():
            data.setdefault("handoff_rules", []).append(name)
            if rule.get("skill"):
                data.setdefault("fragment_guidance", []).append(rule["skill"])
        if 'handoff' in self.registry['questions']:
            from .semantic_handoff import snapshot, select_guidance
            state = snapshot(self)
            self.trace.add('handoff_state', **state)
            name = select_guidance(self,state)
            # A public-read reentry can replace the preceding handoff with ordinary assistance.
            data.setdefault('handoff_rules',[])
            data.setdefault('fragment_guidance',[])
            data['selected_handoff_guidance'] = None
            if name is not None:
                data.setdefault('handoff_rules',[]).append(name)
                data['selected_handoff_guidance'] = {'name':name,'skill':self.registry['handoffs'][name]['skill']}
        return data

    def generation_view(self, progress=None, compact=False):
        if not progress or 'selected_handoff_guidance' not in progress:
            return super().generation_view(progress,compact)
        guide = progress['selected_handoff_guidance']
        visible = super().generation_view({k:v for k,v in progress.items() if k!='selected_handoff_guidance'},compact)
        if guide:
            visible['selected_handoff_guidance'] = {'name':guide['name'],'skill':bounded(guide['skill'],1600 if compact else 3200)}
        if is_scoped(self.registry):
            sources = progress.get("decision_evidence",{}).get("sources",{})
            if sources:
                visible["available_results"] = {role:{"source":info.get("api",info["producer"]),
                    **({"scope":info["scope"]} if "scope" in info else {})} for role,info in sources.items()}
            if lookup := progress.get("usage_lookup"):
                visible["result_usage"] = f"For a role's full usage note, read compiled['{lookup}'][role]."
        return visible

    def generate_message(self, system, progress=None, **kwargs):
        if progress and progress.get('selected_handoff_guidance'):
            system = "Follow selected_handoff_guidance for the current handoff. " + system
        return super().generate_message(system,progress,**kwargs)

    def execute_task(self, library):
        self.initialize(library)
        try:
            self.assemble_available()
        except (Uncovered, ValueError, TypeError, IndexError, KeyError) as exc:
            self.residual_progress = self.handoff(str(exc))
            self.agent("Follow the applicable fragment guidance in compiled_progress. To return a supported "
                       "public read to Jev menu assembly, use one cell yield_read('app__api', {literal public arguments}). "
                       "The program executes the read. Otherwise finish the remaining work with the public tools.",
                       self.residual_progress)

    def summarize(self, *args):
        row = super().summarize(*args)
        events = self.trace.events
        choices = [e for e in events if e["event"] == "decision" and e.get("node") == "fragment/assembly"]
        row["assembly_decisions"] = len(choices)
        row["assembly_focused_menus"] = sum(e["event"] == "assembly_selection" and e["mode"] == "jev" for e in events)
        row["assembly_widens"] = sum(e["event"] == "assembly_widen" or
                                     e["event"] == "assembly_selection" and e["mode"] == "jev_widen" for e in events)
        row["parameter_choices"] -= sum(e["candidates"][e["selected"]]["value"] is not None for e in choices)
        if 'handoff' in getattr(self,'registry',{}).get('questions',{}):
            decisions = [e for e in events if e['event']=='decision' and e.get('node')=='fragment/handoff']
            row['parameter_choices'] -= sum(e['candidates'][e['selected']]['value'] is not None for e in decisions)
            selections = [e for e in events if e['event']=='handoff_selection']
            row['semantic_handoff'] = {k:sum(e.get(k,0) for e in selections) for k in ('calls','input_tokens','output_tokens','model_seconds')}
            row['semantic_handoff'].update(handoffs=len(selections), selected=sum(e['guide'] is not None for e in selections),
                                           eligible=sum(bool(e['eligible']) for e in selections),
                                           guides=sorted({e['guide'] for e in selections if e['guide'] is not None}))
        if self.step_review_threshold is not None:
            if self.step_review_mode == "official_adaptive":
                row["execution_settings"].update(step_review_mode=self.step_review_mode,
                                                 step_review_threshold=self.step_review_threshold,
                                                 step_review_thinking_budget=self.step_review_thinking_budget,
                                                 step_review_mix=0.5)
                choices = [e["response"] for e in events if e["event"] == "model" and e["operation"] == "decide"]
                row["decision_readout_tokens"] = sum(r.get("readout_tokens", 0) for r in choices)
                row["decision_backend_requests"] = sum(r.get("backend_requests", 0) for r in choices)
            else:
                row["execution_settings"].update(step_review_threshold=self.step_review_threshold,
                                                step_review_max_tokens=REVIEW_MAX_TOKENS,
                                                step_review_thinking_budget=REVIEW_THINKING_BUDGET)
            reviews = [e for e in events if e["event"] == "step_review"]
            generated = [e["response"] for e in events if e["event"] == "model" and
                         e["request"].get("purpose") == STEP_REVIEW]
            row["step_review"] = {"calls": len(generated),
                "output_tokens": sum(r.get("output_tokens", 0) for r in generated),
                "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in generated),
                **{key: sum(e["outcome"] == key for e in reviews) for key in ("kept", "changed", "invalid")}}
        return row
