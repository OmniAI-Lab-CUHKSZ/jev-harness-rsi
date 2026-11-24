"""Assemble one executable fragment at a time from the current AppWorld state."""
import ast
import json
import re

from .card_library import ports
from .expressions import parameter_groups, resolve
from .formal_runtime import FormalExecutor
from .fragment_assembly import FIELDS, FragmentAssembly, option, put_ids
from .runtime import Uncovered, abstained, is_open_generation
from .scoped_menu import card_label, consumer_tips, is_scoped


APP_WORDS = {"spotify":r"spotify|songs?|albums?|playlists?|music",
             "file_system":r"file[_ ]system|files?|folders?|directories|directory",
             "phone":r"phone|alarms?|contacts?", "venmo":r"venmo", "simple_note":r"simple[_ ]note|notes?"}
SONG_SOURCES = {"library","album","playlist"}
LOOKUPS = {"spotify__search_albums":"album","spotify__show_playlist_library":"playlist"}


def public_apps(instruction, cards):
    apps = {ports(card)[0] for card in cards.values()} - {"supervisor","general"}
    return {app for app in apps if re.search(r"\b(?:" + APP_WORDS.get(app,re.escape(app)) + r")\b",
                                             instruction,re.I) or app=="file_system" and "~/" in instruction}


def reachable_cards(cards, apps):
    names = {name for name, card in cards.items()
             if not apps or ports(card)[0] in apps | {"general","supervisor"}}
    # Only concrete data dependencies bring in another app. Shared terminal roles would pull in all sources.
    while True:
        missing = {role for name in names for role in ports(cards[name])[1]} - {"answer","effect_receipt"}
        providers = {name for name, card in cards.items() if ports(card)[2] in missing}
        if providers <= names:
            return names
        names |= providers


class CardAssembly(FragmentAssembly):
    def choose(self, name, choices, **facts):
        if self.active_card and not (is_scoped(self.registry) and name == "card_parameter"):
            facts["operation_guidance"] = self.registry["cards"][self.active_card]["skill"]
        return super().choose(name, choices, **facts)

    def initialize(self, library):
        self.registry, self.request = library, self.context["task"]["instruction"]
        self.public, self.tokens, self.seed_rows, self.criteria = {}, {}, {}, {}
        self.sequence, self.completed_cards, self.receipts = [], [], []
        self.derived, self.returned_reads = {}, set()
        self.read_scopes, self.failed_card = {}, None
        self.active_card = None
        self.pool = reachable_cards(library["cards"], public_apps(self.request,library["cards"]))
        self.schemas = {f.get("function",f)["name"]:f.get("function",f)["parameters"]
                        for f in self.context["task"]["functions"]}

    def unavailable(self, name, card):
        _, inputs, output, _ = ports(card)
        if name not in self.pool and (not inputs or not all(r in self.public for r in inputs)):
            return "outside current applications and dependencies"
        missing = [r for r in inputs if r not in self.public]
        if missing:
            return "missing " + ", ".join(missing)
        if output in self.public:
            return "output already available"
        builtin = card.get("builtin")
        if builtin=="play" and len(self.public["song_ids"])!=1:
            return "playback requires exactly one song"
        if builtin in {"filter_genre","sort_songs"} and not self.public["song_ids"]:
            return "empty song selection"
        if builtin=="take_first" and not self.public["ranked_songs"]:
            return "empty ranking"
        if builtin=="filter_genre" and "ranked_songs" in self.public:
            return "ranking already applied to the current scope"
        return None

    def state_view(self):
        def shape(value):
            if isinstance(value,list):
                return {"count":len(value)}
            if isinstance(value,dict):
                return {"fields":list(value)[:12]}
            return str(value)[:180]
        return {"scope":self.public.get("scope"), **({"read_scopes":self.read_scopes} if self.read_scopes else {}),
                "available":{key:shape(value) for key,value in self.public.items()
                if key not in {"scope","selection","song_records"}},
                "applied_conditions":dict(self.criteria), "completed":list(self.completed_cards)}

    def card_option(self, name, card):
        if is_scoped(self.registry):
            return card_label(name, card)
        _, inputs, out, loads = ports(card)
        actual = [f"{k} ({len(self.public[k])} records)" if isinstance(self.public[k],list) else k for k in inputs]
        label = f"{card['description']} Inputs: {', '.join(actual) or 'request and account'}. "
        if loads:
            label += f"Reads: {loads}. "
        return option(label + f"Produces: {out}. {card['skill']}", name)

    def next_card(self):
        choices, excluded = [], {}
        for name, card in self.registry["cards"].items():
            reason = self.unavailable(name,card)
            if reason:
                excluded[name] = reason
                continue
            choices.append(self.card_option(name,card))
        state = self.state_view()
        self.trace.add("card_menu", offered=[c["value"] for c in choices], excluded=excluded, state=state)
        return self.choose("step",choices,**state)

    def forget_derived(self, changed):
        pending = set(changed)
        while pending:
            stale = {out for out,inputs in self.derived.items() if pending.intersection(inputs)}
            for out in stale:
                self.public.pop(out,None)
                self.derived.pop(out)
            pending = stale

    def order_songs(self):
        orders = [option(f"{desc}, {direction}",(k,reverse)) for k,desc in FIELDS.items()
                  for reverse,direction in ((True,"highest or latest first"),(False,"lowest or earliest first"))]
        field, reverse = self.choose("order",orders)
        rows = self.hydrate({field})
        if any(row[field] is None for row in rows):
            raise Uncovered("Some current songs lack the selected sorting field")
        ordered = self.apply_data("ordered_records",rows=rows,field=field,descending=reverse)
        self.criteria.update(field=field,descending=reverse)
        self.set_subset(ordered)
        return self.public["song_ids"]

    def call_endpoint(self, endpoint, arguments, paginate=False):
        app, api = endpoint.split("__",1)
        if endpoint not in self.schemas or endpoint=="supervisor__complete_task" or api in {"login","show_account_passwords"}:
            raise Uncovered("This endpoint is not an installed public data operation")
        props = self.schemas[endpoint].get("properties",{})
        if not isinstance(arguments,dict) or set(arguments)-props.keys() or "access_token" in arguments:
            raise Uncovered("Arguments do not match the available data operation")
        args = dict(arguments)
        if "access_token" in props:
            args["access_token"] = self.token(app)
        if not paginate:
            return self.call(app,api,args,self.active_card or "generated_read")
        rows, page = [], 0
        while True:
            data = self.call(app,api,{**args,"page_index":page,"page_limit":20},self.active_card or "generated_read")
            if not isinstance(data,list):
                raise Uncovered("Expected a page of records")
            rows.extend(data)
            if len(data)<20:
                return rows
            page += 1

    def parameter_context(self, card):
        ctx = {"state":self.public, "task":{"instruction":self.request,"datetime":self.context["task"].get("datetime")},
               "params":{}}
        for key, slot in card.get("parameters",{}).items():
            groups = parameter_groups(slot,ctx)
            skill = card["skill"]+" "+slot.get("skill","")
            for position, choices in enumerate(groups):
                try:
                    ctx["params"][key] = self.choose_inline(slot["question"],choices,skill,ctx["params"])
                    break
                except Uncovered as exc:
                    if position+1 == len(groups) or not abstained(exc):
                        raise
                    self.trace.add("parameter_widen",card=self.active_card,slot=key,remaining=len(groups[position+1]))
        return ctx

    def declared_operation(self, name, card):
        ctx = self.parameter_context(card)
        if "compute" in card:
            return resolve(card["compute"],ctx)
        def run_one(local):
            arguments = resolve(card.get("arguments",{}),local)
            result = self.call_endpoint(card["api"],arguments,card.get("paginate",False))
            if (is_scoped(self.registry) and "each" not in card and
                    card["api"].split("__",1)[1].startswith(("show_","search_","list_","get_"))):
                self.active_read_scope = {"api":card["api"],"arguments":arguments}
            return result
        if "each" not in card:
            return run_one(ctx)
        values = []
        for index,item in enumerate(resolve(card["each"],ctx)):
            value = run_one({**ctx,"item":item})
            values.append(value)
            if not card["api"].split("__",1)[1].startswith(("show_","search_","list_","get_")):
                self.receipts.append({"card":name,"index":index,"item":item,"result":value})
        return values

    def choose_inline(self, question, choices, skill, bindings):
        key = "card_parameter"
        # Per-task prompt view; the installed shared library may be in use by other workers.
        original = self.registry
        if shared_skill := original["skills"].get(key):
            skill = skill + " " + shared_skill
        self.registry = {**original,"questions":{**original["questions"],key:question},
                         "skills":{**original["skills"],key:skill}}
        try:
            return self.choose(key,choices,already_bound=dict(bindings))
        finally:
            self.registry = original

    def perform_card(self, name):
        card = self.registry["cards"][name]
        _,inputs,out,_ = ports(card)
        self.active_card = name
        try:
            if "builtin" not in card:
                value = self.declared_operation(name,card)
            elif name in SONG_SOURCES | {"directory"}:
                self.read_mode(name)
                value = self.public[out]
            elif name=="filter_genre":
                self.filter_genre()
                value = self.public["song_ids"]
            elif name=="sort_songs":
                value = self.order_songs()
            elif name=="take_first":
                numbers = sorted({1} | {int(s) for s in re.findall(r"\b[1-9]\d*\b",self.request)})
                limit = self.choose("quantity",[option(str(k)+" leading song(s)",k) for k in numbers])
                self.criteria["limit"] = limit
                ids = put_ids(self.public["ranked_songs"])[:limit]
                records = {r["song_id"]:r for r in self.public.get("song_records", [])}
                self.set_subset([records.get(sid, self.seed_rows.get(sid, {"song_id":sid})) for sid in ids])
                value = self.public["song_ids"]
            elif name=="titles":
                value = self.apply_data("title_text",rows=self.hydrate({"title"}))
            elif name=="play":
                sid = self.public["song_ids"][0]
                receipt = self.call("spotify","play_music",{"song_id":sid,"access_token":self.token("spotify")},name)
                value = {"song_id":sid,"response":receipt}
            else:
                arguments = {"answer":self.public["answer"]} if name=="submit_answer" else {}
                self.call("supervisor","complete_task",arguments,name)
                value = True
            if value is None:
                raise Uncovered(f"{name} produced no usable value")
            if out=="song_ids":
                value=put_ids(value)
            if card.get("builtin") in ("filter_genre","sort_songs","take_first"):
                self.forget_derived({"song_ids","song_records"})
            self.public[out] = value
            if ("builtin" not in card or name=="titles") and out != "effect_receipt":
                self.derived[out] = set(inputs)
            self.completed_cards.append(name)
            self.trace.add("card_executed",card=name,inputs=list(inputs),output=out)
        except (Uncovered,KeyError,ValueError,IndexError,TypeError) as exc:
            self.failed_card = name
            self.trace.add("card_failed",card=name,reason=str(exc))
            raise
        finally:
            self.active_card = None

    def assemble_available(self):
        while not self.done:
            self.perform_card(self.next_card())

    def read_offers(self):
        offers = {name:card["api"] for name,card in self.registry["cards"].items() if card.get("public_read")
                  and name in self.pool and all(k in self.public for k in card["inputs"])
                  and card["output"] not in self.public}
        for api,source in LOOKUPS.items():
            if (source in self.pool and self.registry["cards"][source].get("builtin") == source
                    and "song_ids" not in self.public):
                offers[source] = api
        return {name:api for name,api in offers.items() if api in self.schemas}

    def delegate_read(self, identifier, arguments):
        offers = self.read_offers()
        names = [n for n,api in offers.items() if n==identifier or api==identifier]
        if len(names)!=1:
            raise Uncovered("Choose one of the named available public reads")
        name = names[0]
        endpoint = offers[name]
        card = self.registry["cards"][name]
        signature = (name,json.dumps(arguments,sort_keys=True))
        if signature in self.returned_reads:
            raise Uncovered("This public read was already returned to the fragment loop")
        self.returned_reads.add(signature)
        if "builtin" in card:
            rows = self.call_endpoint(endpoint,arguments,True)
            record = self.named(rows,name)
            ids = self.members(record,name)
            self.public.update(scope=[name,record["title"]],song_ids=put_ids(ids),
                               collections=[{"kind":name,"id":record[name+"_id"],"title":record["title"],"song_ids":ids}])
            role = "song_ids"
        else:
            self.active_card = name
            try:
                ctx = self.parameter_context(card)
            finally:
                self.active_card = None
            fixed = resolve(card.get("arguments",{}),ctx)
            if any(k in arguments and arguments[k]!=v for k,v in fixed.items()):
                raise Uncovered("The proposed arguments conflict with the declared read scope")
            role = card["output"]
            if role in self.public:
                raise Uncovered("The scoped output already exists")
            self.public[role] = self.call_endpoint(endpoint,{**arguments,**fixed},card.get("paginate",False))
            self.read_scopes[role] = {"api":endpoint,"arguments":{**arguments,**fixed}}
            self.derived[role] = set(card["inputs"])
        self.completed_cards.append(name)
        self.trace.add("fragment_reentry",via="public_read",card=name,api=endpoint,role=role)

    def handoff(self, reason):
        self.trace.add("fragment_handoff",reason=reason,available=list(self.public))
        current = {**self.public, **({"executed_items":self.receipts} if self.receipts else {})}
        relevant = list(dict.fromkeys(self.completed_cards + ([self.failed_card] if self.failed_card else [])))
        usage_role = None
        if is_scoped(self.registry):
            tips = consumer_tips(self.registry, self.public, self.provenance)
            guidance = list(tips.values())[-1:]
            if len(tips) > 1:
                usage_role = "fragment_usage_by_role"
                while usage_role in current:
                    usage_role += "_notes"
                current[usage_role] = tips
            if current:
                self.prime_generated_session(current)
        else:
            if current:
                self.prime_generated_session(current)
            guidance = [self.registry["cards"][name]["handoff_skill"] for name in relevant
                        if self.registry["cards"][name].get("handoff_skill")]
        reads = self.read_offers()
        read_choices = list(reads.values()) if len(reads)==len(set(reads.values())) else reads
        return {"variables":list(current),"completed":list(self.completed_cards),"scope":self.public.get("scope"),
                "unresolved":reason,"available_public_reads":read_choices,
                **({"usage_lookup":usage_role} if usage_role else {}),
                **({"fragment_guidance":guidance} if guidance else {}),
                **({"read_scopes":self.read_scopes} if self.read_scopes else {})}

    def code(self, code, node=None, permit_finish=True):
        if node!="generation_agent" or not code.strip().startswith("yield_read("):
            return FormalExecutor.code(self,code,node,permit_finish)
        try:
            call = ast.parse(code.strip(),mode="eval").body
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!="yield_read" or len(call.args)!=2 or call.keywords:
                raise Uncovered("Use yield_read(endpoint, arguments) as one stand-alone call")
            endpoint, args = map(ast.literal_eval,call.args)
            if not isinstance(endpoint,str) or not isinstance(args,dict):
                raise Uncovered("Expected an endpoint name and public arguments")
            self.delegate_read(endpoint,args)
            self.assemble_available()
        except (Uncovered,KeyError,ValueError,TypeError,IndexError,SyntaxError) as exc:
            refreshed = self.handoff(str(exc))
            if is_scoped(self.registry):
                self.residual_progress.clear()
            self.residual_progress.update(refreshed)
            self.context["history"].append({"returned_read":str(exc)})
        return {"done":self.done,"output":"Public read processed","api_budget_hit":False}

    def execute_task(self, library):
        self.initialize(library)
        try:
            self.assemble_available()
        except (Uncovered,ValueError,TypeError,IndexError,KeyError) as exc:
            self.residual_progress = self.handoff(str(exc))
            self.agent(self.registry["skills"]["residual"]+
                       " To return a supported read to the program, use a standalone Python cell "
                       "yield_read('app__api', {literal public arguments}). The program executes it, attaches "
                       "the actual result, and offers the next fragments. Otherwise finish with the public tools.",
                       self.residual_progress)

    def summarize(self,*args):
        row = super().summarize(*args)
        row["fragments"] = list(self.completed_cards)
        events = self.trace.events
        row["card_menus"] = sum(e["event"]=="card_menu" for e in events)
        row["fragment_decisions"] = sum(e["event"]=="decision" and e.get("node")=="fragment/step" for e in events)
        row["fragment_selections"] = [{"fragment":e["card"],"output":e["output"],"reused":False}
                                      for e in events if e["event"]=="card_executed"]
        row["parameter_choices"] = sum(e["event"]=="decision" and e.get("node")!="fragment/step"
                                       and e["candidates"][e["selected"]]["value"] is not None for e in events)
        path=[]
        for event in events:
            if event["event"]=="model" and is_open_generation(event["operation"], event["request"]):
                if not path or path[-1]!="G":
                    path.append("G")
            elif event["event"]=="card_executed" or event["event"]=="fragment_reentry" and event.get("via")=="public_read":
                path.append(event["card"])
        row["card_path"]=path
        return row
