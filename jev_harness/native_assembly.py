"""Public AppWorld tool execution, binding and local generation."""

from copy import deepcopy
from itertools import product
import json
import re

from .formal_runtime import FormalExecutor, bounded
from .expressions import path_get
from .policy_programs import load_program
from .protocol import PROTOCOL
from .runtime import Uncovered, parse_json
from .retrieval import terms
from .schema_tools import (
    FINISH,
    MANAGED,
    Observation,
    available,
    calculations,
    catalog,
    check_args,
    checklist_plans,
    effect_view,
    expand_binding,
    has_update,
    is_read,
    key,
    leaf_fields,
    literal_plans,
    normalize_action,
    public_calculation,
    public_value,
    ranked_tools,
    record_plans,
    update_fields,
)


DISCOVER = "@discover"
MORE_FRAGMENTS = "@fragments"
ACTION_SKILL = (
    "Choose one step that advances the user's outstanding request. Read the relevant source first "
    "when a target or value is unknown. A matching object alone does not authorize a write. "
    "Check the direction of the action, scope and real receipts before completing."
)
BIND_SKILL = (
    "Choose a binding only if its source, filters and affected records match the instruction. "
    "For a set action choose all and only the requested objects. NONE asks for a local binding."
)
BIND_FORMAT = (
    'Return one JSON object. Put public keyword values in "arguments". The runtime supplies '
    "authentication. Omit both page_index and page_limit to collect all pages. "
    "If either is supplied, read just that public API page, with official defaults for omitted fields. "
    "To apply an operation to observed rows, optionally set "
    '"source":"r1", "where":{"nested.field":"exact value"}, "order_by":"field", '
    '"take":"first" or "last" (omit for all); {"field":"nested.field"} copies a field into one '
    'argument from that same row. To inspect a visible source return {"inspect":{"source":"r1",'
    '"row":0,"field":"nested.field","offset":0}}. A text excerpt is 500 characters. Do not write code. If a prerequisite is missing return '
    '{"search":"short description of the needed read"}.'
)
EDIT_FORMAT = (
    "For a record update, bind the target and the new content or attribute. An identifier alone "
    "does not change the record. To change part of an observed text field, give a source and use "
    '{"replace_text":{"field":"content","old":"exact unique excerpt","new":"replacement"}} '
    "as its argument value. This preserves the rest of the text."
)


def blank_library():
    return {group: {} for group in ("retrieval", "decision", "binding", "fragment", "state", "handoff")}


def preview(value, chars=2000):
    return bounded(value, chars)


def tool_error(text):
    # AppWorld's traceback includes the call expression, which may contain the live token.
    if "Response status code is " in text:
        return "Execution failed. " + text.rsplit("Response status code is ", 1)[1][:340]
    return "Execution failed. " + (text.strip().splitlines() or ["unknown error"])[-1][:240]


def observed_input(obs):
    value = {"ref": obs.ref, "rows": deepcopy(obs.rows), "scope": obs.scope, "complete": obs.complete}
    if obs.arguments is not None:
        value["arguments"] = deepcopy(obs.arguments)
    return value


def run_program(definition, instruction, when, obs, companions=()):
    context = {
        "instruction": instruction,
        "datetime": when,
        "source": obs.source,
        **observed_input(obs),
    }
    context["inputs"] = {o.source: observed_input(o) for o in companions}
    result = load_program(definition["program"])(context)
    return json.loads(json.dumps(result, allow_nan=False))


class NativeTools(FormalExecutor):
    review_nodes = ("native/action", "native/binding", "native/app", "native/fragment_choice")

    def initialize_native(self, library):
        self.lib = {**blank_library(), **library}
        self.tools = catalog(self.context["task"]["functions"])
        self.instruction = self.context["task"]["instruction"]
        self.observed, self.receipts, self.notices = [], [], []
        self.tokens, self.partial = {}, {}
        self.accounts = None
        self.lookup = ""
        self.last_read = {}
        self.state_views = {}
        self.used_transforms = set()
        self.transforms = {}
        self.calculation_refs = {}
        self.binding_sources = {}
        self.last_update = {}
        self.last_write = {}
        self.trace.add("native_protocol", version=PROTOCOL)

    def recorded_tool_io(self, params, response):
        data = {k: v for k, v in response.items() if k not in ("output", "value")}
        val = response.get("value", response.get("output"))
        if params.get("api") in {"login", "show_account_passwords"}:
            data["value"] = {"session_data_received": isinstance(val, (list, dict)) and bool(val)}
        else:
            data["value"] = (
                tool_error(val)
                if isinstance(val, str) and val.startswith("Execution failed.")
                else public_value(val)
            )
        public = public_value(params)
        public.get("arguments", {}).pop("raise_on_failure", None)
        return public, data

    def call(self, app, api, arguments, node=None):
        # AppWorld otherwise returns the same {"message": ...} shape for successful writes and HTTP errors.
        try:
            return super().call(app, api, {**arguments, "raise_on_failure": True}, node)
        except Uncovered as exc:
            if "Execution failed." in str(exc) or "Response status code is " in str(exc):
                raise Uncovered(tool_error(str(exc))) from None
            raise

    def session_token(self, app):
        if app not in self.tokens:
            login = self.tools.get(f"{app}__login", {})
            fields = login.get("parameters", {}).get("properties", {})
            if "username" not in fields or "password" not in fields:
                raise Uncovered(f"No supported public login for {app}")
            field = "phone_number" if "phone" in fields["username"]["description"].lower() else "email"
            if self.accounts is None:
                self.accounts = self.call("supervisor", "show_account_passwords", {}, "session")
            if not isinstance(self.accounts, list):
                raise Uncovered("The credential provider did not return account rows")
            credentials = [r for r in self.accounts if r.get("account_name") == app]
            identity = self.context["task"]["supervisor"].get(field)
            if len(credentials) != 1 or not identity:
                raise Uncovered(f"Cannot bind the public {app} identity")
            result = self.call(
                app, "login", {"username": identity, "password": credentials[0]["password"]}, "session"
            )
            if not isinstance(result, dict) or not result.get("access_token"):
                raise Uncovered(f"The public {app} login did not produce a session")
            self.tokens[app] = result["access_token"]
        return self.tokens[app]

    def facts(self):
        observations = []
        for obs in self.observed[-6:]:
            view = obs.view(self.instruction)
            view["sample"] = [preview(row, 650) for row in view["sample"]]
            if obs.rows and isinstance(obs.rows[0], dict):
                view["fields"] = [field for field, _ in leaf_fields(obs.rows[0])][:35]
            observations.append(view)
        facts = {
            "instruction": self.instruction,
            "datetime": self.context["task"].get("datetime"),
            "observations": observations,
            "tool_receipts": self.receipts[-6:],
            "recent_feedback": self.notices[-2:],
        }
        if self.lookup:
            facts["looking_for"] = self.lookup
        if self.state_views:
            facts["derived_evidence"] = self.state_views
        return facts

    def companion_sets(self, spec):
        sources = spec.get("inputs", [])
        return product(*([o for o in reversed(self.observed) if o.source == source] for source in sources))

    def program(self, group, name, obs, companions=()):
        definition = self.lib[group][name]
        try:
            return run_program(
                definition, self.instruction, self.context["task"].get("datetime"), obs, companions,
            )
        except (
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            ArithmeticError,
            AttributeError,
            re.error,
        ) as exc:
            self.trace.add("program_error", component=group, name=name, error=str(exc)[:240])
            return None

    def ask(self, node, question, options, skill, **extra):
        view = {**self.facts(), **extra}
        try:
            return self.decision(
                question,
                view,
                [{"label": label, "value": v} for label, v in options],
                skill,
                node=node,
                review_step=self.requests + 1 < self.max_calls,
            )
        except Uncovered:
            return None

    def generate_one(self, instruction, data, purpose):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
        response = self._model(
            "generate",
            messages=messages,
            thinking=self.generation_thinking,
            purpose=purpose,
            max_tokens=min(640, self.generation_tokens),
        )
        if response.get("truncated"):
            raise ValueError("Local JSON completion was truncated")
        result = parse_json(response["text"])
        if not isinstance(result, dict):
            raise ValueError("Expected a single structured action")
        return result

    def candidates(self, app=None, query=None):
        return ranked_tools(
            self.tools,
            self.instruction,
            self.observed,
            self.lib["retrieval"],
            app,
            self.lookup if query is None else query,
        )

    def calculation_options(self):
        self.transforms = {}
        self.calculation_refs = {}
        for obs in reversed(self.observed[-4:]):
            if obs.source == "calculation" or not is_read(obs.source):
                continue
            for label, result in calculations(obs, self.instruction):
                if label not in self.used_transforms:
                    self.transforms[label] = result
                    self.calculation_refs[label] = obs.ref
        return [(label, ("calculate", label)) for label in self.transforms]

    def tool_label(self, name):
        tool = self.tools[name]
        desc = tool["description"].split("\n")[0][:160]
        hint = self.lib["decision"].get(name)
        return f"{name}: {desc}" + (f" {hint}" if hint else "")

    def next_action(self):
        ranked = self.candidates()
        action = self.action_page(ranked[:8])
        if action is None:
            action = self.action_page(ranked[8:16])
        if action != DISCOVER:
            return action
        apps = sorted({n.split("__")[0] for n in ranked})
        sections = [(a, a) for a in apps]
        if self.fragments():
            sections.append(("Browse learned operations whose inputs are available", MORE_FRAGMENTS))
        app = self.ask(
            "native/app",
            "Where should we look for the missing operation?",
            sections,
            "Choose by the required source or effect.",
        )
        if app == MORE_FRAGMENTS:
            return self.fragment_pages()
        if app is None:
            return None
        expanded = self.candidates(app=app)
        for i in range(0, len(expanded), 10):
            choice = self.action_page(expanded[i : i + 10], allow_discovery=False)
            if choice is not None:
                return choice
        return None

    def fragment_pages(self):
        options = self.fragments()
        for pos in range(0, len(options), 12):
            choice = self.ask(
                "native/action",
                "Which operation applies to these sources and the current goal?",
                options[pos : pos + 12],
                ACTION_SKILL,
            )
            if choice is not None:
                return choice
        return None

    def local_action(self, endpoint=None):
        view = self.facts()
        if endpoint is not None:
            view["selected_tool"] = self.tools[endpoint]
            if endpoint in self.lib["handoff"]:
                view["guidance"] = self.lib["handoff"][endpoint]
            prompt = "Supply only the missing arguments for the selected public tool. " + BIND_FORMAT
            if update_fields(self.tools[endpoint]):
                prompt += " " + EDIT_FORMAT
                view["editable_fields"] = sorted(update_fields(self.tools[endpoint])[1])
            spec = self.generate_one(prompt, view, "G_binding")
        else:
            # This catalogue view is independent of learned fragment/decision descriptions.
            names = ranked_tools(self.tools, self.instruction, self.observed, query=self.lookup)[:12]
            view["public_tools"] = [self.tools[n] for n in names + [FINISH]]
            prompt = (
                'Take exactly one useful AppWorld action. Set "tool" to a public tool name, or use '
                '{"search":"keywords"} to inspect another part of the public catalogue. '
                'To compute a value from public data you may instead return {"calculate":{"op":"sum",'
                '"source":"r1","field":"amount"}}. Supported ops: count, sum, mean, argmin, argmax, project, '
                "distinct, group and join. You may include where (exact field/value pairs) or separator for join. "
                "Call supervisor__complete_task when all requested work is complete. " + BIND_FORMAT
            )
            if any(update_fields(self.tools[n]) for n in names):
                prompt += " " + EDIT_FORMAT
            spec = self.generate_one(prompt, view, "G_tool")
        if set(spec) == {"inspect"}:
            request = spec["inspect"]
            if not isinstance(request, dict):
                raise ValueError("Inspect requires an observed source")
            obs = next((o for o in self.observed if o.ref == request.get("source")), None)
            row, offset = request.get("row", 0), request.get("offset", 0)
            if (
                obs is None
                or type(row) is not int
                or not 0 <= row < len(obs.rows)
                or type(offset) is not int
                or offset < 0
            ):
                raise ValueError("Invalid source, row or text offset")
            value = path_get(obs.rows[row], request.get("field", ""))
            if value is None:
                raise ValueError("Unknown observation field")
            self.observe(
                "inspection",
                value[offset : offset + 500] if isinstance(value, str) else value,
                f"excerpt of {obs.ref}: " + key(request),
                complete=False,
                sources={obs.ref: {"fields": {"excerpt": request.get("field", "")}}},
            )
            return None
        if endpoint is None and set(spec) == {"calculate"}:
            operation = spec["calculate"]
            result = public_calculation(operation, self.observed)
            self.observe(
                "calculation",
                result,
                "generated operation " + key(operation),
                sources={operation["source"]: {"calculation": operation["op"]}},
            )
            return None
        if "search" in spec:
            if not isinstance(spec["search"], str) or not spec["search"].strip():
                raise ValueError("A discovery query needs text")
            self.lookup = spec["search"][:240]
            self.trace.add("native_discovery", query=self.lookup)
            return None
        spec = normalize_action(spec, endpoint, self.tools)
        selected = spec.get("tool", endpoint)
        if (
            selected not in self.tools
            or (endpoint is not None and selected != endpoint)
            or (selected != FINISH and not available(selected))
        ):
            raise ValueError("Choose one available public operation")
        unknown = set(spec) - {"tool", "arguments", "source", "where", "order_by", "take"}
        if unknown:
            raise ValueError("Unexpected action fields: " + ", ".join(sorted(unknown)))
        if ({"where", "order_by", "take"} & spec.keys() and "source" not in spec) or spec.get("take") not in (
            None,
            "first",
            "last",
        ):
            raise ValueError("A row selector requires a source and a supported take value")
        batch = expand_binding(spec, self.observed)
        self.trace.add("native_generated_binding", tool=selected, spec=spec)
        sources = {}
        if "source" in spec:
            fields = {}
            edits = []
            for arg, expr in spec.get("arguments", {}).items():
                if isinstance(expr, dict) and set(expr) == {"field"}:
                    fields[arg] = expr["field"]
                if isinstance(expr, dict) and set(expr) == {"replace_text"}:
                    fields[arg] = expr["replace_text"]["field"]
                    edits.append(arg)
            sources[spec["source"]] = {"fields": fields} if fields else {"scope_only": True}
            if edits:
                sources[spec["source"]]["edited"] = edits
        return (
            selected,
            batch,
            "generated binding" + (f" from {spec['source']}" if "source" in spec else ""),
            sources,
        )

    def argument_plans(self, endpoint):
        tool = self.tools[endpoint]
        plans = []
        self.binding_sources = {}
        rule = self.lib["binding"].get(endpoint)
        if rule and "program" in rule:
            for obs in reversed(self.observed):
                if obs.source not in rule["sources"]:
                    continue
                for others in self.companion_sets(rule):
                    batch = self.program("binding", endpoint, obs, others)
                    if isinstance(batch, list) and batch:
                        try:
                            for args in batch:
                                check_args(tool, args)
                        except ValueError as exc:
                            self.trace.add(
                                "program_error", component="binding", name=endpoint, error=str(exc)
                            )
                        else:
                            refs = ", ".join(o.ref for o in (obs, *others))
                            plans.append(
                                (
                                    f"{rule['label']} from {refs}",
                                    batch,
                                    {o.ref: {"program_input": endpoint} for o in (obs, *others)},
                                )
                            )
        plans.extend(checklist_plans(tool, self.instruction, self.observed))
        plans.extend(record_plans(tool, self.instruction, self.observed))
        plans.extend(
            (
                f"Use {key(args)}" if args else "Use official defaults, no additional filters or arguments",
                [args],
                {},
            )
            for args in literal_plans(tool, self.instruction)
        )
        if endpoint == FINISH:
            for obs in reversed(self.observed):
                if not isinstance(obs.data, (list, dict)):
                    plans.insert(
                        0,
                        (
                            f"Return result {obs.ref} ({obs.source})",
                            [{"answer": obs.data}],
                            {obs.ref: {"fields": {"answer": ""}}},
                        ),
                    )
                elif isinstance(obs.data, dict):
                    for field, value in leaf_fields(obs.data):
                        if type(value) in (str, int, float):
                            plans.append(
                                (
                                    f"Return {field} from {obs.ref} ({obs.source})",
                                    [{"answer": value}],
                                    {obs.ref: {"fields": {"answer": field}}},
                                )
                            )
        seen = set()
        choices = []
        for label, batch, sources in plans:
            if not batch or not all(
                has_update(tool, args) and self.current_text_edit(tool, args, sources) for args in batch
            ):
                continue
            signature = key(batch)
            if signature not in seen:
                seen.add(signature)
                self.binding_sources[label] = sources
                choices.append((label, batch))
        return choices

    def binding(self, endpoint):
        plans = self.argument_plans(endpoint)
        schema = self.tools[endpoint]["parameters"]
        relevant = set(schema["properties"]) - MANAGED
        if (not relevant and plans) or (len(plans) == 1 and plans[0][1] == [{}] and is_read(endpoint)):
            return endpoint, plans[0][1], plans[0][0]
        if (
            endpoint == FINISH
            and self.receipts
            and not re.search(
                r"\b(what|how many|how much|what is|tell me|name the|give me the number)\b",
                self.instruction,
                re.I,
            )
        ):
            # The action ballot already requested completion. The official default is the action receipt.
            return endpoint, [{}], "finish requested state changes"
        for i in range(0, len(plans), 12):
            page = plans[i : i + 12]
            opts = []
            for label, batch in page:
                display = label if label.startswith("Use ") else f"{label} | e.g. {preview(batch[0], 220)}"
                if update_fields(self.tools[endpoint]):
                    display = f"{label} | effect on first of {len(batch)}: " + key(
                        effect_view(self.tools[endpoint], batch[0], self.observed)
                    )
                opts.append((display, {"tool": endpoint, "arguments": batch, "scope": label}))
            selected = self.ask(
                "native/binding",
                f"Which arguments and scope for {endpoint} match the user's request?",
                opts,
                BIND_SKILL,
                parameters={
                    k: v.get("description", "")[:160]
                    for k, v in schema["properties"].items()
                    if k not in MANAGED
                },
            )
            if selected is not None:
                return (
                    endpoint,
                    selected["arguments"],
                    selected["scope"],
                    self.binding_sources[selected["scope"]],
                )
        if update_fields(self.tools[endpoint]):
            self.trace.add(
                "native_update_fallback",
                tool=endpoint,
                applicable_bindings=len(plans),
                reason="no selected binding supplies a target and change",
            )
        return self.local_action(endpoint)

    def observe(self, source, data, scope, *, complete=True, sources=None, arguments=None):
        obs = Observation(f"r{len(self.observed) + 1}", source, public_value(data), scope, complete,
                          public_value(arguments), tuple(sources or ()))
        self.observed.append(obs)
        for name, spec in self.lib["state"].items():
            if obs.source in spec["sources"]:
                value = self.program("state", name, obs)
                if value is not None:
                    self.state_views[name] = {
                        "from": obs.ref,
                        "meaning": spec["label"],
                        "value": preview(value, 600),
                    }
        self.trace.add(
            "native_observation",
            ref=obs.ref,
            source=source,
            scope=scope,
            count=len(obs.rows),
            value=obs.data,
            complete=obs.complete,
            sources=sources or {},
            **({"arguments": obs.arguments} if obs.arguments is not None else {}),
        )

    def current_text_edit(self, tool, args, sources):
        fields = update_fields(tool)
        if fields and sources:
            target = key({k: args.get(k) for k in sorted(fields[0])})
            for ref, detail in sources.items():
                for field in detail.get("edited", []):
                    if int(ref[1:]) <= self.last_update.get((tool["name"], target, field), -1):
                        return False
        return True

    def perform(self, endpoint, batch, scope, sources=None):
        tool = self.tools[endpoint]
        for args in batch:
            check_args(tool, args)
            if not has_update(tool, args):
                raise ValueError(
                    f"{endpoint} needs at least one editable value for the requested change: "
                    + ", ".join(sorted(update_fields(tool)[1]))
                )
            if not self.current_text_edit(tool, args, sources):
                raise ValueError(
                    "A newer update changed this field. Read the object again before editing its text."
                )
        if not batch or (endpoint == FINISH and len(batch) != 1):
            raise ValueError("An action needs a nonempty binding and at most one task completion")
        app, api = endpoint.split("__", 1)
        props = tool["parameters"]["properties"]
        read = is_read(endpoint)
        packet = (endpoint, key(batch))
        if read and all((endpoint, key(args)) in self.last_read for args in batch):
            raise Uncovered(
                f"{endpoint} with these arguments was already read. Use the existing observation or select a different scope."
            )
        self.trace.add(
            "native_operation",
            tool=endpoint,
            scope=scope,
            count=len(batch),
            arguments=public_value(batch),
            sources=sources or {},
        )
        earlier = self.partial.get(packet, [])
        outputs = list(earlier)
        for i, args in enumerate(batch):
            if not read and i < len(earlier):
                continue
            if read and (endpoint, key(args)) in self.last_read:
                outputs.append(self.last_read[(endpoint, key(args))])
                continue
            bound = dict(args)
            if "access_token" in props:
                bound["access_token"] = self.session_token(app)
            if {"page_index", "page_limit"} <= props.keys():
                result = []
                single_page = "page_index" in bound or "page_limit" in bound
                size = (
                    bound.get("page_limit", props["page_limit"].get("default", 5))
                    if single_page
                    else min(20, props["page_limit"].get("maximum", 20))
                )
                page = bound.get("page_index", props["page_index"].get("default", 0))
                while True:
                    chunk = self.call(app, api, {**bound, "page_index": page, "page_limit": size}, endpoint)
                    if not isinstance(chunk, list):
                        raise Uncovered(
                            f"{endpoint} did not return a page of records: {preview(public_value(chunk), 240)}"
                        )
                    result.extend(chunk)
                    if len(chunk) < size or single_page:
                        break
                    page += 1
            else:
                result = self.call(app, api, bound, endpoint)
            outputs.append(result)
            if read:
                self.last_read[(endpoint, key(args))] = result
            else:
                effect = effect_view(tool, args, self.observed)
                self.trace.add(
                    "native_effect",
                    tool=endpoint,
                    batch_index=i,
                    count=len(batch),
                    effect=effect,
                    sources=sources or {},
                    response=preview(public_value(result), 300),
                )
                if endpoint != FINISH:
                    self.partial[packet] = list(outputs)
                    self.last_read.clear()
                    self.last_write[app] = len(self.observed)
                    receipt = {
                        "tool": endpoint,
                        "arguments": public_value(args),
                        "response": preview(public_value(result), 300),
                    }
                    if update_fields(tool):
                        receipt = {
                            "tool": endpoint,
                            "requested_update": effect,
                            "response": receipt["response"],
                        }
                        targets, editable = update_fields(tool)
                        target = key({k: args.get(k) for k in sorted(targets)})
                        for field in editable & args.keys():
                            self.last_update[(endpoint, target, field)] = len(self.observed)
                    self.receipts.append(receipt)
        self.partial.pop(packet, None)
        if endpoint != FINISH:
            value = outputs[0] if len(outputs) == 1 else outputs
            # Single-page requests retain their explicit scope. The caller can ask for the next page.
            complete = all(
                not ({"page_index", "page_limit"} & a.keys())
                or (
                    a.get("page_index", props.get("page_index", {}).get("default", 0)) == 0
                    and isinstance(v, list)
                    and len(v) < a.get("page_limit", props.get("page_limit", {}).get("default", 5))
                )
                for a, v in zip(batch, outputs)
            )
            self.observe(
                endpoint,
                value,
                scope + " | " + str(preview(batch if len(batch) < 3 else batch[:2], 350)),
                complete=complete,
                sources=sources,
                arguments=batch if read else None,
            )
        self.lookup = ""

    def summarize(self, *args):
        result = super().summarize(*args)
        events = self.trace.events
        result.update(
            protocol=PROTOCOL,
            fragments=sorted({e["source"] for e in events if e["event"] == "native_observation"
                              and e["source"] in getattr(self, "lib", {}).get("fragment", {})}),
            open_calls=sum(e["event"] == "model" and e["operation"] == "generate" for e in events),
            native_ballots=sum(e["event"] == "decision" for e in events),
            open_output_tokens=sum(
                e["response"].get("output_tokens", 0)
                for e in events
                if e["event"] == "model" and e["operation"] == "generate"
            ),
            decision_output_tokens=sum(
                e["response"].get("output_tokens", 0)
                for e in events
                if e["event"] == "model" and e["operation"] == "decide"
            ),
        )
        return result


class NativeGeneration(NativeTools):
    """All-generation ReAct on the same official tool interface."""

    def execute_task(self, library):
        self.initialize_native(library)
        while not self.done:
            try:
                operation = self.local_action()
                if operation:
                    self.perform(*operation)
            except (ValueError, TypeError, Uncovered) as exc:
                self.notices.append(str(exc)[:350])
                self.trace.add("native_feedback", reason=self.notices[-1])
