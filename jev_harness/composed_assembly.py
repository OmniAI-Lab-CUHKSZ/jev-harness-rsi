"""Compose scoped public reads, reusable transformations and official writes."""

import ast
from pathlib import Path
import json
import re

from .composition import (
    PROGRAM_ERRORS, acquire_inputs, already_covered, card_answers, card_plan_key, card_writes,
    contact_plan_available, output_fields, project_batch, request_scope, resolve_card,
    source_plans, validate_output,
)
from .artifacts import cleaned_text
from .native_assembly import ACTION_SKILL, FINISH, NativeTools, blank_library, observed_input, preview
from .policy_programs import load_program
from .record_composition import contact_plans, select_record_subset
from .retrieval import terms
from .runtime import Uncovered
from .schema_tools import available, check_args, is_read, key, public_value


def initial_library():
    path = Path(__file__).with_name("resources") / "composed_h0.json"
    return {**blank_library(), **json.loads(path.read_text(encoding="utf-8"))}


def goal_words(text):
    synonyms = {"track": "song", "played": "play", "listened": "play", "liked": "like",
                "workmate": "coworker", "colleague": "coworker", "housemate": "roommate"}
    return {synonyms.get(word, word) for word in terms(text.replace("co-worker", "coworker"))}


def relevance(spec, instruction):
    goal = goal_words(instruction)
    if cues := spec.get("goal_terms"):
        return max((len(words) for cue in cues if (words := goal_words(cue)) <= goal), default=0)
    return len(goal & goal_words(spec["label"]))


class ResumeAssembly(Exception):
    """A fresh public read lets the outer decision loop continue."""


class ComposedAssembly(NativeTools):
    def initialize_native(self, library):
        super().initialize_native(library)
        self.trace.add("native_composition")
        self.chosen_effects = set()
        self.plan_bindings = {}
        self.card_offers = {}
        self.card_results = {}
        self.used_card_plans = set()
        self.completed_card_plans = []
        self.goal = re.sub(r'["“][^"”]*["”]', "", self.instruction).casefold()
        self.answer_question = bool(re.search(
            r"\b(what|which|how many|how much|tell me|give me the (?:title|number|name))\b", self.goal
        ))
        self.set_request = bool(re.search(r"\b(all|each|every)\b.*\b(from|by|in|within)\b", self.goal))
        self.required = self.required_effects()

    def policy_context(self, observations=()):
        return {
            "instruction": self.instruction,
            "goal": self.goal,
            "datetime": self.context["task"].get("datetime"),
            "inputs": {o.source: observed_input(o) for o in observations},
        }

    def evaluate_policy(self, body, group, name, observations=()):
        try:
            return load_program(body)(self.policy_context(observations))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError, re.error) as exc:
            self.trace.add("program_error", component=group, name=name, error=str(exc)[:240])
            return None

    def required_effects(self):
        required = set()
        for name, spec in self.lib["state"].items():
            if not spec.get("intent"):
                continue
            endpoints = self.evaluate_policy(spec["program"], "state", name)
            if isinstance(endpoints, list):
                required.update(e for e in endpoints if isinstance(e, str) and e in self.tools
                                and available(e) and not is_read(e))
        return required

    def outstanding(self):
        return self.required - self.chosen_effects

    def facts(self):
        return {**super().facts(), "remaining_effects": sorted(self.outstanding()),
                "verified_effects": sorted(self.chosen_effects)}

    def perform(self, endpoint, batch, scope, sources=None):
        projected = "fields" in self.lib["binding"].get(endpoint, {}) or any(
            endpoint in self.card_results.get(ref, {}).get("writes", {}) for ref in sources or {}
        )
        if not batch and endpoint in self.required and projected:
            self.trace.add("native_empty_scope", tool=endpoint, sources=sources or {})
        else:
            super().perform(endpoint, batch, scope, sources)
        if endpoint in self.required:
            self.chosen_effects.add(endpoint)
            self.trace.add("native_goal_progress", completed=endpoint, remaining=sorted(self.outstanding()))

    def read_ready(self, source, batch):
        return next((o for o in reversed(self.observed) if o.source == source and o.complete
                     and o.arguments == batch), None)

    def card_plans(self):
        library = self.lib["fragment"]
        goal = goal_words(self.goal)
        scores = {
            name: max(relevance(card, self.goal), len(goal & goal_words(self.lib["retrieval"].get(name, ""))))
            for name, card in library.items()
        }
        observed = {o.source for o in self.observed if o.complete}
        # Propagate relevance through required producer ports. An alternative collection keeps its own scope.
        for _ in library:
            for name, card in library.items():
                for port in card["composition"]["inputs"].values():
                    sources = port.get("sources", [])
                    if len(sources) == 1:
                        source = sources[0]["source"]
                        if source in scores and source not in observed:
                            scores[source] = max(scores[source], scores[name] - 1)

        self.card_offers.clear()
        ranked = []
        scoped_cards = []
        existing = []
        for name, card in library.items():
            inputs = card["composition"]["inputs"]
            if any("role" in port for port in inputs.values()):
                scoped_cards.append((name, card))
                continue
            yields = output_fields(card)
            for plan in source_plans(self, inputs):
                if (name, card_plan_key(plan)) in self.used_card_plans:
                    continue
                ports = tuple(r.get("ref", r["source"]) for r in plan["reads"].values())
                label = f"{card['label']} | scope: {preview(plan['scope'], 180)}"
                if yields:
                    label += " | yields " + ", ".join(yields)
                action = ("fragment", name, *ports)
                if action in self.card_offers:
                    # Different read arguments may use the same public endpoint.
                    action = (*action, len(self.card_offers))
                ready = sum("ref" in r for r in plan["reads"].values())
                self.card_offers[action] = [plan]
                ranked.append((scores[name], ready, self.card_label(name, label), action))
                if scores[name] > 0:
                    existing.append((name, plan))
        for name, card in scoped_cards:
            candidates = [
                plan for plan in contact_plans(self)
                if (name, card_plan_key(plan)) not in self.used_card_plans and contact_plan_available(self, plan)
            ]
            plans = [plan for plan in candidates if not already_covered(self, name, card, plan, existing)]
            if candidates and not plans:
                self.trace.add("native_card_reuse_existing", name=name)
            if not plans:
                continue
            slots = dict.fromkeys(slot for plan in plans for slot in plan["choices"])
            scopes = [" / ".join(dict.fromkeys(p["choices"][slot] for p in plans if slot in p["choices"]))
                      for slot in slots]
            effects = set().union(*(p["writes"].keys() for p in plans)) - self.chosen_effects
            uses = [endpoint.split("__", 1)[1].replace("_", " ") for endpoint in sorted(effects)]
            calculations = plans[0]["calculations"]
            if "sum" in calculations:
                uses.append(f"sum {calculations['sum']}")
            if "count" in calculations:
                uses.append("count matching records")
            label = f"{card['label']} | current scope: {', '.join(scopes)}"
            if uses:
                label += f" | ready for: {', '.join(uses)}"
            action = ("fragment", name)
            self.card_offers[action] = plans
            ranked.append((max(scores[name], 1), 0, self.card_label(name, label), action))
        return sorted(ranked, key=lambda row: (-row[0], -row[1]))

    def card_label(self, name, label):
        cue = self.lib["decision"].get(name)
        return f"{label} | {cue}" if cue else label

    def fragments(self):
        # Browsing leaves a path to reusable cards with weaker lexical cues.
        actions = [(label, action) for _, _, label, action in self.card_plans()]
        return self.preview_action_choices(actions)

    def decline_card(self, action, reason):
        self.used_card_plans.update((action[1], card_plan_key(p)) for p in self.card_offers.get(action, ()))
        self.trace.add("native_card_exit", name=action[1], reason=reason)

    def execute_card(self, action):
        _, name, *_ = action
        card = self.lib["fragment"][name]
        plans = self.card_offers[action]
        plan = request_scope(self, plans, name)
        self.used_card_plans.add((name, card_plan_key(plan)))
        scoped = "actors" in plan
        loaded = acquire_inputs(self, name, plan, card["label"] if scoped else f"Read for {name}")
        output = card["composition"]["output"]
        if scoped:
            result = select_record_subset(card, plan, loaded, next(iter(plan["fields"])))
            fields = {**output_fields(card), **plan["fields"]}
            validate_output({"fields": fields}, result, loaded)
        else:
            result = resolve_card(self, name, card, loaded)
            if result is None:
                raise Uncovered(f"{name} could not establish its result")
            validate_output(output, result, loaded)
        sources = {obs.ref: {"program_input": name} for obs in loaded.values()}
        scope = plan["scope"] if scoped else ", ".join(f"{obs.ref} {obs.source}" for obs in loaded.values())
        self.observe(name, result, scope, sources=sources)
        ref = self.observed[-1].ref
        self.card_results[ref] = {k: plan[k] for k in ("writes", "calculations") if k in plan}
        self.required.update(plan.get("requested", ()))
        self.completed_card_plans.append((name, {"reads": {
            port: {"source": obs.source, "ref": obs.ref} for port, obs in loaded.items()
        }}))
        self.trace.add("native_card", name=name, inputs=list(sources), output=ref)

    def compiled_bindings(self, endpoint):
        rule = self.lib["binding"].get(endpoint, {})
        resolved = card_writes(self, endpoint)
        if not rule.get("covers"):
            return resolved
        tool = self.tools[endpoint]
        plans = {key(sorted(batch, key=key)): (tool_name, batch, label, sources)
                 for tool_name, batch, label, sources in resolved}
        for obs in [o for o in reversed(self.observed) if o.complete and o.source in rule["sources"]] or [None]:
            if obs is None and rule["sources"]:
                return resolved
            if "fields" in rule:
                try:
                    batch = project_batch(tool, obs, rule["fields"])
                except ValueError:
                    continue
                sources = {obs.ref: {"fields": rule["fields"]}}
                plans.setdefault(key(sorted(batch, key=key)), (endpoint, batch, rule["label"], sources))
                continue
            for companions in self.companion_sets(rule):
                if any(not o.complete for o in companions):
                    continue
                if obs is None:
                    batch = self.evaluate_policy(rule["program"], "binding", endpoint, companions)
                    sources = {o.ref: {"program_input": endpoint} for o in companions}
                else:
                    batch = self.program("binding", endpoint, obs, companions)
                    sources = {o.ref: {"program_input": endpoint} for o in (obs, *companions)}
                if isinstance(batch, list) and batch:
                    try:
                        for args in batch:
                            check_args(tool, args)
                    except ValueError:
                        continue
                    plans.setdefault(key(batch), (endpoint, batch, rule["label"], sources))
        return list(plans.values())

    def compiled_binding(self, endpoint):
        plans = self.compiled_bindings(endpoint)
        return plans[0] if len(plans) == 1 else None

    def answer_bindings(self):
        if not self.answer_question:
            return []
        fields = {
            "title": ("title",), "name": ("name", "title"), "amount": ("amount", "total"),
            "number": ("number", "count", "total"),
        }
        wanted = {"answer"}
        for cue, choices in fields.items():
            if cue in terms(self.goal):
                wanted.update(choices)
        result = card_answers(self)
        for obs in reversed(self.observed):
            if not obs.complete or obs.source not in {*self.lib["fragment"], "calculation"}:
                continue
            if output_fields(self.lib["fragment"].get(obs.source, {})) and not relevance(
                self.lib["fragment"][obs.source], self.goal
            ):
                continue
            values = obs.data.items() if isinstance(obs.data, dict) else (("answer", obs.data),)
            for field, value in values:
                if field in wanted and type(value) in (str, int, float):
                    result.append((f"Verified {field} from {obs.ref} ({obs.source}): {preview(value, 140)}",
                                   {"answer": value}, {obs.ref: {"fields": {"answer": field}}}))
        return result

    def ready_to_finish(self):
        if self.outstanding():
            return False
        if self.answer_question:
            return bool(self.answer_bindings())
        return bool(self.receipts or self.chosen_effects)

    def preview_action_choices(self, actions):
        compositions = {
            value: self.card_offers[value]
            for _, value in actions
            if value in self.card_offers and "actors" in self.card_offers[value][0]
        }
        if not compositions:
            return actions
        related_reads = set()
        previews = {}
        for value, plans in compositions.items():
            sources = {read["source"] for plan in plans for read in plan["reads"].values()}
            related_reads.update(sources)
            apps = " and ".join(sorted({source.split("__", 1)[0] for source in sources}))
            binding = "Read arguments already bound." if len(plans) == 1 else "First choose among the listed scopes."
            previews[value] = (f"Acquires the scoped {apps} inputs itself and returns matched records. "
                               f"{binding} Any write is a separate choice.")
        for _, value in actions:
            if isinstance(value, str) and value in related_reads:
                if value in self.plan_bindings:
                    previews[value] = "Returns this source with bound arguments. The requested follow-up remains."
                else:
                    previews[value] = ("Returns this source alone. Contact matching still remains. "
                                       "Scope may need another choice or generated arguments.")
        return [(f"{label} | Outcome: {previews[value]}" if value in previews else label, value)
                for label, value in actions]

    def action_page(self, names, *, allow_discovery=True, include_cards=True):
        outstanding = self.outstanding()
        answers = self.answer_bindings()
        plans = self.card_plans() if include_cards else []
        cards = [(label, action) for score, _, label, action in plans if score > 0][:3] if not answers else []
        actions = [(label, action) for label, action in cards if "actors" not in self.card_offers[action][0]]
        scoped = [(label, action) for label, action in cards if "actors" in self.card_offers[action][0]]
        self.plan_bindings = {}
        if self.ready_to_finish():
            prompt = answers[0][0] if len(answers) == 1 else "all requested work has been done"
            if len(answers) > 1 and any(self.card_results.values()):
                prompt = "a calculation from the verified records"
            actions.insert(0, (f"Complete the task with {prompt}", FINISH))
        for name in sorted(outstanding):
            projected = "fields" in self.lib["binding"].get(name, {})
            operations = self.compiled_bindings(name)
            card_bound = any(self.card_results.get(ref, {}).get("writes")
                             for _, _, _, sources in operations for ref in sources)
            if operations and (projected or card_bound or len(operations) == 1):
                self.plan_bindings[name] = operations
                if len(operations) == 1:
                    _, batch, label, _ = operations[0]
                    scope = f"{label}, {len(batch)} targets"
                    if batch:
                        scope += f"; e.g. {preview(batch[0], 120)}"
                else:
                    scope = f"Choose among {len(operations)} observed object sets"
                actions.append((f"{self.tool_label(name)} | {scope}", name))
        meaningful = terms(self.goal)
        for name in names:
            plan = self.compiled_binding(name) if is_read(name) else None
            if plan and self.read_ready(name, plan[1]):
                continue
            if name in self.plan_bindings or name in self.chosen_effects:
                continue
            if name in outstanding and (self.set_request or "fields" in self.lib["binding"].get(name, {})):
                # The all-from scope needs a constructed group or an explicit recovered operation.
                continue
            if not is_read(name) and self.required and name not in outstanding:
                continue
            title = self.tool_label(name)
            if plan:
                self.plan_bindings[name] = [plan]
                title += f" | grounded arguments {preview(plan[1][0], 120)}"
            elif not (meaningful & terms(title)) and actions:
                continue
            actions.append((title, name))
        if len(cards) == len(scoped) and not answers and (not outstanding or not self.set_request):
            actions = self.calculation_options()[:2] + actions
        actions[:0] = scoped[:max(0, (14 if allow_discovery else 15) - len(actions))]
        cards = [(label, value) for label, value in actions if value in self.card_offers and include_cards]
        self.trace.add("native_fragment_retrieval", eligible=[p[3] for p in plans], page=[a for _, a in cards])
        if allow_discovery:
            actions = actions[:14] + [("Look for another relevant public operation", "@discover")]
        else:
            actions = actions[:15]
        action = self.ask(
            "native/action", "What step advances the user's remaining goal?", self.preview_action_choices(actions),
            ACTION_SKILL + " " + self.lib["decision"].get("action", ""),
        )
        if action is None and cards:
            for _, offered in cards:
                self.decline_card(offered, "No applicable card on this action page")
            return self.action_page(names, allow_discovery=allow_discovery, include_cards=False)
        return action

    def binding(self, endpoint):
        if endpoint in self.plan_bindings:
            plans = self.plan_bindings[endpoint]
            if len(plans) == 1:
                return plans[0]
            for start in range(0, len(plans), 12):
                options = [
                    (f"{label} from {', '.join(sources)}, {len(batch)} targets: {preview(batch[:3], 200)}",
                     {"tool": endpoint, "arguments": batch, "scope": label, "sources": sources})
                    for _, batch, label, sources in plans[start : start + 12]
                ]
                selected = self.ask("native/binding", f"Which observed set applies to {endpoint}?", options,
                                    "Choose the complete set that matches the user's scope.")
                if selected is not None:
                    return endpoint, selected["arguments"], selected["scope"], selected["sources"]
            raise Uncovered(f"No supported object set was selected for {endpoint}")
        if (
            endpoint == FINISH and self.chosen_effects and not self.outstanding()
            and not self.answer_question and not self.receipts
        ):
            return FINISH, [{}], "finish verified empty effect scopes"
        if endpoint == FINISH and self.answer_question:
            answers = self.answer_bindings()
            if len(answers) == 1:
                label, args, sources = answers[0]
            else:
                selected = self.ask("native/binding", "Which verified value answers the user?",
                                    [(label, (label, args, sources)) for label, args, sources in answers[:12]],
                                    "Use an answer that covers the requested scope.")
                if selected is None:
                    return self.local_action(endpoint)
                label, args, sources = selected
            return endpoint, [args], label, sources
        return super().binding(endpoint)

    def recorded_tool_io(self, params, response):
        if "code" not in params:
            return super().recorded_tool_io(params, response)
        view = public_value(response)
        if isinstance(view.get("output"), str):
            view["output"] = cleaned_text(view["output"])
        return {**params, "code": cleaned_text(params["code"])}, view

    def code(self, code, node=None, permit_finish=True):
        if not code.strip().startswith("yield_read("):
            result = super().code(code, node, permit_finish)
            # Raw Python batches expose request provenance, without per-action success.
            # Keep control in that session after any possible effect.
            self.residual_dirty |= any(a["method"].upper() not in {"GET", "HEAD"}
                                       for a in result.get("actions", []))
            return result
        try:
            if self.residual_dirty:
                raise ValueError("Continue in this Python session after state changes so accepted work is preserved.")
            call = ast.parse(code.strip(), mode="eval").body
            if not isinstance(call, ast.Call) or len(call.args) != 2 or call.keywords:
                raise ValueError("Use yield_read(endpoint, arguments) as one stand-alone operation")
            endpoint, args = map(ast.literal_eval, call.args)
            if not isinstance(endpoint, str) or endpoint not in self.tools or not is_read(endpoint) or not available(endpoint):
                raise ValueError("Pass a public read endpoint to yield_read")
            if self.read_ready(endpoint, [args]):
                raise ValueError("That read is already in compiled['observations']. Use the transferred result.")
            self.perform(endpoint, [args], "Public read delegated by residual generation")
        except (SyntaxError, TypeError, ValueError, Uncovered) as exc:
            self.context["history"].append({"read_feedback": str(exc)})
            return {"output": str(exc)}
        self.trace.add("native_residual_reentry", ref=self.observed[-1].ref, source=endpoint)
        raise ResumeAssembly

    def residual(self, reason, endpoint=None):
        self.residual_dirty = False
        self.trace.add("native_residual", reason=reason, endpoint=endpoint, remaining=sorted(self.outstanding()))
        observations = {o.ref: o.data for o in self.observed}
        self.prime_generated_session({"observations": observations, "receipts": self.receipts})
        progress = {
            "variables": ["observations", "receipts"],
            "observations": [{"ref": o.ref, "tool": o.source, "scope": o.scope, "count": len(o.rows)}
                             for o in self.observed[-8:]],
            "remaining_official_operations": sorted(self.outstanding()),
            "unresolved": reason,
        }
        active_cards = [o.source for o in reversed(self.observed) if o.ref in self.card_results]
        advice = [self.lib["handoff"][name] for name in
                  list(dict.fromkeys([endpoint, *sorted(self.outstanding()), *active_cards, "residual"]))
                  if name in self.lib["handoff"]][:2]
        try:
            self.agent(
                " Use compiled['observations'][ref] for the recorded result with that reference. "
                "Check the exact source and scope before using a record. compiled['receipts'] lists successful writes. "
                "For a missing public read that could enable a decision, you may return the one stand-alone operation "
                "yield_read('app__tool', {literal_arguments}). The outer executor performs the read. "
                + " ".join(advice),
                progress,
            )
        except ResumeAssembly:
            pass

    def execute_task(self, library):
        self.initialize_native(library)
        while not self.done:
            action = self.next_action()
            if action is None:
                self.residual("No relevant next operation was selected")
                continue
            try:
                if isinstance(action, tuple):
                    if action[0] == "calculate":
                        label = action[1]
                        self.used_transforms.add(label)
                        self.observe("calculation", self.transforms[label], label,
                                     sources={self.calculation_refs[label]: {"calculation": label}})
                    else:
                        self.execute_card(action)
                else:
                    operation = self.binding(action)
                    if operation:
                        self.perform(*operation)
            except (*PROGRAM_ERRORS, Uncovered) as exc:
                reason = str(exc)[:350]
                if isinstance(action, tuple) and action[0] == "fragment":
                    self.decline_card(action, reason)
                    continue
                self.notices.append(reason)
                self.trace.add("native_feedback", reason=reason)
                if isinstance(action, str):
                    self.residual(reason, action)
