"""Compose scoped public reads, reusable transformations and official writes."""

import ast
from pathlib import Path
import json
import re

from .dynamic_cards import already_covered, dynamic_answers, dynamic_label, dynamic_plans, dynamic_writes, execute_dynamic
from .expressions import path_get
from .formal_optimizer import cleaned_text
from .native_assembly import ACTION_SKILL, FINISH, NativeAssembly, blank_library, observed_input, preview
from .policy_programs import load_program
from .record_composition import record_composition
from .retrieval import terms
from .runtime import Uncovered
from .schema_tools import Observation, accepts, available, check_args, has_update, is_read, key, public_value


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


def project_batch(tool, observation, fields):
    """Bind the whole read or write scope from a complete, selected observation."""
    if not observation.complete:
        raise ValueError("A complete object set is required")
    batch = {}
    for row in observation.rows:
        args = {parameter: path_get(row, field) for parameter, field in fields.items()}
        if any(value is None for value in args.values()):
            raise ValueError("An object is missing a field required by the tool")
        check_args(tool, args)
        if not has_update(tool, args):
            raise ValueError("A projected update must include the changed value")
        batch.setdefault(key(args), args)
    return list(batch.values())


class ResumeAssembly(Exception):
    """A fresh public read lets the outer decision loop continue."""


class ComposedAssembly(NativeAssembly):
    def initialize_native(self, library):
        super().initialize_native(library)
        self.trace.add("native_composition")
        self.chosen_effects = set()
        self.plan_bindings = {}
        self.dynamic_offers = {}
        self.dynamic_results = {}
        self.used_dynamic_plans = set()
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
            endpoint in self.dynamic_results.get(ref, {}).get("writes", {}) for ref in sources or {}
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

    def read_batch(self, source, planner, loaded):
        if isinstance(planner, dict):
            obs = next((o for o in loaded if o.source == planner["source"]), None)
            if obs is not None:
                return project_batch(self.tools[source], obs, planner["fields"])
            return None
        batch = self.evaluate_policy(planner, "fragment.reads", source, loaded)
        if not isinstance(batch, list) or not batch:
            return None
        for args in batch:
            check_args(self.tools[source], args)
        return batch

    def sources_for(self, spec, primary):
        """Check known inputs and stage reads whose arguments need an earlier result."""
        loaded, pending = [], set()
        for source in (primary, *spec.get("inputs", [])):
            planner = spec.get("reads", {}).get(source)
            if planner is not None:
                if source not in self.tools or not available(source) or not is_read(source):
                    yield None
                    return
                depends_on_pending = (
                    planner["source"] in pending if isinstance(planner, dict) else bool(pending)
                )
                if not depends_on_pending:
                    try:
                        batch = self.read_batch(source, planner, loaded)
                    except ValueError:
                        batch = None
                    if batch is None:
                        yield None
                        return
                    ready = self.read_ready(source, batch)
                    if ready is not None:
                        loaded.append(ready)
                    elif batch:
                        pending.add(source)
                    else:
                        loaded.append(Observation("", source, [], "Empty selected input", arguments=[]))
                else:
                    pending.add(source)
                yield source
                continue
            obs = next((o for o in reversed(self.observed) if o.source == source and o.complete), None)
            if obs is None:
                yield None
                return
            loaded.append(obs)
            yield obs.ref

    def card_plans(self):
        plans = []
        library = self.lib["fragment"]
        scores = {name: relevance(spec, self.goal) for name, spec in library.items()}
        for name, spec in library.items():
            if ("dynamic" in spec or "composition" in spec) and (hint := self.lib["retrieval"].get(name)):
                scores[name] = max(scores[name], len(goal_words(hint) & goal_words(self.goal)))
        # Relevance can reach a required producer with no live result yet. Primary sources
        # listed as alternatives keep their own goal/scope evidence.
        observed = {o.source for o in self.observed if o.complete}
        for _ in range(len(library)):
            for name, spec in library.items():
                if "dynamic" in spec or "composition" in spec:
                    continue
                producers = spec.get("inputs", []) + (spec["sources"] if len(spec["sources"]) == 1 else [])
                for source in set(producers) - observed:
                    if source in scores:
                        scores[source] = max(scores[source], scores[name] - 1)
        self.dynamic_offers.clear()
        for name, spec in library.items():
            if "dynamic" in spec or "composition" in spec:
                continue
            for primary in spec["sources"]:
                ports = tuple(self.sources_for(spec, primary))
                if None in ports:
                    continue
                action = ("fragment", name, *ports)
                if action in self.used_transforms:
                    continue
                score = scores[name]
                ready = sum(p.startswith("r") and p[1:].isdecimal() for p in ports)
                scope = next((o.scope for o in self.observed if o.ref == ports[0]), primary)
                label = f"{spec['label']} | scope: {preview(scope, 180)}"
                if spec.get("produces"):
                    label += " | yields " + ", ".join(spec["produces"])
                plans.append((score, ready, label, action))
        existing = [plan for plan in plans if plan[0] > 0]
        for name, spec in library.items():
            if "composition" not in spec and ("dynamic" not in spec or scores[name] <= 0):
                continue
            downstream = any(scores[other] > 0 and name in [*consumer["sources"], *consumer.get("inputs", ())]
                             for other, consumer in library.items() if other != name)
            if "composition" in spec:
                prepared, proposed = record_composition(self, spec)
                if not proposed:
                    continue
                found = dynamic_plans(self, name, prepared, downstream=downstream, proposed=proposed)
            else:
                prepared = spec
                found = dynamic_plans(self, name, prepared, downstream=downstream)
            choices = [plan for plan in found if not already_covered(self, prepared, plan, existing)]
            if found and not choices:
                self.trace.add("native_dynamic_reuse_existing", name=name)
            if choices:
                self.dynamic_offers[name] = (prepared, choices)
                label = dynamic_label(self, prepared, choices)
                if cue := self.lib["decision"].get(name):
                    label += f" | {cue}"
                plans.append((max(scores[name], 1), 0, label, ("dynamic", name)))
        return sorted(plans, key=lambda row: (-row[0], -row[1]))

    def fragments(self):
        # Browsing leaves a path to reusable cards with weak lexical cues.
        return [(label, action) for _, _, label, action in self.card_plans()]

    def execute_card(self, action):
        kind, name, *ports = action
        spec = self.lib["fragment"][name]
        if kind == "dynamic":
            prepared, plans = self.dynamic_offers[name]
            execute_dynamic(self, name, prepared, plans)
            return
        by_ref = {o.ref: o for o in self.observed}
        loaded = []
        for port in ports:
            if port in by_ref:
                obs = by_ref[port]
            else:
                batch = self.read_batch(port, spec["reads"][port], loaded)
                if batch is None:
                    raise Uncovered(f"{name}: insufficient public parameters for {port}")
                obs = self.read_ready(port, batch)
                if obs is None:
                    planner = spec["reads"][port]
                    sources = {o.ref: {"program_input": name} for o in loaded}
                    if isinstance(planner, dict):
                        origin = next(o for o in loaded if o.source == planner["source"])
                        sources = {origin.ref: {"fields": planner["fields"]}}
                    if batch:
                        self.perform(port, batch, f"Read for {name}", sources)
                    else:
                        self.observe(port, [], "Empty selected input", sources=sources, arguments=[])
                    obs = self.observed[-1]
            if not obs.complete:
                raise Uncovered(f"{name}: {obs.ref} is only a partial observation")
            loaded.append(obs)
        result = self.resolve_fragment(name, loaded[0], loaded[1:])
        if result is None or (result == [] and not any(
            name in rule["sources"] and "fields" in rule for rule in self.lib["binding"].values()
        )):
            raise Uncovered(f"{name} could not establish its result")
        rows = result if isinstance(result, list) else [result]
        if any(path_get(row, field) is None or not accepts({"type": kind}, path_get(row, field))
               for row in rows for field, kind in spec.get("produces", {}).items()):
            raise Uncovered(f"{name} did not supply its declared public fields")
        sources = {o.ref: {"program_input": name} for o in loaded}
        observed_scope = ", ".join(f"{o.ref} {o.source}" for o in loaded)
        self.observe(name, result, observed_scope, sources=sources)
        self.trace.add("native_card", name=name, inputs=list(sources), output=self.observed[-1].ref)

    def compiled_bindings(self, endpoint):
        rule = self.lib["binding"].get(endpoint, {})
        resolved = dynamic_writes(self, endpoint)
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
        result = dynamic_answers(self)
        for obs in reversed(self.observed):
            if not obs.complete or obs.source not in {*self.lib["fragment"], "calculation"}:
                continue
            if self.lib["fragment"].get(obs.source, {}).get("produces") and not relevance(
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
            value: self.dynamic_offers[value[1]][1]
            for _, value in actions
            if isinstance(value, tuple) and value[0] == "dynamic" and "composition" in self.lib["fragment"][value[1]]
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

    def action_page(self, names, *, allow_discovery=True, include_compositions=True):
        outstanding = self.outstanding()
        answers = self.answer_bindings()
        plans = self.card_plans()
        cards = [(label, action) for score, _, label, action in plans if score > 0 and action[0] == "fragment"][:3]
        if answers:
            cards = []
        actions = list(cards)
        self.plan_bindings = {}
        if self.ready_to_finish():
            prompt = answers[0][0] if len(answers) == 1 else "all requested work has been done"
            if len(answers) > 1 and self.dynamic_results:
                prompt = "a calculation from the verified records"
            actions.insert(0, (f"Complete the task with {prompt}", FINISH))
        for name in sorted(outstanding):
            projected = "fields" in self.lib["binding"].get(name, {})
            operations = self.compiled_bindings(name)
            dynamic = any(ref in self.dynamic_results for _, _, _, sources in operations for ref in sources)
            if operations and (projected or dynamic or len(operations) == 1):
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
        if not cards and not answers and (not outstanding or not self.set_request):
            actions = self.calculation_options()[:2] + actions
        capacity = 14 if allow_discovery else 15
        room = min(3, max(0, capacity - len(actions)))
        dynamic_cards = ([(label, action) for _, _, label, action in plans if action[0] == "dynamic"
                          and (include_compositions or "composition" not in self.lib["fragment"][action[1]])][:room]
                         if not answers else [])
        actions[:0] = dynamic_cards
        self.trace.add("native_fragment_retrieval", eligible=[p[3] for p in plans],
                       page=[a for _, a in [*dynamic_cards, *cards]])
        if allow_discovery:
            actions = actions[:14] + [("Look for another relevant public operation", "@discover")]
        else:
            actions = actions[:15]
        action = self.ask(
            "native/action", "What step advances the user's remaining goal?", self.preview_action_choices(actions),
            ACTION_SKILL + " " + self.lib["decision"].get("action", ""),
        )
        if action is None and include_compositions:
            declined = [value[1] for _, value in dynamic_cards if "composition" in self.lib["fragment"][value[1]]]
            for name in declined:
                self.decline_dynamic(name, "No applicable card on this action page")
            if declined:
                return self.action_page(names, allow_discovery=allow_discovery, include_compositions=False)
        return action

    def decline_dynamic(self, name, reason):
        _, plans = self.dynamic_offers[name]
        self.used_dynamic_plans.update((name, key(plan)) for plan in plans)
        self.trace.add("native_dynamic_exit", name=name, reason=reason)

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
        active_cards = [o.source for o in reversed(self.observed) if o.ref in self.dynamic_results]
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
                    if action[0] != "dynamic":
                        self.used_transforms.add(action)
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
            except (TypeError, ValueError, Uncovered) as exc:
                reason = str(exc)[:350]
                if isinstance(action, tuple) and action[0] == "dynamic":
                    self.decline_dynamic(action[1], reason)
                    continue
                self.notices.append(reason)
                self.trace.add("native_feedback", reason=reason)
                if isinstance(action, str):
                    self.residual(reason, action)
