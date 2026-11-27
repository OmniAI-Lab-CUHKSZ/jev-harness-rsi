"""Validate cards and build their named inputs from public observations and reads."""

from copy import deepcopy
from decimal import Decimal, DecimalException
from itertools import groupby, product
import json
import re

from .expressions import path_get
from .policy_programs import load_program, validate_program
from .retrieval import terms
from .runtime import MAX_REAL_OPTIONS, Uncovered
from .schema_tools import MANAGED, Observation, accepts, available, check_args, has_update, is_read, key


PROGRAM_ERRORS = (ArithmeticError, AttributeError, IndexError, KeyError, NameError, TypeError, ValueError, re.error)
TYPES = {"integer", "number", "string", "object", "array", "boolean"}


def output_fields(card):
    return card.get("composition", {}).get("output", {}).get("fields", {})


def validate_projection(fields, tool, declared):
    schema = tool["parameters"]
    if (
        not isinstance(fields, dict) or not fields
        or not all(isinstance(field, str) and field for field in fields.values())
        or fields.keys() - (schema["properties"].keys() - MANAGED)
        or (set(schema.get("required", [])) - MANAGED) - fields.keys()
        or not set(fields.values()) <= declared.keys()
    ):
        raise ValueError("Project declared output fields to the official tool's public parameters")


def validate_card(card, tools, fragments):
    if not isinstance(card, dict) or not {"label", "program", "composition"} <= card.keys() or (
        card.keys() - {"label", "program", "composition", "goal_terms"}
    ):
        raise ValueError("A card needs label, program and composition")
    if not isinstance(card["label"], str) or not card["label"].strip():
        raise ValueError("A card needs a useful choice label")
    if "goal_terms" in card and (not isinstance(card["goal_terms"], list) or not card["goal_terms"] or
        not all(isinstance(term, str) and term.strip() for term in card["goal_terms"])):
        raise ValueError("goal_terms must be a nonempty list of goal phrases")
    validate_program(card["program"], location="program")
    spec = card["composition"]
    if not isinstance(spec, dict) or not {"inputs", "output"} <= spec.keys() or (
        spec.keys() - {"inputs", "output", "choices", "equivalent", "scope"}
    ):
        raise ValueError("Composition needs named inputs and one output")
    inputs, output = spec["inputs"], spec["output"]
    if not isinstance(inputs, dict) or not inputs or not all(isinstance(name, str) and name.isidentifier() for name in inputs):
        raise ValueError("Name each input of the composition")
    if "scope" in spec:
        scope = spec["scope"]
        if (not isinstance(scope, dict) or set(scope) != {"noun", "ports"}
                or not isinstance(scope["noun"], str) or not terms(scope["noun"])
                or not isinstance(scope["ports"], dict) or not scope["ports"] or scope["ports"].keys() - inputs.keys()
                or any(not isinstance(label, str) or not terms(label) for label in scope["ports"].values())):
            raise ValueError("Scope needs a collection noun and labels for the selectable input ports")
    if not isinstance(output, dict) or set(output) not in ({"fields"}, {"value"}, {"subset"}, {"subset", "fields"}):
        raise ValueError("Declare output fields, a scalar type or a subset of a named input")
    if "fields" in output and (not isinstance(output["fields"], dict) or not output["fields"]
        or any(not isinstance(field, str) or not field.strip() or not isinstance(kind, str) or kind not in TYPES
               for field, kind in output["fields"].items())):
        raise ValueError("Output fields need JSON types")
    if "value" in output and output["value"] not in ("integer", "number", "string", "boolean"):
        raise ValueError("A scalar needs a scalar JSON type")
    if "subset" in output and (not isinstance(output["subset"], str) or output["subset"] not in inputs):
        raise ValueError("A subset needs a declared input")
    aliases = spec.get("equivalent", [])
    if not isinstance(aliases, list) or not all(isinstance(name, str) and name.isidentifier() for name in aliases):
        raise ValueError("Name equivalent cards by their identifier")
    choices = spec.get("choices", {})
    if not isinstance(choices, dict) or any(
        not isinstance(name, str) or not name.isidentifier() or not isinstance(choice, dict)
        or set(choice) != {"input", "fields", "question"}
        or not isinstance(choice["input"], str) or choice["input"] not in inputs
        or not isinstance(choice["question"], str) or not choice["question"].strip()
        or not isinstance(choice["fields"], list) or not choice["fields"]
        or not all(isinstance(path, str) and path.strip() for path in choice["fields"])
        for name, choice in choices.items()
    ):
        raise ValueError("Each choice needs a named input, public fields and a question")
    roles = {name: part.get("role") for name, part in inputs.items() if isinstance(part, dict) and "role" in part}
    if roles:
        from .record_composition import FILTER_INPUTS

        if roles != FILTER_INPUTS or inputs != {name: {"role": role} for name, role in roles.items()} or (
            output.get("subset") != "rows" or choices or "scope" in spec
        ):
            raise ValueError("The contact adapter needs both supported roles and a subset of rows")
        return
    earlier = {}
    for name, part in inputs.items():
        alternatives = part.get("sources") if isinstance(part, dict) and set(part) == {"sources"} else None
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("An input needs public source alternatives")
        for index, source in enumerate(alternatives):
            if not isinstance(source, dict) or set(source) not in ({"source"}, {"source", "read"}):
                raise ValueError("Declare a source and an optional read")
            endpoint = source["source"]
            if not isinstance(endpoint, str) or endpoint not in tools.keys() | fragments.keys() | {"calculation", "inspection"}:
                raise ValueError("Unknown input source in proposed harness")
            if "read" not in source:
                continue
            if endpoint not in tools or not available(endpoint) or not is_read(endpoint):
                raise ValueError("A card read needs a published read endpoint")
            rule = source["read"]
            if isinstance(rule, str):
                validate_program(rule, location=f"composition.inputs.{name}.sources[{index}].read")
            elif isinstance(rule, list) and rule:
                for args in rule:
                    check_args(tools[endpoint], args)
            elif (isinstance(rule, dict) and set(rule) == {"input", "fields"}
                  and isinstance(rule["input"], str) and rule["input"] in earlier):
                for prior in earlier[rule["input"]]:
                    validate_projection(rule["fields"], tools[endpoint], output_fields(fragments.get(prior["source"], {})))
            else:
                raise ValueError("Use public argument objects, a planner or a projection from an earlier input")
        earlier[name] = alternatives


def project_batch(tool, observation, fields):
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


def input_values(observations):
    return {
        name: {"source": obs.source, "ref": obs.ref, "rows": deepcopy(obs.rows), "scope": obs.scope,
               "complete": obs.complete, "arguments": deepcopy(obs.arguments)}
        for name, obs in observations.items()
    }


def read_arguments(engine, source, rule, loaded):
    if isinstance(rule, dict):
        return project_batch(engine.tools[source], loaded[rule["input"]], rule["fields"])
    if isinstance(rule, list):
        batch = rule
    else:
        state = {**engine.policy_context(), "inputs": input_values(loaded)}
        try:
            batch = load_program(rule)(state)
        except PROGRAM_ERRORS as exc:
            raise ValueError(f"Could not bind {source}: {exc}") from exc
    if not isinstance(batch, list) or not batch:
        raise ValueError(f"No supported arguments for {source}")
    for args in batch:
        check_args(engine.tools[source], args)
    return batch


def bind_inputs(engine, alternatives):
    reads, loaded, pending = {}, {}, set()
    for port, entry in alternatives.items():
        source = entry["source"]
        if "read" not in entry:
            existing = engine.read_ready(source)
            if existing is None:
                return None
            reads[port] = {"source": source, "ref": existing.ref}
            loaded[port] = existing
            continue
        if source not in engine.tools or not available(source) or not is_read(source):
            return None
        rule = entry["read"]
        if isinstance(rule, dict) and rule["input"] not in alternatives:
            return None
        deferred = rule["input"] in pending if isinstance(rule, dict) else isinstance(rule, str) and pending
        if deferred:
            reads[port] = entry
            pending.add(port)
            continue
        try:
            batch = read_arguments(engine, source, rule, loaded)
        except ValueError:
            return None
        reads[port] = {**entry, "arguments": batch}
        existing = engine.read_ready(source, batch)
        if existing is not None:
            loaded[port] = existing
        elif batch:
            pending.add(port)
        else:
            loaded[port] = Observation("", source, [], "Empty selected input", arguments=[])
    first = next(iter(reads.values()))
    scope = loaded[next(iter(reads))].scope if "ref" in first else first["source"]
    return {"reads": reads, "scope": scope}


def input_scopes(goal, scope):
    """Use stated collection members, or offer distinct scopes when the wording is ambiguous."""
    noun, ports = terms(scope["noun"]), scope["ports"]
    labels = {port: terms(label) for port, label in ports.items()}
    vocabulary = noun.union(*labels.values(), {"all", "and", "any", "my", "only", "or", "the", "your"})
    phrases = re.split(r"[.;?!]|\b(?:across|among|from|in|within|of|with|that|which|were)\b", goal)
    scopes = {}
    if re.search(r"\bnot\s+(?:in|from|within|among|across)\b", goal):
        return ()
    for phrase in phrases:
        if not noun <= terms(phrase):
            continue
        if re.search(r"\b(?:except|excluding|without|but not|neither)\b", phrase):
            return ()
        words = [next(iter(terms(word)), word) for word in re.findall(r"[^\W_]+", phrase)]
        blocks = [set(values) for allowed, values in groupby(words, key=vocabulary.__contains__) if allowed]
        named = tuple(port for port, words in labels.items() if any(noun | words <= block for block in blocks))
        included = named or tuple(ports)
        scopes[included] = None
        if re.search(r"\bor\b", phrase) or not named and not re.search(r"\ball\b", phrase):
            scopes.update(dict.fromkeys((port,) for port in included))
    return tuple(scopes)


def source_plans(engine, inputs, scope=None):
    scopes = input_scopes(engine.goal, scope) if scope is not None else ((),)
    for selected in scopes:
        active = {name: part for name, part in inputs.items()
                  if scope is None or name not in scope["ports"] or name in selected}
        for combination in product(*(part["sources"] for part in active.values())):
            plan = bind_inputs(engine, dict(zip(active, combination, strict=True)))
            if plan is None:
                continue
            if scope is not None:
                plan["scope"] = " + ".join(scope["ports"][port] for port in selected) + " " + scope["noun"]
            yield plan


def card_plan_key(plan):
    # Deferred reads keep their identity when an earlier input becomes available.
    reads = {port: ({k: v for k, v in read.items() if k != "arguments"} if "read" in read else read)
             for port, read in plan["reads"].items()}
    return key({**plan, "reads": reads})


def declared_sources(card):
    return {entry["source"] for part in card["composition"]["inputs"].values()
            for entry in part.get("sources", [])}


def run_card(card, state, loaded, selections=()):
    values = {**state, "inputs": input_values(loaded)}
    if selections:
        values.update(selections=deepcopy(selections), selected=deepcopy(selections[-1]))
    return json.loads(json.dumps(load_program(card["program"])(values), allow_nan=False))


def fragment_record(observation, index, fields):
    if type(index) is not int or not 0 <= index < len(observation.rows):
        raise ValueError("A proposed record index is outside the observation")
    values = {field: path_get(observation.rows[index], field) for field in fields}
    if all(value is None for value in values.values()):
        raise ValueError(f"No declared field is visible in {observation.ref}[{index}]")
    display = {}
    for field, value in values.items():
        text = json.dumps(value, ensure_ascii=False)
        display[field] = value if len(text) <= 500 else text[:499] + "…"
    return {"source": observation.source, "ref": observation.ref, "row": index}, display


def fragment_ballot(name, questions, request, loaded, instruction):
    if (not isinstance(request, dict) or not {"role", "rows"} <= request.keys()
            or request.keys() - {"role", "rows", "using"}):
        raise ValueError("Choose a declared question role and its observed rows")
    role, rows, using = request["role"], request["rows"], request.get("using", {})
    if not isinstance(role, str) or role not in questions or not isinstance(using, dict) or using.keys() - questions.keys():
        raise ValueError("Use declared roles for the requested choice and its reference records")
    if (not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REAL_OPTIONS
            or any(type(row) is not int for row in rows) or len(set(rows)) != len(rows)):
        raise ValueError(f"A ballot needs 1 to {MAX_REAL_OPTIONS} distinct observed rows")
    slot = questions[role]
    observation = loaded[slot["input"]]
    candidates = []
    for index in rows:
        record, fields = fragment_record(observation, index, slot["fields"])
        candidates.append({"label": f"{observation.ref}[{index}] " + json.dumps(fields, ensure_ascii=False),
                           "value": {**record, "fragment": name, "role": role}})
    evidence = {"instruction": instruction, "source": observation.source, "source_complete": observation.complete}
    if using:
        references = []
        for anchor in sorted(using):
            template = questions[anchor]
            record, fields = fragment_record(loaded[template["input"]], using[anchor], template["fields"])
            references.append({**record, "fields": fields})
        evidence["reference_records"] = references
    return role, slot["question"], observation, candidates, evidence


def resolve_card(engine, name, card, loaded):
    state = engine.policy_context()
    questions = card["composition"].get("choices", {})
    selections, asked = [], set()
    result = run_card(card, state, loaded)
    while isinstance(result, dict) and set(result) == {"$choose"}:
        try:
            role, question, obs, candidates, evidence = fragment_ballot(
                name, questions, result["$choose"], loaded, engine.instruction
            )
        except ValueError as exc:
            engine.trace.add("native_fragment_choice", name=name, status="unavailable", reason=str(exc))
            raise Uncovered(str(exc)) from exc
        refs = [(r["ref"], r["row"]) for r in evidence.get("reference_records", [])]
        detail = {"role": role, "ref": obs.ref, "source": obs.source}
        if refs:
            detail["using"] = refs
        identity = (role, obs.ref, tuple(sorted(c["value"]["row"] for c in candidates)), tuple(refs))
        if identity in asked:
            reason = "This question already has an answer for these records"
            engine.trace.add("native_fragment_choice", name=name, status="repeated", reason=reason, **detail)
            raise Uncovered(reason)
        asked.add(identity)
        try:
            selection = engine.decision(
                question, evidence, candidates,
                "Use the public record that fills the requested role. Choose NONE if none fits or the evidence leaves a tie.",
                node="native/fragment_choice", review_step=engine.requests + 1 < engine.max_calls,
            )
        except Uncovered as exc:
            engine.trace.add("native_fragment_choice", name=name, status="unbound", reason=str(exc), **detail)
            raise
        index = selection["row"]
        selections.append({"source": obs.source, "ref": obs.ref, "row": index, "value": obs.rows[index], "role": role})
        result = run_card(card, state, loaded, selections)
        status = "continued" if isinstance(result, dict) and set(result) == {"$choose"} else (
            "resolved" if result is not None else "no_result"
        )
        engine.trace.add("native_fragment_choice", name=name, row=index, status=status, **detail)
    return result


def validate_output(output, result, loaded):
    if "subset" in output:
        originals = {key(row) for row in loaded[output["subset"]].rows}
        if not isinstance(result, list) or any(key(row) not in originals for row in result):
            raise ValueError("The card must return a subset of the selected records")
    if "fields" in output:
        rows = result if isinstance(result, list) else [result]
        if any(not isinstance(row, dict) or any(
            not accepts({"type": kind}, path_get(row, field)) for field, kind in output["fields"].items()
        ) for row in rows):
            raise ValueError("The card did not produce its declared fields")
    if "value" in output and not accepts({"type": output["value"]}, result):
        raise ValueError("The card did not produce its declared scalar")


def write_literals(engine, endpoint, rule):
    values = load_program(rule["arguments"])(engine.policy_context()) if "arguments" in rule else {}
    schema = engine.tools[endpoint]["parameters"]
    if (not isinstance(values, dict) or values.keys() & rule["fields"].keys()
            or values.keys() - (schema["properties"].keys() - MANAGED)
            or set(schema.get("required", ())) - MANAGED - (rule["fields"].keys() | values.keys())
            or any(not accepts(schema["properties"][k], v) for k, v in values.items())):
        raise ValueError("A write needs valid public arguments")
    return values


def contact_plan_available(engine, plan):
    try:
        for read in plan["reads"].values():
            source = read["source"]
            if not available(source) or not is_read(source):
                return False
            for args in read["arguments"]:
                check_args(engine.tools[source], args)
        for endpoint in plan["writes"].keys() - engine.chosen_effects:
            write_literals(engine, endpoint, plan["writes"][endpoint])
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        return False
    return True


def request_scope(engine, plans, name):
    questions = {
        "records": ("Which official records cover the requested direction?",
                    "Match the direction to the user's words. Both directions is the union."),
        "contacts": ("Which contact relationships define the requested set?", ""),
        "date": ("Which date interval, inclusive, matches the user's request?", ""),
    }
    for slot, (question, skill) in questions.items():
        if not any(slot in plan.get("choices", {}) for plan in plans):
            continue
        choices = list(dict.fromkeys(plan.get("choices", {}).get(slot) for plan in plans))
        if len(choices) == 1:
            selected, mode = choices[0], "program"
        else:
            labels = [(v or "Unrestricted", v or "") for v in choices]
            answer = engine.ask("native/binding", question, labels, skill, fragment=name)
            if answer is None:
                raise Uncovered(f"No applicable value for {slot}")
            selected, mode = answer or None, "Jev"
        plans = [p for p in plans if p.get("choices", {}).get(slot) == selected]
        if selected is not None:
            engine.trace.add("native_card_slot", name=name, slot=slot, value=selected, mode=mode)
    if len(plans) != 1:
        raise Uncovered("The selected scope did not describe a unique plan")
    return plans[0]


def input_read(engine, read, loaded):
    if "ref" in read:
        obs = engine.observed[int(read["ref"][1:]) - 1]
        if not engine.current_input(obs):
            raise Uncovered(f"{obs.source} needs a fresh observation")
        return obs, None
    source = read["source"]
    args = read["arguments"] if "arguments" in read else read_arguments(engine, source, read["read"], loaded)
    return engine.read_ready(source, args), args


def card_input_refs(engine, plan):
    """Find relevant evidence without making any of the planned reads."""
    loaded = {}
    for index, (port, read) in enumerate(plan["reads"].items()):
        if "arguments" not in read and isinstance(read.get("read"), str) and len(loaded) < index:
            continue
        try:
            obs, _ = input_read(engine, read, loaded)
        except (KeyError, ValueError, Uncovered):
            continue
        if obs is not None:
            loaded[port] = obs
    return tuple((port, obs.ref) for port, obs in loaded.items())


def acquire_inputs(engine, name, plan, label):
    loaded = {}
    for port, read in plan["reads"].items():
        source = read["source"]
        obs, args = input_read(engine, read, loaded)
        if obs is None:
            rule = read.get("read")
            sources = ({loaded[rule["input"]].ref: {"fields": rule["fields"]}} if isinstance(rule, dict)
                       else {o.ref: {"program_input": name} for o in loaded.values()} if rule else {})
            if args:
                engine.perform(source, args, label, sources)
            else:
                engine.observe(source, [], "Empty selected input", sources=sources, arguments=[])
            obs = engine.observed[-1]
        if not obs.complete:
            raise Uncovered(f"{source} is only a partial observation")
        loaded[port] = obs
    return loaded


def card_writes(engine, endpoint):
    if endpoint not in engine.outstanding():
        return []
    choices = []
    for obs in reversed(engine.observed):
        rule = engine.card_results.get(obs.ref, {}).get("writes", {}).get(endpoint)
        if rule is None:
            continue
        try:
            literals = write_literals(engine, endpoint, rule)
            rows = {}
            for item in obs.rows:
                args = {**literals, **{arg: path_get(item, field) for arg, field in rule["fields"].items()}}
                check_args(engine.tools[endpoint], args)
                rows.setdefault(key(args), args)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
            continue
        choices.append((endpoint, list(rows.values()), obs.scope, {obs.ref: {"fields": rule["fields"]}}))
    return choices


def card_answers(engine):
    answers = []
    for obs in reversed(engine.observed):
        for operation, field in engine.card_results.get(obs.ref, {}).get("calculations", {}).items():
            if operation == "sum":
                try:
                    total = sum((Decimal(str(path_get(row, field))) for row in obs.rows), Decimal(0))
                    answer = float(total)
                except (DecimalException, ValueError, OverflowError):
                    continue
                if not accepts({"type": "number"}, answer):
                    continue
                label = f"Sum {field} for {obs.scope}"
            else:
                answer, label, field = len(obs.rows), f"Count records for {obs.scope}", None
            answers.append((f"{label} ({answer})", {"answer": answer},
                            {obs.ref: {"calculation": operation, "field": field}}))
    return answers


def read_bundle(engine, plan):
    calls = set()
    for read in plan["reads"].values():
        args = read.get("arguments")
        if "ref" in read:
            args = next(o.arguments for o in engine.observed if o.ref == read["ref"])
        if args is None:
            return None
        calls.add((read["source"], key(sorted(args, key=key))))
    return calls


def already_covered(engine, name, card, plan, existing):
    aliases = set(card["composition"].get("equivalent", ())) - {name}
    effects = engine.outstanding()
    if not aliases or engine.answer_question or not effects or effects - plan.get("writes", {}).keys():
        return False
    asked = read_bundle(engine, plan)
    if asked is None:
        return False
    for alias, old in [*existing, *engine.completed_card_plans]:
        if alias not in aliases:
            continue
        if all(engine.lib["binding"].get(e, {}).get("covers")
               and alias in engine.lib["binding"][e]["sources"] for e in effects) and read_bundle(engine, old) == asked:
            return True
    return False
