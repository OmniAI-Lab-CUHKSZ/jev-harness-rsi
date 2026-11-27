"""Plan parameterized fragments from public observations and tool signatures."""

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal

from .expressions import path_get
from .native_assembly import FINISH, preview
from .policy_programs import load_program
from .record_composition import select_record_subset
from .runtime import Uncovered
from .schema_tools import MANAGED, accepts, available, check_args, is_read, key


def validate_dynamic(spec, tools):
    config = spec["dynamic"]
    if not isinstance(config, dict) or config.keys() != {"plan", "slots", "results"}:
        raise ValueError("A dynamic fragment needs a plan, slots and result descriptions")
    if any(name in spec for name in ("choice", "choices", "reads")):
        raise ValueError("Declare dynamic reads and choices in the dynamic plan")
    load_program(config["plan"])
    slots, results = config["slots"], config["results"]
    if not isinstance(slots, dict) or not isinstance(results, dict) or not results:
        raise ValueError("Declare the available result types and named slots")
    for slot in slots.values():
        if (not isinstance(slot, dict) or slot.keys() - {"question", "skill"}
                or not isinstance(slot.get("question"), str) or not slot["question"].strip()
                or not isinstance(slot.get("skill", ""), str)):
            raise ValueError("Each slot needs a question and may provide local guidance")
    for result in results.values():
        if not isinstance(result, dict) or result.keys() - {"fields", "writes", "calculations", "equivalent"}:
            raise ValueError("Describe a result's fields and the operations that can consume it")
        fields = result.get("fields")
        if not isinstance(fields, dict) or not fields or any(
            not field or kind not in {"integer", "number", "string", "object", "array", "boolean"}
            for field, kind in fields.items()
        ):
            raise ValueError("A dynamic result needs public field paths and types")
        calculations = result.get("calculations", {})
        if (not isinstance(calculations, dict) or calculations.keys() - {"count", "sum"}
                or ("count" in calculations and calculations["count"] is not True)
                or ("sum" in calculations and fields.get(calculations["sum"]) not in {"integer", "number"})):
            raise ValueError("Use count or sum of a declared numeric field")
        equivalent = result.get("equivalent", [])
        if not isinstance(equivalent, list) or not all(isinstance(name, str) and name.isidentifier()
                                                      for name in equivalent):
            raise ValueError("Name any equivalent existing fragment by its identifier")
        writes = result.get("writes", {})
        if not isinstance(writes, dict):
            raise ValueError("Describe writes by their published endpoint")
        for endpoint, rule in writes.items():
            if endpoint not in tools or not available(endpoint) or is_read(endpoint):
                raise ValueError("A dynamic consumer needs an official write endpoint")
            if (not isinstance(rule, dict) or rule.keys() - {"fields", "arguments"}
                    or not isinstance(rule.get("fields"), dict) or not rule["fields"]):
                raise ValueError("Bind a write to fields on this result type")
            schema = tools[endpoint]["parameters"]
            projection = rule["fields"]
            if (projection.keys() - (schema["properties"].keys() - MANAGED)
                    or set(projection.values()) - fields.keys()):
                raise ValueError("A write projection must use declared result and API fields")
            if "arguments" in rule:
                load_program(rule["arguments"])
            elif set(schema.get("required", ())) - MANAGED - projection.keys():
                raise ValueError("Supply the write's required arguments")
    return config


def write_literals(engine, endpoint, rule):
    literals = load_program(rule["arguments"])(engine.policy_context()) if "arguments" in rule else {}
    if not isinstance(literals, dict) or literals.keys() & rule["fields"].keys():
        raise ValueError("A dynamic write must bind its literals independently of record fields")
    schema = engine.tools[endpoint]["parameters"]
    if (literals.keys() - (schema["properties"].keys() - MANAGED)
            or set(schema.get("required", ())) - MANAGED - (rule["fields"].keys() | literals.keys())
            or any(not accepts(schema["properties"][k], v) for k, v in literals.items())):
        raise ValueError("A dynamic write needs valid public arguments")
    return literals


def read_available(engine, sources, read):
    if not isinstance(read, dict) or read.get("source") not in sources:
        return False
    if read.keys() == {"source", "ref"}:
        return any(o.ref == read["ref"] and o.source == read["source"] and o.complete for o in engine.observed)
    if read.keys() != {"source", "arguments"}:
        return False
    source, batch = read["source"], read["arguments"]
    if source not in engine.tools or not available(source) or not is_read(source) or not isinstance(batch, list) or not batch:
        return False
    for arguments in batch:
        check_args(engine.tools[source], arguments)
    return True


def dynamic_plans(engine, name, spec, *, downstream=False, proposed=None):
    config = spec["dynamic"]
    if proposed is None:
        context = {**engine.policy_context(engine.observed), "remaining_effects": sorted(engine.outstanding()),
                   "answer_question": engine.answer_question}
        try:
            proposed = load_program(config["plan"])(context)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
            engine.trace.add("native_dynamic_skip", name=name, reason=str(exc)[:250])
            return []
    if not isinstance(proposed, list):
        return []
    checked = {}
    for plan in proposed:
        try:
            if not isinstance(plan, dict) or not {"result", "reads", "choices"} <= plan.keys():
                continue
            result = config["results"][plan["result"]]
            reads, choices = plan["reads"], plan["choices"]
            if not isinstance(reads, dict) or not reads or not isinstance(choices, dict):
                continue
            if choices.keys() - config["slots"].keys() or any(not isinstance(v, str) or not v.strip()
                                                              for v in choices.values()):
                continue
            if not all(read_available(engine, spec["sources"], read) for read in reads.values()):
                continue
            intended = set(plan["requested"]) - engine.chosen_effects if "composition" in spec else engine.outstanding()
            writes = set(result.get("writes", {})) & intended
            if not (writes or downstream or (engine.answer_question and result.get("calculations"))):
                continue
            for endpoint in writes:
                write_literals(engine, endpoint, result["writes"][endpoint])
            identity = key(plan)
            if (name, identity) not in engine.used_dynamic_plans:
                checked.setdefault(identity, plan)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
            engine.trace.add("native_dynamic_skip", name=name, reason=str(exc)[:250])
    return list(checked.values())


def dynamic_label(engine, spec, plans):
    config = spec["dynamic"]
    scope = []
    for slot in config["slots"]:
        labels = list(dict.fromkeys(plan["choices"][slot] for plan in plans if slot in plan["choices"]))
        if labels:
            scope.append(labels[0] if len(labels) == 1 else " / ".join(labels[:3]))
    results = [config["results"][plan["result"]] for plan in plans]
    intended = (set(plans[0]["requested"]) - engine.chosen_effects if "composition" in spec
                else engine.outstanding())
    effects = sorted({endpoint for result in results for endpoint in result.get("writes", {})} & intended)
    uses = [endpoint.split("__", 1)[1].replace("_", " ") for endpoint in effects]
    if engine.answer_question:
        sums = sorted({result["calculations"]["sum"] for result in results if "sum" in result.get("calculations", {})})
        uses.extend(f"sum {field}" for field in sums)
        if any(result.get("calculations", {}).get("count") for result in results):
            uses.append("count matching records")
    details = (f" | current scope: {', '.join(scope)}" if scope else "")
    if uses:
        details += f" | ready for: {', '.join(uses)}"
    return spec["label"] + details


def already_covered(engine, spec, plan, existing):
    """Keep a declared equivalent only for the same grounded reads and requested effects."""
    result = spec["dynamic"]["results"][plan["result"]]
    effects = engine.outstanding()
    if engine.answer_question or not effects or effects - result.get("writes", {}).keys():
        return False
    aliases = set(result.get("equivalent", ()))
    reads = list(plan["reads"].values())
    if not aliases or any("arguments" not in read for read in reads):
        return False
    expected = {(r["source"], key(sorted(r["arguments"], key=key))) for r in reads}
    bindings = engine.lib["binding"]
    by_ref = {obs.ref: obs for obs in engine.observed}
    scopes = [action for _, _, _, action in existing if action[1] in aliases]
    scopes.extend(("fragment", event["name"], *event["inputs"]) for event in engine.trace.events
                  if event["event"] == "native_card" and event["name"] in aliases and event["output"] in by_ref)
    for _, name, *ports in scopes:
        if name not in aliases or not all(bindings.get(e, {}).get("covers") and
                                          name in bindings[e]["sources"] for e in effects):
            continue
        declared = engine.lib["fragment"][name]
        loaded, requests = [], set()
        for port in ports:
            if port in by_ref:
                obs = by_ref[port]
                source, args = obs.source, obs.arguments
            else:
                source = port
                try:
                    args = engine.read_batch(source, declared["reads"][source], loaded)
                except (KeyError, ValueError):
                    break
                obs = engine.read_ready(source, args)
            if not args:
                break
            requests.add((source, key(sorted(args, key=key))))
            if obs is not None:
                loaded.append(obs)
        else:
            if requests == expected:
                return True
    return False


def choose_plan(engine, name, config, plans):
    for slot, spec in config["slots"].items():
        groups = defaultdict(list)
        for plan in plans:
            groups[plan["choices"].get(slot)].append(plan)
        if len(groups) == 1:
            value = next(iter(groups))
            if value is not None:
                engine.trace.add("native_dynamic_slot", name=name, slot=slot, value=value, mode="program")
            continue
        options = [(value or "Unrestricted", value or "") for value in groups]
        value = engine.ask("native/binding", spec["question"], options, spec.get("skill", ""), fragment=name)
        if value is None:
            raise Uncovered(f"No applicable value for {slot}")
        plans = groups[value or None]
        engine.trace.add("native_dynamic_slot", name=name, slot=slot, value=value, mode="Jev")
    if len(plans) != 1:
        raise Uncovered("The selected slots did not describe a unique executable plan")
    return plans[0]


def execute_dynamic(engine, name, spec, plans):
    config = spec["dynamic"]
    plan = choose_plan(engine, name, config, plans)
    inputs, refs = {}, []
    for port, read in plan["reads"].items():
        source = read["source"]
        if "ref" in read:
            observation = next(o for o in engine.observed if o.ref == read["ref"])
        else:
            args = read["arguments"]
            observation = engine.read_ready(source, args)
            if observation is None:
                engine.perform(source, args, spec["label"])
                observation = engine.observed[-1]
        if not observation.complete:
            raise Uncovered(f"Incomplete dynamic input {source}")
        inputs[port] = deepcopy(observation.data)
        refs.append(observation.ref)

    result = config["results"][plan["result"]]
    if "composition" in spec:
        records = select_record_subset(spec["program"], plan, inputs, next(iter(result["fields"])))
    else:
        values = {**engine.policy_context(), "reads": inputs, "parameters": deepcopy(plan.get("parameters", {}))}
        records = load_program(spec["program"])(values)
    fields = {**spec.get("produces", {}), **result["fields"]}
    if not isinstance(records, list) or any(not isinstance(row, dict) or any(
        path_get(row, field) is None or not accepts({"type": kind}, path_get(row, field))
        for field, kind in fields.items()
    ) for row in records):
        raise Uncovered(f"{name} did not produce its declared records")
    scope = ", ".join(plan["choices"].values()) or spec["label"]
    engine.observe(name, records, scope, sources={ref: {"program_input": name} for ref in refs})
    ref = engine.observed[-1].ref
    engine.dynamic_results[ref] = result
    if "composition" in spec:
        engine.required.update(plan["requested"])
    engine.used_dynamic_plans.add((name, key(plan)))
    engine.trace.add("native_card", name=name, inputs=refs, output=ref, result_type=plan["result"])


def dynamic_writes(engine, endpoint):
    if endpoint not in engine.outstanding():
        return []
    choices = []
    for obs in reversed(engine.observed):
        if not (rule := engine.dynamic_results.get(obs.ref, {}).get("writes", {}).get(endpoint)):
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


def dynamic_answers(engine):
    answers = []
    for obs in reversed(engine.observed):
        calculations = engine.dynamic_results.get(obs.ref, {}).get("calculations", {})
        for operation, value in calculations.items():
            if operation == "sum":
                computed = sum((Decimal(str(path_get(row, value))) for row in obs.rows), Decimal(0))
                if not computed.is_finite():
                    continue
                answer = float(computed)
                label = f"Sum {value} for {obs.scope} ({preview(answer, 100)})"
            else:
                answer = len(obs.rows)
                label = f"Count records for {obs.scope} ({answer})"
            answers.append((label, {"answer": answer}, {obs.ref: {"calculation": operation,
                                                                  "field": value if operation == "sum" else None}}))
    return answers
