"""Compile official tool choices and argument alternatives from public observations."""

from dataclasses import dataclass
from itertools import product
import json
import re

from .expressions import path_get
from .retrieval import terms


FINISH = "supervisor__complete_task"
MANAGED = {"access_token", "page_limit", "page_index"}
READ_VERBS = {"show", "search", "get", "list", "check", "find", "retrieve", "is", "does"}


def key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def public_value(value):
    if isinstance(value, list):
        return [public_value(v) for v in value]
    if isinstance(value, dict):
        return {
            k: public_value(v)
            for k, v in value.items()
            if k
            not in {"access_token", "refresh_token", "password", "verification_code", "card_number", "cvv"}
        }
    return value


def is_read(name):
    return name.split("__")[-1].split("_")[0] in READ_VERBS or name.endswith("_exists")


def leaf_fields(row, prefix=""):
    if not isinstance(row, dict):
        return
    for k, v in row.items():
        field = prefix + k
        if isinstance(v, dict):
            yield from leaf_fields(v, field + ".")
        elif v is not None:
            yield field, v


def accepts(schema, value):
    if "anyOf" in schema:
        return any(accepts(s, value) for s in schema["anyOf"])
    types = {
        "string": (str,),
        "integer": (int,),
        "number": (float, int),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
        "null": (type(None),),
    }
    kind = schema.get("type")
    if kind in types and type(value) not in types[kind]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get(
        "maxLength", float("inf")
    ):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return schema.get("minimum", float("-inf")) <= value <= schema.get(
            "maximum", float("inf")
        ) and schema.get("exclusiveMinimum", float("-inf")) < value < schema.get(
            "exclusiveMaximum", float("inf")
        )
    if isinstance(value, list) and "items" in schema:
        return all(accepts(schema["items"], v) for v in value)
    return True


def check_args(tool, arguments):
    schema = tool["parameters"]
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    unexpected = set(arguments) - (schema["properties"].keys() - {"access_token"})
    if unexpected:
        raise ValueError(
            f"Unknown arguments for {tool['name']}: {', '.join(sorted(unexpected))}. "
            "Put source, where, order_by and take beside arguments."
        )
    missing = set(schema.get("required", ())) - MANAGED - arguments.keys()
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(sorted(missing)))
    for k, v in arguments.items():
        if not accepts(schema["properties"][k], v):
            raise ValueError(f"Argument {k} does not match its public schema")


def normalize_action(spec, expected, tools):
    """Accept one standard function call and equivalent placements of observation selectors."""
    if set(spec) == {"function"} or set(spec) == {"type", "function"}:
        spec = spec["function"]
    if not isinstance(spec, dict):
        raise ValueError("Expected a single public function call")
    spec = dict(spec)
    if "name" in spec and "tool" not in spec:
        spec["tool"] = spec.pop("name")
    name = spec.get("tool", expected)
    if name not in tools:
        return spec
    args = spec.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    args = dict(args)
    props = tools[name]["parameters"]["properties"]
    for k in ("source", "where", "order_by", "take"):
        if k in args and k not in props:
            if k in spec and spec[k] != args[k]:
                raise ValueError(f"Conflicting {k} selectors")
            spec[k] = args.pop(k)
    if "access_token" in props:
        args.pop("access_token", None)
    return {**spec, "arguments": args}


def catalog(functions):
    return {f["name"]: f for row in functions if (f := row.get("function", row))}


def available(name):
    return name.split("__")[-1] not in {"login", "show_account_passwords"} and name != FINISH


def ranked_tools(tools, instruction, history=(), hints=None, app=None, query=""):
    words = terms(instruction)
    lookup = terms(query)
    seen = {o.source for o in history}
    live = set()
    for obs in history[-3:]:
        if obs.rows and isinstance(obs.rows[0], dict):
            live.update(k.replace(".", "_") for k, _ in leaf_fields(obs.rows[0]))

    def score(name):
        tool = tools[name]
        domain, api = name.split("__")
        ids, description = terms(api), terms(tool["description"])
        extra = terms((hints or {}).get(name, ""))
        params = tool["parameters"]["properties"].keys() - MANAGED
        needed = set(tool["parameters"].get("required", [])) - MANAGED
        missing = [p for p in needed if p.endswith("_id") and p not in live]
        matches = words & (ids | description | extra)
        s = (
            5 * len(words & ids)
            + 3 * len(matches)
            + 2 * len(words & terms(" ".join(params)))
            + 6 * len(words & extra)
            + 7 * int(terms(domain) <= words)
            + 15 * len(lookup & (ids | description | extra | terms(domain)))
            - 8 * len(missing)
        )
        if not history and is_read(name):
            s += 4
        if history and matches:
            s += 3 * len(needed & live)
        if name in seen:
            s -= 13
        return s

    names = (name for name in tools if available(name) and (app is None or name.startswith(app + "__")))
    return sorted(names, key=lambda n: (-score(n), n))


@dataclass
class Observation:
    ref: str
    source: str
    data: object
    scope: str
    complete: bool = True
    arguments: list | None = None

    @property
    def rows(self):
        return self.data if isinstance(self.data, list) else [self.data]

    def view(self, instruction):
        rows = self.rows
        query = terms(instruction)

        def relevance(row):
            return len(terms(str(row)) & query)

        matches = sorted(rows, key=relevance, reverse=True)[:2] if rows else []
        sample = []
        for row in matches + rows[:1]:
            if row not in sample:
                sample.append(row)
        return {
            "source": self.source,
            "ref": self.ref,
            "scope": self.scope,
            "complete": self.complete,
            "count": len(rows),
            "sample": sample,
        }


def literals(instruction, tool):
    quotes = re.findall(r'"([^"]+)"', instruction.replace("“", '"').replace("”", '"'))
    numbers = [
        float(n) if "." in n else int(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", instruction)
    ]
    words = re.findall(r"[A-Za-z][\w&-]*", instruction)
    nouns = {n.rstrip("s") for n in tool["name"].split("__")[-1].split("_")[1:]}
    modifiers = [
        a
        for a, b in zip(words, words[1:])
        if b.casefold().rstrip("s") in nouns
        and a.casefold() not in {"all", "the", "my", "of", "those", "which", "these", "a", "an", "each"}
    ]
    names = [
        w
        for w in words[1:]
        if w[0].isupper()
        and w.casefold() not in {tool["name"].split("__")[0].replace("_", ""), "i", "send", "the", "my", "if"}
    ]
    return quotes, numbers, list(dict.fromkeys(modifiers)), list(dict.fromkeys(names))


def literal_plans(tool, instruction):
    props = tool["parameters"]["properties"]
    required = set(tool["parameters"].get("required", [])) - MANAGED
    quoted, numbers, modifiers, names = literals(instruction, tool)
    text = instruction.casefold()
    slots = {}
    for name, schema in props.items():
        if name in MANAGED or name.endswith("_id"):
            continue
        values = []
        if "enum" in schema:
            values = [
                v
                for v in schema["enum"]
                if isinstance(v, str) and re.search(r"\b" + re.escape(v) + r"\b", text)
            ]
        elif schema.get("type") in {"integer", "number"}:
            relevant = bool(terms(name) & terms(text))
            if name.startswith("min_"):
                relevant &= bool(re.search(r"at least|minimum|no fewer|or more", text))
            if name.startswith("max_"):
                relevant &= bool(re.search(r"at most|maximum|no more|or fewer", text))
            if relevant or name in required:
                values = numbers
        elif schema.get("type") == "string" and name not in {"sort_by", "access_token", "password"}:
            if name.endswith("phone_number") or schema.get("format") in {"phone", "phone-number"}:
                values = [
                    n.strip()
                    for n in re.findall(r"(?<!\w)\+?\d[\d ().-]{5,}\d(?!\w)", instruction)
                    if 7 <= len(re.sub(r"\D", "", n)) <= 15
                ]
            elif name.endswith("email") or schema.get("format") == "email":
                values = re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", instruction)
            elif name.endswith("path"):
                values = [q for q in quoted if "/" in q and "<" not in q]
            elif name == "query":
                values = quoted + modifiers + names
            elif (
                name in required or terms(name) & terms(text) or (modifiers and schema.get("default") is None)
            ):
                values = quoted + modifiers
        if values:
            slots[name] = list(dict.fromkeys(v for v in values if accepts(schema, v)))[:5]
    # Distinct interpretations of textual filters stay visible, e.g. free-text search vs a named filter.
    dimensions = list(slots)
    result = []
    for values in product(*[([None] if name not in required else []) + slots[name] for name in dimensions]):
        args = {k: v for k, v in zip(dimensions, values) if v is not None}
        if required - args.keys() or (
            "query" in args and args["query"] in [v for k, v in args.items() if k != "query"]
        ):
            continue
        result.append(args)
        if len(result) >= 80:
            break
    return sorted(result, key=lambda a: (-len(a), "query" in a))


def field_for(name, schema, row, source):
    fields = dict(leaf_fields(row))
    exact = [p for p, v in fields.items() if p.replace(".", "_") == name and accepts(schema, v)]
    if len(exact) == 1:
        return exact[0]
    if (
        name.endswith("_id")
        and name[:-3] in terms(source)
        and "id" in fields
        and accepts(schema, fields["id"])
    ):
        return "id"
    return None


def groups(obs, instruction):
    rows = [row for row in obs.rows if isinstance(row, dict)]
    if not rows:
        return
    matches = {}
    query = terms(instruction)
    for i, row in enumerate(rows):
        for path, value in leaf_fields(row):
            if not isinstance(value, str) or len(value) > 100:
                continue
            overlap = terms(value) & query
            if overlap and (
                path.endswith(("name", "title", "email")) or value.casefold() in instruction.casefold()
            ):
                matches.setdefault((path, value), []).append(i)
    subsets = [(f"{p} = {v}", indexes) for (p, v), indexes in matches.items()]
    subsets.sort(key=lambda pair: (-len(terms(pair[0]) & query), len(pair[1]), pair[0]))
    whole_scope = "all records" if obs.complete else "all records in this observed excerpt"
    for label, indices in subsets[:8] + [(whole_scope, list(range(len(rows))))]:
        yield label, [rows[i] for i in indices]


def record_plans(tool, instruction, observations):
    props = tool["parameters"]["properties"]
    required = [p for p in tool["parameters"].get("required", []) if p not in MANAGED]
    if not required:
        return []
    results = []
    for obs in reversed(observations):
        for scope, rows in groups(obs, instruction):
            fields = {p: field_for(p, props[p], rows[0], obs.source) for p in required}
            if not all(fields.values()):
                continue
            batch = []
            for row in rows:
                args = {p: path_get(row, f) for p, f in fields.items()}
                if not all(accepts(props[p], args[p]) for p in required):
                    break
                if args not in batch:
                    batch.append(args)
            else:
                results.append(
                    (
                        f"{obs.ref} {scope}; {len(batch)} distinct bindings via {fields}",
                        batch,
                        {obs.ref: {"fields": fields}},
                    )
                )
    return results


def bound_plans(tool, instruction, observations):
    return [(label, batch) for label, batch, _ in record_plans(tool, instruction, observations)]


def update_fields(tool):
    """AppWorld update endpoints separate record identifiers from editable payload."""
    if not tool["name"].split("__")[-1].startswith("update_"):
        return None
    props = tool["parameters"]["properties"]
    required = set(tool["parameters"].get("required", []))
    targets = {p for p in required if p.endswith(("_id", "_path"))}
    return targets, props.keys() - MANAGED - targets


def has_update(tool, args):
    fields = update_fields(tool)
    return fields is None or any(k in args and args[k] is not None for k in fields[1])


def checklist_plans(tool, instruction, observations):
    """Only offer a reversible single-line edit on a uniquely named, observed checklist item."""
    fields = update_fields(tool)
    if not fields or not fields[0]:
        return
    targets, editable = fields
    props = tool["parameters"]["properties"]
    # Explicit goals are idempotent. For other wording let Jev see the two possible settings.
    direction = re.sub(r'["“][^"”]*["”]', "", instruction).casefold()
    clear = re.search(
        r"\bas\s+(?:not\s+done|incomplete|unchecked)\b|^\s*(?:please\s+)?(?:uncheck|unset)\b", direction
    )
    set_mark = re.search(r"\bas\s+(?:done|complete|completed|checked)\b|^\s*(?:please\s+)?check\b", direction)
    desired = (" ",) if clear and not set_mark else ("x",) if set_mark and not clear else ("x", " ")
    for obs in reversed(observations):
        if obs.source.split("__")[0] != tool["name"].split("__")[0]:
            continue
        for row in obs.rows:
            if not isinstance(row, dict) or not targets <= row.keys():
                continue
            for field in sorted(editable & row.keys()):
                body = row[field]
                if not isinstance(body, str) or not accepts(props[field], body):
                    continue
                matches = list(re.finditer(r"(?m)^([ \t]*(?:[-*+] )?\[)([ xX])(\][ \t]+)([^\r\n]+)", body))
                entries = [m[4].strip().casefold() for m in matches]
                for m, entry in zip(matches, entries):
                    if (
                        not entry
                        or entries.count(entry) != 1
                        or not re.search(r"(?<!\w)" + re.escape(entry) + r"(?!\w)", instruction, re.I)
                    ):
                        continue
                    for value in desired:
                        mark = m[2] if m[2].lower() == value else value
                        edited = body[: m.start(2)] + mark + body[m.end(2) :]
                        args = {**{k: row[k] for k in sorted(targets)}, field: edited}
                        label = f"{obs.ref} set '[{mark}] {m[4].strip()}' in {field}, preserving other text"
                        yield label, [args], {obs.ref: {"fields": {k: k for k in args}, "edited": [field]}}


def effect_view(tool, args, observations):
    """Describe values submitted to a public operation, before a success receipt is known."""

    def short(value, n=120):
        return value if not isinstance(value, str) or len(value) <= n else value[: n - 1] + "…"

    fields = update_fields(tool)
    if not fields:
        return {k: short(v) for k, v in public_value(args).items()}
    targets, editable = fields
    identity = {k: args[k] for k in sorted(targets & args.keys())}
    result = {"target": identity or "current account", "set": {}}
    for k in sorted(editable & args.keys() & public_value(args).keys()):
        val = args[k]
        change = {"to": short(val)}
        # A before value must belong to the same public app and exactly the same object.
        if identity:
            for obs in reversed(observations):
                if obs.source.split("__")[0] != tool["name"].split("__")[0] or not is_read(obs.source):
                    continue
                values = [
                    r[k]
                    for r in obs.rows
                    if isinstance(r, dict)
                    and k in r
                    and all(r.get(field) == v for field, v in identity.items())
                ]
                if not values or any(v != values[0] for v in values):
                    continue
                old = values[0]
                change.update(previous_ref=obs.ref, previous=short(old), same_value=old == val)
                if isinstance(old, str) and isinstance(val, str) and old != val:
                    offset = next(
                        (i for i, (a, b) in enumerate(zip(old, val)) if a != b), min(len(old), len(val))
                    )
                    begin = max(0, offset - 24)
                    change.update(offset=offset, previous=short(old[begin:], 95), to=short(val[begin:], 95))
                break
        result["set"][k] = change
    return result


def unique_plans(plans):
    seen = set()
    result = []
    for label, batch in plans:
        signature = key(batch)
        if batch and signature not in seen:
            seen.add(signature)
            result.append((label, batch))
    return result


def calculations(obs, instruction):
    """Offer data operations only on actual fields. Each choice carries an executable result."""
    if not isinstance(obs.data, list) or not obs.data or not isinstance(obs.data[0], dict):
        return []
    text = instruction.casefold()
    latest = re.search(r"\b(latest|last|newest|most recent|maximum|most|largest|longest|highest)\b", text)
    first = re.search(r"\b(earliest|oldest|minimum|least|smallest|shortest|lowest)\b", text)
    total = re.search(r"\b(total|sum|average|mean|how many|count)\b", text)
    format_values = re.search(r"\b(unique|distinct|comma.separated|group by)\b", text)
    if not (latest or first or total or format_values):
        return []
    date_order = bool(re.search(r"\b(latest|last|newest|recent|earliest|oldest)\b", text))
    choices = []
    for scope, rows in groups(obs, instruction):
        source = f"{obs.ref} ({obs.source}), {scope}, n={len(rows)}"
        fields = dict(leaf_fields(rows[0]))
        if total and re.search(r"\b(how many|count)\b", text):
            choices.append((2, "Count records in " + source, len(rows)))
        for field, example in fields.items():
            numbers = type(example) in (float, int) and not field.endswith(("_id", ".id")) and field != "id"
            dates = isinstance(example, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}", example))
            if not (numbers or dates or (format_values and isinstance(example, str))):
                continue
            pairs = [(path_get(r, field), r) for r in rows]
            valid = [(v, r) for v, r in pairs if v is not None]
            if not valid or any(
                type(v) not in (int, float) if numbers else not isinstance(v, str) for v, r in valid
            ):
                continue
            rank = 4 * len(terms(field) & terms(instruction)) + (2 if scope != "all records" else 0)
            if latest or first:
                if (date_order and not dates) or (not date_order and not numbers):
                    continue
                best = (max if latest else min)(v for v, _ in valid)
                winners = [r for v, r in valid if v == best]
                if len(winners) == 1:
                    label = (
                        "Latest/highest" if latest else "Earliest/lowest"
                    ) + f" record by {field} in {source}"
                    choices.append((rank + int(field == "created_at"), label, winners[0]))
            if total and numbers:
                if len(valid) != len(rows):
                    continue
                values = [v for v, _ in valid]
                average = bool(re.search(r"\b(average|mean)\b", text))
                value = sum(values) / len(values) if average else sum(values)
                choices.append((rank, ("Average " if average else "Sum ") + field + " in " + source, value))
            if format_values and isinstance(example, str):
                values = [v for v, _ in valid]
                if "group by" in text:
                    result = {}
                    for v, r in valid:
                        result.setdefault(v, []).append(r)
                    label = "Group records by "
                elif re.search(r"comma.separated", text):
                    result = ", ".join(values)
                    label = "Join with commas: "
                else:
                    result = list(dict.fromkeys(values))
                    label = "Distinct values of "
                choices.append((rank, label + field + " in " + source, result))
    choices.sort(key=lambda row: -row[0])
    return [(label, value) for _, label, value in choices]


def public_calculation(spec, observations):
    if not isinstance(spec, dict) or set(spec) - {"op", "source", "field", "where", "separator"}:
        raise ValueError("Use a single calculation with op, source, and optional field/where/separator")
    obs = next((o for o in observations if o.ref == spec.get("source")), None)
    if obs is None:
        raise ValueError("Calculation requires an observed source")
    rows = obs.rows
    where = spec.get("where", {})
    if not isinstance(where, dict):
        raise ValueError("where expects exact field/value pairs")
    rows = [r for r in rows if all(path_get(r, k) == v for k, v in where.items())]
    op = spec.get("op")
    if op == "count":
        return len(rows)
    field = spec.get("field")
    if not isinstance(field, str) or not rows:
        raise ValueError("Calculation needs rows and a field")
    values = [path_get(r, field) for r in rows]
    if any(v is None for v in values):
        raise ValueError("Missing field in the selected scope")
    if op in ("sum", "mean") and all(type(v) in (int, float) for v in values):
        return sum(values) if op == "sum" else sum(values) / len(values)
    if op in ("argmin", "argmax"):
        best = (min if op == "argmin" else max)(values)
        winners = [r for v, r in zip(values, rows) if v == best]
        if len(winners) != 1:
            raise ValueError("Tied ordering; disambiguate the record")
        return winners[0]
    if op == "project":
        return values
    if op == "distinct":
        return list({key(v): v for v in values}.values())
    if (
        op == "join"
        and all(isinstance(v, str) for v in values)
        and isinstance(spec.get("separator", ", "), str)
    ):
        return spec.get("separator", ", ").join(values)
    if op == "group" and all(type(v) in (str, int, float, bool) for v in values):
        grouped = {}
        for value, row in zip(values, rows):
            grouped.setdefault(str(value), []).append(row)
        return grouped
    raise ValueError("Unsupported operation or field type for the calculation")


def expand_binding(spec, observations):
    """Resolve one generated binding, optionally projecting every row in an observed set."""
    if not isinstance(spec, dict):
        raise ValueError("A tool action must be an object")
    args = spec.get("arguments", {})
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    if "source" not in spec:
        return [args]
    obs = next((o for o in observations if o.ref == spec["source"]), None)
    if not obs:
        raise ValueError("Unknown observation reference")
    rows = list(obs.rows)
    condition = spec.get("where")
    if condition is not None:
        if (
            not isinstance(condition, dict)
            or not condition
            or any(not isinstance(v, (str, int, float, bool)) for v in condition.values())
        ):
            raise ValueError("where takes exact field/value pairs")
        rows = [r for r in rows if all(path_get(r, k) == v for k, v in condition.items())]
    if "order_by" in spec:
        path = spec["order_by"]
        rows.sort(key=lambda r: path_get(r, path))
    if spec.get("take") in ("first", "last") and rows:
        if (
            len(rows) > 1
            and "order_by" in spec
            and path_get(rows[-1 if spec["take"] == "last" else 0], spec["order_by"])
            == path_get(rows[-2 if spec["take"] == "last" else 1], spec["order_by"])
        ):
            raise ValueError("Tied ordering; disambiguate the record")
        rows = [rows[0] if spec["take"] == "first" else rows[-1]]
    if not rows:
        raise ValueError("No records match the specified binding")

    def resolve(row, value):
        if isinstance(value, dict) and set(value) == {"field"}:
            return path_get(row, value["field"])
        if isinstance(value, dict) and set(value) == {"replace_text"}:
            edit = value["replace_text"]
            if (
                not isinstance(edit, dict)
                or set(edit) != {"field", "old", "new"}
                or not all(isinstance(v, str) for v in edit.values())
            ):
                raise ValueError("replace_text needs string field, old, new")
            body = path_get(row, edit["field"])
            if not isinstance(body, str) or not edit["old"] or body.count(edit["old"]) != 1:
                raise ValueError("replace_text needs one unique, nonempty occurrence in the observed field")
            return body.replace(edit["old"], edit["new"], 1)
        return value

    return [{k: resolve(row, v) for k, v in args.items()} for row in rows]
