"""Small program operations used by compiled bindings and candidate builders.

All operations execute ordinary code. None calls a language model.
New reusable operations can be added here independently of prompt skills.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import re


def path_get(value, path, default=None):
    for part in path.split(".") if path else []:
        try:
            value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return default
    return value


def _flatten(items):
    return [x for item in items for x in (item if isinstance(item, list) else [item])]


def _unique(items, key=None):
    result, seen = [], set()
    for item in items:
        identity = path_get(item, key) if key else item
        identity = json.dumps(identity, sort_keys=True, default=str)
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result


def _regex(text, pattern, group=1, default=None):
    match = re.search(pattern, text, re.I | re.S)
    return match.group(group) if match else default


def _add_minutes(time, minutes):
    value = datetime.strptime(time, "%H:%M") + timedelta(minutes=float(minutes))
    return value.strftime("%H:%M")


def _shift_minutes(instruction):
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "half": 0.5}
    match = re.search(r"(\d+(?:\.\d+)?|a|an|one|two|three|half)\s*(hour|minute)s?\s*(later|earlier)", instruction, re.I)
    if not match:
        raise ValueError("Time shift is outside the compiled parameter parser")
    number, unit, direction = (v.lower() for v in match.groups())
    amount = words[number] if number in words else float(number)
    return amount * (60 if unit == "hour" else 1) * (-1 if direction == "earlier" else 1)


def _replace_line(content, line, replacement):
    lines = content.splitlines(keepends=True)
    selected = line.rstrip("\r\n")
    for index, text in enumerate(lines):
        if text.rstrip("\r\n") == selected:
            ending = "\r\n" if text.endswith("\r\n") else "\n" if text.endswith("\n") else ""
            lines[index] = replacement + ending
            return "".join(lines)
    raise ValueError("Selected line no longer exists in the note")


OPS = {
    "get": lambda x, key, default=None: path_get(x, str(key), default),
    "first": lambda xs: xs[0] if xs else None,
    "last": lambda xs: xs[-1] if xs else None,
    "flatten": _flatten,
    "unique": _unique,
    "length": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": lambda xs, key=None, reverse=False: sorted(xs, key=(lambda x: path_get(x, key)) if key else None, reverse=reverse),
    "concat": lambda *xs: _flatten(xs),
    "join": lambda xs, separator=", ": separator.join(str(x) for x in xs),
    "split_lines": lambda text: text.splitlines(),
    "strip": lambda text: text.strip(),
    "lower": lambda text: text.lower(),
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "contains": lambda container, value: value in container,
    "not": lambda x: not x,
    "and": lambda *xs: all(xs),
    "or": lambda *xs: any(xs),
    "regex": _regex,
    "replace": lambda text, old, new: text.replace(old, new),
    "replace_line": _replace_line,
    "add_minutes": _add_minutes,
    "shift_minutes": _shift_minutes,
    "json": lambda value: json.dumps(value, ensure_ascii=False, default=str),
    "int": int,
    "float": float,
    "str": str,
}


def resolve(spec, context):
    if isinstance(spec, list):
        return [resolve(value, context) for value in spec]
    if not isinstance(spec, dict):
        return spec
    if "ref" in spec:
        sentinel = object()
        result = path_get(context, spec["ref"], sentinel)
        if result is sentinel:
            if "default" in spec:
                return resolve(spec["default"], context)
            raise KeyError("Missing compiled binding: " + spec["ref"])
        return result
    if "op" in spec:
        return OPS[spec["op"]](*resolve(spec.get("args", []), context),
                               **resolve(spec.get("kwargs", {}), context))
    if "template" in spec:
        values = {**context, **resolve(spec.get("values", {}), context)}
        return spec["template"].format_map(values)
    if "map" in spec:
        definition = spec["map"]
        return [resolve(definition["value"], {**context, definition.get("item", "item"): item})
                for item in resolve(definition["source"], context)]
    if "filter" in spec:
        definition = spec["filter"]
        return [item for item in resolve(definition["source"], context)
                if resolve(definition["where"], {**context, definition.get("item", "item"): item})]
    return {key: resolve(value, context) for key, value in spec.items()}


def build_candidates(definition, context):
    result = []
    if isinstance(definition, list):
        return resolve(definition, context)
    rows = resolve(definition["source"], context)
    for index, item in enumerate(rows):
        local = {**context, "item": item, "index": index}
        if "where" in definition and not resolve(definition["where"], local):
            continue
        result.append({"label": resolve(definition.get("label", {"op": "json", "args": [{"ref": "item"}]}), local),
                       "value": resolve(definition.get("value", {"ref": "item"}), local)})
    return result
