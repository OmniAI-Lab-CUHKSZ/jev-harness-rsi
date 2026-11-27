"""Small pure-Python policies that Sol can replace inside a versioned harness."""
import ast
from calendar import month_abbr, month_name
from datetime import date, timedelta
from functools import lru_cache
import re


def regex(pattern, text):
    return bool(re.search(pattern, text, re.I))


def captures(pattern, text):
    return re.findall(pattern, text, re.I)


def first_calendar_day(when, days):
    """Lower date for an inclusive window ending on the task's public date."""
    if type(days) is not int or days < 1:
        raise ValueError("Use a positive number of calendar days")
    return (date.fromisoformat(when[:10]) - timedelta(days=days - 1)).isoformat()


def calendar_windows(goal, when):
    """Resolve public dates, distinguishing a missing restriction from an unsupported one."""
    text = goal.casefold()
    period = re.search(r"\b(?:this|ongoing|current)\s+(month|year)\b", text)
    temporal = r"\b(?:since|yesterday|today|last|past|before|after|between|recently)\b"
    if not period and not re.search(temporal, text):
        return [(None, None)]
    today = date.fromisoformat(when[:10])
    months = {name.casefold(): i for names in (month_abbr, month_name) for i, name in enumerate(names) if name}
    explicit = re.search(r"\bsince\s+(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s+(\d{4}))?", text)
    if explicit and explicit[2] in months:
        start = date(int(explicit[3] or today.year), months[explicit[2]], int(explicit[1]))
        return [(start.isoformat(), today.isoformat())] if start <= today else []
    days = re.search(r"\b(?:last|past)\s+([1-9]\d*)\s+(?:calendar\s+)?days?\b", text)
    if days:
        count = int(days[1])
        included = (first_calendar_day(when, count), today.isoformat())
        if re.search(r"\b(?:including|counting)\s+today\b|\btoday\s+included\b", text):
            return [included]
        yesterday = today - timedelta(days=1)
        excluded = ((today - timedelta(days=count)).isoformat(), yesterday.isoformat())
        return [excluded] if re.search(r"\bexcluding\s+today\b", text) else [included, excluded]
    yesterday = (today - timedelta(days=1)).isoformat()
    if re.search(r"\byesterday or today\b|\btoday or yesterday\b", text):
        return [(yesterday, today.isoformat())]
    if re.search(r"\byesterday\b", text):
        return [(yesterday, yesterday)]
    if period and not re.search(temporal, text):
        first = today.replace(month=1, day=1) if period[1] == "year" else today.replace(day=1)
        return [(first.isoformat(), today.isoformat())]
    return []


FUNCTIONS = {"len":len,"any":any,"all":all,"sorted":sorted,"min":min,"max":max,"sum":sum,
             "enumerate":enumerate,"zip":zip,"str":str,"int":int,"float":float,"list":list,
             "dict":dict,"set":set,"isinstance":isinstance,"regex":regex,"captures":captures,
             "first_calendar_day":first_calendar_day,"calendar_windows":calendar_windows}
METHODS = {"get","keys","items","values","startswith","endswith","lower","casefold","strip","split","replace",
           "join","append","extend","add","count"}
NODES = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.For,ast.If,ast.IfExp,ast.Break,
         ast.Continue,ast.Pass,ast.Expr,ast.Assign,ast.AugAssign,ast.Name,ast.Constant,ast.Compare,ast.BoolOp,
         ast.BinOp,ast.UnaryOp,ast.Subscript,ast.Slice,ast.List,ast.Tuple,ast.Set,ast.Dict,ast.ListComp,
         ast.SetComp,ast.DictComp,ast.GeneratorExp,ast.comprehension,ast.Call,ast.keyword,ast.Attribute,
         ast.Load,ast.Store,ast.And,ast.Or,ast.Not,ast.Eq,ast.NotEq,ast.In,ast.NotIn,ast.Is,ast.IsNot,
         ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub)


def policy_capabilities():
    return {
        "functions": sorted(FUNCTIONS),
        "methods": sorted(METHODS),
        "operator_nodes": sorted(node.__name__ for node in NODES
                                 if issubclass(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop))),
    }


@lru_cache(maxsize=64)
def load_program(source):
    if not isinstance(source,str) or not 1<=len(source)<=5000:
        raise ValueError("Expected the compact body of a Python policy")
    text="def policy(state):\n"+ "\n".join("    "+line for line in source.splitlines())
    try:
        tree=ast.parse(text)
        program=compile(tree,"<harness policy>","exec")
    except SyntaxError as exc:
        line = max(1, (exc.lineno or 2) - 1)
        raise ValueError(f"line {line}: Policy syntax: {exc.msg}") from exc
    entry=tree.body[0]
    if len(tree.body)!=1 or not isinstance(entry,ast.FunctionDef):
        raise ValueError("Expected one policy function")
    for node in ast.walk(tree):
        error = None
        if not isinstance(node, NODES) or (isinstance(node, ast.FunctionDef) and node is not entry):
            error = f"Unsupported policy operation: {type(node).__name__}"
        elif isinstance(node, ast.Name) and node.id.startswith("_"):
            error = "Policy variables use public names"
        elif isinstance(node, ast.Attribute) and node.attr not in METHODS:
            error = f"Unsupported policy method: .{node.attr}"
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                if target.id not in FUNCTIONS:
                    error = f"Unsupported policy function: {target.id}()"
            elif isinstance(target, ast.Attribute):
                if target.attr not in METHODS:
                    error = f"Unsupported policy method: .{target.attr}()"
            else:
                error = "Call a named policy function or method"
        if error is not None:
            line = getattr(node, "lineno", None)
            if line is None:
                # Operator nodes have no position of their own.
                line = next((parent.lineno for parent in ast.walk(tree)
                             if hasattr(parent, "lineno") and node in ast.iter_child_nodes(parent)), 2)
            raise ValueError(f"line {max(1, line - 1)}: {error}")
    namespace={"__builtins__":{},**FUNCTIONS}
    exec(program,namespace)
    return namespace["policy"]


def validate_program(source, *, location):
    if not isinstance(source, str):
        raise ValueError(f"{location}: Expected a Python policy body as a string")
    try:
        load_program(source)
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc
