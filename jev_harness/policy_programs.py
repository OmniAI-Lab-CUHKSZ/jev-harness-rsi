"""Small pure-Python policies that Sol can replace inside a versioned harness."""
import ast
from calendar import month_abbr, month_name
from copy import deepcopy
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
    temporal = r"\b(?:since|yesterday|today|last|past|before|after|between|recently)\b"
    if not re.search(temporal, text):
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


@lru_cache(maxsize=64)
def load_program(source):
    if not isinstance(source,str) or not 1<=len(source)<=5000:
        raise ValueError("Expected the compact body of a Python policy")
    text="def policy(state):\n"+ "\n".join("    "+line for line in source.splitlines())
    try:
        tree=ast.parse(text)
    except SyntaxError as exc:
        raise ValueError("Policy syntax: "+str(exc)) from exc
    entry=tree.body[0]
    if len(tree.body)!=1 or not isinstance(entry,ast.FunctionDef):
        raise ValueError("Expected one policy function")
    for node in ast.walk(tree):
        if not isinstance(node,NODES) or (isinstance(node,ast.FunctionDef) and node is not entry):
            raise ValueError("Unsupported policy operation: "+type(node).__name__)
        if isinstance(node,ast.Name) and node.id.startswith("_"):
            raise ValueError("Policy variables use public names")
        if isinstance(node,ast.Attribute) and node.attr not in METHODS:
            raise ValueError("Unsupported policy method: "+node.attr)
        if isinstance(node,ast.Call) and not (isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS
                                               or isinstance(node.func,ast.Attribute) and node.func.attr in METHODS):
            raise ValueError("Use built-in operations or the public pattern helpers")
    namespace={"__builtins__":{},**FUNCTIONS}
    exec(compile(tree,"<harness policy>","exec"),namespace)
    return namespace["policy"]


def run_program(library, name, state, trace):
    code=library.get("programs",{}).get(name)
    if not code:
        return None
    try:
        return load_program(code)(deepcopy(state))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError,re.error) as exc:
        trace.add("program_error",policy=name,error=str(exc)[:300])
        return None
