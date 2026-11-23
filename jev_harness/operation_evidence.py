"""Index public program operations in the current Train batch, retaining source locations."""
import ast
import re
from collections import defaultdict

from .formal_optimizer import cleaned_text
from .runtime import is_open_generation


def names(node, context):
    return {v.id for v in ast.walk(node) if isinstance(v, ast.Name) and isinstance(v.ctx, context)}


def endpoints(node):
    result = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            owner = call.func.value
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "apis":
                if owner.attr not in {"api_docs", "supervisor"} and call.func.attr != "login":
                    result.add(owner.attr + "__" + call.func.attr)
    return result


def operations(node):
    for x in ast.walk(node):
        if isinstance(x, ast.Compare):
            for op in x.ops:
                if isinstance(op, (ast.In, ast.NotIn)):
                    yield "membership", "exclude" if isinstance(op, ast.NotIn) else "include", x
        elif isinstance(x, ast.BinOp) and isinstance(x.op, (ast.BitAnd, ast.BitOr)):
            kind = "membership" if isinstance(x.op, ast.BitAnd) else "union"
            yield kind, "intersection" if kind == "membership" else kind, x
        elif isinstance(x, ast.Call):
            name = x.func.attr if isinstance(x.func, ast.Attribute) else getattr(x.func, "id", "")
            if name in {"intersection", "difference", "sorted", "sum", "min", "max"}:
                yield "membership" if name in {"intersection", "difference"} else name, name, x


def public_actions(response):
    found = []
    for a in response.get("actions", []):
        path = re.sub(r"/[0-9]+(?=/|$)", "/{id}", a.get("url", "").split("?")[0])
        if "/auth/" in path or "password" in path:
            continue
        operation = a.get("method", "").upper() + " " + path
        if operation not in found:
            found.append(operation)
    return found


def cell_sites(code, dependencies):
    """Lightweight source tracking for common assignments, loops, comprehensions and collection updates."""
    tree = ast.parse(code)
    lines = code.splitlines()
    sites = []
    def sources(node):
        found = endpoints(node)
        for name in names(node, ast.Load):
            found.update(dependencies.get(name, ()))
        return found

    def visit(stmt, boundary, scope):
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            for name in names(stmt.target, ast.Store):
                dependencies[name] = sources(stmt.iter) | scope
        # Control flow is traversed in source order; only the current statement is attributed here.
        nested = [part for _, val in ast.iter_fields(stmt) if isinstance(val, list)
                  for part in val if isinstance(part, (ast.stmt, ast.ExceptHandler))]
        exprs = [part for part in ast.iter_child_nodes(stmt) if isinstance(part, ast.expr)]
        for expr in exprs:
            for api in sorted(endpoints(expr)):
                target = stmt.targets[0] if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 else None
                var = target.id if isinstance(target,ast.Name) and isinstance(expr,ast.Call) else None
                sites.append((api, "API", "call", stmt.lineno, boundary.end_lineno, var))
            for kind, variant, op in operations(expr):
                for api in sorted(sources(op) | scope):
                    target = stmt.targets[0] if isinstance(stmt, ast.Assign) and stmt.targets else None
                    var = target.id if isinstance(target, ast.Name) else None
                    sites.append((api, kind, variant, op.lineno, boundary.end_lineno, var))
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value:
            deps = sources(stmt.value) | scope
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                for name in names(target, ast.Store):
                    dependencies[name] = deps.copy()
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.attr in {"add", "append", "extend", "update"}:
                    deps = set(scope)
                    for arg in call.args:
                        deps.update(sources(arg))
                    dependencies.setdefault(call.func.value.id, set()).update(deps)
        if nested:
            control = sources(stmt.iter) if isinstance(stmt, (ast.For, ast.AsyncFor)) else set()
            for child in nested:
                visit(child, boundary, scope | control)

    for stmt in tree.body:
        visit(stmt, stmt, set())
    uses = defaultdict(list)
    for x in ast.walk(tree):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
            uses[x.id].append(x.lineno)
    for api, kind, variant, line, until, var in sites:
        info = {"variant":variant, "line":line, "through_line":until, "variable":var,
                "code":cleaned_text(lines[line-1].strip())[:220]}
        consumers = [n for n in uses.get(var, ()) if n > line]
        if consumers:
            n = min(consumers)
            info["next_source_use"] = {"line":n,"code":cleaned_text(lines[n-1].strip())[:180]}
        yield api, kind, info


def batch_operations(task_events):
    groups = defaultdict(lambda: defaultdict(list))
    counts = {"generated_cells":0, "parsed_cells":0, "unparsed":[]}
    for task, events in task_events.items():
        deps, tokens = {}, None
        for idx, event in enumerate(events):
            if event["event"] == "model" and is_open_generation(event.get("operation"), event.get("request", {})):
                tokens = event["response"].get("output_tokens")
            if event["event"] != "tool_batch" or event.get("node") != "generation_agent":
                continue
            counts["generated_cells"] += 1
            actions = public_actions(event.get("response", {}))
            try:
                sites = list(cell_sites(event["code"], deps))
                counts["parsed_cells"] += 1
            except (SyntaxError, ValueError, RecursionError):
                counts["unparsed"].append({"task":task, "event":idx, "actions":actions})
                sites = []
            for api, kind, info in sites:
                groups[(api,kind)][task].append({"event":idx, **info, "cell_output_tokens":tokens,
                    "cell_reported_error":"Execution failed." in str(event.get("response",{}).get("output","")),
                    "observed_actions":actions})
            tokens = None
    rows = [{"api":api,"operation":kind,"tasks":list(tasks),"sites":dict(tasks)}
            for (api,kind),tasks in groups.items()]
    rows.sort(key=lambda r:(-len(r["tasks"]), r["operation"] == "API",
                            -max(s["cell_output_tokens"] or 0 for ss in r["sites"].values() for s in ss),
                            r["api"], r["operation"]))
    return rows, counts
