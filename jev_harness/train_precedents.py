"""Find earlier Train events that can explain a concrete step in the current batch."""
from collections import defaultdict

from .formal_optimizer import cleaned_text
from .operation_evidence import batch_operations, public_actions
from .runtime import is_open_generation


def indexed_steps(task, events, operations):
    # A name in generated source can occur in an unexecuted branch. Keep HTTP
    # observations distinct so that Sol can check what the old path actually did.
    source = defaultdict(set)
    for group in operations:
        for site in group["sites"].get(task, ()):
            api = group["api"]
            if group["operation"] == "API":
                source[site["event"]].add("api " + api)
            else:
                source[site["event"]].add(f"operation {api} {group['operation']} {site['variant']}")

    steps = {}
    tokens = None
    parse_error = None
    menu, offered = None, ()
    for i, event in enumerate(events):
        kind = event["event"]
        signals = set()
        cell_tokens, error = None, False
        if kind == "model" and is_open_generation(event.get("operation"), event.get("request", {})):
            tokens = event["response"].get("output_tokens")
        elif kind == "tool_batch" and event.get("node") == "generation_agent":
            signals.update(source[i])
            response = event.get("response", {})
            for action in public_actions(response):
                if not action.partition(" ")[2].startswith(("/supervisor/", "/api_docs/")):
                    signals.add("observed " + action)
            error = "Execution failed." in str(response.get("output", ""))
            cell_tokens, tokens = tokens, None
            if parse_error:
                signals.add(parse_error)
                parse_error = None
        elif kind == "generation_parse_error":
            parse_error = "parse " + cleaned_text(event.get("error", ""))[:140]
            signals.add(parse_error)
            cell_tokens, tokens, error = tokens, None, True
        elif kind == "program_error" and event.get("policy"):
            signals.add("policy " + event["policy"])
            error = True
        elif kind == "api" and event["app"] not in ("api_docs", "supervisor") and event["api"] != "login":
            signals.add(f"api {event['app']}__{event['api']}")
            error = "Execution failed." in str(event.get("response", {}).get("output", ""))
        elif kind == "adapter_login":
            signals.add(f"login {event['app']} {event.get('identity_field', '')}".strip())
            error = event.get("success") is False
        elif kind in ("card_executed", "card_failed", "parameter_state", "parameter_bindings"):
            card = event.get("card")
            if card and card not in ("submit_effect", "submit_answer"):
                signals.add("card " + card)
                signals.update("slot " + card + "/" + slot for slot in event.get("bound", {}))
                signals.update("role " + role for role in event.get("inputs", ()))
                if isinstance(event.get("output"), str):
                    signals.add("role " + event["output"])
            error = kind == "card_failed"
        elif kind in ("handoff_state", "generated_read_observation"):
            roles = event.get("values", {}) if kind == "handoff_state" else event.get("roles", ())
            signals.update("role " + role for role in roles)
            if kind == "handoff_state":
                parse_error = None
        elif kind == "handoff_selection" and event.get("guide"):
            signals.add("guide " + event["guide"])
        elif kind == "card_menu":
            offered = event.get("offered", ())
            menu = event.get("assembly")
            if menu:
                signals.update(f"offered {menu}/{name}" for name in offered if name not in ("submit_effect", "submit_answer"))
        elif kind == "decision" and event.get("node") == "fragment/step":
            choices, selected = event.get("candidates", ()), event.get("selected")
            if type(selected) is int and 0 <= selected < len(choices):
                card = choices[selected].get("value")
                if isinstance(card, str) and card in offered and card not in ("submit_effect", "submit_answer"):
                    signals.add("card " + card)
                    if menu:
                        signals.add(f"offered {menu}/{card}")
            menu, offered = None, ()
        if signals:
            steps[i] = {"kind":kind, "signals":signals, "cell_output_tokens":cell_tokens, "reported_error":error}
    return steps


class TrainPrecedents:
    def __init__(self, current, operations, past):
        self.current = {task:indexed_steps(task, es, operations) for task, es in current.items()}
        self.sizes = {task:len(es) for task, es in current.items()}
        self.records, self.steps = {}, {}
        self.by_signal = defaultdict(list)
        # The caller supplies only traces from completed Train executions.
        for doc, events in past:
            task = doc.split("/", 1)[1]
            summary = next((e for e in reversed(events) if e["event"] == "summary"), {})
            self.records[doc] = {"trace":doc, "task":task, "goal":events[0].get("instruction", ""),
                                 "task_success":summary.get("success"), "trace_output_tokens":summary.get("output_tokens"),
                                 "events":len(events)}
            groups, _ = batch_operations({task:events})
            self.steps[doc] = indexed_steps(task, events, groups)
            for index, step in self.steps[doc].items():
                for signal in step["signals"]:
                    self.by_signal[signal].append((doc, index))

    def related(self, task, event):
        if task not in self.current or type(event) is not int or not 0 <= event < self.sizes[task]:
            raise ValueError("Use a current Train task and a real event index.")
        result = {"task":task, "event":event, "matches":[]}
        if not self.records:
            return {**result, "status":"no_prior_train"}
        steps = self.current[task]
        if event not in steps:
            nearby = sorted(steps, key=lambda i:(abs(i-event), i))[:3]
            return {**result, "status":"no_indexed_operation", "nearby_events":sorted(nearby)}
        here = steps[event]
        signals = here["signals"]
        if here["kind"] == "tool_batch":
            # Repairing the preceding parse error and performing this business
            # operation are different questions. The error itself is queryable.
            signals = {s for s in signals if not s.startswith("parse ")} or signals
        pairs = {p for signal in signals for p in self.by_signal.get(signal, ())}
        best = {}
        scenario = task.rsplit("_", 1)[0]
        for doc, i in pairs:
            record, step = self.records[doc], self.steps[doc][i]
            other = record["task"]
            group = other.rsplit("_", 1)[0]
            common = signals & step["signals"]
            observed = sum(s.startswith(("observed ", "operation ", "slot ")) for s in common)
            # Prefer a working site, then specificity, independent scope and
            # generated work that might be saved. Source-only matches stay visible.
            rank = (record["task_success"] is True and not step["reported_error"],
                    observed, len(common), not step["reported_error"], group != scenario, other != task,
                    step["cell_output_tokens"] or 0)
            # Several versions or variants of a goal provide one scenario's evidence.
            order = (tuple(-v for v in rank), doc, i)
            if group not in best or order < best[group][0]:
                relation = "different_scenario" if group != scenario else "same_scenario" if other != task else "earlier_same_task"
                row = {"trace":doc, "event":i, "relation":relation, "shared":sorted(common),
                       "past_kind":step["kind"], "past_reported_error":step["reported_error"],
                       "past_cell_output_tokens":step["cell_output_tokens"],
                       "past_observed_actions":sorted(s[9:] for s in step["signals"] if s.startswith("observed ")),
                       "past_task_success":record["task_success"], "past_goal":record["goal"]}
                best[group] = (order, row)
        result.update(status="related_train" if best else "no_shared_operation",
                      current_kind=here["kind"], current_cell_output_tokens=here["cell_output_tokens"],
                      matches=[row for _,row in sorted(best.values(), key=lambda pair:pair[0])[:3]])
        return result

    def suggestions(self, rows):
        """Start near the last substantive step; look earlier for a successful comparison."""
        found = []
        if not self.records:
            return found
        targets = [row for row in rows if not row["success"]]
        if not targets:
            targets = sorted((row for row in rows if row["output_tokens"]),
                             key=lambda row:-row["output_tokens"])[:1]
        for row in targets:
            task = row["task"]
            selected = None
            for i, step in reversed(self.current[task].items()):
                if step["kind"] == "adapter_login" and not step["reported_error"]:
                    continue
                matches = self.related(task, i)["matches"]
                if not matches:
                    continue
                example = matches[0]
                if selected is None:
                    selected = (i, example)
                if example["past_task_success"] is True and not example["past_reported_error"]:
                    selected = (i, example)
                    break
            if selected:
                i, example = selected
                found.append({"task":task, "event":i, "past_trace":example["trace"], "past_event":example["event"],
                              "relation":example["relation"], "past_task_success":example["past_task_success"],
                              "past_kind":example["past_kind"], "past_reported_error":example["past_reported_error"],
                              "past_goal":example["past_goal"], "past_cell_output_tokens":example["past_cell_output_tokens"],
                              "past_observed_actions":example["past_observed_actions"][:6],
                              "shared":example["shared"][:4]})
        return found
