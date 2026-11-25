"""Give the editor one live choice at a changed decision in a successful earlier Train trace."""
import json

from formal_run import read, save

from .formal_optimizer import cleaned_text, read_events
from .semantic_assembly import SemanticAssembly
from .trace_comparison import ParentPath, compare_path
from .transport import WorkerError, WorkerUnavailable


def recorded_decision(events, index):
    """A native ballot can be followed by adaptive review before the recorded selection."""
    request = events[index].get("request", {})
    for event in events[index + 1:]:
        if event["event"] == "model" and event["operation"] == "decide" and all(
                event["request"].get(k) == request.get(k) for k in ("options", "question", "state", "skill")):
            continue
        if event["event"] == "step_review":
            continue
        if event["event"] == "decision" and [str(c["label"]) for c in event["candidates"]] == request.get("options"):
            return event
        break
    return None


def official_request(page, args, settings):
    args = dict(args)
    page["native_widens"] = page["widen"] and page["review"] and args.get("thinking") != "auto"
    if page["review"] and settings.get("step_review_threshold") is not None:
        if settings.get("step_review_mode") != "official_adaptive" or page["budget"] <= 0:
            return None
        if args.get("thinking") != "auto":
            if page["calls_left"] < 2:
                return None
            args.update(thinking="auto", threshold=settings["step_review_threshold"],
                        think_budget=min(page["budget"], settings["step_review_thinking_budget"]))
    return args


def ballot_outcome(answer, choices, prior, page, *, page_indices=False):
    selected = answer.get("index")
    probabilities = answer.get("probabilities") or []
    native = answer.get("native_probabilities") or probabilities
    if (len(probabilities) != len(choices) or len(native) != len(choices)
            or type(selected) is not int or not 0 <= selected < len(choices)):
        return {"status":"invalid_response"}
    widened = page["native_widens"] and native.index(max(native)) == len(choices)-1
    if widened:
        selected, probabilities = len(choices)-1, native
    chosen = choices[selected]
    same = (chosen["label"] == prior["label"] if page_indices else
            json.dumps(chosen["value"], sort_keys=True) == json.dumps(prior["value"], sort_keys=True))
    result = dict(status="observed", candidate_choice=chosen, same_choice=same, confidence=probabilities[selected])
    if max(probabilities) < page["minimum"] and (widened or not answer.get("thinking_used")):
        result.update(status="policy_abstained", same_choice=None)
    return result


def changed_ballot(events, parent, candidate, functions, settings):
    page = {}

    class CapturePage(SemanticAssembly):
        def decision(self, question, evidence, candidates, skill="", policy=None, node=None, **mode):
            spent = sum(e["response"].get("output_tokens", 0) for e in self.trace.events if e["event"] == "model")
            page.clear()
            page.update(node=node, choices=candidates, widen=mode.get("widen_on_none", False),
                        review=node == "fragment/step" and mode.get("review_step", False),
                        budget=self.output_budget - spent, calls_left=self.max_calls - self.requests,
                        minimum=(policy or {}).get("minimum_probability", 0))
            return super().decision(question, evidence, candidates, skill, policy, node, **mode)

    def capture(before, old, after, new):
        if before == after == "decide":
            page["request"] = dict(new)

    path = compare_path(events, parent, candidate, functions, settings, inspect_change=capture,
                        executor_type=CapturePage)
    if "request" not in page or path["matched"] >= parent.matched:
        return None
    change = path["first_change"]
    index = change["event"]
    before = recorded_decision(events, index)
    if not before or before["node"] != page["node"]:
        return None
    args = page["request"]
    choices = [*page["choices"], {"value": None, "label": args["options"][-1]}]
    if [str(c["label"]) for c in choices] != args["options"]:
        return None
    # Ask the official fused endpoint once. A focused menu that natively abstains
    # widens before review in the online executor, even if the endpoint also deliberated.
    page["request"] = official_request(page, args, settings)
    if page["request"] is None:
        return None
    return change, before, page, choices


def historical_choice(bench, proposal, candidate, old, paths, worker, target):
    if target.exists():
        return read(target)
    records = bench.precedents.records
    archive = set(bench.archived)
    ordered = [v["trace"] for v in proposal.get("history_use", [])]
    ordered.extend(v["trace"] for v in paths["focus_history"]["matches"])
    ordered.extend(row["trace"] for row in old.values())
    ordered.extend(sorted(archive))
    for doc in dict.fromkeys(ordered):
        if doc not in archive or records[doc]["task_success"] is not True:
            continue
        task = records[doc]["task"]
        if task in bench.tasks:
            continue
        ref_doc, events, parent = bench.reference_paths[task]
        if ref_doc == doc:
            change = old[task].get("first_change") or {}
            if change.get("parent") != "decide" or change.get("candidate") != "decide":
                continue
        else:
            events = read_events(bench.paths[doc])
            parent = ParentPath(events, bench.lib, bench.study.functions, bench.study.execution)
        ballot = changed_ballot(events, parent, candidate, bench.study.functions, bench.study.execution)
        if ballot is None:
            continue
        change, before, page, choices = ballot
        prior = before["candidates"][before["selected"]]
        req = page["request"]
        old_labels = [str(c["label"]) for c in before["candidates"]]
        new_labels = req["options"]
        result = {"status":"outcome_unconfirmed", "model_calls":1, "version":"draft", "trace":doc,
                  "event":change["event"], "node":page["node"], "goal":cleaned_text(events[0]["instruction"]),
                  "question":cleaned_text(req["question"]), "parent_choice":prior,
                  "changed_fields":change["fields"],
                  "added_options":[cleaned_text(o)[:300] for o in new_labels if o not in old_labels][:4],
                  "removed_options":[cleaned_text(o)[:300] for o in old_labels if o not in new_labels][:4]}
        model = worker.worker("model")
        # A lost reply or interrupted round must not cause a second billed request.
        save(target, result)
        try:
            answer = model.request("decide", **req)
        except (WorkerError, WorkerUnavailable, OSError) as exc:
            result["error"] = cleaned_text(str(exc))[:300]
        else:
            result.update({k:answer[k] for k in ("input_tokens","output_tokens","reasoning_tokens",
                          "backend_requests","seconds","thinking_used") if k in answer})
            local_indices = [c["value"] for c in choices[:-1]] == list(range(len(choices)-1))
            result.update(ballot_outcome(answer, choices, prior, page, page_indices=local_indices))
        save(target, result)
        return result
    result = {"status":"no_comparable_solved_train_choice", "model_calls":0, "version":"draft"}
    save(target, result)
    return result
