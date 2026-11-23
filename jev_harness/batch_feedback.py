"""Compare the other current Train tasks while Sol can still edit its candidate."""
import json
from itertools import zip_longest
from pathlib import Path

from .formal_optimizer import read_events
from .runtime import is_open_generation
from .semantic_workbench import PROMPT, SemanticWorkbench
from .transport import WorkerError, WorkerUnavailable


def prefix_view(events):
    requests, choices, cards = [], [], []
    generation = None
    for i, event in enumerate(events):
        kind = event["event"]
        if kind == "probe_generation_boundary":
            generation = (i, event["messages"])
            break
        if kind == "model":
            if is_open_generation(event["operation"], event["request"]):
                generation = (i, event["request"]["messages"])
                break
            if event["operation"] == "decide":
                requests.append((i, event["request"]))
        elif kind == "decision":
            value = event["candidates"][event["selected"]]["value"]
            choices.append((i, [event["node"], value]))
        elif kind == "card_executed":
            cards.append(event["card"])
    return requests, choices, cards, generation


def generation_payload(messages):
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            value = json.loads(message["content"])
        except (ValueError, KeyError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def read_list(progress):
    if isinstance(progress, dict):
        reads = progress.get("available_public_reads")
        if isinstance(reads, dict):
            return [f"{name} = {api}" for name, api in reads.items()]
        if isinstance(reads, list):
            return reads
    return None  # The execution harness may truncate the progress shown to 27B.


def prefix_changes(parent, candidate):
    old_requests, old_choices, old_cards, old_generation = parent
    new_requests, new_choices, new_cards, new_generation = candidate
    changes = {}
    for old, new in zip_longest(old_requests, new_requests):
        before, after = old[1] if old else {}, new[1] if new else {}
        if before != after:
            request = {
                "parent_event": old[0] if old else None, "candidate_event": new[0] if new else None,
                "fields": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))}
            if "options" in request["fields"]:
                old_opts, new_opts = before.get("options", []), after.get("options", [])
                request["added_option_examples"] = [s[:150] for s in new_opts if s not in old_opts][:2]
                request["removed_option_examples"] = [s[:150] for s in old_opts if s not in new_opts][:2]
            changes["first_changed_Jev_request"] = request
            break
    for old, new in zip_longest(old_choices, new_choices):
        if old is None or new is None or old[1] != new[1]:
            changes["first_changed_Jev_choice"] = {
                "parent": old[1] if old else None, "candidate": new[1] if new else None,
                "parent_event": old[0] if old else None, "candidate_event": new[0] if new else None}
            break
    if old_cards != new_cards:
        changes["executed_cards"] = {"parent": old_cards, "candidate": new_cards}
    old_event, old_messages = old_generation or (None, None)
    new_event, new_messages = new_generation or (None, None)
    if old_messages == new_messages:
        return changes
    detail = {"parent_event": old_event, "candidate_event": new_event}
    changes["first_generation_input"] = detail
    if old_messages is None or new_messages is None:
        detail["change"] = "new handoff" if old_messages is None else "no candidate handoff observed"
        return changes
    before, after = generation_payload(old_messages), generation_payload(new_messages)
    detail["fields"] = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if [m for m in old_messages if m.get("role") == "system"] != [m for m in new_messages if m.get("role") == "system"]:
        detail["fields"].append("system")
    old_progress, new_progress = before.get("compiled_progress"), after.get("compiled_progress")
    if isinstance(old_progress, dict) and isinstance(new_progress, dict):
        detail["progress_fields"] = sorted(k for k in old_progress.keys() | new_progress.keys()
                                           if old_progress.get(k) != new_progress.get(k))
    old_reads, new_reads = read_list(old_progress), read_list(new_progress)
    if old_reads is not None and new_reads is not None and old_reads != new_reads:
        changes["public_reads"] = {"added": [s for s in new_reads if s not in old_reads],
                                   "removed": [s for s in old_reads if s not in new_reads]}
        if not any(changes["public_reads"].values()):
            changes["public_reads"]["order_changed"] = True
    return changes


class BatchFeedbackWorkbench(SemanticWorkbench):
    trial_reminder = ("This is turn 8 of 12. Try a candidate now to see the other tasks in this batch. "
                      "You can then inspect a changed sibling or revise the patch.")
    prompt = PROMPT + """
For the current batch, each trial also returns a parent/candidate comparison on every task through the
first generation handoff or task completion. These prefixes use real Jev decisions and public tools.
The before/after success and total generated tokens of the last official revision, when available, are in
batch[].last_measured_round, including rejected revisions. The current trial cannot supply a new whole-task
score. Start a provisional patch by turn 9 so you can inspect its effects before the final reply.
To read or search, send one action such as {"read":[{"doc":"current/<task>","start":0,"count":6}]}
or {"find":{"doc":"events/<task>","text":"card_failed"}}. Use the returned event indices to read the raw trace.
Look at previously solved sibling tasks and at a different scenario in this same batch. If an unrelated
Jev choice or generation input changes, inspect the paired event locations or probe that sibling directly.
Consider the responsible assembly, decision, state, adapter, card or handoff. Edit the parts supported by
the observations. In the proposal reason, explain the layer you chose and any observed spillover. If a wrong
menu led to generation, compare repairing the menu or decision with a scoped handoff. Use the evidence to
choose between them. All six are alternatives, with no quota by component."""

    def __init__(self, study, lib, rows, history, folder, sources):
        super().__init__(study, lib, rows, history, folder, sources)
        self.parent_prefixes = {task: prefix_view(read_events(self.paths["current/" + task]))
                                for task in self.tasks}
        self.previous_trial = None
        for result_file in sorted(folder.glob("probe_*/result.json"), reverse=True):
            result = json.loads(result_file.read_text(encoding="utf-8"))
            report = result.get("current_batch_prefixes", [])
            if ("comparison_objects" in result and {r["task"] for r in report} == self.tasks
                    and all("changes" in r for r in report)):
                self.previous_trial = (result["comparison_objects"], report)
                self.register_prefix_traces(report)
                break
        # A revisit has the same task IDs as its first visit, whose results remain in history.
        current_visit = folder.parent.parent.name
        for revision in reversed(history):
            if "batch" in revision:
                visit = revision.get("visit", f"b{revision['batch']:02d}")
                if visit != current_visit:
                    continue
            candidates = self.paired_paths(revision.get("train_traces", []))
            parents = self.paired_paths(revision.get("parent_traces", []))
            if not self.tasks or candidates.keys() != self.tasks or parents.keys() != self.tasks:
                continue
            for item in self.index:
                task = item["task"]
                before, after = parents[task], candidates[task]
                left, right = (json.loads(p.parents[1].joinpath("record.json").read_text(encoding="utf-8"))
                               for p in (before, after))
                item["last_measured_round"] = {
                    "parent": revision["parent"], "candidate": revision["candidate"], "accepted": revision["accepted"],
                    "success_before_after": [left["success"], right["success"]],
                    "total_generated_tokens_before_after": [left["output_tokens"], right["output_tokens"]],
                    "parent_trace": before.parts[-4] + "/" + task, "candidate_trace": after.parts[-4] + "/" + task,
                    "prefix_changes": prefix_changes(prefix_view(read_events(before)), prefix_view(read_events(after)))}
            break

    def paired_paths(self, traces):
        paths = [self.study.root / Path(t.replace("\\", "/")) for t in traces]
        return {p.parents[1].name: p for p in paths}

    def register_prefix_traces(self, report):
        for row in report:
            if "candidate_trace" in row:
                doc = row["candidate_trace"]
                self.paths[doc] = self.folder / doc.removeprefix("probe/") / "trace.jsonl"
        self.docs["trace_directory"] = sorted(self.paths)

    def find_evidence(self, query):
        queries = query if isinstance(query, list) else [query]
        results = []
        for item in queries:
            if not isinstance(item, dict):
                raise ValueError('Use {"find":{"doc":"events/<task>","text":"card_failed"}}')
            docs = item.get("docs", [item.get("doc")])
            text = item.get("text", item.get("query"))
            if not isinstance(text, str) or not text or not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
                raise ValueError('Supply a document and search text, for example {"doc":"dsl","text":"decision"}')
            for doc in docs:
                hits = super().find_evidence({"doc": doc, "text": text, "start": item.get("start", 0)})
                results.append({"doc": doc, "hits": hits})
                if len(results) == 5:
                    return results
        return results

    def probe(self, proposal, task):
        from formal_run import save
        focus = super().probe(proposal, task)
        self.last_probed = None
        folder = self.folder / f"probe_{self.calls:02d}"
        # The same executable objects can be investigated with a different focus or stop event.
        if self.previous_trial is not None and self.previous_trial[0] == proposal["objects"]:
            report = self.previous_trial[1]
        else:
            candidate, _ = self.install(self.lib, proposal, self.sources, self.study.functions)
            def compare(item, worker):
                name = item["task"]
                dest = folder / "batch" / name
                if name == task and proposal.get("until") is None:
                    path, result = folder / "trace.jsonl", focus
                else:
                    label = "_".join(("batch_prefix", self.study.root.name,
                                      *folder.relative_to(self.study.root).parts, name))
                    try:
                        engine = self.probe_type(worker.worker("model"), worker.worker("environment"),
                                                 **self.study.execution)
                        engine.set_probe_target(None)
                        result = engine.run_prefix(name, candidate, dest, self.study.seed, label)
                    except (WorkerError, WorkerUnavailable) as exc:
                        worker.close_worker("environment")
                        worker.close_worker("model")
                        return {"task": name, "status": "trial unavailable", "error": str(exc)[:240]}
                    path = dest / "trace.jsonl"
                doc = "probe/" + path.parent.relative_to(self.folder).as_posix()
                return {"task": name, "parent_success": item["success"], "parent_total_tokens": item["output_tokens"],
                        "candidate_stop": result["stop"], "candidate_signalled_completion": result["completed"],
                        "prefix_Jev_calls": result["native_calls"], "prefix_generated_tokens": result["output_tokens"],
                        "parent_trace": "current/" + name, "candidate_trace": doc,
                        "changes": prefix_changes(self.parent_prefixes[name], prefix_view(read_events(path)))}

            report = self.study.execute_jobs(self.index, compare)
            self.register_prefix_traces(report)
            if all("changes" in row for row in report):
                self.previous_trial = (proposal["objects"], report)
        focus["current_batch_prefixes"] = report
        save(folder / "result.json", {**focus, "comparison_objects": proposal["objects"]})
        self.last_probed = self.probe_key(proposal, task)
        return focus
