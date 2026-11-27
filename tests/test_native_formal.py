import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jev_harness.artifacts import read, save
from jev_harness import native_formal, native_replay
from jev_harness.native_revision import EditRound, TrainSample, evidence, evidence_page, revise
from jev_harness.policy_programs import load_program, policy_capabilities


def test_ordered_dev_cold_start_two_rounds_and_dev_rollback(tmp_path, monkeypatch):
    # These synthetic identifiers never go to AppWorld or a model service.
    tasks = lambda scenario: [f"{scenario}_{i}" for i in (1, 2, 3)]
    manifest = {"dev_tasks": tasks("dev"), "normal_holdout_tasks": tasks("heldout"),
                "train_batches": [{"batch": i, "tasks": tasks(f"train{i}")} for i in (1, 2, 3)]}
    study = Mock(root=tmp_path, seed=42, execution={}, test_started=False,
                 settings={"optimizer": {"rounds": 2}, "confirmation_scoring": "two_repeats_only"})
    h0 = native_formal.initial_library()
    worker = SimpleNamespace(gpu="fake:0")
    study.execute_preferred.side_effect = lambda jobs, run, pref: [run(job, worker) for job in jobs]
    episodes, edits = [], []

    def run_one(task, stage, variant, library, *, workers):
        directory = tmp_path / "runs" / stage / task
        if (directory / "record.json").exists():
            return read(directory / "record.json")
        episodes.append((stage, task))
        assert variant == ("A" if stage in ("dev_react", "test_react") else "B")
        if stage.startswith("test_"):
            assert study.test_started and study.close_account.called
        if variant == "A":
            assert library == native_formal.blank_library()
        if stage == "dev_h00":
            assert library == h0
        passed = stage not in {"b01r01", "dev_b02r01"}
        trace = directory / "attempt_1" / "trace.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"event": "task", "instruction": "Synthetic task"}) + "\n", encoding="utf-8")
        row = dict(task_id=task, success=passed, seed=42, gpu=workers.gpu, variant=variant, seconds=1,
                   input_tokens=1, output_tokens=4, open_output_tokens=4, decision_output_tokens=0,
                   api_calls=1, open_calls=1, native_ballots=0, model_calls={"G_tool": 1},
                   trace_path=str(trace.relative_to(tmp_path)))
        save(directory / "record.json", row)
        return row

    study.run_task.side_effect = run_one

    def edit(study, ids, *, current):
        name = current.folder.name
        assert all(sample.row["task_id"].startswith("train") for sample in current.samples.values())
        assert all(sample.row["variant"] == "B" for sample in current.samples.values())
        if name.endswith("r01"):
            current_batch = {k: s for k, s in current.samples.items() if not k.startswith("history/")}
            assert set(current_batch) == {f"h0_decision/{t}" for t in ids}
            assert all(s.trace.parents[2].name == name[:3] + "_entry" for s in current_batch.values())
        edits.append(name)
        if name in ("b01r01", "b01r02"):
            assert not current.solved and not any(k.startswith("history/") for k in current.samples)
            assert current.parent == h0
        if name == "b01r02":
            assert current.previous_round["outcome"] == "rejected"
            assert all(f"previous_candidate/{t}" in current.samples for t in ids)
            assert all(current.samples[f"h0_decision/{t}"].trace.parents[2].name == "b01_entry" for t in ids)
        if name == "b02r02":
            assert current.previous_round["outcome"] == "accepted"
            assert all(f"previous_parent/{t}" in current.samples for t in ids)
            assert all(current.samples[f"h0_decision/{t}"].trace.parents[2].name == "b02r01" for t in ids)
        if name == "b03r01":
            assert set(current.parent["decision"]) == h0["decision"].keys() | {"b01r02"}
            assert set(current.solved) == set(tasks("train1") + tasks("train2"))
        child = deepcopy(current.parent)
        objects = []
        if name in ("b01r01", "b01r02", "b02r01"):
            child["decision"][name] = "synthetic cue"
            objects = [{"kind": "decision", "name": name}]
        save(current.folder / "proposal.json", {"reason": name, "objects": objects})
        return child

    monkeypatch.setattr(native_formal, "revise", edit)
    native_formal.run_formal(study, manifest, "react-dev")
    assert [stage for stage, _ in episodes] == ["dev_react"] * 3 and not edits
    native_formal.run_formal(study, manifest, "h0-dev")
    assert [stage for stage, _ in episodes] == ["dev_react"] * 3 + ["dev_h00"] * 3 and not edits
    native_formal.run_formal(study, manifest, "test")
    assert edits == [f"b{batch:02d}r{rnd:02d}" for batch in (1, 2, 3) for rnd in (1, 2)]
    checkpoints = read(tmp_path / "checkpoints.json")
    assert [s["next_version"] for s in checkpoints] == ["b01r02"] * 3
    assert [s["round_outcomes"] for s in checkpoints] == [
        ["rejected", "accepted"], ["accepted", "no_change"], ["no_change", "no_change"]]
    assert checkpoints[0]["dev_selection"]["parent_score"] == checkpoints[0]["dev_selection"]["candidate_score"]
    assert not checkpoints[1]["dev_selection"]["accepted"] and checkpoints[2]["dev_selection"] is None
    assert len({(stage, task) for stage, task in episodes}) == len(episodes)
    expected = {1: ["b01_entry", "b01r01", "b01r02"], 2: ["b02_entry", "b02r01"], 3: ["b03_entry"]}
    for batch, stages in expected.items():
        for task in tasks(f"train{batch}"):
            assert [stage for stage, task_id in episodes if task_id == task] == stages
    assert len([stage for stage, _ in episodes if stage.startswith("dev_")]) == 12
    assert read(tmp_path / "test_report.json")["version"] == "b01r02"
    assert sum(stage.startswith("test_") for stage, task in episodes) == 6


@pytest.mark.parametrize("final_reply", ["empty_patch", "read", "invalid_patch", "invalid_program"])
def test_editor_can_mine_and_open_local_generation_with_only_the_parent_trace(tmp_path, final_reply):
    from jev_harness.protocol import design_manifest

    task = design_manifest()["train_batches"][0]["tasks"][0]
    events = [{"event": "task", "instruction": "Synthetic task"},
              {"event": "model", "operation": "generate",
               "request": {"purpose": "G_binding", "messages": [{"role": "user", "content":
                           json.dumps({"selected_tool": {"name": "example__send"}})}]},
               "response": {"output_tokens": 37}}]
    trace = tmp_path / "entry.jsonl"
    trace.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    save(tmp_path / "offline/public_functions.json", [])
    library = native_formal.blank_library()
    label = "h0_decision/" + task
    row = {"task_id": task, "success": False, "open_calls": 1, "output_tokens": 37}
    current = EditRound(tmp_path / "offline/b01r01", library, {label: TrainSample(trace, row, library)}, {})
    study = Mock(root=tmp_path)
    sol = study.solver.return_value
    read_request = json.dumps({"read": {"doc": label, "start": 1, "count": 1}})
    replies = {
        "empty_patch": json.dumps({"objects": [], "reason": "Synthetic example"}),
        "read": read_request,
        "invalid_patch": json.dumps({"objects": "invalid"}),
        "invalid_program": json.dumps({"objects": [{
            "kind": "fragment", "name": "broken", "value": {
                "label": "Broken computation",
                "program": "break",
                "composition": {
                    "inputs": {"data": {"sources": [{"source": "calculation"}]}},
                    "output": {"value": "integer"},
                },
            },
        }]}),
    }
    sol.turn.side_effect = [read_request] * 11 + [replies[final_reply]]
    assert revise(study, [task], current=current) == library
    assert read(current.folder / "candidate.json") == library
    reason = "Synthetic example" if final_reply == "empty_patch" else (
        "Skipped after 12 draft replies without a valid Sol proposal."
    )
    assert read(current.folder / "proposal.json") == {"objects": [], "reason": reason}
    assert revise(study, [task], current=current) == library
    calls = sol.turn.call_args_list
    assert len(calls) == 12
    assert calls[-1].args[2].name == "turn_12"
    assert "Final response: submit a proposal now" in calls[-1].args[1]
    prompt, page = [json.loads(c.args[1]) for c in calls[:2]]
    assert set(prompt["Train evidence"]) == {label} and set(prompt["trace roles"]) == {"h0_decision"}
    assert prompt["Train evidence"][label]["open_sites"] == [[1, "G_binding", "example__send", 37]]
    assert "G_binding" in str(page) and "example__send" in str(page)
    assert prompt["first_proposal_by_reply"] == 9
    assert prompt["policy_language"] == policy_capabilities()
    blocked = json.loads(calls[9].args[1])
    assert "Submit a draft before requesting another evidence page." in blocked["error"]
    assert "rows" not in blocked
    study.start_workers.assert_not_called()


def test_editor_repairs_early_draft_after_a_located_policy_error(tmp_path, monkeypatch):
    from jev_harness.protocol import design_manifest

    task = design_manifest()["train_batches"][0]["tasks"][0]
    label = "h0_decision/" + task
    trace = tmp_path / "entry.jsonl"
    trace.write_text('{"event":"task","instruction":"Synthetic task"}\n', encoding="utf-8")
    tool = {"name": "example__show_records", "description": "Read example records", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
    }}
    save(tmp_path / "offline/public_functions.json", [tool])
    library = native_formal.blank_library()
    row = {"task_id": task, "success": False, "open_calls": 0, "output_tokens": 0}
    current = EditRound(tmp_path / "offline/b01r01", library, {label: TrainSample(trace, row, library)}, {})
    study = Mock(root=tmp_path, execution={})
    sol = study.solver.return_value
    monkeypatch.setattr(native_replay, "first_change", lambda *args: {"status": "unchanged"})
    read_request = {"read": {"doc": label, "start": 0, "count": 1}}
    source = {"source": tool["name"], "read": "query = state['goal']\nreturn [{'query': query.rstrip('.')}]"}
    card = {"label": "Count observed requests", "program": "return len(state['inputs']['rows']['rows'])",
            "composition": {"inputs": {"rows": {"sources": [source]}}, "output": {"value": "integer"}}}
    broken = {"reason": "Synthetic example", "objects": [{"kind": "fragment", "name": "count_rows", "value": card}]}
    fixed = deepcopy(broken)
    fixed_card = fixed["objects"][0]["value"]
    fixed_card["composition"]["inputs"]["rows"]["sources"][0]["read"] = source["read"].replace("rstrip", "strip")
    sol.turn.side_effect = [json.dumps(reply) for reply in
                            [*([read_request] * 8), broken, read_request, fixed, fixed]]

    assert revise(study, [task], current=current)["fragment"]["count_rows"] == fixed_card
    assert read(current.folder / "proposal.json") == fixed
    calls = sol.turn.call_args_list
    assert [c.args[2].name for c in calls[-4:]] == ["turn_9", "turn_10", "turn_11", "review"]
    request = json.loads(calls[8].args[1])
    assert "Submit the smallest complete draft now" in request["instruction"]
    assert request["draft_replies_remaining"] == 4
    error = json.loads(calls[9].args[1])
    assert ("fragment/count_rows: composition.inputs.rows.sources[0].read: "
            "line 2: Unsupported policy method: .rstrip()") in error["error"]
    assert error["draft_replies_remaining"] == 3
    assert json.loads(calls[10].args[1])["doc"] == label
    study.start_workers.assert_not_called()


def test_policy_operator_error_uses_the_body_line():
    with pytest.raises(ValueError, match=r"line 2: Unsupported policy operation: Mod"):
        load_program("value = 10\nreturn value % 3")


def test_only_matching_composition_runs_can_resume(tmp_path, monkeypatch):
    from jev_harness.protocol import RESOURCES

    shared = read(RESOURCES / "experiment.json")
    settings = native_formal.configuration(shared)
    connections = {"model_nodes": {"fake": {"args": ["--model", settings["model_path"]]}}}
    study_factory = Mock()
    monkeypatch.setattr(native_formal, "SharedStudy", study_factory)
    monkeypatch.setattr(native_formal, "run_formal", Mock())
    args = SimpleNamespace(root=tmp_path, max_gpus=6, through="h0-dev")
    save(tmp_path / "configuration.json", {**settings, "protocol": "appworld_native_tools_v2"})
    save(tmp_path / "runs/dev_react/t/record.json", {"saved": True})
    with pytest.raises(ValueError, match="fresh output directory"):
        native_formal.launch(args, shared, connections)
    study_factory.assert_not_called()
    save(tmp_path / "configuration.json", settings)
    native_formal.launch(args, shared, connections)
    assert settings["train_sampling"] == "entry_harness_only"
    assert settings["optimizer"]["draft_reply_limit"] == 12
    assert settings["lanes_per_gpu"] == 2 and settings["cuda_graph_from_start"]
    assert read(tmp_path / "runs/dev_react/t/record.json") == {"saved": True}
    assert study_factory.call_count == 1


def test_old_train_index_opens_actual_events_and_excludes_dev(tmp_path):
    from jev_harness.protocol import design_manifest

    split = design_manifest()
    task = split["train_batches"][0]["tasks"][0]
    trace = tmp_path / "events.jsonl"
    trace.write_text('{"event":"task","instruction":"Synthetic example"}\n'
                     '{"event":"native_feedback","reason":"example"}\n', encoding="utf-8")
    row = dict(task_id=task, success=True, output_tokens=24, open_calls=2)
    sample = TrainSample(trace, row, native_formal.blank_library())
    doc = "history/first/" + task
    docs, panel = evidence({doc: sample})
    assert panel[doc]["success"] and "steps" not in panel[doc]
    assert "native_feedback" in str(evidence_page(docs, {"doc": doc, "start": 1, "count": 1}))
    assert evidence_page(docs, {"doc": "index"})["rows"][0]["text"] == doc
    with pytest.raises(ValueError, match="Use doc=index"):
        evidence_page(docs, {"doc": "missing"})
    with pytest.raises(ValueError, match="Only official Train"):
        evidence({"hidden": TrainSample(trace, {**row, "task_id": split["dev_tasks"][0]}, sample.library)})


def test_each_old_solved_case_is_listed_at_its_first_change_with_shared_24_budget(tmp_path, monkeypatch):
    trace = tmp_path / "prefix.jsonl"
    trace.write_text('{"event":"task","instruction":"Synthetic example"}\n', encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    save(catalog, [])
    study = Mock(workers=[Mock()])
    study.workers[0].worker.return_value = "mock-decider"
    visited = []

    def change(events, candidate, functions, execution, model=None):
        visited.append(model)
        return {"status": "reselected" if model else "needs_real_choice"}

    monkeypatch.setattr(native_replay, "first_change", change)
    cases = {str(i): (trace, catalog, {}) for i in range(26)}
    report = native_replay.solved_preview(study, {}, cases)
    assert set(report) == set(cases)
    assert visited.count("mock-decider") == 24
    assert [r["status"] for r in report.values()].count("needs_real_choice") == 2


def test_confirmed_train_trace_stays_attached_to_the_arm_that_produced_it(tmp_path):
    result = tmp_path / "batches/b01/b01r01/paired_selection.json"
    runs = []
    for arm, passed in (("parent", False), ("candidate", True)):
        trace = tmp_path / "runs" / ("b01r01_review_" + arm) / "case_1/attempt_1/trace.jsonl"
        relative = str(trace.relative_to(tmp_path))
        save(trace.parents[1] / "record.json", {"task_id": "case_1", "success": passed, "trace_path": relative})
        runs.append({"arm": arm, "confirmation_pass": 1, "trace_path": relative})
    save(result, {"repeats": {"case_1": runs}})
    parent, child = {}, {"decision": {"store__send": "new"}}
    samples = native_formal.confirmed_train_samples(tmp_path, result, parent, child)
    assert samples["b01r01_confirmation1_parent/case_1"].library == parent
    success = samples["b01r01_confirmation1_candidate/case_1"]
    assert success.row["success"] and success.library == child
