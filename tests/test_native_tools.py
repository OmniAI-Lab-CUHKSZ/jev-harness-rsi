import json
from unittest.mock import Mock

import pytest

from jev_harness.formal_runtime import BudgetExhausted
from jev_harness.native_assembly import DISCOVER, FINISH, MORE_FRAGMENTS, NativeAssembly, NativeGeneration, blank_library
from jev_harness.native_evidence import fragment_evidence
from jev_harness.native_replay import first_change
from jev_harness.native_revision import apply_edits, trial_task
from jev_harness.runtime import Trace
from jev_harness.schema_tools import (
    Observation,
    bound_plans,
    checklist_plans,
    calculations,
    check_args,
    expand_binding,
    effect_view,
    has_update,
    literal_plans,
    normalize_action,
    public_calculation,
)


def tool(name, description, fields=None, required=()):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": fields or {}, "required": list(required)},
    }


STRING = {"type": "string", "description": "Text"}
INTEGER = {"type": "integer", "description": "Number"}
TOOLS = [
    tool("store__show_orders", "Show the orders."),
    tool(
        "store__send_item",
        "Send an item to the recipient.",
        {"item_id": INTEGER, "message": STRING},
        ["item_id", "message"],
    ),
    tool("store__request_item", "Ask a recipient to send an item.", {"item_id": INTEGER}, ["item_id"]),
    tool(FINISH, "Mark the task as completed", {"answer": STRING}),
]


def choose(index, count):
    return {
        "index": index,
        "probabilities": [int(i == index) for i in range(count)],
        "mode": "D",
        "input_tokens": 11,
        "output_tokens": 0,
    }


class World:
    def __init__(self):
        self.calls = []

    def request(self, op, **kwargs):
        if op == "prepare":
            return {
                "instruction": "Read the order and send its item a personalized message.",
                "functions": TOOLS,
                "supervisor": {},
                "datetime": "2026-01-01",
            }
        if op == "evaluate":
            return {
                "success": self.calls
                == [
                    ("show_orders", {}),
                    ("send_item", {"item_id": 12, "message": "Enjoy your order!"}),
                    ("complete_task", {}),
                ],
                "failures": [],
            }
        assert op == "call"
        api, args = kwargs["api"], dict(kwargs["arguments"])
        assert args.pop("raise_on_failure")
        self.calls.append((api, args))
        result = [{"item_id": 12}] if api == "show_orders" else {"message": "success"}
        return {"value": result, "done": api == "complete_task", "api_calls": 1, "api_total": len(self.calls)}


def scripted(operation, **req):
    if operation == "decide":
        options = req["options"]
        if req["question"].startswith("Which arguments"):
            return choose(len(options) - 1, len(options))
        facts = json.loads(req["state"])
        target = (
            "Complete the task now"
            if facts["tool_receipts"]
            else "store__send_item"
            if facts["observations"]
            else "store__show_orders"
        )
        index = next(i for i, opt in enumerate(options) if opt.startswith(target))
        return choose(index, len(options))
    return {
        "mode": "G",
        "output_tokens": 15,
        "text": '{"source":"r1","arguments":{"item_id":{"field":"item_id"},"message":"Enjoy your order!"}}',
    }


def setup_engine():
    model = Mock()
    model.request.side_effect = scripted
    return NativeAssembly(
        model,
        World(),
        max_calls=12,
        step_review_mode="official_adaptive",
        step_review_threshold=0.60,
        step_review_thinking_budget=256,
    )


def test_unregistered_tool_then_local_value_then_real_decision_and_finish():
    e = setup_engine()
    row = e.run("synthetic")
    assert row["success"] and row["open_calls"] == 1 and row["native_ballots"] == 3
    assert row["question_generation_calls"] == row["candidate_generation_calls"] == 0
    assert row["model_requests"] == 4 and row["output_tokens"] == 15
    requests = e.model.request.call_args_list
    assert requests[-1].args == ("decide",)
    assert requests[0].kwargs["thinking"] == "auto" and requests[0].kwargs["think_budget"] == 256
    payload = json.loads(
        next(c.kwargs["messages"][1]["content"] for c in requests if c.args == ("generate",))
    )
    assert payload["selected_tool"]["name"] == "store__send_item" and "public_tools" not in payload


def test_single_write_needs_intent_and_none_retrieves_more_tools():
    items = [tool(f"app__show_{i}", f"Read thing {i}") for i in range(20)]
    items.append(tool("app__delete", "Delete everything"))
    e = setup_engine()
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Read thing", "functions": items}, "history": []}
    e.initialize_native({})
    pages = []

    def on_action(op, **req):
        pages.append(req["options"])
        return choose(len(pages[0]) - 1 if len(pages) == 1 else 0, len(req["options"]))

    e.model.request.side_effect = on_action
    assert e.next_action() not in pages[0] and len(pages) == 2
    e.model.request.side_effect = lambda op, **req: choose(len(req["options"]) - 1, len(req["options"]))
    assert e.action_page(["app__delete"], allow_discovery=False) is None
    assert not e.environment.calls


def test_literal_filter_and_distinct_opposite_effects():
    search = tool(
        "music__search_artists",
        "Search artists",
        {
            "genre": {**STRING, "default": None},
            "min_follower_count": INTEGER,
            "max_follower_count": INTEGER,
            "query": STRING,
        },
    )
    possibilities = literal_plans(search, "Follow all the classical artists with at least 22 followers.")
    assert {"genre": "classical", "min_follower_count": 22} in possibilities
    assert all("max_follower_count" not in args for args in possibilities)
    assert all(not ("genre" in p and p.get("genre") == p.get("query")) for p in possibilities)


def test_literal_contact_bindings_need_an_actual_phone_number_or_email_address():
    search = tool(
        "app__search_messages", "Read messages", {"phone_number": STRING, "email": STRING, "query": STRING}
    )
    assert all(
        "phone_number" not in a and "email" not in a
        for a in literal_plans(search, "Reply to the phone text messages.")
    )
    choices = literal_plans(search, "Read messages from +1 (555) 123-4567 or alex@example.org.")
    assert any(a.get("phone_number") == "+1 (555) 123-4567" for a in choices)
    assert any(a.get("email") == "alex@example.org" for a in choices)


def test_joint_record_fields_preserve_lists_and_check_runtime_types():
    source = Observation("r1", "pay__show_sent", "ignored", "outgoing")
    source.data = [
        {"amount": 4, "receiver": {"email": "a@example.org"}, "tags": ["x", "y"]},
        {"amount": 8, "receiver": {"email": "b@example.org"}, "tags": []},
    ]
    payout = tool(
        "pay__transfer",
        "Send money",
        {"receiver_email": STRING, "amount": INTEGER, "tags": {"type": "array", "items": STRING}},
        ["receiver_email", "amount", "tags"],
    )
    plans = bound_plans(payout, "Pay them", [source])
    assert plans[0][1] == [
        {"receiver_email": "a@example.org", "amount": 4, "tags": ["x", "y"]},
        {"receiver_email": "b@example.org", "amount": 8, "tags": []},
    ]
    spec = {
        "source": "r1",
        "where": {"receiver.email": "b@example.org"},
        "arguments": {"amount": {"field": "amount"}},
    }
    assert expand_binding(spec, [source]) == [{"amount": 8}]
    with pytest.raises(ValueError):
        check_args(payout, {"receiver_email": "b@example.org", "amount": True, "tags": []})
    with pytest.raises(ValueError, match="No records"):
        expand_binding({**spec, "where": {"receiver.email": "c@example.org"}}, [source])
    with pytest.raises(ValueError, match="Tied"):
        expand_binding(
            {"source": "r1", "take": "last", "order_by": "amount"},
            [Observation("r1", "x", [{"amount": 4}, {"amount": 4}], "all")],
        )


def test_historical_changed_ballot_really_reselected_and_unchanged_costs_nothing():
    e = setup_engine()
    assert e.run("synthetic")["success"]
    fake = Mock()
    fake.request.side_effect = scripted
    settings = dict(
        max_calls=12,
        step_review_threshold=0.6,
        step_review_mode="official_adaptive",
        step_review_thinking_budget=256,
    )
    lib = blank_library()
    assert first_change(e.trace.events, lib, TOOLS, settings, fake)["status"] == "unchanged"
    fake.request.assert_not_called()
    lib["decision"]["store__send_item"] = "Send the already selected object."
    result = first_change(e.trace.events, lib, TOOLS, settings, fake)
    assert result["status"] == "reselected" and result["same_displayed_choice"]
    assert fake.request.call_count == 1 and fake.request.call_args.args == ("decide",)


def test_offline_fragment_gets_a_real_source_and_no_unselected_guidance_enters_generation():
    e = setup_engine()
    e.context = {"task": {"instruction": "Find the latest order", "functions": TOOLS}}
    e.trace = Trace()
    lib = apply_edits(
        blank_library(),
        {
            "objects": [
                {
                    "kind": "fragment",
                    "name": "latest_order",
                    "value": {
                        "sources": ["store__show_orders"],
                        "label": "Get the latest order",
                        "program": "return state['rows'][-1:]",
                    },
                },
                {"kind": "decision", "name": "store__request_item", "value": "unused decision guidance"},
            ]
        },
        {t["name"]: t for t in TOOLS},
    )
    e.initialize_native(lib)
    e.observe("store__show_orders", [{"item_id": 1}, {"item_id": 2}], "all")
    assert e.fragments()[0][1] == ("fragment", "latest_order", "r1")
    assert e.program("fragment", "latest_order", e.observed[0]) == [{"item_id": 2}]
    for endpoint in ("store__send_item", None):
        e.generate_one = Mock(return_value={"search": "missing prerequisite"})
        e.local_action(endpoint)
        assert "unused decision guidance" not in str(e.generate_one.call_args)


def test_generation_arm_shares_exec_and_multiwrite_retry_resumes_at_first_failed_row():
    e = NativeGeneration(Mock(), Mock())
    e.trace = Trace()
    e.context = {"task": {"instruction": "Deliver", "functions": TOOLS}}
    e.initialize_native({})
    batch = [{"item_id": n, "message": "hi"} for n in (7, 8)]
    e.call = Mock(side_effect=[{"ok": True}, ValueError("temporary"), {"ok": True}])
    with pytest.raises(ValueError, match="temporary"):
        e.perform("store__send_item", batch, "from two rows")
    e.perform("store__send_item", batch, "from two rows")
    assert [c.args[2]["item_id"] for c in e.call.call_args_list] == [7, 8, 8]
    assert len(e.receipts) == 2
    assert e.observed[-1].data == [{"ok": True}, {"ok": True}]
    params, response = e.recorded_tool_io(
        {"api": "login", "arguments": {"password": "secret"}},
        {"output": "secret", "value": {"access_token": "secret"}, "done": False},
    )
    assert "secret" not in str((params, response))


def test_both_modes_can_compute_extrema_and_detect_ties_from_observed_values():
    rows = [
        {"name": "First", "amount": 4, "created_at": "2020-12-01"},
        {"name": "Second", "amount": 9, "created_at": "2021-02-03"},
    ]
    obs = Observation("r1", "app__show_records", rows, "a complete list")
    choices = dict(calculations(obs, "Find the latest record by creation date."))
    chosen = next(v for k, v in choices.items() if "created_at" in k)
    assert (
        chosen
        == public_calculation({"op": "argmax", "source": "r1", "field": "created_at"}, [obs])
        == rows[1]
    )
    assert public_calculation({"op": "sum", "source": "r1", "field": "amount"}, [obs]) == 13
    with pytest.raises(ValueError, match="Tied"):
        public_calculation(
            {"op": "argmax", "source": "r1", "field": "amount"},
            [Observation("r1", "read", rows[:1] * 2, "all")],
        )


def test_rejected_public_write_is_feedback_and_never_becomes_a_success_receipt():
    e = setup_engine()
    e.prepare_options()
    e.trace, e.done = Trace(), False
    e.context = {"task": {"instruction": "Send item", "functions": TOOLS}, "history": []}
    e.initialize_native({})
    e.environment = Mock()
    e.environment.request.return_value = {
        "value": "Execution failed. generated expression with SECRET\n"
        "Exception: Response status code is 409:\nItem does not exist",
        "api_calls": 1,
        "api_total": 1,
        "done": False,
    }
    from jev_harness.runtime import Uncovered

    with pytest.raises(Uncovered, match="409"):
        e.perform("store__send_item", [{"item_id": 99, "message": "Hi"}], "from public input")
    assert not e.receipts and not e.observed
    assert "SECRET" not in str(e.trace.events)
    assert all(event["event"] != "native_effect" for event in e.trace.events)
    assert e.api_total == 1


def test_public_function_json_preserves_real_tool_arguments_and_resolves_nested_binding_controls():
    send = TOOLS[1]
    args = {
        "item_id": {"field": "item_id"},
        "message": "thanks",
        "source": "r1",
        "take": "last",
        "order_by": "created_at",
    }
    request = normalize_action(
        {"function": {"name": send["name"], "arguments": json.dumps(args)}}, None, {send["name"]: send}
    )
    assert request["source"] == "r1" and request["take"] == "last" and "source" not in request["arguments"]
    # Public schema takes precedence when an application actually owns that parameter name.
    native = tool("app__copy", "Copy from source", {"source": STRING}, ["source"])
    request = normalize_action(
        {"name": native["name"], "arguments": {"source": "my file"}}, None, {native["name"]: native}
    )
    assert request["arguments"] == {"source": "my file"} and "source" not in request
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_action(
            {"tool": send["name"], "source": "r1", "arguments": {"source": "r2"}}, None, {send["name"]: send}
        )


def test_specific_public_page_is_honored_and_marked_partial():
    read = tool(
        "app__show_items",
        "Read items",
        {
            "page_index": {"type": "integer", "minimum": 0},
            "page_limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    )
    e = setup_engine()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Read items", "functions": [read]}}
    e.initialize_native({})
    e.call = Mock(return_value=[{"item_id": 5}])
    e.perform(read["name"], [{"page_index": 3, "page_limit": 1}], "explicit page")
    assert e.call.call_args.args[2] == {"page_index": 3, "page_limit": 1} and not e.observed[-1].complete
    e.perform(read["name"], [{"page_limit": 1}], "top result only")
    assert e.call.call_args.args[2] == {"page_index": 0, "page_limit": 1}
    assert e.call.call_count == 2 and not e.observed[-1].complete


def test_implicit_full_scan_and_explicit_page_index_follow_different_public_scopes():
    read = tool(
        "app__show_items",
        "Read items",
        {
            "page_index": {"type": "integer", "default": 0},
            "page_limit": {"type": "integer", "minimum": 1, "maximum": 3, "default": 2},
        },
    )
    e = setup_engine()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Read the items", "functions": [read]}}
    e.initialize_native({})
    e.call = Mock(side_effect=[[1, 2, 3], [4], [3, 4]])
    e.perform(read["name"], [{}], "entire collection")
    assert e.observed[-1].data == [1, 2, 3, 4] and e.observed[-1].complete
    e.perform(read["name"], [{"page_index": 1}], "second official page")
    assert e.call.call_args.args[2] == {"page_index": 1, "page_limit": 2} and not e.observed[-1].complete


def test_learned_join_waits_for_both_observations_and_additional_fragments_remain_reachable():
    e = setup_engine()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Get the items referenced by my notes", "functions": TOOLS}}
    spec = {
        "sources": ["store__show_orders"],
        "inputs": ["notes__show_items"],
        "label": "Select referenced items",
        "program": "ids = [n['id'] for n in state['inputs']['notes__show_items']['rows']]\nreturn [r for r in state['rows'] if r['id'] in ids]",
    }
    lib = apply_edits(
        blank_library(),
        {"objects": [{"kind": "fragment", "name": "match_notes", "value": spec}]},
        {t["name"]: t for t in TOOLS + [tool("notes__show_items", "Read notes")]},
    )
    e.initialize_native(lib)
    e.observe("store__show_orders", [{"id": 3}, {"id": 4}], "orders")
    assert not e.fragments()
    e.observe("notes__show_items", [{"id": 4}], "notes")
    assert e.fragments()[0][1] == ("fragment", "match_notes", "r1", "r2")
    assert e.program("fragment", "match_notes", e.observed[0], [e.observed[1]]) == [{"id": 4}]
    for i in range(14):
        lib["fragment"][f"other_{i}"] = {**spec, "label": f"Other transformation {i}"}
    pages = []

    def select_page(node, question, options, skill):
        pages.append(options)
        return None if len(pages) == 1 else options[-1][1]

    e.ask = select_page
    assert e.fragment_pages() == pages[-1][-1][1] and list(map(len, pages)) == [12, 3]


def test_homepage_can_offer_zero_and_browse_the_same_fragment_on_request():
    e = setup_engine()
    e.trace = Trace()
    notes, sms = "notes__show_note", "phone__search_messages"
    e.context = {"task": {"instruction": "Use my note to reply to the message about music", "functions": TOOLS}}
    e.initialize_native(blank_library())
    spec = {"sources": [notes], "inputs": [sms], "label": "Find film titles in a note for a message",
            "goal_terms": ["film", "something to watch"], "program": "return state['rows']"}
    e.lib["fragment"]["films"] = spec
    e.observe(notes, [{"title": "Film notes"}], "films")
    assert not e.fragments()
    e.observe(sms, [{"message": "I like films."}], "films")
    e.lookup = "film"
    pages = []

    def browse(node, question, options, skill, **kwargs):
        pages.append([value for _, value in options])
        return {1: DISCOVER, 2: MORE_FRAGMENTS}.get(len(pages), options[0][1])

    e.ask = browse
    chosen = e.next_action()
    assert chosen == ("fragment", "films", "r1", "r2")
    assert chosen not in pages[0] and MORE_FRAGMENTS not in pages[0]
    assert MORE_FRAGMENTS in pages[1] and chosen in pages[2]
    entry = fragment_evidence(e.trace.events, e.lib)["fragments"]["films"]
    assert entry["eligible"] == 1 and entry["recommended"] == 0
    e.ask = lambda node, question, options, skill: [value for _, value in options]
    e.instruction = "Reply from the note about something to sit down and watch"
    assert chosen in e.action_page([])
    e.instruction = "Give me the film titles"
    assert chosen in e.action_page([])
    with pytest.raises(ValueError, match="goal_terms"):
        apply_edits(blank_library(), {"objects": [{"kind": "fragment", "name": "films",
            "value": {**spec, "goal_terms": "film"}}]}, {notes: {}, sms: {}})
    del spec["goal_terms"]
    assert chosen in e.action_page([])
    e.instruction = "Find the messages in my notes"
    assert chosen not in e.action_page([])
    for name in ("second", "third"):
        e.lib["fragment"][name] = dict(spec)
    e.instruction = "Give me the film titles"
    assert len(e.fragments()) == 3
    assert len([v for v in e.action_page([]) if isinstance(v, tuple) and v[0] == "fragment"]) == 2


def test_adding_irrelevant_fragment_preserves_solved_prefix_and_new_recommendation_reselects():
    e = setup_engine()
    assert e.run("synthetic")["success"]
    candidate = blank_library()
    candidate["fragment"]["sample"] = {
        "sources": ["store__show_orders"], "label": "Select a greeting for the order",
        "program": "return state['rows']", "goal_terms": ["film"],
    }
    config = dict(max_calls=12, step_review_mode="official_adaptive", step_review_threshold=0.60,
                  step_review_thinking_budget=256)
    e.model.request.reset_mock()
    assert first_change(e.trace.events, candidate, TOOLS, config, e.model)["status"] == "unchanged"
    e.model.request.assert_not_called()
    candidate["fragment"]["sample"]["goal_terms"] = ["personalized message"]
    change = first_change(e.trace.events, candidate, TOOLS, config, e.model)
    assert change["status"] == "reselected" and change["same_semantic_choice"]
    e.model.request.assert_called_once()


def test_partial_updates_show_a_real_delta_including_false_and_zero():
    update = tool(
        "notes__update_entry",
        "Edit a note",
        {"entry_id": INTEGER, "content": STRING, "pinned": {"type": "boolean"}, "position": INTEGER},
        ["entry_id"],
    )
    assert not has_update(update, {"entry_id": 7})
    assert has_update(update, {"entry_id": 7, "pinned": False})
    assert has_update(update, {"entry_id": 7, "position": 0})
    assert has_update(TOOLS[2], {"item_id": 7})
    e = setup_engine()
    e.trace = Trace()
    e.context = {"task": {"instruction": 'Please unset "Pack tent".', "functions": [update]}}
    e.initialize_native({})
    body = "Plans\r\n- [X] Pack tent\r\n- [ ] Charge phone\r\n"
    e.observe("notes__show_entry", {"entry_id": 7, "content": body}, "one note")
    e.observe("other__show_entry", {"entry_id": 7, "content": "wrong app"}, "other app")
    plans = e.argument_plans(update["name"])
    assert len(plans) == 1
    label, batch = plans[0]
    assert batch == [{"entry_id": 7, "content": body.replace("[X]", "[ ]")}]
    delta = effect_view(update, batch[0], e.observed)
    assert delta["target"] == {"entry_id": 7} and delta["set"]["content"]["previous_ref"] == "r1"
    assert delta["set"]["content"]["offset"] == body.index("X")
    with pytest.raises(ValueError, match="editable value"):
        e.perform(update["name"], [{"entry_id": 7}], "model attempted empty update")
    assert not e.environment.calls and not e.receipts
    e.call = Mock(return_value={"message": "updated"})
    sources = e.binding_sources[label]
    e.perform(update["name"], batch, label, sources)
    assert not e.argument_plans(update["name"]) and "requested_update" in e.receipts[0]
    with pytest.raises(ValueError, match="newer update"):
        e.perform(update["name"], batch, label, sources)
    assert e.call.call_count == 1


def test_ambiguous_checklist_is_left_to_explicit_binding_and_edit_must_be_unique():
    obs = Observation("r1", "work__show_task", {"task_id": 9, "body": "[ ] Call Sam\n[ ] Call Sam"}, "task")
    update = tool("work__update_task", "Edit", {"task_id": INTEGER, "body": STRING}, ["task_id"])
    e = setup_engine()
    e.trace = Trace()
    e.context = {"task": {"instruction": 'Finish "Call Sam"', "functions": [update]}}
    e.initialize_native({})
    e.observed = [obs]
    assert e.argument_plans(update["name"]) == []
    expr = {
        "source": "r1",
        "arguments": {
            "task_id": {"field": "task_id"},
            "body": {"replace_text": {"field": "body", "old": "[ ] Call Sam", "new": "[x] Call Sam"}},
        },
    }
    with pytest.raises(ValueError, match="unique"):
        expand_binding(expr, [obs])
    obs.data["body"] = "Tomorrow\n[ ] Call Sam"
    assert expand_binding(expr, [obs]) == [{"task_id": 9, "body": "Tomorrow\n[x] Call Sam"}]


def test_already_satisfied_checkbox_is_idempotent_and_ambiguous_goal_offers_both():
    update = tool("work__update_task", "Edit", {"task_id": INTEGER, "body": STRING}, ["task_id"])
    obs = Observation("r1", "work__show_task", {"task_id": 9, "body": "[X] Call Sam"}, "task")
    done = list(checklist_plans(update, 'Mark "Call Sam" as done.', [obs]))
    assert len(done) == 1 and done[0][1][0]["body"] == "[X] Call Sam"
    ambiguous = list(checklist_plans(update, 'Change the status of "Call Sam".', [obs]))
    assert {batch[0]["body"] for _, batch, _ in ambiguous} == {"[X] Call Sam", "[ ] Call Sam"}


def test_real_flow_credits_projections_and_never_credits_a_skipped_or_rejected_write():
    e = setup_engine()
    library = blank_library()
    library["fragment"]["greeting"] = {
        "sources": ["store__show_orders"],
        "label": "Prepare the greeting",
        "goal_terms": ["personalized message"],
        "program": "return [{'item_id': state['rows'][0]['item_id'], 'message': 'Enjoy your order!'}]",
    }

    def decide(op, **req):
        if op == "decide":
            if req["question"] == "Which order did the customer mean?":
                assert req["thinking"] == "auto" and req["options"][0].startswith("r1[0]")
                assert set(json.loads(req["state"])) == {"instruction", "source", "source_complete"}
                return choose(0, len(req["options"]))
            if req["question"].startswith("Which arguments"):
                return choose(0, len(req["options"]))
            facts = json.loads(req["state"])
            source = [o["source"] for o in facts["observations"]]
            start = (
                "Complete the task now"
                if facts["tool_receipts"]
                else "store__send_item"
                if "greeting" in source
                else "Prepare the greeting"
                if source
                else "store__show_orders"
            )
            return choose(
                next(i for i, k in enumerate(req["options"]) if k.startswith(start)), len(req["options"])
            )
        raise AssertionError("This task can be compiled")

    e.model.request.side_effect = decide
    result = e.run("synthetic", library)
    assert result["success"] and result["open_calls"] == 0
    flow = fragment_evidence(e.trace.events, library)["fragments"]["greeting"]
    assert (flow["offered"], flow["chosen"], len(flow["produced"])) == (1, 1, 1)
    (use,) = flow["accepted_uses"]
    assert use["role"] == "explicit_argument_fields" and use["fields"] == {
        "item_id": "item_id",
        "message": "message",
    }
    failed = [ev for ev in e.trace.events if ev["event"] != "native_effect"]
    assert not fragment_evidence(failed, library)["fragments"]["greeting"]["accepted_uses"]

    # Adding an internal selection to a previously solved path must trigger a real reselection
    # at that boundary, without comparing the chosen order to the old next tool.
    initial = library["fragment"]["greeting"]
    revised = apply_edits(library, {"objects": [{"kind": "fragment", "name": "greeting", "value": {
        **initial,
        "choice": {"source": "store__show_orders", "fields": ["item_id"],
                   "question": "Which order did the customer mean?"},
        "program": "if 'selected' not in state:\n    return {'$choose': [0]}\n"
                   "return [{'item_id': state['selected']['value']['item_id'], 'message': 'Enjoy your order!'}]",
    }}]}, {t["name"]: t for t in TOOLS})
    config = dict(max_calls=12, step_review_mode="official_adaptive", step_review_threshold=0.60,
                  step_review_thinking_budget=256)
    assert first_change(e.trace.events, library, TOOLS, config)["status"] == "unchanged"
    before = first_change(e.trace.events, revised, TOOLS, config)
    assert before["status"] == "needs_real_choice" and before["new_node"] == "native/fragment_choice"
    after = first_change(e.trace.events, revised, TOOLS, config, e.model)
    assert after["status"] == "reselected" and after["comparison"] == "different_decision_roles"
    assert after["same_displayed_choice"] is None and "same_semantic_choice" not in after
    e = setup_engine()
    e.model.request.side_effect = decide
    new = e.run("synthetic", revised)
    assert new["success"] and new["native_ballots"] == result["native_ballots"] + 1
    assert new["open_calls"] == new["candidate_generation_calls"] == new["question_generation_calls"] == 0
    assert first_change(e.trace.events, revised, TOOLS, config)["status"] == "unchanged"
    (slot,) = fragment_evidence(e.trace.events, revised)["fragments"]["greeting"]["internal_choices"]
    assert slot["status"] == "resolved" and slot["ref"] == "r1" and slot["row"] == 0


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [True], [1], list(range(16)), None])
def test_unusable_fragment_rows_do_not_call_jev_or_become_observations(indices):
    e = setup_engine()
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Choose an order", "functions": TOOLS}}
    spec = {"sources": ["store__show_orders"], "label": "Order",
            "choice": {"source": "store__show_orders", "fields": ["item_id"], "question": "Which order?"},
            "program": "return {'$choose': " + repr(indices) + "}"}
    e.initialize_native({"fragment": {"order": spec}})
    obs = Observation("r1", "store__show_orders", [{"item_id": 12}], "orders")
    assert e.resolve_fragment("order", obs) is None
    assert e.trace.events[-1]["status"] == "unavailable"
    e.model.request.assert_not_called()


def test_fragment_none_abstains_even_if_there_is_only_one_offered_row():
    e = setup_engine()
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Use the other order", "functions": TOOLS}}
    spec = {"sources": ["store__show_orders"], "label": "Order",
            "choice": {"source": "store__show_orders", "fields": ["item_id"], "question": "Which order?"},
            "program": "return {'$choose': [0]}"}
    e.initialize_native({"fragment": {"order": spec}})
    e.model.request.side_effect = lambda op, **req: choose(len(req["options"]) - 1, len(req["options"]))
    obs = Observation("r1", "store__show_orders", [{"item_id": 12}], "orders")
    assert e.resolve_fragment("order", obs) is None
    assert e.trace.events[-1]["status"] == "unbound" and e.model.request.call_count == 1


def test_fragment_can_ask_again_using_only_the_relevant_previous_record():
    source = "store__show_orders"
    slot = {"source": source, "fields": ["item_id"], "question": "Which order is referred to?"}
    spec = {
        "sources": [source], "label": "Resolve related orders", "choices": {"start": slot, "related": slot},
        "program": """answers = state.get('selections', [])
if not answers:
    return {'$choose': {'role': 'start', 'rows': [0, 1]}}
if len(answers) <= 2:
    reference = answers[-1]['row']
    return {'$choose': {'role': 'related', 'rows': [0, 1], 'using': {'start': reference}}}
return [a['value']['item_id'] for a in answers]""",
    }
    library = apply_edits(blank_library(), {"objects": [{"kind": "fragment", "name": "related", "value": spec}]},
                          {t["name"]: t for t in TOOLS})
    obs = Observation("r1", source, [{"item_id": 4, "private_context": "irrelevant"}, {"item_id": 7}], "guess")
    e = setup_engine()
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Find the related orders", "functions": TOOLS}}
    e.initialize_native(library)
    e.model.request.side_effect = [choose(1, 3), choose(0, 3), choose(1, 3)]
    assert e.resolve_fragment("related", obs) == [7, 4, 7]
    views = [json.loads(c.kwargs["state"]) for c in e.model.request.call_args_list]
    assert "reference_records" not in views[0]
    assert [v["reference_records"] for v in views[1:]] == [
        [{"source": source, "ref": "r1", "row": i, "fields": {"item_id": item}}]
        for i, item in [(1, 7), (0, 4)]
    ]
    assert all("guess" not in json.dumps(v) and "private_context" not in json.dumps(v) for v in views)
    steps = fragment_evidence(e.trace.events, library)["fragments"]["related"]["internal_choices"]
    assert [(step["role"], step["status"]) for step in steps] == [
        ("start", "continued"), ("related", "continued"), ("related", "resolved")
    ]

    # The third question consumes the same task budget as the first two.
    e.prepare_options()
    e.trace = Trace()
    e.max_calls = 2
    e.model.request.reset_mock()
    e.model.request.side_effect = [choose(1, 3), choose(0, 3)]
    with pytest.raises(BudgetExhausted):
        e.resolve_fragment("related", obs)
    assert e.model.request.call_count == 2


def test_fragment_does_not_pay_twice_for_an_answered_question_with_reordered_rows():
    e = setup_engine()
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"instruction": "Pick an order", "functions": TOOLS}}
    source = "store__show_orders"
    spec = {"sources": [source], "label": "Order", "choice": {
        "source": source, "fields": ["item_id"], "question": "Which order?"},
        "program": "return {'$choose': [1,0] if state.get('selected') else [0,1]}"}
    e.initialize_native({"fragment": {"order": spec}})
    e.model.request.side_effect = lambda op, **req: choose(0, len(req["options"]))
    assert e.resolve_fragment("order", Observation("r1", source, [{"item_id": 0}, {"item_id": 1}], "")) is None
    assert e.model.request.call_count == 1 and e.trace.events[-1]["status"] == "repeated"


def test_historical_replay_reaches_the_first_changed_internal_question_after_matching_the_first():
    e = setup_engine()
    source = "store__show_orders"
    slot = {"source": source, "fields": ["item_id"], "question": "Which record?"}
    spec = {"sources": [source], "label": "Prepare the greeting", "goal_terms": ["personalized message"], "choices": {
        "target": slot, "item": slot, "owner": slot},
        "program": """answers = state.get('selections', [])
if not answers:
    return {'$choose': {'role': 'target', 'rows': [0]}}
if len(answers) == 1:
    return {'$choose': {'role': 'item', 'rows': [0], 'using': {'target': answers[0]['row']}}}
return [{'item_id': state['selected']['value']['item_id'], 'message': 'Enjoy your order!'}]"""}
    lib = blank_library()
    lib["fragment"]["greeting"] = spec

    def decide(op, **req):
        assert op == "decide"
        options = req["options"]
        if req["question"] == "Which record?" or req["question"].startswith("Which arguments"):
            return choose(0, len(options))
        facts = json.loads(req["state"])
        sources = [obs["source"] for obs in facts["observations"]]
        label = ("Complete the task now" if facts["tool_receipts"] else "store__send_item" if "greeting" in sources
                 else "Prepare the greeting" if sources else "store__show_orders")
        return choose(next(i for i, option in enumerate(options) if option.startswith(label)), len(options))

    e.model.request.side_effect = decide
    assert e.run("synthetic", lib)["success"]
    config = dict(max_calls=12, step_review_mode="official_adaptive", step_review_threshold=0.60,
                  step_review_thinking_budget=256)
    events = e.trace.events
    assert first_change(events, lib, TOOLS, config)["status"] == "unchanged"
    before = [i for i, v in enumerate(events) if v["event"] == "model" and v["request"].get("question") == "Which record?"]
    assert len(before) == 2
    child = {**lib, "fragment": {"greeting": {**spec, "program": spec["program"].replace("'role': 'item'", "'role': 'owner'")}}}
    e.model.request.reset_mock()
    change = first_change(events, child, TOOLS, config, e.model)
    assert change["event"] == before[1] and change["status"] == "reselected"
    assert change["comparison"] == "different_decision_roles"
    assert change["previous_question"]["role"] == "item" and change["new_question"]["role"] == "owner"
    assert change["same_displayed_choice"] is None and "same_semantic_choice" not in change
    assert e.model.request.call_count == 1


def test_skipped_fragment_can_disagree_with_an_officially_accepted_literal():
    lib = blank_library()
    lib["fragment"]["shortlist"] = {
        "sources": ["app__show_catalogue"],
        "label": "Shortlist",
        "program": "return [{'message': state['rows'][0]['title']}]",
    }
    events = [
        {"event": "task", "instruction": "Send titles"},
        {"event": "native_protocol", "version": "v2"},
        {
            "event": "native_observation",
            "ref": "r1",
            "source": "app__show_catalogue",
            "value": {"title": "A"},
            "scope": "public data",
            "complete": True,
            "count": 1,
        },
        {
            "event": "decision",
            "node": "native/action",
            "selected": 1,
            "candidates": [
                {"label": "shortlist", "value": ["fragment", "shortlist", "r1"]},
                {"label": "send", "value": "app__send"},
            ],
        },
        {"event": "native_operation", "tool": "app__send", "arguments": [{"message": "A, B"}]},
        {
            "event": "native_effect",
            "tool": "app__send",
            "batch_index": 0,
            "sources": {},
            "effect": {"message": "A, B"},
        },
    ]
    f = fragment_evidence(events, lib)["fragments"]["shortlist"]
    assert not f["chosen"] and not f["produced"] and not f["accepted_uses"]
    assert f["skipped_output_checks"][0]["comparable_fields"]["message"] == {
        "same": False,
        "fragment_value": "A",
        "submitted": "A, B",
    }
    spec = lib["fragment"]["shortlist"]
    spec["choice"] = {"source": "app__show_catalogue", "fields": ["title"], "question": "Which title?"}
    spec["program"] = "return {'$choose': [0]}"
    flow = fragment_evidence(events, lib)["fragments"]["shortlist"]
    assert flow["skipped_output_checks"][0]["status"] == "needs_fragment_choice"
    assert not flow["produced"] and not flow["accepted_uses"]


def test_scope_and_program_inputs_are_distinguished_from_proven_field_use():
    lib = blank_library()
    lib["fragment"]["sample"] = {
        "sources": ["app__show"],
        "label": "Sample",
        "program": "return state['rows']",
    }
    events = [
        {"event": "native_protocol", "version": "v2"},
        {
            "event": "native_observation",
            "source": "sample",
            "ref": "r1",
            "count": 1,
            "scope": "records",
            "complete": True,
            "value": [{"id": 3}],
        },
        {
            "event": "native_observation",
            "source": "inspection",
            "ref": "r2",
            "count": 1,
            "scope": "project",
            "complete": False,
            "value": 3,
            "sources": {"r1": {"fields": {"field": "id"}}},
        },
    ]
    for ref, detail in [
        ("r1", {"scope_only": True}),
        ("r1", {"program_input": "join"}),
        ("r2", {"fields": {"answer": ""}}),
    ]:
        events.extend(
            [
                {"event": "native_operation", "tool": FINISH, "arguments": [{"answer": 3}]},
                {
                    "event": "native_effect",
                    "tool": FINISH,
                    "batch_index": 0,
                    "effect": {"answer": 3},
                    "sources": {ref: detail},
                },
            ]
        )
    uses = fragment_evidence(events, lib)["fragments"]["sample"]["accepted_uses"]
    assert [x["role"] for x in uses] == [
        "row_scope_only",
        "input_to_binding_program",
        "upstream_of_bound_source",
    ]
    literal = {
        "event": "native_effect",
        "tool": FINISH,
        "batch_index": 0,
        "effect": {"answer": 3},
        "sources": {},
    }
    identical = [
        *events[:3],
        {"event": "native_operation", "tool": FINISH, "arguments": [{"answer": 3}]},
        literal,
    ]
    assert fragment_evidence(identical, lib)["fragments"]["sample"]["accepted_uses"] == []


def test_draft_trial_selects_only_an_affected_current_training_task():
    parent = blank_library()
    lib = blank_library()
    lib["fragment"]["joint"] = {"sources": ["app__show_x"], "inputs": ["app__show_y"]}
    panel = {
        "h0_decision/a": {"success": False, "steps": [], "observed_sources": ["app__show_x"]},
        "h0_decision/b": {"success": True, "steps": [], "observed_sources": ["app__show_x", "app__show_y"]},
    }
    impacts = {k: {"status": "unchanged"} for k in ("a", "b")}
    assert trial_task(["a"], impacts, panel, parent, lib) is None
    assert trial_task(["a", "b"], impacts, panel, parent, lib) == "b"


def test_score_command_runs_three_independent_arms_without_editor(tmp_path, monkeypatch):
    import argparse
    from formal_run import read, save
    from jev_harness import native_pilot

    root = tmp_path / "scores"
    parent = blank_library()
    child = blank_library()
    child["decision"]["store__send_item"] = "Deliver the selected order."
    save(tmp_path / "entry.json", parent)
    save(tmp_path / "candidate.json", child)
    calls = []

    class Study:
        def __init__(self, *args, **kwargs):
            self.groups = {"local:0": object()}

        def start_workers(self):
            pass

        def paired_model_info(self):
            pass

        def status(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def execute_preferred(self, jobs, fn, affinity):
            return [fn(job, object()) for job in jobs]

        def run_task(self, task, stage, arm, library, workers):
            calls.append((task, stage, arm, library))
            fn = root / "runs" / stage / task / "attempt_1" / "trace.jsonl"
            fn.parent.mkdir(parents=True)
            fn.write_text(json.dumps({"event": "task", "instruction": "Read"}) + "\n", encoding="utf-8")
            row = {
                k: 0
                for k in (
                    "output_tokens",
                    "open_calls",
                    "native_ballots",
                    "open_output_tokens",
                    "decision_output_tokens",
                    "api_calls",
                    "seconds",
                )
            }
            row.update(
                task_id=task, success=True, error=None, gpu="local:0", trace_path=str(fn.relative_to(root))
            )
            save(fn.parent.parent / "record.json", row)
            return row

    monkeypatch.setattr(native_pilot, "NativeStudy", Study)
    parser = argparse.ArgumentParser()
    native_pilot.configure(parser)
    args = parser.parse_args(
        [
            "--connections",
            "x",
            "--root",
            str(root),
            "--train",
            "07b42fd_1",
            "--check",
            "cf6abd2_2",
            "--phase",
            "score",
            "--entry-library",
            str(tmp_path / "entry.json"),
            "--candidate-library",
            str(tmp_path / "candidate.json"),
        ]
    )
    native_pilot.run_pilot(args, {"seed": 10, "execution": {}}, {"model_nodes": {}})
    assert len(calls) == len(read(root / "pilot_results.json")) == 6
    assert all(lib == child if stage == "candidate" else lib == parent for _, stage, _, lib in calls)
    assert len(list((root / "runs").glob("*/*/fragment_flow.json"))) == 6


def test_composed_reads_and_cards_form_one_scoped_answer_without_generation():
    from jev_harness.composed_assembly import ComposedAssembly

    tools = [
        tool("store__show_orders", "List owned orders."),
        tool("store__show_order", "Read order details.", {"order_id": INTEGER}, ["order_id"]),
        tool(FINISH, "Answer or complete.", {"answer": STRING}),
    ]
    task = {"task_id": "synthetic", "datetime": "2026-01-01", "supervisor": {}, "functions": tools,
            "instruction": "What is the title of my largest order?"}
    calls = []

    def environment(op, **params):
        if op == "prepare":
            return task
        if op == "evaluate":
            return {"success": calls[-1] == ("complete_task", {"answer": "Large"}), "failures": []}
        assert op == "call"
        endpoint, args = params["api"], dict(params["arguments"])
        args.pop("raise_on_failure")
        calls.append((endpoint, args))
        if endpoint == "show_orders":
            value = [{"order_id": 1}, {"order_id": 2}]
        elif endpoint == "show_order":
            value = {"title": "Large" if args["order_id"] == 2 else "Small", "size": args["order_id"]}
        else:
            value = {"status": "success"}
        return {"value": value, "done": endpoint == "complete_task", "api_total": len(calls)}

    def model(op, **req):
        assert op == "decide"
        opt = req["options"]
        start = "collect the owned orders" if not calls else "find the largest order" if len(calls) == 1 else "Complete the task"
        return choose(next(i for i, label in enumerate(opt) if label.startswith(start)), len(opt))

    lib = blank_library()
    lib["fragment"] = {
        "order_ids": {
            "label": "collect the owned orders",
            "sources": ["store__show_orders"],
            "reads": {"store__show_orders": "return [{}]"},
            "produces": {"order_id": "integer"},
            "program": 'return state["rows"] if state["arguments"] == [{}] else None',
        },
        "largest_order": {
            "label": "find the largest order",
            "goal_terms": ["largest order"],
            "sources": ["order_ids"],
            "inputs": ["store__show_order"],
            "reads": {"store__show_order": 'return state["inputs"]["order_ids"]["rows"]'},
            "produces": {"title": "string"},
            "program": 'rows = state["inputs"]["store__show_order"]["rows"]\n'
                       'return {"title": max((r["size"], r["title"]) for r in rows)[1]}',
        },
    }
    executor = ComposedAssembly(Mock(request=model), Mock(request=environment))
    row = executor.run("synthetic", lib)
    assert row["success"] and row["open_calls"] == 0 and row["native_ballots"] == 3
    assert calls == [("show_orders", {}), ("show_order", {"order_id": 1}), ("show_order", {"order_id": 2}),
                     ("complete_task", {"answer": "Large"})]
    flow = fragment_evidence(executor.trace.events, lib)
    assert flow["fragments"]["order_ids"]["accepted_uses"][0]["role"] == "upstream_of_bound_source"
    assert first_change(executor.trace.events, lib, tools, {}, executor_type=ComposedAssembly)["status"] == "unchanged"


def test_composed_effects_share_scope_and_require_both_receipts():
    from jev_harness.composed_assembly import ComposedAssembly

    fields = {"payment_id": INTEGER}
    tools = [tool("store__like_payment", "Like a payment.", fields, fields),
             tool("store__comment_payment", "Comment on a payment.", fields, fields),
             tool(FINISH, "Complete the task.")]
    goal = 'Comment on and like all payments from my coworkers.'
    library = blank_library()
    library["state"]["effect_intents"] = {
        "label": "recognize two independent writes", "sources": [], "intent": True,
        "program": 'return ["store__comment_payment", "store__like_payment"]'
    }
    for name in ("store__comment_payment", "store__like_payment"):
        library["binding"][name] = {
            "sources": ["eligible_payments"], "covers": True,
            "label": "Only payments from the scoped set", "program": 'return state["rows"]',
        }

    def respond(op, **req):
        assert op == "call"
        return {"value": {"message": "accepted"}, "done": False, "api_calls": 1}

    model = Mock()
    model.request.side_effect = lambda op, **req: choose(0, len(req["options"]))
    executor = ComposedAssembly(model, Mock(request=respond))
    executor.prepare_options()
    executor.trace, executor.done = Trace(), False
    executor.context = {"task": {"functions": tools, "instruction": goal}, "history": []}
    executor.initialize_native(library)
    assert executor.action_page(["store__like_payment", "store__comment_payment"], allow_discovery=False) is None
    assert not model.request.called
    executor.observe("eligible_payments", [{"payment_id": 10}, {"payment_id": 20}], "complete coworker join")
    assert executor.action_page([], allow_discovery=False) == "store__comment_payment"
    executor.perform(*executor.binding("store__comment_payment"))
    assert not executor.ready_to_finish()
    assert executor.action_page([], allow_discovery=False) == "store__like_payment"
    executor.perform(*executor.binding("store__like_payment"))
    assert executor.ready_to_finish()
    assert [(r["tool"], r["arguments"]["payment_id"]) for r in executor.receipts] == [
        ("store__comment_payment", 10), ("store__comment_payment", 20),
        ("store__like_payment", 10), ("store__like_payment", 20),
    ]


def test_composed_read_delegation_resumes_and_preserves_existing_observations():
    from jev_harness.composed_assembly import ComposedAssembly

    captured = []
    tools = [tool("store__show_orders", "List orders."), tool(FINISH, "Finish.")]

    def environment(op, **req):
        captured.append((op, req))
        if op == "bind_progress":
            return {"names": ["observations", "receipts"]}
        return {"value": [{"order_id": 4}], "done": False, "api_calls": 1}

    def model(op, **req):
        assert op == "generate"
        return {"mode": "G", "text": 'yield_read("store__show_orders", {})', "output_tokens": 12}

    executor = ComposedAssembly(Mock(request=model), Mock(request=environment))
    executor.prepare_options()
    executor.trace, executor.seed, executor.done = Trace(), 0, False
    executor.context = {"task": {"functions": tools, "instruction": "Read orders", "supervisor": {},
                                "datetime": "2026-01-01"}, "history": []}
    executor.initialize_native(blank_library())
    executor.observe("calculation", {"count": 2}, "prior scoped count")
    executor.residual("A different source is needed")
    assert captured[0] == ("bind_progress", {"values": {"observations": {"r1": {"count": 2}}, "receipts": []}})
    assert captured[1][0] == "call"
    assert executor.observed[-1].source == "store__show_orders"
    assert executor.trace.events[-1]["event"] == "native_residual_reentry"


def test_composed_editor_accepts_typed_acquisition_and_blocks_writes_from_read_ports():
    tools = {t["name"]: t for t in TOOLS}
    spec = {"label": "Read open orders", "sources": ["store__show_orders"],
            "produces": {"item_id": "integer"}, "reads": {"store__show_orders": "return [{}]"},
            "program": 'return state["rows"]'}
    edit = {"objects": [{"kind": "fragment", "name": "open_orders", "value": spec}]}
    assert apply_edits(blank_library(), edit, tools, composed=True)["fragment"]["open_orders"] == spec
    spec.update(sources=["store__send_item"], reads={"store__send_item": "return [{}]"})
    with pytest.raises(ValueError, match="declared public read source"):
        apply_edits(blank_library(), edit, tools, composed=True)


def test_dynamic_editor_and_menu_ground_a_single_plan_without_extra_model_questions():
    from jev_harness.composed_assembly import ComposedAssembly

    spec = {
        "label": "Read order for the requested item",
        "sources": ["store__show_orders"],
        "program": 'return state["reads"]["orders"]',
        "dynamic": {
            "plan": ('return [{"result":"item", "choices":{"scope":"ordered items"}, '
                     '"reads":{"orders":{"source":"store__show_orders", "arguments":[{}]}}}]'),
            "slots": {"scope": {"question": "Which items?"}},
            "results": {"item": {"fields": {"item_id": "integer"}, "writes": {
                "store__send_item": {"fields": {"item_id": "item_id"},
                                     "arguments": 'return {"message":"Enjoy your order!"}'}
            }}},
        },
    }
    proposal = {"objects": [
        {"kind": "fragment", "name": "order_items", "value": spec},
        {"kind": "decision", "name": "order_items", "value": "Use actual orders."},
        {"kind": "state", "name": "send_intent", "value": {
            "label": "Requested item delivery", "sources": [], "intent": True,
            "program": 'return ["store__send_item"] if "send" in state["goal"] else []',
        }},
    ]}
    tools = {t["name"]: t for t in TOOLS}
    library = apply_edits(blank_library(), proposal, tools, composed=True)
    engine = ComposedAssembly(Mock(), World())
    engine.trace, engine.done = Trace(), False
    engine.context = {"task": {"functions": TOOLS, "instruction": "Send the item from the order"}, "history": []}
    engine.initialize_native(library)
    assert any("Use actual orders." in label for _, _, label, _ in engine.card_plans())
    engine.ask = Mock(side_effect=AssertionError("There is only one scope to bind"))
    engine.observe("store__show_orders", [{"item_id": 12}], "Published orders", arguments=[{}])
    engine.execute_card(("dynamic", "order_items"))
    (endpoint, batch, _, source), = engine.compiled_bindings("store__send_item")
    assert endpoint == "store__send_item"
    assert batch == [{"item_id": 12, "message": "Enjoy your order!"}]
    assert source == {"r2": {"fields": {"item_id": "item_id"}}}
    engine.ask.assert_not_called()

    engine.context["task"]["instruction"] = "Review the order's phone number"
    engine.initialize_native(library)
    assert not engine.card_plans()


def test_calendar_scope_distinguishes_no_restriction_from_unsupported_dates():
    from jev_harness.policy_programs import calendar_windows

    when = "2023-06-01 12:00:00"
    assert calendar_windows("Approve all pending requests", when) == [(None, None)]
    assert calendar_windows("Payments in the last 7 days, including today", when) == [("2023-05-26", "2023-06-01")]
    assert calendar_windows("Payments since 1st Feb of this year", when) == [("2023-02-01", "2023-06-01")]
    assert calendar_windows("Payments in the ongoing year", when) == [("2023-01-01", "2023-06-01")]
    assert calendar_windows("Payments in the current month", when) == [("2023-06-01", "2023-06-01")]
    assert calendar_windows("Payments after the last company outing", when) == []


def test_endpoint_independent_filter_preserves_both_transaction_directions_and_original_fields():
    from jev_harness.composed_assembly import ComposedAssembly

    contacts, payments, like = "phone__search_contacts", "venmo__show_transactions", "venmo__like_transaction"
    tools = [
        tool(contacts, "Search phone contacts", {"relationship": STRING}),
        tool(payments, "Show transactions", {
            "direction": {**STRING, "enum": ["sent", "received"]}, "min_created_at": STRING,
        }),
        tool(like, "Like a transaction", {"transaction_id": INTEGER}, ["transaction_id"]),
    ]
    proposal = {"objects": [{"kind": "fragment", "name": "match_people", "value": {
        "label": "Match counterparties to the selected contacts",
        "composition": {"inputs": {"rows": "counterparty_records", "people": "contact_identities"},
                        "output": "matching_records"},
        "program": (
            'emails = {person[state["people_key"][0]] for person in state["people"]}\n'
            'actor, field = state["record_key"]\n'
            'return [row for row in state["rows"] if row[actor][field] in emails]'
        ),
    }}]}
    parent = blank_library()
    child = apply_edits(parent, proposal, {t["name"]: t for t in tools}, composed=True)
    model = Mock(request=lambda operation, **request: choose(0, len(request["options"])))
    engine = ComposedAssembly(model, Mock())
    engine.trace = Trace()
    engine.prepare_options()
    task = {"task_id": "example", "datetime": "2023-06-01", "supervisor": {},
            "instruction": "Like all Venmo transactions in the ongoing year to and from my roommates."}
    engine.seed, engine.done = 0, False
    engine.trace.add("task", **task, seed=engine.seed)
    engine.context = {"task": {**task, "functions": tools}, "history": []}

    engine.initialize_native(parent)
    engine.next_action()
    change = first_change(engine.trace.events, child, tools, {}, model,
                          executor_type=ComposedAssembly, continue_after_choice=True)
    assert change["status"] == "reselected" and change["continuation"]["after"] == "call"
    assert change["proposed_reads"][0]["sources"] == [contacts, payments, payments]

    engine.ask = Mock(return_value=payments)
    engine.action_page([payments])
    prior_request = engine.ask.call_args
    engine.initialize_native(child)
    engine.ask = Mock(side_effect=[None, payments])
    assert engine.action_page([payments]) == payments
    offered = {value: label for label, value in engine.ask.call_args_list[0].args[2]}
    assert "Acquires the scoped phone and venmo inputs itself" in offered[("dynamic", "match_people")]
    assert "Read arguments already bound." in offered[("dynamic", "match_people")]
    assert "Contact matching still remains." in offered[payments]
    assert "another choice or generated arguments" in offered[payments]
    assert engine.ask.call_args_list[0].args[3] == prior_request.args[3]
    assert engine.ask.call_args == prior_request
    assert not engine.notices

    engine.initialize_native(child)
    engine.ask = Mock(side_effect=AssertionError("All of these values are determined by the task"))
    engine.observe(contacts, [{"email": "roommate@example.test"}], "Roommates",
                   arguments=[{"relationship": "roommate"}])
    sent = {"transaction_id": 1, "receiver": {"email": "roommate@example.test"},
            "created_at": "2023-04-22", "amount": 5}
    received = {"transaction_id": 2, "sender": {"email": "roommate@example.test"},
                "created_at": "2023-05-28", "amount": 9}
    engine.observe(payments, [sent], "Sent transactions",
                   arguments=[{"direction": "sent", "min_created_at": "2023-01-01"}])
    engine.observe(payments, [received, {**received, "transaction_id": 3, "created_at": "2022-12-31"}],
                   "Received transactions", arguments=[{"direction": "received", "min_created_at": "2023-01-01"}])
    assert engine.card_plans()[0][3] == ("dynamic", "match_people")
    engine.execute_card(("dynamic", "match_people"))
    assert engine.observed[-1].rows == [sent, received]
    assert engine.compiled_bindings(like)[0][1] == [{"transaction_id": 1}, {"transaction_id": 2}]

    engine.context["task"]["instruction"] = "I would like all the Venmo transactions with my roommates."
    engine.initialize_native(child)
    assert not engine.card_plans()


@pytest.mark.parametrize("planner", ['return None', 'return [{"contact_id": "invalid"}]'])
def test_composed_prechecks_a_companion_before_showing_the_card(planner):
    from jev_harness.composed_assembly import ComposedAssembly

    tools = [*TOOLS, tool("store__show_contact", "Read one contact.", {"contact_id": INTEGER}, ["contact_id"])]
    e = ComposedAssembly(Mock(), Mock())
    e.trace = Trace()
    e.context = {"task": {"functions": tools, "instruction": "Read order and contact"}, "history": []}
    library = blank_library()
    library["fragment"]["read_contact"] = {
        "label": "Select order contact", "sources": ["store__show_orders"], "inputs": ["store__show_contact"],
        "reads": {"store__show_contact": planner}, "program": 'return state["rows"]',
    }
    e.initialize_native(library)
    e.observe("store__show_orders", [{"order_id": 1}], "Complete order read")
    assert not e.card_plans()
    assert not e.environment.request.called
    assert not e.model.request.called


@pytest.mark.parametrize("matched_ids", [[12, 12, 13], []])
def test_composed_projected_sets_drive_official_reads_and_writes(matched_ids):
    from jev_harness.composed_assembly import ComposedAssembly

    tools = [
        tool("store__show_orders", "List orders."),
        tool("store__show_order", "Read details of an order.", {"order_id": INTEGER}, ["order_id"]),
        tool("store__send_item", "Send the matched item.", {"item_id": INTEGER}, ["item_id"]),
        tool(FINISH, "Complete the task."),
    ]
    calls = []
    task = {"task_id": "synthetic", "functions": tools, "instruction": "Send all matched order items.",
            "datetime": "2026-01-01", "supervisor": {}}
    effects = [("send_item", {"item_id": n}) for n in sorted(set(matched_ids))]

    def environment(op, **req):
        if op == "prepare":
            return task
        if op == "evaluate":
            return {"success": [c for c in calls if c[0] == "send_item"] == effects, "failures": []}
        name = req["api"]
        args = {k: v for k, v in req["arguments"].items() if k != "raise_on_failure"}
        calls.append((name, args))
        data = {"message": "Done"}
        if name == "show_orders":
            data = [{"order_id": n} for n in matched_ids or [99]]
        if name == "show_order":
            data = {"item": {"id": args["order_id"]}}
        return {"value": data, "done": name == "complete_task", "api_total": len(calls)}

    def model(op, **req):
        assert op == "decide"
        options = req["options"]
        for prefix in ("Filter order items", "store__send_item:", "Complete the task", "Read all order IDs"):
            index = next((i for i, opt in enumerate(options) if opt.startswith(prefix)), None)
            if index is not None:
                return choose(index, len(options))
        raise AssertionError(options)

    library = blank_library()
    library["fragment"] = {
        "order_ids": {
            "label": "Read all order IDs", "sources": ["store__show_orders"], "produces": {"order_id": "integer"},
            "reads": {"store__show_orders": "return [{}]"},
            "program": 'return state["rows"]',
        },
        "matched": {
            "label": "Filter order items", "sources": ["order_ids"], "inputs": ["store__show_order"],
            "reads": {"store__show_order": {"source": "order_ids", "fields": {"order_id": "order_id"}}},
            "produces": {"item_id": "integer"},
            "program": 'return [{"item_id": r["item"]["id"]} for r in state["inputs"]["store__show_order"]["rows"]'
                       ' if r["item"]["id"] != 99]',
        },
    }
    library["binding"]["store__send_item"] = {
        "label": "Send matched items", "covers": True, "sources": ["matched"], "fields": {"item_id": "item_id"},
    }
    library["state"]["intent"] = {
        "label": "Send matched orders", "intent": True, "sources": [], "program": 'return ["store__send_item"]',
    }
    apply_edits(library, {"objects": []}, {t["name"]: t for t in tools}, composed=True)

    e = ComposedAssembly(Mock(request=model), Mock(request=environment))
    row = e.run("synthetic", library)
    assert row["success"] and row["open_calls"] == 0 and row["native_ballots"] == 4
    read_ids = sorted(set(matched_ids)) or [99]
    assert calls == [
        ("show_orders", {}), *(("show_order", {"order_id": n}) for n in read_ids),
        *effects, ("complete_task", {}),
    ]
    if matched_ids:
        event = next(t for t in e.trace.events if t["event"] == "native_effect" and t["tool"] == "store__send_item")
        result = next(t for t in e.trace.events if t["event"] == "native_card" and t["name"] == "matched")
        assert event["sources"] == {result["output"]: {"fields": {"item_id": "item_id"}}}
    else:
        assert any(t["event"] == "native_empty_scope" for t in e.trace.events)


def test_composed_asks_jev_when_complete_projections_disagree():
    from jev_harness.composed_assembly import ComposedAssembly

    tools = [tool("store__send_item", "Send an item.", {"item_id": INTEGER}, ["item_id"])]
    model = Mock()
    model.request.side_effect = [choose(0, 2), choose(1, 3)]
    e = ComposedAssembly(model, Mock())
    e.prepare_options()
    e.trace = Trace()
    e.context = {"task": {"functions": tools, "instruction": "Send the requested items."}, "history": []}
    library = blank_library()
    library["binding"]["store__send_item"] = {
        "label": "Send scoped items", "covers": True, "sources": ["matched"], "fields": {"item_id": "item_id"},
    }
    library["state"]["intent"] = {
        "label": "Send requested items", "intent": True, "sources": [], "program": 'return ["store__send_item"]',
    }
    e.initialize_native(library)
    e.observe("matched", [{"item_id": 12}, {"item_id": 18}], "Order A")
    e.observe("matched", [{"item_id": 18}, {"item_id": 12}], "Order A in a different order")
    e.observe("matched", [{"item_id": 15}], "Order B")
    endpoint = e.action_page([], allow_discovery=False)
    assert endpoint == "store__send_item"
    assert e.binding(endpoint)[1:] == (
        [{"item_id": 18}, {"item_id": 12}], "Send scoped items", {"r2": {"fields": {"item_id": "item_id"}}}
    )
    assert model.request.call_count == 2 and not e.environment.request.called
