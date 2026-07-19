import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from agent.outcome_events import query_outcome_events
from agent.tracing import TraceRecorder
from hermes_state import SessionDB
from run_agent import AIAgent


def _tool_call(call_id, name="terminal", **arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _message(*calls):
    return SimpleNamespace(tool_calls=list(calls))


@pytest.fixture
def agent(tmp_path):
    tool_defs = [
        {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}
        for name in ("terminal", "read_file")
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        value = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    value._turn_tracer = TraceRecorder("session-1", "turn-1", "hash", trace_id="trace-1")
    value._shadow_outcome_candidates = {}
    return value


def test_sequential_and_concurrent_queue_equivalent_candidates_without_changing_results(agent):
    pytest_result = json.dumps({"output": "1 passed", "exit_code": 0, "error": None})
    seq_messages = []
    seq_call = _tool_call("call-seq", command="pytest tests/unit", background=False)
    with patch("run_agent.handle_function_call", return_value=pytest_result):
        agent._execute_tool_calls_sequential(_message(seq_call), seq_messages, "task")

    seq_candidate = next(iter(agent._shadow_outcome_candidates.values()))
    assert seq_candidate.event_type == "verification_passed"
    assert seq_messages[0]["content"] == pytest_result

    agent._shadow_outcome_candidates = {}
    concurrent_messages = []
    concurrent_calls = (
        _tool_call("call-concurrent", command="pytest tests/unit", background=False),
        _tool_call("other", name="read_file", path="README.md"),
    )

    def invoke(name, *_args, **_kwargs):
        return pytest_result if name == "terminal" else "unchanged read result"

    with patch.object(agent, "_invoke_tool", side_effect=invoke):
        agent._execute_tool_calls_concurrent(_message(*concurrent_calls), concurrent_messages, "task")

    concurrent_candidate = next(iter(agent._shadow_outcome_candidates.values()))
    assert concurrent_candidate.event_type == seq_candidate.event_type
    assert concurrent_candidate.producer == seq_candidate.producer
    assert concurrent_messages[0]["content"] == pytest_result
    assert concurrent_messages[1]["content"] == "unchanged read result"


def test_capture_excludes_background_review_background_and_malformed_results(agent):
    valid_result = json.dumps({"exit_code": 0, "output": "ok", "error": None})
    agent._memory_write_origin = "background_review"
    agent._queue_shadow_outcome("one", "terminal", {"command": "pytest", "background": False}, valid_result)
    agent._memory_write_origin = "assistant_tool"
    agent.platform = "background_review"
    agent._queue_shadow_outcome("two", "terminal", {"command": "pytest", "background": False}, valid_result)
    agent.platform = "cli"
    agent._queue_shadow_outcome("three", "terminal", {"command": "pytest", "background": True}, valid_result)
    agent._queue_shadow_outcome("four", "terminal", {"command": "pytest", "background": False}, "not json")
    agent._queue_shadow_outcome("five", "terminal", {"command": "pytest", "background": False}, json.dumps({"exit_code": 2}))
    assert agent._shadow_outcome_candidates == {}


def test_trace_record_precedes_append_and_append_precedes_mark_persisted(agent, tmp_path, monkeypatch):
    calls = []
    db = MagicMock(db_path=tmp_path / "state.db")
    db.record_trace.side_effect = lambda *_args: calls.append("record_trace")
    agent._session_db = db
    candidate_result = json.dumps({"exit_code": 1, "output": "failed", "error": None})
    agent._queue_shadow_outcome("call-1", "terminal", {"command": "python -m pytest tests", "background": False}, candidate_result)
    monkeypatch.setattr(run_agent, "append_producer_outcome_event", lambda *_args: calls.append("append"))
    original_mark = agent._turn_tracer.mark_persisted
    agent._turn_tracer.mark_persisted = lambda: (calls.append("mark_persisted"), original_mark())[1]

    agent._persist_turn_trace()

    assert calls == ["record_trace", "append", "mark_persisted"]


def test_append_failure_is_structured_and_does_not_prevent_trace_persistence(agent, tmp_path, monkeypatch):
    db = MagicMock(db_path=tmp_path / "state.db")
    agent._session_db = db
    agent._queue_shadow_outcome(
        "call-1", "terminal", {"command": "pytest", "background": False}, json.dumps({"exit_code": 0})
    )
    monkeypatch.setattr(
        run_agent, "append_producer_outcome_event", MagicMock(side_effect=RuntimeError("append failed"))
    )
    warning = MagicMock()
    monkeypatch.setattr(run_agent.logger, "warning", warning)
    mark = MagicMock()
    agent._turn_tracer.mark_persisted = mark

    agent._persist_turn_trace()

    db.record_trace.assert_called_once()
    mark.assert_called_once()
    assert warning.call_args.kwargs["extra"]["stage"] == "append"
    assert set(warning.call_args.kwargs["extra"]) == {
        "stage", "error_class", "pending_count", "attempt_count",
    }


def test_append_failure_retries_on_later_turn_without_candidate_misattribution(agent, tmp_path, monkeypatch):
    db = MagicMock(db_path=tmp_path / "state.db")
    agent._session_db = db
    agent._queue_shadow_outcome(
        "old-call", "terminal", {"command": "pytest"}, json.dumps({"exit_code": 0})
    )
    append = MagicMock(side_effect=[RuntimeError("temporary"), SimpleNamespace()])
    monkeypatch.setattr(run_agent, "append_producer_outcome_event", append)

    agent._persist_turn_trace()
    old_candidate = next(iter(agent._shadow_outcome_candidates.values()))
    assert old_candidate.trace_id == "trace-1"

    agent._start_shadow_outcome_turn(
        TraceRecorder("session-1", "turn-2", "hash", trace_id="trace-2")
    )
    agent._persist_turn_trace()

    assert [call.args[1].trace_id for call in append.call_args_list] == ["trace-1", "trace-1"]
    assert agent._shadow_outcome_candidates == {}


def test_successful_append_is_removed_and_not_retried_on_later_turn(agent, tmp_path, monkeypatch):
    agent._session_db = MagicMock(db_path=tmp_path / "state.db")
    agent._queue_shadow_outcome(
        "call-1", "terminal", {"command": "pytest"}, json.dumps({"exit_code": 0})
    )
    append = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr(run_agent, "append_producer_outcome_event", append)

    agent._persist_turn_trace()
    agent._start_shadow_outcome_turn(
        TraceRecorder("session-1", "turn-2", "hash", trace_id="trace-2")
    )
    agent._persist_turn_trace()

    assert append.call_count == 1
    assert agent._shadow_outcome_candidates == {}


def test_contradictory_queued_event_is_quarantined_and_logs_without_identifiers(agent, monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(run_agent.logger, "warning", warning)
    arguments = {"command": "pytest", "background": False}
    agent._queue_shadow_outcome("call-1", "terminal", arguments, json.dumps({"exit_code": 0}))
    agent._queue_shadow_outcome("call-1", "terminal", arguments, json.dumps({"exit_code": 1}))
    agent._queue_shadow_outcome("call-1", "terminal", arguments, json.dumps({"exit_code": 0}))

    assert agent._shadow_outcome_candidates == {}
    assert warning.call_count == 1
    extra = warning.call_args.kwargs["extra"]
    assert extra["stage"] == "queue_conflict"
    assert extra["error_class"] == "ShadowOutcomeConflict"
    assert not ({"trace_id", "event_id", "tool_call_id"} & set(extra))


def test_pending_queue_is_deterministically_bounded(agent, monkeypatch):
    monkeypatch.setattr(run_agent, "_MAX_PENDING_SHADOW_OUTCOMES", 2)
    warning = MagicMock()
    monkeypatch.setattr(run_agent.logger, "warning", warning)
    for call_id in ("first", "second", "third"):
        agent._queue_shadow_outcome(
            call_id, "terminal", {"command": "pytest"}, json.dumps({"exit_code": 0})
        )

    assert [candidate.tool_call_id for candidate in agent._shadow_outcome_candidates.values()] == [
        "second", "third",
    ]
    assert warning.call_args.kwargs["extra"]["stage"] == "queue_capacity"


def test_record_trace_failure_never_appends_or_marks_and_logs_structurally(agent, monkeypatch):
    db = MagicMock()
    db.record_trace.side_effect = RuntimeError("write failed")
    agent._session_db = db
    agent._queue_shadow_outcome(
        "call-1", "terminal", {"command": "pytest", "background": False}, json.dumps({"exit_code": 0})
    )
    append = MagicMock()
    monkeypatch.setattr(run_agent, "append_producer_outcome_event", append)
    warning = MagicMock()
    monkeypatch.setattr(run_agent.logger, "warning", warning)
    mark = MagicMock()
    agent._turn_tracer.mark_persisted = mark

    agent._persist_turn_trace()

    append.assert_not_called()
    mark.assert_not_called()
    assert agent._shadow_outcome_candidates
    assert warning.call_args.kwargs["extra"]["stage"] == "record_trace"
    assert set(warning.call_args.kwargs["extra"]) == {
        "stage", "error_class", "pending_count", "attempt_count",
    }


def test_duplicate_trace_persistence_does_not_inflate_outcomes(agent, tmp_path):
    db = SessionDB(tmp_path / "state.db")
    agent._session_db = db
    agent._queue_shadow_outcome(
        "call-1", "terminal", {"command": "pytest tests", "background": False}, json.dumps({"exit_code": 0})
    )

    agent._persist_turn_trace()
    # Simulate replay of the same completed trace and structural event.
    agent._turn_tracer._persisted = False
    agent._persist_turn_trace()

    events = query_outcome_events(db.db_path, trace_id="trace-1")
    assert len(events) == 1
    assert events[0].event_type == "verification_passed"
    db.close()


def test_new_turn_preserves_pending_candidate_with_original_trace_attribution(agent):
    agent._queue_shadow_outcome(
        "call-1", "terminal", {"command": "pytest", "background": False}, json.dumps({"exit_code": 0})
    )
    assert agent._shadow_outcome_candidates

    agent._start_shadow_outcome_turn(TraceRecorder("session-1", "turn-2", "hash", trace_id="trace-2"))

    assert [candidate.trace_id for candidate in agent._shadow_outcome_candidates.values()] == ["trace-1"]
    assert agent._turn_tracer.trace_id == "trace-2"
