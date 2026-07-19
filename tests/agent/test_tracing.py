import hashlib
import json

import pytest

from agent.tracing import TraceRecorder, active_tracer, hash_user_message


def test_hash_user_message_is_stable_and_content_free():
    assert hash_user_message("secret") == hashlib.sha256(b"secret").hexdigest()
    assert hash_user_message({"b": 2, "a": 1}) == hash_user_message({"a": 1, "b": 2})


def test_nested_spans_capture_structure_counts_and_errors():
    recorder = TraceRecorder("session", "turn", hash_user_message("private task"), trace_id="trace")
    with recorder.span("turn", "root") as root:
        with recorder.span("model_call", "llm_api_call") as model:
            pass
        with pytest.raises(ValueError):
            with recorder.span("tool_call", "terminal", {"arg_count": 1}):
                raise ValueError("private failure text")

    recorder.finish("failed")
    rows = recorder.to_span_rows()
    assert rows[1]["parent_span_id"] == root.span_id
    assert rows[2]["parent_span_id"] == root.span_id
    assert rows[2]["status"] == "error"
    assert rows[2]["error_class"] == "ValueError"
    assert recorder.total_model_calls == 1
    assert recorder.total_tool_calls == 1
    assert "private failure text" not in json.dumps(rows)


def test_active_tracer_requires_a_trace_recorder():
    class Subject:
        pass

    subject = Subject()
    subject._turn_tracer = object()
    assert active_tracer(subject) is None
    subject._turn_tracer = TraceRecorder("s", "t", "h")
    assert active_tracer(subject) is subject._turn_tracer
