import json

from agent.tracing import TraceRecorder
from hermes_state import SessionDB
from run_agent import AIAgent


def test_session_db_creates_trace_schema_and_persists_parented_spans(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    recorder = TraceRecorder("session", "turn", "hash", trace_id="trace")
    with recorder.span("turn", "root") as root:
        with recorder.span("tool_call", "terminal", {"arg_count": 2}):
            pass
    recorder.finish()

    db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    trace = db._conn.execute("SELECT * FROM traces WHERE trace_id='trace'").fetchone()
    spans = db._conn.execute(
        "SELECT * FROM spans WHERE trace_id='trace' ORDER BY start_time, span_id"
    ).fetchall()
    indexes = {row[1] for row in db._conn.execute("PRAGMA index_list('spans')")}
    db.close()

    assert trace["status"] == "completed"
    assert trace["total_tool_calls"] == 1
    assert len(spans) == 2
    assert spans[1]["parent_span_id"] == root.span_id
    assert json.loads(spans[1]["metadata_json"]) == {"arg_count": 2}
    assert "idx_spans_trace_start" in indexes


def test_record_trace_replaces_trace_and_span_set_atomically(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    trace = {
        "trace_id": "trace", "session_id": "s", "turn_id": "t",
        "user_message_hash": "h", "start_time": 1, "end_time": 2,
        "status": "completed", "total_wall_ms": 1000,
        "total_model_calls": 0, "total_tool_calls": 1, "total_subagents": 0,
    }
    span = {
        "span_id": "old", "trace_id": "trace", "parent_span_id": None,
        "span_type": "tool_call", "name": "terminal", "start_time": 1,
        "end_time": 2, "status": "completed", "error_class": None,
        "metadata_json": "{}",
    }
    db.record_trace(trace, [span])
    replacement = dict(span, span_id="new", name="write_file")
    db.record_trace(trace, [replacement])
    rows = db._conn.execute("SELECT span_id FROM spans WHERE trace_id='trace'").fetchall()
    db.close()

    assert [row["span_id"] for row in rows] == ["new"]


def test_agent_persist_turn_trace_closes_root_and_is_idempotent(tmp_path):
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = SessionDB(tmp_path / "state.db")
    agent._turn_tracer = TraceRecorder("session", "turn", "hash", trace_id="agent-trace")
    agent._turn_root_span_cm = agent._turn_tracer.span("turn", "root")
    agent._turn_root_span_cm.__enter__()

    agent._persist_turn_trace()
    agent._persist_turn_trace()

    trace_count = agent._session_db._conn.execute(
        "SELECT COUNT(*) FROM traces WHERE trace_id='agent-trace'"
    ).fetchone()[0]
    spans = agent._session_db._conn.execute(
        "SELECT status, end_time FROM spans WHERE trace_id='agent-trace'"
    ).fetchall()
    agent._session_db.close()

    assert trace_count == 1
    assert len(spans) == 1
    assert spans[0]["status"] == "completed"
    assert spans[0]["end_time"] is not None
