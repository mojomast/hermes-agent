import json

from agent.tracing import TraceRecorder, hash_user_message
from agent.training_episodes import export_episodes_jsonl, episode_summary, iter_episodes, trace_to_episode
from hermes_state import SessionDB


def _record_fixture_trace(path, *, error=False):
    db = SessionDB(path)
    try:
        recorder = TraceRecorder(session_id="sess", turn_id="turn", user_message_hash=hash_user_message("secret user task"))
        with recorder.span("turn", "root"):
            with recorder.span("model_call", "llm_api_call"):
                pass
            with recorder.span("tool_call", "terminal", {"api_key": "sk-test", "arg_count": 1}) as span:
                if error:
                    span.status = "error"
                    span.error_class = "CommandFailed"
            with recorder.span("final_answer", "turn.final_answer", {"completed": not error, "response_len": 42}):
                pass
        recorder.finish("completed" if not error else "failed")
        db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
        return recorder.trace_id
    finally:
        db.close()


def test_trace_to_episode_extracts_reward_and_redacts_metadata(tmp_path):
    db_path = tmp_path / "state.db"
    trace_id = _record_fixture_trace(db_path)
    episode = next(iter_episodes(db_path=db_path, limit=1))

    assert episode.schema_version == "training_episode.v1"
    assert episode.trace_id == trace_id
    assert episode.ready_for_training is True
    assert episode.reward > 0
    assert any(s.name == "trace_completed" and s.value is True for s in episode.outcome_signals)
    terminal_step = next(s for s in episode.steps if s.name == "terminal")
    assert terminal_step.metadata["api_key"] == "[REDACTED]"
    assert "secret user task" not in json.dumps(episode.to_dict())


def test_raw_content_keys_redact_non_secret_private_text():
    private_result = "private project note: launch codename velvet-mango"
    private_stdout = "stdout contained internal customer migration details"
    private_prompt = "prompt asked about a confidential family matter"
    trace = {
        "trace_id": "trace-raw-redaction",
        "session_id": "sess",
        "turn_id": "turn",
        "user_message_hash": "hash",
        "start_time": 1.0,
        "end_time": 2.0,
        "total_wall_ms": 1000,
        "total_model_calls": 1,
        "status": "completed",
    }
    spans = [
        {
            "span_type": "tool_call",
            "name": "terminal",
            "status": "completed",
            "start_time": 1.0,
            "end_time": 1.1,
            "error_class": None,
            "metadata_json": json.dumps(
                {
                    "result": private_result,
                    "stdout": private_stdout,
                    "nested": {"prompt": private_prompt},
                }
            ),
        },
        {
            "span_type": "final_answer",
            "name": "turn.final_answer",
            "status": "completed",
            "start_time": 1.1,
            "end_time": 2.0,
            "error_class": None,
            "metadata_json": json.dumps({"completed": True}),
        },
    ]

    episode_blob = json.dumps(trace_to_episode(trace, spans).to_dict())

    assert "[REDACTED_RAW_PAYLOAD]" in episode_blob
    assert private_result not in episode_blob
    assert private_stdout not in episode_blob
    assert private_prompt not in episode_blob


def test_failed_tool_call_lowers_reward_and_blocks_training(tmp_path):
    db_path = tmp_path / "state.db"
    _record_fixture_trace(db_path, error=True)
    episode = next(iter_episodes(db_path=db_path, limit=1))

    assert episode.ready_for_training is False
    assert any(s.name == "trace_error_count" and s.value >= 1 for s in episode.outcome_signals)
    assert episode.reward < 1.0


def test_export_jsonl_and_summary_report_latency(tmp_path):
    db_path = tmp_path / "state.db"
    _record_fixture_trace(db_path)
    out = tmp_path / "episodes.jsonl"

    export = export_episodes_jsonl(out, db_path=db_path, limit=10)
    summary = episode_summary(db_path=db_path, limit=10)

    assert export["episode_count"] == 1
    assert export["elapsed_ms"] >= 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["schema_version"] == "training_episode.v1"
    assert summary["episode_count"] == 1
    assert summary["ready_for_training_count"] == 1
    assert summary["conversion_latency_ms"] >= 0
