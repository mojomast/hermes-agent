import json

from agent.tracing import TraceRecorder, hash_user_message
from agent.training_episodes import (
    export_episodes_jsonl,
    episode_summary,
    iter_episodes,
    replay_eval_episodes,
    replay_episode_eval,
    trace_to_episode,
)
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
    private_response = "assistant response included patient name lavender otter"
    private_transcript = "call transcript included home address marmalade lane"
    private_model_output = "model output included private investor note teal fox"
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
                    "tool_response": private_response,
                    "chat_history_transcript": private_transcript,
                    "modelOutput": private_model_output,
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
    assert private_response not in episode_blob
    assert private_transcript not in episode_blob
    assert private_model_output not in episode_blob


def test_nested_list_metadata_redacts_raw_payloads_and_secrets():
    private_prompt = "prompt asked for a confidential acquisition codename blue-raccoon"
    secret_token = "sk-test-private-token-123456"
    trace = {
        "trace_id": "trace-list-redaction",
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
            "name": "execute_code",
            "status": "completed",
            "start_time": 1.0,
            "end_time": 1.1,
            "error_class": None,
            "metadata_json": json.dumps(
                {
                    "events": [
                        {"prompt": private_prompt},
                        {"api_key": secret_token},
                    ]
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
    assert "[REDACTED]" in episode_blob
    assert private_prompt not in episode_blob
    assert secret_token not in episode_blob


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


def test_replay_eval_episodes_passes_ready_trace(tmp_path):
    db_path = tmp_path / "state.db"
    _record_fixture_trace(db_path)

    result = replay_eval_episodes(db_path=db_path, limit=10, min_reward=1.0)

    assert result["schema_version"] == "training_episode_replay_eval.v1"
    assert result["ok"] is True
    assert result["episode_count"] == 1
    assert result["evaluated_count"] == 1
    assert result["passed_count"] == 1
    assert result["failed_count"] == 0
    assert result["ready_for_training_count"] == 1
    assert result["privacy"]["raw_content_exported"] is False
    assert result["privacy"]["finding_count"] == 0
    assert result["failures"] == []
    assert result["elapsed_ms"] >= 0


def test_replay_eval_episodes_flags_failed_trace(tmp_path):
    db_path = tmp_path / "state.db"
    _record_fixture_trace(db_path, error=True)

    result = replay_eval_episodes(db_path=db_path, limit=10, min_reward=1.0)

    assert result["ok"] is False
    assert result["episode_count"] == 1
    assert result["passed_count"] == 0
    assert result["failed_count"] == 1
    assert result["failures"]
    assert result["failures"][0]["ready_for_training"] is False
    assert "not_ready_for_training" in result["failures"][0]["reasons"]
    assert "reward_below_min" in result["failures"][0]["reasons"]


def test_replay_eval_episodes_can_include_compact_episode_rows(tmp_path):
    db_path = tmp_path / "state.db"
    _record_fixture_trace(db_path)

    result = replay_eval_episodes(db_path=db_path, limit=10, min_reward=1.0, include_episodes=True)

    assert "episodes" in result
    assert len(result["episodes"]) == 1
    row = result["episodes"][0]
    assert row["episode_id"].startswith("episode:")
    assert row["trace_id"]
    assert row["passed"] is True
    assert "steps" not in row
    assert "metadata" not in json.dumps(row)


def test_replay_eval_privacy_regression_forbidden_substrings_not_exported():
    private_result = "private launch codename velvet-mango"
    private_prompt = "confidential family matter orange-badger"
    private_stdout = "internal migration note silver-ferret"
    trace = {
        "trace_id": "trace-privacy-replay",
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
            "metadata_json": json.dumps({"result": private_result, "stdout": private_stdout, "nested": {"prompt": private_prompt}}),
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

    episode = trace_to_episode(trace, spans)
    row = replay_episode_eval(episode, min_reward=1.0, forbidden_substrings=[private_result, private_prompt, private_stdout])
    blob = json.dumps(episode.to_dict())

    assert row["privacy_finding_count"] == 0
    assert row["passed"] is True
    assert private_result not in blob
    assert private_prompt not in blob
    assert private_stdout not in blob
    assert "[REDACTED_RAW_PAYLOAD]" in blob


def test_export_jsonl_privacy_regression_redacts_raw_payloads(tmp_path):
    private_result = "PRIVATE_EXPORT_RESULT_MANGO"
    private_stdout = "PRIVATE_EXPORT_STDOUT_OTTER"
    private_response = "PRIVATE_EXPORT_RESPONSE_LAVENDER"
    private_prompt = "PRIVATE_EXPORT_PROMPT_VELVET"
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        recorder = TraceRecorder(session_id="sess", turn_id="turn", user_message_hash=hash_user_message("private export prompt"))
        with recorder.span("turn", "root"):
            with recorder.span("tool_call", "terminal", {"result": private_result, "stdout": private_stdout, "tool_response": private_response, "nested": {"prompt": private_prompt}, "api_key": "sk-test-private"}):
                pass
            with recorder.span("final_answer", "turn.final_answer", {"completed": True}):
                pass
        recorder.finish("completed")
        db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    finally:
        db.close()

    out = tmp_path / "episodes.jsonl"
    export_episodes_jsonl(out, db_path=db_path, limit=10)
    text = out.read_text(encoding="utf-8")
    row = json.loads(text.splitlines()[0])

    assert "[REDACTED_RAW_PAYLOAD]" in text
    assert "[REDACTED]" in text
    assert row["privacy"]["raw_content_exported"] is False
    for private_value in [private_result, private_stdout, private_response, private_prompt, "private export prompt", "sk-test-private"]:
        assert private_value not in text
