import json

from agent.episode_retrieval import (
    behavioral_hints_for_task,
    episode_features,
    format_behavioral_hints_for_prompt,
    retrieve_similar_episodes,
    task_frame_from_text,
)
from agent.tracing import TraceRecorder, hash_user_message
from agent.training_episodes import iter_episodes
from hermes_state import SessionDB


PRIVATE_RESULT = "PRIVATE_HINT_RESULT_MANGO"
PRIVATE_STDOUT = "PRIVATE_HINT_STDOUT_OTTER"
PRIVATE_PROMPT = "PRIVATE_HINT_PROMPT_VELVET"


def _record_trace(path, *, name="terminal", error=False, metadata=None, user_message="secret user task"):
    db = SessionDB(path)
    try:
        recorder = TraceRecorder(session_id="sess", turn_id="turn", user_message_hash=hash_user_message(user_message))
        with recorder.span("turn", "root"):
            with recorder.span("model_call", "llm_api_call"):
                pass
            with recorder.span("tool_call", name, metadata or {"arg_count": 1}) as span:
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


def test_task_frame_from_text_exports_only_structural_categories():
    frame = task_frame_from_text(f"fix dirty git tree with pytest and {PRIVATE_PROMPT}")
    blob = json.dumps(frame.to_dict())

    assert "git_dirty_tree" in frame.categories
    assert "focused_tests" in frame.categories
    assert "tool:terminal" in frame.structural_features
    assert PRIVATE_PROMPT not in blob
    assert "fix dirty git" not in blob


def test_retrieves_only_ready_high_reward_training_episodes(tmp_path):
    db_path = tmp_path / "state.db"
    successful_id = _record_trace(db_path, name="terminal")
    _record_trace(db_path, name="terminal", error=True)

    result = retrieve_similar_episodes(
        db_path=db_path,
        task_text="commit dirty tree after focused pytest verification",
        limit=5,
        min_reward=1.0,
    )

    assert result["schema_version"] == "episode_retrieval.v1"
    assert result["matches_count"] == 1
    assert result["matches"][0]["episode_id"] == f"episode:{successful_id}"
    assert result["matches"][0]["ready_for_training"] is True
    assert result["matches"][0]["reward"] >= 1.0
    assert "tool:terminal" in result["matches"][0]["matched_features"]
    assert result["privacy"]["raw_content_exported"] is False
    assert result["privacy"]["includes_raw_episode_payloads"] is False


def test_episode_features_and_hints_do_not_export_raw_private_metadata(tmp_path):
    db_path = tmp_path / "state.db"
    _record_trace(
        db_path,
        name="terminal",
        metadata={
            "result": PRIVATE_RESULT,
            "stdout": PRIVATE_STDOUT,
            "nested": {"prompt": PRIVATE_PROMPT},
            "api_key": "sk-test-private-token",
        },
        user_message="private launch task codename should not leak",
    )
    episode = next(iter_episodes(db_path=db_path, limit=1))

    features = episode_features(episode)
    hints = behavioral_hints_for_task(
        db_path=db_path,
        task_text=f"privacy redaction training episode work {PRIVATE_PROMPT}",
        limit=5,
        min_reward=1.0,
    )
    blob = json.dumps({"features": features, "hints": hints}, sort_keys=True)

    assert hints["schema_version"] == "behavioral_hints.v1"
    assert hints["hints"]
    assert hints["privacy"]["raw_content_exported"] is False
    assert hints["privacy"]["includes_raw_episode_payloads"] is False
    assert PRIVATE_RESULT not in blob
    assert PRIVATE_STDOUT not in blob
    assert PRIVATE_PROMPT not in blob
    assert "private launch task" not in blob
    assert "metadata" not in blob
    assert "steps" not in blob


def test_behavioral_hints_are_compact_and_prompt_format_is_bounded(tmp_path):
    db_path = tmp_path / "state.db"
    _record_trace(db_path, name="terminal")
    _record_trace(db_path, name="read_file")
    _record_trace(db_path, name="patch")

    result = behavioral_hints_for_task(
        db_path=db_path,
        task_text="implement code edit with git status and pytest verification",
        limit=10,
        min_reward=1.0,
    )
    rendered = format_behavioral_hints_for_prompt(result, max_hints=3, max_chars=400)

    assert len(result["hints"]) <= 3
    assert len(json.dumps(result)) < 6000
    assert all(len(hint["text"]) <= 240 for hint in result["hints"])
    assert rendered.startswith("<behavioral_hints")
    assert rendered.endswith("</behavioral_hints>")
    assert len(rendered) <= 400


def test_behavioral_hints_handles_empty_db(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()

    result = behavioral_hints_for_task(db_path=db_path, task_text="pytest git task", limit=5, min_reward=1.0)

    assert result["matches_count"] == 0
    assert result["hints"] == []
    assert result["privacy"]["includes_raw_episode_payloads"] is False
