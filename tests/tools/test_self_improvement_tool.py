import json

import pytest

import tools.self_improvement_tool as self_improvement_tool
from agent.tracing import TraceRecorder, hash_user_message
from hermes_state import SessionDB
from model_tools import discover_builtin_tools
from tools.registry import registry
from toolsets import resolve_toolset


def test_self_improvement_tools_register_and_resolve():
    discover_builtin_tools()
    assert registry.get_entry("training_episodes") is not None
    assert registry.get_entry("semantic_code") is not None
    assert "training_episodes" in resolve_toolset("self_improvement")
    assert "semantic_code" in resolve_toolset("coding")
    assert "training_episodes" in resolve_toolset("full") or "training_episodes" in resolve_toolset("all")


def test_semantic_code_tool_smoke(tmp_path):
    discover_builtin_tools()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tool = registry.get_entry("semantic_code")
    result = json.loads(tool.handler({"project_path": str(project), "operation": "definition", "query": "f"}))
    assert result["success"] is True
    assert result["data"]["definitions"][0]["line"] == 1


def test_training_episodes_tool_replay_eval(tmp_path):
    discover_builtin_tools()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        recorder = TraceRecorder(session_id="sess", turn_id="turn", user_message_hash=hash_user_message("tool replay eval fixture"))
        with recorder.span("turn", "root"):
            with recorder.span("model_call", "llm_api_call"):
                pass
            with recorder.span("tool_call", "terminal", {"arg_count": 1}):
                pass
            with recorder.span("final_answer", "turn.final_answer", {"completed": True}):
                pass
        recorder.finish("completed")
        db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    finally:
        db.close()

    tool = registry.get_entry("training_episodes")
    result = json.loads(tool.handler({"operation": "replay_eval", "db_path": str(db_path), "limit": 10, "min_reward": 1.0, "require_ready": True, "include_episodes": True}))

    assert result["success"] is True
    data = result["data"]
    assert data["schema_version"] == "training_episode_replay_eval.v1"
    assert data["ok"] is True
    assert data["episode_count"] == 1
    assert data["passed_count"] == 1
    assert data["privacy"]["finding_count"] == 0
    assert len(data["episodes"]) == 1


def test_training_episodes_schema_exposes_replay_eval_args():
    discover_builtin_tools()
    entry = registry.get_entry("training_episodes")
    props = entry.schema["parameters"]["properties"]

    assert "min_reward" in props
    assert "require_ready" in props
    assert "include_episodes" in props
    assert "task_text" in props
    assert "replay_eval" in props["operation"]["enum"]
    assert "behavioral_hints" in props["operation"]["enum"]
    assert "outcome_summary" in props["operation"]["enum"]
    assert "contrastive_hints" in props["operation"]["enum"]
    assert "contrastive_replay_eval" in props["operation"]["enum"]
    assert "shadow_recurrence" in props["operation"]["enum"]
    assert "max_negative_reward" in props


def test_training_episodes_shadow_recurrence_is_bounded_read_only_and_private(tmp_path, monkeypatch):
    discover_builtin_tools()
    captured = {}
    monkeypatch.setattr(
        self_improvement_tool,
        "evaluate_shadow_recurrence",
        lambda db_path, *, limit: captured.update(db_path=db_path, limit=limit) or {
            "schema_version": "shadow_recurrence_report.v1",
            "policy_version": "shadow_lesson_lifecycle.v1",
            "shadow_only": True,
            "activation_allowed": False,
            "prompt_modified": False,
            "historical_event_count": 9,
            "effective_event_count": 7,
            "candidate_lesson_count": 2,
            "activation_eligible_lesson_count": 1,
            "active_lesson_count": 0,
            "captured_verifier_counts": {"effective_pass": 2},
            "taxonomy_counts": {"wrong_scope": 3},
            "lessons": [{"lesson_id": "lesson:secret", "taxonomy_code": "wrong_scope"}],
            "privacy": {"raw_content_exported": False},
        },
    )
    result = json.loads(registry.get_entry("training_episodes").handler({
        "operation": "shadow_recurrence", "db_path": str(tmp_path / "missing.db"), "limit": 5000,
    }))
    assert result["success"] is True
    assert captured["db_path"] == tmp_path / "missing.db"
    assert captured["limit"] == 200
    assert result["data"]["shadow_only"] is True
    assert result["data"]["activation_allowed"] is False
    assert result["data"]["prompt_modified"] is False
    assert result["data"]["historical_event_count"] == 9
    assert result["data"]["candidate_lesson_count"] == 2
    blob = json.dumps(result["data"]).lower()
    for forbidden in ("lesson_id", "lesson:", "taxonomy", "trace", "session", "event_id", "relation_id", "digest", "wrong_scope"):
        assert forbidden not in blob
    assert not (tmp_path / "missing.db").exists()


def test_training_episodes_contrastive_operations_are_shadow_only(tmp_path):
    discover_builtin_tools()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()
    tool = registry.get_entry("training_episodes")

    summary = json.loads(tool.handler({"operation": "outcome_summary", "db_path": str(db_path)}))
    hints = json.loads(tool.handler({"operation": "contrastive_hints", "db_path": str(db_path), "task_text": "new capability subsystem"}))
    replay = json.loads(tool.handler({"operation": "contrastive_replay_eval", "db_path": str(db_path), "task_text": "new capability subsystem"}))

    assert summary["success"] is True and summary["data"]["event_count"] == 0
    assert hints["data"]["schema_version"] == "contrastive_episode_retrieval.v1"
    assert replay["data"]["schema_version"] == "contrastive_replay_eval.v1"
    assert replay["data"]["shadow_only"] is True
    assert replay["data"]["prompt_modified"] is False
    assert replay["data"]["privacy_eval_valid"] is False
    assert replay["data"]["ok"] is False


def test_training_episodes_tool_forwards_max_negative_reward(tmp_path):
    discover_builtin_tools()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    recorder = TraceRecorder(session_id="s", turn_id="t", user_message_hash="h", trace_id="threshold")
    with recorder.span("turn", "root"):
        with recorder.span("tool_call", "terminal"):
            pass
        with recorder.span("final_answer", "turn.final_answer", {"completed": True}):
            pass
    recorder.finish("completed")
    db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    db.close()
    tool = registry.get_entry("training_episodes")
    result = json.loads(tool.handler({
        "operation": "contrastive_hints", "db_path": str(db_path),
        "task_text": "debug failure", "max_negative_reward": -100,
    }))
    assert result["data"]["negative_matches"] == []


@pytest.mark.parametrize("operation,target_name", [
    ("contrastive_hints", "retrieve_contrastive_episodes"),
    ("contrastive_replay_eval", "contrastive_replay_eval"),
])
def test_registered_contrastive_handlers_forward_all_arguments(monkeypatch, operation, target_name):
    discover_builtin_tools()
    captured = {}
    monkeypatch.setattr(self_improvement_tool, target_name, lambda **kwargs: captured.update(kwargs) or {})

    result = json.loads(registry.get_entry("training_episodes").handler({
        "operation": operation,
        "db_path": "/tmp/shadow-state.db",
        "task_text": "operator shadow task",
        "limit": 17,
        "min_reward": 2.5,
        "max_negative_reward": -3.5,
    }))

    assert result["success"] is True
    assert str(captured.pop("db_path")) == "/tmp/shadow-state.db"
    assert captured == {
        "task_text": "operator shadow task",
        "positive_limit": 17,
        "negative_limit": 17,
        "corrected_limit": 17,
        "min_positive_reward": 2.5,
        "max_negative_reward": -3.5,
    }


@pytest.mark.parametrize("field,value", [("min_reward", "nan"), ("max_negative_reward", "inf")])
def test_registered_tool_surfaces_non_finite_threshold_errors(field, value):
    discover_builtin_tools()
    result = json.loads(registry.get_entry("training_episodes").handler({
        "operation": "contrastive_hints", "db_path": "/tmp/unused.db",
        "task_text": "debug", field: value,
    }))
    assert field in result["error"]


@pytest.mark.parametrize("minimum,maximum", [(1.0, 1.0), (1.0, 2.0)])
def test_registered_contrastive_tool_rejects_overlapping_thresholds(minimum, maximum):
    discover_builtin_tools()
    result = json.loads(registry.get_entry("training_episodes").handler({
        "operation": "contrastive_replay_eval", "db_path": "/tmp/unused.db",
        "task_text": "debug", "min_reward": minimum,
        "max_negative_reward": maximum,
    }))
    assert "max_negative_reward" in result["error"]


def test_contrastive_tool_description_is_operator_shadow_inspection():
    discover_builtin_tools()
    entry = registry.get_entry("training_episodes")
    assert "operator-invoked shadow inspection" in entry.description
    assert "not automatically applied" in entry.description


def test_training_episodes_tool_behavioral_hints_redacts_private_payloads(tmp_path):
    discover_builtin_tools()
    db_path = tmp_path / "state.db"
    private_result = "PRIVATE_TOOL_HINT_RESULT_MANGO"
    private_stdout = "PRIVATE_TOOL_HINT_STDOUT_OTTER"
    private_prompt = "PRIVATE_TOOL_HINT_PROMPT_VELVET"
    db = SessionDB(db_path)
    try:
        recorder = TraceRecorder(session_id="sess", turn_id="turn", user_message_hash=hash_user_message("tool behavioral hint private fixture"))
        with recorder.span("turn", "root"):
            with recorder.span("model_call", "llm_api_call"):
                pass
            with recorder.span("tool_call", "terminal", {"result": private_result, "stdout": private_stdout, "nested": {"prompt": private_prompt}}):
                pass
            with recorder.span("final_answer", "turn.final_answer", {"completed": True}):
                pass
        recorder.finish("completed")
        db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    finally:
        db.close()

    tool = registry.get_entry("training_episodes")
    result = json.loads(tool.handler({
        "operation": "behavioral_hints",
        "db_path": str(db_path),
        "task_text": f"dirty git pytest privacy task {private_prompt}",
        "limit": 10,
        "min_reward": 1.0,
    }))
    blob = json.dumps(result, sort_keys=True)

    assert result["success"] is True
    data = result["data"]
    assert data["schema_version"] == "behavioral_hints.v1"
    assert data["matches_count"] == 1
    assert data["hints"]
    assert data["privacy"]["raw_content_exported"] is False
    assert data["privacy"]["includes_raw_episode_payloads"] is False
    assert private_result not in blob
    assert private_stdout not in blob
    assert private_prompt not in blob
    assert "steps" not in blob
    assert "metadata" not in blob


def test_training_episodes_tool_behavioral_hints_handles_empty_db(tmp_path):
    discover_builtin_tools()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()

    tool = registry.get_entry("training_episodes")
    result = json.loads(tool.handler({"operation": "behavioral_hints", "db_path": str(db_path), "task_text": "pytest git", "limit": 5}))

    assert result["success"] is True
    assert result["data"]["matches_count"] == 0
    assert result["data"]["hints"] == []


def test_semantic_code_tool_references_default_omits_context(tmp_path):
    discover_builtin_tools()
    project = tmp_path / "private"
    project.mkdir()
    private_line = "SECRET_SNIPPET = 'PRIVATE_TOOL_CONTEXT_MANGO'"
    (project / "secret.py").write_text(f"{private_line}\ndef use_secret():\n    return SECRET_SNIPPET\n", encoding="utf-8")

    tool = registry.get_entry("semantic_code")
    result = json.loads(tool.handler({"project_path": str(project), "operation": "references", "query": "SECRET_SNIPPET"}))
    blob = json.dumps(result)

    assert result["success"] is True
    assert result["data"]["total"] >= 2
    assert all(ref.get("context", "") == "" for ref in result["data"]["references"])
    assert "PRIVATE_TOOL_CONTEXT_MANGO" not in blob
