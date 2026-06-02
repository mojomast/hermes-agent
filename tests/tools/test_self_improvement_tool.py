import json

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
    assert "replay_eval" in props["operation"]["enum"]


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
