import json

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
