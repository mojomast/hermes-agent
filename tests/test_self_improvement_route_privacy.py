import json

import pytest


@pytest.mark.asyncio
async def test_semantic_index_route_default_omits_reference_context(tmp_path):
    from hermes_cli.web_server import get_semantic_index

    project = tmp_path / "private-route"
    project.mkdir()
    private_line = "SECRET_SNIPPET = 'PRIVATE_ROUTE_CONTEXT_MANGO'"
    (project / "secret.py").write_text(f"{private_line}\ndef use_secret():\n    return SECRET_SNIPPET\n", encoding="utf-8")

    result = await get_semantic_index(project_path=str(project), operation="references", query="SECRET_SNIPPET")
    blob = json.dumps(result)

    assert result["total"] >= 2
    assert all(ref.get("context", "") == "" for ref in result["references"])
    assert "PRIVATE_ROUTE_CONTEXT_MANGO" not in blob


@pytest.mark.asyncio
async def test_training_episode_summary_route_is_summary_only(monkeypatch):
    from hermes_cli import web_server

    captured = {}

    def fake_episode_summary(*, limit):
        captured["limit"] = limit
        return {
            "schema_version": "training_episode.v1",
            "episode_count": 1,
            "ready_for_training_count": 1,
            "recent_rewards": [4.47],
            "avg_reward": 4.47,
            "conversion_latency_ms": 0,
        }

    monkeypatch.setattr("agent.training_episodes.episode_summary", fake_episode_summary)

    result = await web_server.get_training_episode_summary(limit=999999)
    blob = json.dumps(result)

    assert captured["limit"] == 1000
    assert result["episode_count"] == 1
    for forbidden_key in ["episodes", "steps", "metadata", "prompt", "content", "stdout", "stderr", "args", "arguments"]:
        assert forbidden_key not in blob
