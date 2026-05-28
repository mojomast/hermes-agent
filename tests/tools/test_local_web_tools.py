import json
from unittest.mock import MagicMock, patch

import pytest


def test_web_search_uses_configured_searxng_endpoint_and_normalizes_results(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
        "backend": "searxng",
        "searxng": {
            "base_url": "http://searxng.local:8080",
            "search_path": "/search",
            "default_limit": 2,
            "timeout": 7,
        },
    })

    response = MagicMock()
    response.json.return_value = {
        "results": [
            {"title": "First", "url": "https://example.com/1", "content": "Snippet one", "score": 0.9},
            {"title": "Second", "url": "https://example.com/2", "snippet": "Snippet two"},
            {"title": "Third", "url": "https://example.com/3", "content": "too many"},
        ]
    }
    response.raise_for_status.return_value = None

    with patch.object(web_tools.httpx, "get", return_value=response) as mock_get:
        payload = json.loads(web_tools.web_search_tool("hermes agent", limit=2))

    assert payload["success"] is True
    assert payload["provider"] == "searxng"
    assert payload["data"]["web"] == [
        {
            "title": "First",
            "url": "https://example.com/1",
            "description": "Snippet one",
            "snippet": "Snippet one",
            "position": 1,
            "rank": 1,
            "score": 0.9,
        },
        {
            "title": "Second",
            "url": "https://example.com/2",
            "description": "Snippet two",
            "snippet": "Snippet two",
            "position": 2,
            "rank": 2,
        },
    ]
    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == "http://searxng.local:8080/search"
    assert mock_get.call_args.kwargs["params"]["q"] == "hermes agent"
    assert mock_get.call_args.kwargs["params"]["format"] == "json"
    assert mock_get.call_args.kwargs["timeout"] == 7


def test_fetch_url_rejects_binary_urls_before_fetch(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"fetch_url": {"max_content_chars": 1000}})
    result = json.loads(web_tools.fetch_url_tool("https://example.com/file.zip"))

    assert result["ok"] is False
    assert result["success"] is False
    assert "binary" in result["error"].lower()


def test_fetch_url_extracts_markdown_and_metadata(monkeypatch):
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"fetch_url": {"max_content_chars": 1000}})

    fake_trafilatura = MagicMock()
    fake_trafilatura.fetch_url.return_value = "<html><title>Demo</title><article>Hello world</article></html>"
    fake_trafilatura.extract.return_value = '{"title":"Demo","author":"Mojo","date":"2026-05-15","text":"# Demo\\n\\nHello world"}'

    with patch.dict("sys.modules", {"trafilatura": fake_trafilatura}):
        result = json.loads(web_tools.fetch_url_tool("https://example.com/post", output_format="markdown"))

    assert result["ok"] is True
    assert result["success"] is True
    assert result["content"] == "# Demo\n\nHello world"
    assert result["metadata"]["title"] == "Demo"
    assert result["metadata"]["author"] == "Mojo"
    fake_trafilatura.fetch_url.assert_called_once()
    assert fake_trafilatura.extract.call_args.kwargs["output_format"] == "json"
    assert fake_trafilatura.extract.call_args.kwargs["with_metadata"] is True


def test_browser_task_forwards_to_configured_worker(monkeypatch):
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "_load_browser_config", lambda: {
        "task_worker_url": "http://browser-worker:8765",
        "task_timeout": 12,
        "task_default_max_steps": 4,
    })

    response = MagicMock()
    response.json.return_value = {
        "ok": True,
        "final_url": "https://example.com/done",
        "transcript": [{"step": 1, "action": "goto", "url": "https://example.com"}],
        "extracted": {"text": "done"},
    }
    response.raise_for_status.return_value = None

    with patch.object(browser_tool.requests, "post", return_value=response) as mock_post:
        result = json.loads(browser_tool.browser_task(
            goal="click the button",
            start_url="https://example.com",
            domain_whitelist=["example.com"],
        ))

    assert result["ok"] is True
    assert result["success"] is True
    assert result["extracted"]["text"] == "done"
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "http://browser-worker:8765/task"
    assert mock_post.call_args.kwargs["json"]["goal"] == "click the button"
    assert mock_post.call_args.kwargs["json"]["max_steps"] == 4
    assert mock_post.call_args.kwargs["timeout"] == 12
