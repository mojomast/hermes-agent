"""Toolset exposure tests for local web and high-level browser tools."""

from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset


def test_web_toolset_exposes_search_extract_and_local_fetch():
    tools = set(resolve_toolset("web"))

    assert {"web_search", "web_extract", "fetch_url"}.issubset(tools)
    assert "fetch_url" in TOOLSETS["web"]["tools"]


def test_browser_toolset_exposes_primitives_search_and_browser_task():
    tools = set(resolve_toolset("browser"))

    assert "browser_task" in tools
    assert "browser_task" in TOOLSETS["browser"]["tools"]
    assert "web_search" in tools


def test_hermes_core_exposes_web_fetch_and_browser_task():
    core = set(_HERMES_CORE_TOOLS)

    assert {"web_search", "fetch_url", "browser_task"}.issubset(core)


def test_platform_toolsets_expose_local_web_and_browser_task():
    for name in ("hermes-cli", "hermes-acp", "hermes-api-server"):
        tools = set(resolve_toolset(name))
        assert {"web_search", "fetch_url", "browser_task"}.issubset(tools), name
