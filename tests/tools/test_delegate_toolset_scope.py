"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from tools.delegate_tool import _narrow_child_toolsets, _strip_blocked_tools


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = _narrow_child_toolsets(
            requested,
            parent_toolsets=set(parent.enabled_toolsets),
            parent_enabled=parent.enabled_toolsets,
        )

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        requested = ["terminal", "web"]
        scoped = _narrow_child_toolsets(
            requested,
            parent_toolsets=set(parent.enabled_toolsets),
            parent_enabled=parent.enabled_toolsets,
        )

        assert sorted(scoped) == ["terminal", "web"]

    def test_no_toolsets_requested_inherits_parent(self):
        """When toolsets is None/empty, child inherits parent's set."""
        parent_toolsets = ["terminal", "file", "web"]
        child = _narrow_child_toolsets(
            None,
            parent_toolsets=set(parent_toolsets),
            parent_enabled=parent_toolsets,
        )
        assert "terminal" in child
        assert "file" in child
        assert "web" in child

    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        requested = ["web", "browser"]
        scoped = _narrow_child_toolsets(
            requested,
            parent_toolsets=set(parent.enabled_toolsets),
            parent_enabled=parent.enabled_toolsets,
        )

        assert scoped == []

    def test_empty_requested_toolsets_inherits_parent(self):
        """Empty requested toolset lists are treated like omitted toolsets."""
        child = _narrow_child_toolsets(
            [],
            parent_toolsets={"terminal", "file", "web"},
            parent_enabled=["terminal", "file", "web"],
        )

        assert sorted(child) == ["file", "terminal", "web"]

    def test_parent_enabled_none_uses_derived_parent_toolsets(self):
        """Parents with all toolsets enabled use the derived effective scope."""
        child = _narrow_child_toolsets(
            None,
            parent_toolsets={"terminal", "file"},
            parent_enabled=None,
        )

        assert sorted(child) == ["file", "terminal"]

    def test_preserve_mcp_toolsets_when_requested(self):
        """Narrowed children retain parent MCP toolsets when MCP inheritance is enabled."""
        child = _narrow_child_toolsets(
            ["terminal"],
            parent_toolsets={"terminal", "mcp-becomussy"},
            parent_enabled=["terminal", "mcp-becomussy"],
            inherit_mcp_toolsets=True,
        )

        assert child == ["terminal", "mcp-becomussy"]

    def test_disable_mcp_preservation_keeps_requested_scope_only(self):
        child = _narrow_child_toolsets(
            ["terminal"],
            parent_toolsets={"terminal", "mcp-becomussy"},
            parent_enabled=["terminal", "mcp-becomussy"],
            inherit_mcp_toolsets=False,
        )

        assert child == ["terminal"]
