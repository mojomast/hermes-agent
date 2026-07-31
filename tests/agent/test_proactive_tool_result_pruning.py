"""Regression tests for deterministic proactive tool-result projection."""

from copy import deepcopy
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _PRUNED_TOOL_PLACEHOLDER
from agent.model_metadata import estimate_messages_tokens_rough

LARGE_WINDOW = 1_000_000


def _compressor(**kw):
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=LARGE_WINDOW):
        return ContextCompressor(**defaults)


def _assistant_call(cid, args='{"command":"ls"}'):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": cid,
            "type": "function",
            "function": {"name": "terminal", "arguments": args},
        }],
    }


def _build(n_pairs, big_indices, big_chars=20_000):
    messages = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        messages.append(_assistant_call(cid))
        content = chr(65 + i % 26) * big_chars if i in big_indices else "ok"
        messages.append({"role": "tool", "tool_call_id": cid, "content": content})
    return messages


def _tool(messages, cid):
    return next(m for m in messages if m.get("tool_call_id") == cid)


def test_prunes_old_results_below_full_compression_trigger_without_llm():
    compressor = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    assert compressor.should_compress(prompt_tokens=120_000) is False
    messages = _build(8, {0, 1, 2})
    canonical = deepcopy(messages)

    with patch.object(compressor, "_generate_summary") as summarize:
        result, pruned = compressor.prune_tool_results_only(messages, current_tokens=120_000)

    summarize.assert_not_called()
    assert pruned == 3
    assert result is not messages
    assert messages == canonical
    assert estimate_messages_tokens_rough(result) < estimate_messages_tokens_rough(messages)
    for cid in ("call_0", "call_1", "call_2"):
        assert len(_tool(result, cid)["content"]) < 20_000
        assert _tool(result, cid)["content"] != _PRUNED_TOOL_PLACEHOLDER


def test_disabled_and_below_trigger_are_identity_noops():
    messages = _build(8, {0, 1, 2})
    disabled = _compressor()
    assert disabled.proactive_prune_tokens == 0
    result, pruned = disabled.prune_tool_results_only(messages, current_tokens=500_000)
    assert result is messages
    assert pruned == 0

    enabled = _compressor(proactive_prune_tokens=48_000)
    result, pruned = enabled.prune_tool_results_only(messages, current_tokens=10_000)
    assert result is messages
    assert pruned == 0


def test_recent_tail_is_exactly_preserved_and_structure_ids_survive():
    compressor = _compressor(
        proactive_prune_tokens=1,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    messages = _build(8, {0, 6, 7})
    tail_before = deepcopy(messages[-4:])
    roles = [m["role"] for m in messages]
    result, pruned = compressor.prune_tool_results_only(messages, current_tokens=2)

    assert pruned == 1
    assert result[-4:] == tail_before
    assert [m["role"] for m in result] == roles
    assert [m.get("tool_call_id") for m in result] == [m.get("tool_call_id") for m in messages]
    assert [tc["id"] for m in result for tc in m.get("tool_calls", [])] == [
        tc["id"] for m in messages for tc in m.get("tool_calls", [])
    ]
    assert len(_tool(result, "call_0")["content"]) < 20_000


def test_result_floor_defaults_to_8000_and_clamps_to_200():
    assert _compressor().proactive_prune_min_result_chars == 8000
    assert _compressor(proactive_prune_min_result_chars=0).proactive_prune_min_result_chars == 8000
    assert _compressor(proactive_prune_min_result_chars=50).proactive_prune_min_result_chars == 200
    assert _compressor(proactive_prune_min_result_chars=-1).proactive_prune_min_result_chars == 200


def test_duplicate_elision_never_rewrites_protected_tail():
    compressor = _compressor(
        proactive_prune_tokens=1,
        proactive_prune_min_reclaim_tokens=0,
    )
    messages = _build(8, set())
    duplicate = "D" * 20_000
    for message in messages:
        if message.get("role") == "tool":
            message["content"] = duplicate
    tail_before = deepcopy(messages[-4:])

    result, pruned = compressor.prune_tool_results_only(messages, current_tokens=2)

    assert pruned > 0
    assert result[-4:] == tail_before


def test_full_compression_prune_keeps_legacy_200_char_floor():
    compressor = _compressor()
    messages = _build(8, set())
    _tool(messages, "call_0")["content"] = "Q" * 300
    result, pruned = compressor._prune_old_tool_results(messages, protect_tail_count=4)
    assert pruned >= 1
    assert len(_tool(result, "call_0")["content"]) < 300


def test_reclaim_hysteresis_defaults_to_4096_and_rejects_small_projection():
    compressor = _compressor(proactive_prune_tokens=1)
    assert compressor.proactive_prune_min_reclaim_tokens == 4096
    messages = _build(8, {0}, big_chars=9_000)
    result, pruned = compressor.prune_tool_results_only(messages, current_tokens=2)
    assert result is messages
    assert pruned == 0


def test_reclaim_hysteresis_commits_large_projection():
    compressor = _compressor(proactive_prune_tokens=1)
    messages = _build(8, {0}, big_chars=20_000)
    result, pruned = compressor.prune_tool_results_only(messages, current_tokens=2)
    assert result is not messages
    assert pruned == 1


def test_projection_is_idempotent_and_second_pass_is_identity_noop():
    compressor = _compressor(
        proactive_prune_tokens=1,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    messages = _build(8, {0, 1, 2})
    first, n1 = compressor.prune_tool_results_only(messages, current_tokens=2)
    second, n2 = compressor.prune_tool_results_only(first, current_tokens=2)
    assert n1 == 3
    assert n2 == 0
    assert second is first
