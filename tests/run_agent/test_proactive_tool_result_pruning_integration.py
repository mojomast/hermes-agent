"""Run-loop integration regressions for proactive tool-result projection."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.config import DEFAULT_CONFIG
from run_agent import AIAgent


def _tool_defs():
    return [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _bare_agent(compressor):
    agent = AIAgent.__new__(AIAgent)
    agent.context_compressor = compressor
    agent._persist_session = MagicMock(return_value=True)
    return agent


def test_proactive_projection_is_opt_in_in_default_config():
    compression = DEFAULT_CONFIG["compression"]
    assert compression["proactive_prune_tokens"] == 0
    assert compression["proactive_prune_min_result_chars"] == 8000
    assert compression["proactive_prune_min_reclaim_tokens"] == 4096


def test_proactive_projection_config_is_parsed_into_builtin_compressor():
    config = {
        "compression": {
            "proactive_prune_tokens": "48000",
            "proactive_prune_min_result_chars": 12000.0,
            "proactive_prune_min_reclaim_tokens": "2048",
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
        )

    assert agent.context_compressor.proactive_prune_tokens == 48000
    assert agent.context_compressor.proactive_prune_min_result_chars == 12000
    assert agent.context_compressor.proactive_prune_min_reclaim_tokens == 2048


def test_post_tool_projection_persists_raw_messages_before_replacing_memory():
    raw = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "raw payload"},
    ]
    projected = [raw[0], {**raw[1], "content": "[projected]"}]
    events = []

    def prune(messages, *, current_tokens):
        events.append(("prune", messages, current_tokens))
        return projected, 1

    compressor = SimpleNamespace(
        proactive_prune_tokens=100,
        prune_tool_results_only=prune,
    )
    agent = _bare_agent(compressor)
    def persist(messages, history):
        events.append(("persist", list(messages), history))
        return True

    agent._persist_session.side_effect = persist
    history = [{"role": "user", "content": "earlier"}]

    result = agent._proactively_prune_tool_results(
        raw, current_tokens=101, conversation_history=history
    )

    assert result is projected
    assert [event[0] for event in events] == ["persist", "prune"]
    assert events[0][1][1]["content"] == "raw payload"
    agent._persist_session.assert_called_once_with(raw, history)


def test_post_tool_projection_is_noop_when_disabled_or_below_trigger():
    prune = MagicMock(return_value=([{"role": "tool", "content": "projected"}], 1))
    raw = [{"role": "tool", "content": "raw"}]

    for trigger, current_tokens in ((0, 1000), (1001, 1000)):
        agent = _bare_agent(
            SimpleNamespace(
                proactive_prune_tokens=trigger,
                prune_tool_results_only=prune,
            )
        )
        assert agent._proactively_prune_tool_results(raw, current_tokens) is raw
        agent._persist_session.assert_not_called()

    prune.assert_not_called()


def test_post_tool_projection_failure_fails_open_after_raw_persistence():
    compressor = SimpleNamespace(
        proactive_prune_tokens=1,
        prune_tool_results_only=MagicMock(side_effect=RuntimeError("projection failed")),
    )
    agent = _bare_agent(compressor)
    raw = [{"role": "tool", "content": "raw"}]

    assert agent._proactively_prune_tool_results(raw, current_tokens=2) is raw
    agent._persist_session.assert_called_once_with(raw, None)


def test_post_tool_projection_requires_confirmed_canonical_ledger_write():
    compressor = SimpleNamespace(
        proactive_prune_tokens=1,
        prune_tool_results_only=MagicMock(return_value=([{"role": "tool"}], 1)),
    )
    agent = _bare_agent(compressor)
    agent._persist_session.return_value = False
    raw = [{"role": "tool", "content": "raw"}]

    assert agent._proactively_prune_tool_results(raw, current_tokens=2) is raw
    compressor.prune_tool_results_only.assert_not_called()


def test_durable_first_flush_does_not_duplicate_rows_on_later_persistence():
    raw = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "raw payload"},
    ]
    projected = [raw[0], {**raw[1], "content": "[projected]"}]
    compressor = SimpleNamespace(
        proactive_prune_tokens=1,
        prune_tool_results_only=MagicMock(return_value=(projected, 1)),
    )
    agent = AIAgent.__new__(AIAgent)
    agent.context_compressor = compressor
    agent.persist_session = True
    agent._session_db = MagicMock()
    agent._last_flushed_db_idx = 0
    agent.session_id = "session-1"
    agent.platform = "cli"
    agent.model = "test-model"
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._save_session_log = MagicMock()

    result = agent._proactively_prune_tool_results(raw, current_tokens=2)
    assert result is projected
    assert agent._session_db.append_message.call_count == 2

    later = result + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    agent._persist_session(later)

    assert agent._session_db.append_message.call_count == 4
    persisted_contents = [
        call.kwargs["content"] for call in agent._session_db.append_message.call_args_list
    ]
    assert persisted_contents == [None, "raw payload", "done", "next"]
