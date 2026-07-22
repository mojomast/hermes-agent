import copy
import json

from agent.behavioral_adaptation import AdaptationDecision
from agent.tracing import TraceRecorder
from run_agent import AIAgent


def _agent():
    agent = object.__new__(AIAgent)
    agent.ephemeral_system_prompt = "CALLER_EPHEMERAL_CANARY"
    agent._behavioral_adaptation_decision = AdaptationDecision(
        suffix="<behavioral-hint>Check structural prerequisites before acting.</behavioral-hint>",
        telemetry={
            "schema_version": "behavioral_adaptation_decision.v1", "policy_version": "behavioral_adaptation.v1",
            "config_state": "enabled", "cohort": "treatment", "eligible": True, "activated": True,
            "reason": "activated", "taxonomy_code": "prerequisite_not_checked", "hint_char_count": 84,
            "privacy_contract_valid": True, "raw_content_stored": False, "identifiers_stored": False,
        },
    )
    return agent


def test_api_assembly_suffix_is_ephemeral_and_does_not_mutate_inputs(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_BEHAVIORAL_ADAPTATION", raising=False)
    agent = _agent()
    cached = "CACHED_SYSTEM"
    messages = [{"role": "user", "content": "hello"}]
    history = copy.deepcopy(messages)
    caller_ephemeral = agent.ephemeral_system_prompt

    effective = agent._effective_system_for_api(cached)
    assert effective.startswith(cached)
    assert caller_ephemeral in effective
    assert "behavioral-hint" in effective
    assert cached == "CACHED_SYSTEM"
    assert messages == history
    assert agent.ephemeral_system_prompt == caller_ephemeral
    assert agent._behavioral_adaptation_decision.suffix not in json.dumps(messages)


def test_runtime_kill_switch_is_checked_at_each_request_assembly(monkeypatch):
    agent = _agent()
    monkeypatch.setenv("HERMES_DISABLE_BEHAVIORAL_ADAPTATION", "true")
    killed = agent._effective_system_for_api("SYSTEM")
    assert "behavioral-hint" not in killed
    assert "CALLER_EPHEMERAL_CANARY" in killed
    monkeypatch.setenv("HERMES_DISABLE_BEHAVIORAL_ADAPTATION", "false")
    assert "behavioral-hint" in agent._effective_system_for_api("SYSTEM")


def test_decision_telemetry_uses_existing_bounded_trace_span(monkeypatch, tmp_path):
    agent = object.__new__(AIAgent)
    agent._behavioral_adaptation_config = {"enabled": False}
    agent._behavioral_adaptation_foreground = True
    agent.session_id = "CANARY_SESSION"
    agent._session_db = None
    agent._turn_tracer = TraceRecorder("s", "t", "h")
    agent._decide_behavioral_adaptation_for_turn("CANARY_RAW_TEXT")
    rows = agent._turn_tracer.to_span_rows()
    row = next(r for r in rows if r["name"] == "behavioral_adaptation_decision")
    metadata = json.loads(row["metadata_json"])
    assert metadata["activated"] is False
    assert "CANARY_RAW_TEXT" not in row["metadata_json"]
    assert "CANARY_SESSION" not in row["metadata_json"]


def test_one_policy_decision_is_reused_without_requery(monkeypatch):
    agent = _agent()
    original = agent._behavioral_adaptation_decision
    assert agent._effective_system_for_api("S").count("behavioral-hint") == 2  # open + close tag
    assert agent._effective_system_for_api("S").count("behavioral-hint") == 2
    assert agent._behavioral_adaptation_decision is original
