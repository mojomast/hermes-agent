import json
import sqlite3

import pytest

from agent.behavioral_adaptation import (
    MAX_SUFFIX_CHARS,
    decide_behavioral_adaptation,
    parse_behavioral_adaptation_config,
    treatment_for_session,
)
from agent.outcome_events import append_outcome_event, retract_outcome_event, OPERATOR_RELATION_PRODUCER
from hermes_cli.config import DEFAULT_CONFIG


def _report(eligible=True, taxonomy="wrong_scope"):
    return {
        "schema_version": "shadow_recurrence_report.v1",
        "policy_version": "shadow_lesson_lifecycle.v1",
        "policy": {
            "observed_minimum_roots": 1,
            "candidate_minimum_roots": 2,
            "eligibility_minimum_roots": 3,
            "eligibility_minimum_sessions": 2,
            "strict_corrected_pair_required": True,
        },
        "privacy": {
            "raw_content_exported": False,
            "raw_user_text_read": False,
            "raw_prompt_read": False,
            "raw_tool_payloads_read": False,
            "evidence_digests_exported": False,
            "store_identifiers_exported": False,
            "taxonomy_codes_allowlisted": True,
            "deterministic_pseudonymous_lesson_ids": True,
            "pseudonymous_lesson_ids_linkable_across_reports": True,
        },
        "lessons": [{
            "lesson_id": "CANARY_PRIVATE_STORE_ID",
            "taxonomy_code": taxonomy,
            "lifecycle_state": 3 if eligible else 2,
            "state": "activation_eligible" if eligible else "candidate",
            "effective_root_count": 3,
            "session_count": 2,
            "valid_strict_corrected_pair_count": 1 if eligible else 0,
            "activation_eligible": eligible,
        }],
    }


def _eligible_db(path):
    def event(trace, event_id, *, taxonomy="wrong_scope", created=1, **kw):
        return append_outcome_event(
            path, trace_id=trace, event_id=event_id, taxonomy_code=taxonomy,
            confidence=.99, created_at=created, **kw,
        )

    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, session_id TEXT)")
        con.executemany("INSERT INTO traces VALUES (?,?)", [("a", "s1"), ("b", "s1"), ("c", "s2"), ("fixed", "s2"), ("operator", "s2")])
    originals = []
    for i, trace in enumerate(("a", "b", "c")):
        originals.append(event(trace, f"neg-{trace}", event_type="wrong_scope", source="user", polarity="negative", created=i + 1))
    event("fixed", "correction", event_type="user_correction", source="user", polarity="positive",
          supersedes_event_id=originals[0].event_id, created=10)
    event("fixed", "pass", taxonomy=None, event_type="verification_passed", source="verifier", polarity="positive", created=11)


def _treatment_session(percent=50):
    for i in range(10000):
        value = f"session-{i}"
        if treatment_for_session(value, percent):
            return value
    raise AssertionError("no treatment session")


def test_config_is_strict_default_off_and_literal_true_only():
    assert DEFAULT_CONFIG["behavioral_adaptation"] == {"enabled": False, "treatment_percent": 50}
    assert parse_behavioral_adaptation_config(None).enabled is False
    assert parse_behavioral_adaptation_config({}).enabled is False
    assert parse_behavioral_adaptation_config({"enabled": "true"}).valid is False
    assert parse_behavioral_adaptation_config({"enabled": 1}).valid is False
    assert parse_behavioral_adaptation_config({"enabled": True, "treatment_percent": "50"}).valid is False
    cfg = parse_behavioral_adaptation_config({"enabled": True, "treatment_percent": 50})
    assert cfg.enabled is True and cfg.valid is True


def test_treatment_control_assignment_is_deterministic_and_stable():
    values = [treatment_for_session("stable-session", 50) for _ in range(10)]
    assert len(set(values)) == 1
    assert treatment_for_session("anything", 0) is False
    assert treatment_for_session("anything", 100) is True
    cohorts = {treatment_for_session(f"s-{i}", 50) for i in range(100)}
    assert cohorts == {False, True}


def test_only_relevant_activation_eligible_strict_pair_generates_one_fixed_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.behavioral_adaptation.evaluate_shadow_recurrence", lambda _: _report())
    decision = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 100}, session_id="s",
        task_text="The prior answer was wrong; confirm the requested scope", db_path=tmp_path / "x.db",
        foreground=True,
    )
    assert decision.suffix and decision.telemetry["activated"] is True
    assert decision.telemetry["taxonomy_code"] == "wrong_scope"
    assert len(decision.suffix) <= MAX_SUFFIX_CHARS
    assert "CANARY_PRIVATE_STORE_ID" not in json.dumps(decision.as_dict())
    assert "requested scope" in decision.suffix


def test_control_shadow_and_irrelevant_turns_never_modify_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.behavioral_adaptation.evaluate_shadow_recurrence", lambda _: _report())
    control = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 0}, session_id="s",
        task_text="wrong scope", db_path=tmp_path / "x", foreground=True,
    )
    irrelevant = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 100}, session_id="s",
        task_text="write a poem", db_path=tmp_path / "x", foreground=True,
    )
    assert control.suffix is None and control.telemetry["cohort"] == "control"
    assert irrelevant.suffix is None and irrelevant.telemetry["reason"] == "no_relevant_eligible_pair"


def test_malformed_privacy_contract_and_db_errors_fail_closed(monkeypatch, tmp_path):
    bad = _report()
    bad["privacy"]["raw_prompt_read"] = True
    monkeypatch.setattr("agent.behavioral_adaptation.evaluate_shadow_recurrence", lambda _: bad)
    decision = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 100}, session_id="s", task_text="wrong scope",
        db_path=tmp_path / "x", foreground=True,
    )
    assert decision.suffix is None and decision.telemetry["reason"] == "privacy_contract_invalid"
    monkeypatch.setattr("agent.behavioral_adaptation.evaluate_shadow_recurrence", lambda _: (_ for _ in ()).throw(sqlite3.DatabaseError("CANARY_DB_SECRET")))
    failed = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 100}, session_id="s", task_text="wrong scope",
        db_path=tmp_path / "x", foreground=True,
    )
    assert failed.suffix is None and failed.telemetry["reason"] == "evidence_unavailable"
    assert "CANARY_DB_SECRET" not in json.dumps(failed.as_dict())


def test_foreground_only_and_no_raw_text_or_identifiers_in_decision(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.behavioral_adaptation.evaluate_shadow_recurrence", lambda _: _report())
    canary = "CANARY_RAW_USER_TEXT_9831"
    decision = decide_behavioral_adaptation(
        config={"enabled": True, "treatment_percent": 100}, session_id="CANARY_SESSION_ID",
        task_text=f"wrong scope {canary}", db_path=tmp_path / "x", foreground=False,
    )
    blob = json.dumps(decision.as_dict())
    assert decision.suffix is None and decision.telemetry["reason"] == "non_foreground"
    assert canary not in blob and "CANARY_SESSION_ID" not in blob


def test_real_retraction_and_later_failure_invalidate_strict_pair(tmp_path):
    path = tmp_path / "state.db"
    _eligible_db(path)
    session = _treatment_session()
    kwargs = dict(config={"enabled": True, "treatment_percent": 50}, session_id=session,
                  task_text="wrong scope", db_path=path, foreground=True)
    assert decide_behavioral_adaptation(**kwargs).suffix
    append_outcome_event(path, trace_id="fixed", event_id="later-fail", event_type="verification_failed",
                         source="verifier", polarity="negative", confidence=1, created_at=12)
    assert decide_behavioral_adaptation(**kwargs).suffix is None

    # A fresh eligible database loses eligibility when one recurrence root is retracted.
    path2 = tmp_path / "retracted.db"
    _eligible_db(path2)
    operator = append_outcome_event(path2, trace_id="operator", event_id="operator", event_type="assumption_invalidated",
                                    source="operator", polarity="neutral", confidence=1)
    retract_outcome_event(path2, source_event_id=operator.event_id, target_event_id="neg-c",
                          producer=OPERATOR_RELATION_PRODUCER)
    kwargs["db_path"] = path2
    assert decide_behavioral_adaptation(**kwargs).suffix is None
