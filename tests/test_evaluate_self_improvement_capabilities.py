import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.evaluate_self_improvement_capabilities as eval_script
from hermes_state import SessionDB
from scripts.evaluate_self_improvement_capabilities import (
    FIXTURE_PRIVACY_CANARIES,
    evaluate_synthetic_shadow_recurrence,
    run_eval,
)


def test_evaluate_self_improvement_capabilities_includes_fixture_replay_eval():
    result = run_eval(limit=10, min_reward=1.0)

    assert "fixture_replay_eval" in result
    replay = result["fixture_replay_eval"]
    assert replay["schema_version"] == "training_episode_replay_eval.v1"
    assert replay["episode_count"] == 2
    assert replay["passed_count"] == 1
    assert replay["privacy"]["finding_count"] == 0

    shadow = result["fixture_contrastive_replay_eval"]
    assert shadow["schema_version"] == "contrastive_replay_eval.v1"
    assert shadow["shadow_only"] is True
    assert shadow["prompt_modified"] is False
    for metric in (
        "eligible_episode_count", "positive_match_count", "negative_match_count",
        "corrected_match_count", "generated_hint_count", "unsupported_hint_count",
        "privacy_finding_count", "taxonomy_counts", "recurrence_counts", "latency_ms",
    ):
        assert metric in shadow
    assert shadow["privacy_canary_coverage_count"] == len(set(FIXTURE_PRIVACY_CANARIES))
    assert shadow["expected_canary_coverage_count"] == len(set(FIXTURE_PRIVACY_CANARIES))
    assert shadow["planted_canary_count"] == len(set(FIXTURE_PRIVACY_CANARIES))
    assert shadow["canary_coverage_ok"] is True
    assert shadow["privacy_eval_valid"] is True
    assert shadow["negative_match_count"] >= 1
    assert shadow["corrected_match_count"] >= 1
    assert shadow["positive_match_count"] >= 1
    assert shadow["generated_hint_count"] >= 1
    assert shadow["unsupported_hint_count"] == 0
    assert shadow["recurrence_counts"]["duplicate_existing_capability"] >= 2
    assert "capability_extension_ideation" in shadow["task_frame"]["categories"]
    blob = json.dumps(shadow)
    assert not any(canary in blob for canary in FIXTURE_PRIVACY_CANARIES)
    assert shadow["ok"] is True

    recurrence = result["fixture_shadow_recurrence_eval"]
    assert recurrence["schema_version"] == "shadow_recurrence_eval.v1"
    assert recurrence["shadow_only"] is True
    assert recurrence["activation_allowed"] is False
    assert recurrence["prompt_modified"] is False
    assert recurrence["classified_candidate_count"] == 2
    assert recurrence["authorized_persist_count"] == 1
    assert recurrence["idempotent_retry_count"] == 1
    assert recurrence["authorization_finding_count"] == 0
    assert recurrence["idempotency_finding_count"] == 0
    assert recurrence["parity_finding_count"] == 0
    assert recurrence["privacy_finding_count"] == 0
    assert recurrence["historical_event_count"] > recurrence["effective_event_count"]
    assert recurrence["effective_retraction_count"] == 1
    assert recurrence["observed_transition_count"] >= 1
    assert recurrence["candidate_transition_count"] >= 1
    assert recurrence["activation_eligible_transition_count"] >= 1
    assert recurrence["verified_pair_count"] >= 1
    assert recurrence["downgrade_transition_count"] >= 1
    assert recurrence["recovery_transition_count"] >= 1
    assert recurrence["privacy_canary_coverage_count"] == len(set(FIXTURE_PRIVACY_CANARIES))
    assert recurrence["expected_canary_coverage_count"] == len(set(FIXTURE_PRIVACY_CANARIES))
    assert recurrence["canary_coverage_ok"] is True
    assert recurrence["ok"] is True
    recurrence_blob = json.dumps(recurrence)
    assert not any(canary in recurrence_blob for canary in FIXTURE_PRIVACY_CANARIES)
    for raw_key in ("trace_id", "event_id", "relation_id", "evidence_digest", "lesson_id"):
        assert raw_key not in recurrence_blob


def test_synthetic_recurrence_fixture_uses_independent_trusted_mistake_roots(tmp_path):
    db_path = tmp_path / "recurrence.db"

    section = evaluate_synthetic_shadow_recurrence(
        db_path,
        limit=200,
        planted_canary_count=len(set(FIXTURE_PRIVACY_CANARIES)),
    )

    # Synthetic manual user evidence must satisfy the production gates rather
    # than relying on verifier/evaluator events or repeated traces/sessions.
    with sqlite3.connect(db_path) as con:
        roots = con.execute(
            "SELECT e.trace_id,t.session_id,e.source,e.polarity,e.confidence,e.taxonomy_code "
            "FROM outcome_events e JOIN traces t ON t.trace_id=e.trace_id "
            "WHERE e.event_type='duplicate_proposal' ORDER BY e.event_id"
        ).fetchall()
    assert len(roots) == 4
    assert len({row[0] for row in roots}) == 4
    assert len({row[1] for row in roots}) == 4
    assert all(row[2:6] == ("user", "negative", .99, "duplicate_existing_capability") for row in roots)
    assert section["activation_eligible_transition_count"] == 1
    assert section["verified_pair_count"] == 1
    assert section["effective_retraction_count"] == 1
    assert section["activation_eligible_lesson_count"] == 1
    assert section["active_lesson_count"] == 0


def test_real_db_adds_bounded_count_only_shadow_recurrence(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    SessionDB(db_path).close()
    calls = []
    canonical = eval_script.evaluate_shadow_recurrence
    def fake_recurrence(path, *, limit):
        calls.append((path, limit))
        if path != db_path:
            return canonical(path, limit=limit)
        return {
            "schema_version": "shadow_recurrence_report.v1", "policy_version": "shadow_lesson_lifecycle.v1",
            "shadow_only": True, "activation_allowed": False, "prompt_modified": False,
            "historical_event_count": 4, "effective_event_count": 3, "candidate_lesson_count": 1,
            "activation_eligible_lesson_count": 1, "active_lesson_count": 0,
        }
    monkeypatch.setattr(eval_script, "evaluate_shadow_recurrence", fake_recurrence)
    result = run_eval(db_path=db_path, limit=10)
    assert calls[-1] == (db_path, 10)
    section = result["shadow_recurrence"]
    assert section["schema_version"] == "shadow_recurrence_eval.v1"
    assert section["historical_event_count"] == 4
    assert section["activation_eligible_lesson_count"] == 1
    assert "lessons" not in section and "lesson_id" not in json.dumps(section)
    assert section["shadow_only"] is True
    assert section["activation_allowed"] is False
    assert section["prompt_modified"] is False


def test_real_db_contrastive_eval_uses_default_task_limits_and_thresholds(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    SessionDB(db_path).close()
    calls = []
    monkeypatch.setattr(
        eval_script,
        "contrastive_replay_eval",
        lambda **kwargs: calls.append(kwargs) or {
            "privacy_canary_coverage_count": len(set(FIXTURE_PRIVACY_CANARIES)),
            "expected_canary_coverage_count": len(set(FIXTURE_PRIVACY_CANARIES)),
            "planted_canary_count": len(set(FIXTURE_PRIVACY_CANARIES)),
            "canary_coverage_ok": True,
            "privacy_eval_valid": True,
            "negative_match_count": 1,
            "corrected_match_count": 1,
            "positive_match_count": 1,
            "generated_hint_count": 1,
            "unsupported_hint_count": 0,
            "recurrence_counts": {"duplicate_existing_capability": 2},
            "task_frame": {"categories": ["capability_extension_ideation"]},
            "shadow_only": True,
            "prompt_modified": False,
            "ok": True,
        },
    )

    run_eval(db_path=db_path, limit=10, min_reward=2.0, max_negative_reward=-4.0)

    real_call = calls[-1]
    assert real_call["task_text"] == eval_script.DEFAULT_CONTRASTIVE_TASK_TEXT
    assert real_call["positive_limit"] == 10
    assert real_call["negative_limit"] == 10
    assert real_call["corrected_limit"] == 10
    assert real_call["min_positive_reward"] == 2.0
    assert real_call["max_negative_reward"] == -4.0
    assert "expected_canary_count" not in real_call
    assert "planted_canary_count" not in real_call


@pytest.mark.parametrize("field,value", [("min_reward", math.nan), ("max_negative_reward", math.inf)])
def test_run_eval_rejects_non_finite_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        run_eval(**{field: value})


@pytest.mark.parametrize("limit", [-1, 0, 201])
def test_run_eval_rejects_out_of_range_limit(limit):
    with pytest.raises(ValueError, match="limit"):
        run_eval(limit=limit)


@pytest.mark.parametrize("min_reward,max_negative_reward", [(1, 1), (1, 2)])
def test_run_eval_rejects_overlapping_reward_thresholds(min_reward, max_negative_reward):
    with pytest.raises(ValueError, match="max_negative_reward"):
        run_eval(min_reward=min_reward, max_negative_reward=max_negative_reward)


def test_eval_cli_accepts_limit_10_and_rejects_negative_limit(tmp_path):
    script = str(Path(eval_script.__file__).resolve())
    python = str(Path(sys.executable).absolute())
    valid = subprocess.run(
        [python, script, "--limit", "10", "--output-json", str(tmp_path / "report.json")],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert valid.returncode == 0, valid.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["fixture_contrastive_replay_eval"]["ok"] is True

    invalid = subprocess.run(
        [python, script, "--limit", "-1"],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert invalid.returncode != 0
    assert "limit" in invalid.stderr
