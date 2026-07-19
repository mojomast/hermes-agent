import json
import sqlite3

import pytest

from agent.outcome_events import (
    OPERATOR_RELATION_PRODUCER,
    TAXONOMY_CODES,
    append_outcome_event,
    append_outcome_event_relation,
    retract_outcome_event,
)
from agent.shadow_recurrence import (
    LESSON_POLICY_VERSION,
    SHADOW_RECURRENCE_SCHEMA_VERSION,
    evaluate_shadow_recurrence,
)


def _trace(path, trace_id, session_id):
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS traces (trace_id TEXT PRIMARY KEY, session_id TEXT)")
        con.execute("INSERT INTO traces VALUES (?,?)", (trace_id, session_id))


def _negative(path, trace_id, event_id, taxonomy="wrong_scope", created_at=1):
    return append_outcome_event(
        path, trace_id=trace_id, event_id=event_id, event_type="wrong_scope",
        source="user", polarity="negative", confidence=.99,
        taxonomy_code=taxonomy, created_at=created_at,
    )


def _pair(path, original_trace, replacement_trace, prefix, taxonomy="wrong_scope", verifier="verification_passed"):
    original = _negative(path, original_trace, prefix + "-original", taxonomy, 10)
    append_outcome_event(
        path, trace_id=replacement_trace, event_id=prefix + "-correction",
        event_type="user_correction", source="user", polarity="positive", confidence=.99,
        taxonomy_code=taxonomy, supersedes_event_id=original.event_id, created_at=11,
    )
    return append_outcome_event(
        path, trace_id=replacement_trace, event_id=prefix + "-verify",
        event_type=verifier, source="verifier",
        polarity="positive" if verifier == "verification_passed" else "negative",
        confidence=1, created_at=12,
    )


def _item(report, taxonomy="wrong_scope"):
    return next(item for item in report["lessons"] if item["taxonomy_code"] == taxonomy)


def test_absent_observed_candidate_thresholds_and_dedup(tmp_path):
    path = tmp_path / "state.db"
    report = evaluate_shadow_recurrence(path)
    assert not path.exists()
    assert report["schema_version"] == SHADOW_RECURRENCE_SCHEMA_VERSION
    assert report["policy_version"] == LESSON_POLICY_VERSION
    assert len(report["lessons"]) == len(TAXONOMY_CODES)
    assert _item(report)["lifecycle_state"] == 0

    _negative(path, "root-a", "a")
    # Retries/duplicate evidence rows on one root do not add recurrence.
    _negative(path, "root-a", "a-duplicate", created_at=2)
    assert _item(evaluate_shadow_recurrence(path))["lifecycle_state"] == 1
    _negative(path, "root-b", "b", created_at=3)
    item = _item(evaluate_shadow_recurrence(path))
    assert item["lifecycle_state"] == 2
    assert item["state"] == "candidate"
    assert item["effective_root_count"] == 2


def test_three_roots_require_two_non_null_sessions_and_current_pair(tmp_path):
    path = tmp_path / "state.db"
    for trace_id in ("a", "b", "c"):
        _trace(path, trace_id, "one-session")
        _negative(path, trace_id, trace_id)
    item = _item(evaluate_shadow_recurrence(path))
    assert item["effective_root_count"] == 3
    assert item["session_count"] == 1
    assert item["activation_eligible"] is False

    _trace(path, "replacement", "second-session")
    with sqlite3.connect(path) as con:
        con.execute("UPDATE traces SET session_id='second-session' WHERE trace_id='c'")
    # A verified correction remains a separate gate, not an independent mistake root.
    _pair(path, "a", "replacement", "pair")
    item = _item(evaluate_shadow_recurrence(path))
    assert item["effective_root_count"] == 3
    assert item["session_count"] == 2
    assert item["valid_strict_corrected_pair_count"] == 1
    assert item["activation_eligible"] is True
    assert item["state"] == "activation_eligible"
    assert item["active"] is False


def test_no_pair_blocks_and_later_fail_then_pass_changes_eligibility(tmp_path):
    path = tmp_path / "state.db"
    for index, session in enumerate(("s1", "s1", "s2")):
        trace = f"root-{index}"
        _trace(path, trace, session)
        _negative(path, trace, f"neg-{index}")
    assert _item(evaluate_shadow_recurrence(path))["activation_eligible"] is False

    _trace(path, "replacement", "s2")
    _pair(path, "root-0", "replacement", "pair")
    assert _item(evaluate_shadow_recurrence(path))["activation_eligible"] is True
    append_outcome_event(
        path, trace_id="replacement", event_id="later-fail", event_type="verification_failed",
        source="verifier", polarity="negative", confidence=1, created_at=13,
    )
    assert _item(evaluate_shadow_recurrence(path))["activation_eligible"] is False
    append_outcome_event(
        path, trace_id="replacement", event_id="later-pass", event_type="verification_passed",
        source="verifier", polarity="positive", confidence=1, created_at=14,
    )
    assert _item(evaluate_shadow_recurrence(path))["activation_eligible"] is True


def test_retraction_downgrades_and_report_is_private(tmp_path):
    path = tmp_path / "state.db"
    target = _negative(path, "private-trace", "target")
    operator = append_outcome_event(
        path, trace_id="operator-trace", event_id="operator", event_type="assumption_invalidated",
        source="operator", polarity="neutral", confidence=1,
    )
    before = evaluate_shadow_recurrence(path)
    lesson_id = _item(before)["lesson_id"]
    retract_outcome_event(
        path, source_event_id=operator.event_id, target_event_id=target.event_id,
        producer=OPERATOR_RELATION_PRODUCER,
    )
    after = evaluate_shadow_recurrence(path)
    assert _item(after)["lifecycle_state"] == 0
    assert _item(after)["historical_root_count"] == 1
    assert _item(after)["lesson_id"] == lesson_id
    blob = json.dumps(after)
    assert "private-trace" not in blob and "operator-trace" not in blob
    for forbidden in ("trace_id", "session_id", "event_id", "relation_id"):
        assert forbidden not in blob
    assert "CANARY_RAW_USER_TEXT" not in blob
    assert after["shadow_only"] is True
    assert after["activation_allowed"] is False
    assert after["prompt_modified"] is False
    assert after["privacy"]["raw_content_exported"] is False


def test_counts_noise_ordering_and_bounded_limit(tmp_path):
    path = tmp_path / "state.db"
    _negative(path, "wanted", "wanted", "wrong_scope", 1)
    for index in range(205):
        append_outcome_event(
            path, trace_id=f"noise-{index}", event_id=f"noise-{index:03}",
            event_type="verification_failed", source="system", polarity="negative",
            confidence=0, created_at=100 + index,
        )
    report = evaluate_shadow_recurrence(path, limit=999)
    assert report["historical_event_count"] == 206
    assert report["effective_event_count"] == 206
    assert report["captured_verifier_counts"] == {
        "historical_pass": 0, "historical_fail": 0,
        "effective_pass": 0, "effective_fail": 0,
    }
    assert _item(report)["effective_root_count"] == 1
    assert [x["taxonomy_code"] for x in report["lessons"]] == sorted(TAXONOMY_CODES)
    with pytest.raises(ValueError, match="limit"):
        evaluate_shadow_recurrence(path, limit=0)


def test_only_trusted_user_mistakes_advance_recurrence(tmp_path):
    path = tmp_path / "state.db"
    noisy = [
        ("low", "user", "negative", .89, "wrong_scope"),
        ("system", "system", "negative", 1, "wrong_scope"),
        ("evaluator", "evaluator", "negative", 1, "wrong_scope"),
        ("operator", "operator", "negative", 1, "wrong_scope"),
        ("neutral", "user", "neutral", 1, "wrong_scope"),
        ("positive", "user", "positive", 1, "wrong_scope"),
        ("unrelated", "user", "negative", 1, "unsupported_claim"),
    ]
    for event_id, source, polarity, confidence, taxonomy in noisy:
        append_outcome_event(
            path, trace_id=event_id, event_id=event_id, event_type="wrong_scope",
            source=source, polarity=polarity, confidence=confidence, taxonomy_code=taxonomy,
        )
    assert _item(evaluate_shadow_recurrence(path))["effective_root_count"] == 0
    _negative(path, "trusted", "trusted")
    assert _item(evaluate_shadow_recurrence(path))["effective_root_count"] == 1


def test_retraction_of_retractor_restores_recurrence_root(tmp_path):
    path = tmp_path / "state.db"
    target = _negative(path, "mistake", "mistake")
    retractor = append_outcome_event(
        path, trace_id="audit-1", event_id="retractor", event_type="assumption_invalidated",
        source="operator", polarity="neutral", confidence=1,
    )
    retract_outcome_event(path, source_event_id=retractor.event_id, target_event_id=target.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    assert _item(evaluate_shadow_recurrence(path))["effective_root_count"] == 0
    undo = append_outcome_event(
        path, trace_id="audit-2", event_id="undo", event_type="assumption_invalidated",
        source="operator", polarity="neutral", confidence=1,
    )
    retract_outcome_event(path, source_event_id=undo.event_id, target_event_id=retractor.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    assert _item(evaluate_shadow_recurrence(path))["effective_root_count"] == 1


def test_corrects_relation_preserves_original_recurrence_root(tmp_path):
    path = tmp_path / "state.db"
    original = _negative(path, "mistake", "mistake")
    correction = append_outcome_event(
        path, trace_id="replacement", event_id="correction", event_type="user_correction",
        source="user", polarity="positive", confidence=.99, taxonomy_code="wrong_scope",
        supersedes_event_id=original.event_id,
    )
    append_outcome_event_relation(
        path, source_event_id=correction.event_id, target_event_id=original.event_id,
        relation_kind="corrects", producer=OPERATOR_RELATION_PRODUCER,
    )
    assert _item(evaluate_shadow_recurrence(path))["effective_root_count"] == 1
