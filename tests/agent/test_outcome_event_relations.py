import sqlite3

import pytest

from agent.outcome_events import (
    OPERATOR_RELATION_PRODUCER,
    append_outcome_event,
    append_outcome_event_relation,
    outcome_evidence_summaries,
    outcome_summary,
    query_effective_outcome_events,
    query_historical_outcome_events,
    query_outcome_correction_pairs,
    retract_outcome_event,
)


def event(path, event_id, trace_id="trace", **overrides):
    values = dict(event_type="verification_passed", source="verifier", polarity="positive", confidence=1.0)
    values.update(overrides)
    return append_outcome_event(path, trace_id=trace_id, event_id=event_id, **values)


def test_relation_store_is_append_only_idempotent_and_validated(tmp_path):
    path = tmp_path / "state.db"
    event(path, "new")
    event(path, "old")
    relation = append_outcome_event_relation(
        path, source_event_id="new", target_event_id="old", relation_kind="replaces",
        producer=OPERATOR_RELATION_PRODUCER,
    )
    assert append_outcome_event_relation(
        path, source_event_id="new", target_event_id="old", relation_kind="replaces",
        producer=OPERATOR_RELATION_PRODUCER,
    ) == relation
    with sqlite3.connect(path) as con:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE outcome_event_relations SET relation_kind='corrects'")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM outcome_event_relations")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT OR REPLACE INTO outcome_event_relations VALUES (?,?,?,?,?,?)", (
                relation.relation_id, "old", "new", "replaces", OPERATOR_RELATION_PRODUCER, 1,
            ))
    for kwargs, match in [
        ({"source_event_id": "new", "target_event_id": "new", "relation_kind": "corrects", "producer": OPERATOR_RELATION_PRODUCER}, "self"),
        ({"source_event_id": "missing", "target_event_id": "old", "relation_kind": "corrects", "producer": OPERATOR_RELATION_PRODUCER}, "endpoint"),
        ({"source_event_id": "new", "target_event_id": "old", "relation_kind": "unknown", "producer": OPERATOR_RELATION_PRODUCER}, "relation kind"),
        ({"source_event_id": "new", "target_event_id": "old", "relation_kind": "corrects", "producer": "user"}, "unauthorized producer"),
    ]:
        with pytest.raises(ValueError, match=match):
            append_outcome_event_relation(path, **kwargs)
    with pytest.raises(ValueError, match="cycle"):
        append_outcome_event_relation(path, source_event_id="old", target_event_id="new", relation_kind="corrects", producer=OPERATOR_RELATION_PRODUCER)
    with pytest.raises(ValueError, match="unauthorized producer"):
        retract_outcome_event(path, source_event_id="new", target_event_id="old",
                              producer=OPERATOR_RELATION_PRODUCER)


def test_effective_and_historical_queries_and_summaries(tmp_path):
    path = tmp_path / "state.db"
    old = event(path, "old", event_type="verification_failed", polarity="negative", created_at=1)
    new = event(path, "new", created_at=2)
    append_outcome_event_relation(path, source_event_id=new.event_id, target_event_id=old.event_id,
                                  relation_kind="replaces", producer=OPERATOR_RELATION_PRODUCER)
    assert query_effective_outcome_events(path, trace_id="trace") == [new]
    assert query_historical_outcome_events(path, trace_id="trace") == [new, old]
    assert outcome_summary(path, trace_id="trace")["event_count"] == 1
    assert outcome_summary(path, trace_id="trace", historical=True)["event_count"] == 2
    assert outcome_evidence_summaries(path, ["trace"])["trace"].strict_verifier_status == "verification_passed"


def test_corrects_is_non_suppressing_and_suppressed_suppressor_restores_target(tmp_path):
    path = tmp_path / "state.db"
    original = event(path, "original", "old", event_type="wrong_scope", source="user",
                     polarity="negative", confidence=.99, taxonomy_code="wrong_scope", created_at=1)
    correction = event(path, "correction", "new", event_type="user_correction", source="user",
                       confidence=.99, taxonomy_code="wrong_scope", supersedes_event_id=original.event_id,
                       created_at=2)
    append_outcome_event_relation(
        path, source_event_id=correction.event_id, target_event_id=original.event_id,
        relation_kind="corrects", producer=OPERATOR_RELATION_PRODUCER,
    )
    assert {row.event_id for row in query_effective_outcome_events(path)} == {"original", "correction"}

    retractor = event(path, "retractor", "audit-1", event_type="assumption_invalidated",
                      source="operator", polarity="neutral", created_at=3)
    retract_outcome_event(path, source_event_id=retractor.event_id, target_event_id=original.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    assert "original" not in {row.event_id for row in query_effective_outcome_events(path)}

    undo = event(path, "undo", "audit-2", event_type="assumption_invalidated",
                 source="operator", polarity="neutral", created_at=4)
    retract_outcome_event(path, source_event_id=undo.event_id, target_event_id=retractor.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    effective = {row.event_id for row in query_effective_outcome_events(path)}
    assert "original" in effective and "retractor" not in effective
    assert outcome_summary(path)["event_count"] == 3
    assert outcome_evidence_summaries(path, ["old"])["old"].event_count == 1


def test_operator_retraction_is_structural_and_invalidates_latest_verifier(tmp_path):
    path = tmp_path / "state.db"
    original = event(path, "original", "old", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="wrong_scope", created_at=1)
    correction = event(path, "correction", "new", event_type="user_correction", source="user", confidence=.99, taxonomy_code="wrong_scope", supersedes_event_id=original.event_id, created_at=2)
    verifier = event(path, "verifier", "new", created_at=3)
    operator = event(path, "operator-action", "audit", event_type="assumption_invalidated", source="operator", polarity="neutral", created_at=4)
    assert len(query_outcome_correction_pairs(path, ["new"])) == 1
    retract_outcome_event(path, source_event_id=operator.event_id, target_event_id=verifier.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    assert query_outcome_correction_pairs(path, ["new"]) == []
    assert verifier not in query_effective_outcome_events(path, trace_id="new")
    assert verifier in query_historical_outcome_events(path, trace_id="new")
    with pytest.raises(TypeError):
        retract_outcome_event(path, source_event_id=operator.event_id, target_event_id=correction.event_id,
                              producer=OPERATOR_RELATION_PRODUCER, reason="raw text forbidden")


def test_retracted_newest_verifier_falls_back_to_latest_effective_verifier(tmp_path):
    path = tmp_path / "state.db"
    original = event(path, "original", "old", event_type="wrong_scope", source="user",
                     polarity="negative", confidence=.99, taxonomy_code="wrong_scope", created_at=1)
    event(path, "correction", "new", event_type="user_correction", source="user", confidence=.99,
          taxonomy_code="wrong_scope", supersedes_event_id=original.event_id, created_at=2)
    event(path, "older-pass", "new", created_at=3)
    newest = event(path, "newest-fail", "new", event_type="verification_failed",
                   polarity="negative", created_at=4)
    assert query_outcome_correction_pairs(path, ["new"]) == []
    operator = event(path, "operator", "audit", event_type="assumption_invalidated", source="operator",
                     polarity="neutral", created_at=5)
    retract_outcome_event(path, source_event_id=operator.event_id, target_event_id=newest.event_id,
                          producer=OPERATOR_RELATION_PRODUCER)
    assert len(query_outcome_correction_pairs(path, ["new"])) == 1
    assert outcome_evidence_summaries(path, ["new"])["new"].strict_verifier_status == "verification_passed"


@pytest.mark.parametrize("retracted_id", ["original", "correction"])
def test_retracting_either_correction_pair_endpoint_invalidates_pair(tmp_path, retracted_id):
    path = tmp_path / "state.db"
    original = event(path, "original", "old", event_type="wrong_scope", source="user",
                     polarity="negative", confidence=.99, taxonomy_code="wrong_scope", created_at=1)
    correction = event(path, "correction", "new", event_type="user_correction", source="user",
                       confidence=.99, taxonomy_code="wrong_scope", supersedes_event_id=original.event_id,
                       created_at=2)
    event(path, "verifier", "new", created_at=3)
    operator = event(path, "operator-action", "audit", event_type="assumption_invalidated",
                     source="operator", polarity="neutral", created_at=4)
    append_outcome_event_relation(
        path, source_event_id=correction.event_id, target_event_id=original.event_id,
        relation_kind="corrects", producer=OPERATOR_RELATION_PRODUCER,
    )
    assert len(query_outcome_correction_pairs(path, ["new"])) == 1
    retract_outcome_event(
        path, source_event_id=operator.event_id, target_event_id=retracted_id,
        producer=OPERATOR_RELATION_PRODUCER,
    )
    assert query_outcome_correction_pairs(path, ["new"]) == []
    assert {item.event_id for item in query_historical_outcome_events(path)} >= {"original", "correction"}


def test_effective_queries_filter_before_cap_and_missing_read_does_not_create(tmp_path):
    path = tmp_path / "state.db"
    assert query_effective_outcome_events(path) == []
    assert query_historical_outcome_events(path) == []
    assert not path.exists()
    for index in range(205):
        event(path, f"noise-{index:03}", f"noise-{index}", created_at=100 + index)
    wanted = event(path, "wanted", "wanted", created_at=1)
    assert query_effective_outcome_events(path, trace_id="wanted", limit=500) == [wanted]


def test_effective_graph_resolution_is_complete_beyond_query_cap(tmp_path):
    path = tmp_path / "state.db"
    target = event(path, "target", "target", event_type="assumption_invalidated",
                   source="operator", polarity="neutral", created_at=1)
    previous = target
    for index in range(201):
        current = event(path, f"control-{index:03}", f"audit-{index}",
                        event_type="assumption_invalidated", source="operator",
                        polarity="neutral", created_at=index + 2)
        retract_outcome_event(path, source_event_id=current.event_id, target_event_id=previous.event_id,
                              producer=OPERATOR_RELATION_PRODUCER)
        previous = current
    # An odd number of effective-suppression links leaves the oldest target suppressed;
    # resolution must not truncate the dependency chain at MAX_QUERY_LIMIT (200).
    assert query_effective_outcome_events(path, trace_id="target", limit=500) == []
