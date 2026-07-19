import sqlite3
from dataclasses import replace

import pytest

from agent.outcome_events import (
    OutcomeEvent,
    OutcomeEvidenceSummary,
    append_outcome_event,
    append_producer_outcome_event,
    init_outcome_events,
    outcome_summary,
    outcome_summaries,
    outcome_evidence_summaries,
    query_outcome_correction_pairs,
    query_outcome_events,
)
from agent.shadow_outcome_capture import ShadowOutcomeCandidate, classify_foreground_pytest
from hermes_state import SessionDB


def _trace_db(path):
    db = SessionDB(path)
    db.record_trace({
        "trace_id": "trace-1", "session_id": "s", "turn_id": "t",
        "user_message_hash": "opaque", "start_time": 1.0, "end_time": 2.0,
        "status": "completed", "total_wall_ms": 100,
        "total_model_calls": 1, "total_tool_calls": 0, "total_subagents": 0,
    }, [])
    db.close()


def test_init_is_additive_idempotent_and_append_only(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    init_outcome_events(path)
    init_outcome_events(path)
    event = append_outcome_event(
        path, trace_id="trace-1", event_type="user_correction", source="user",
        polarity="negative", confidence=0.99,
        taxonomy_code="duplicate_existing_capability",
        evidence_digest="sha256:" + "a" * 64,
    )
    assert isinstance(event, OutcomeEvent)
    init_outcome_events(path)
    assert query_outcome_events(path, trace_id="trace-1") == [event]
    with sqlite3.connect(path) as con:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE outcome_events SET confidence=.1 WHERE event_id=?", (event.event_id,))
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM outcome_events WHERE event_id=?", (event.event_id,))


def test_validation_rejects_unknown_values_ranges_digest_and_trace(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    init_outcome_events(path)
    base = dict(trace_id="trace-1", event_type="user_correction", source="user", polarity="negative", confidence=.8)
    for key, value in [
        ("event_type", "made_up"), ("source", "model"), ("polarity", "bad"),
        ("confidence", 1.1), ("taxonomy_code", "unknown"),
        ("evidence_digest", "raw correction wording"),
    ]:
        args = dict(base)
        args[key] = value
        with pytest.raises(ValueError):
            append_outcome_event(path, **args)
    with pytest.raises(ValueError):
        append_outcome_event(path, **{**base, "trace_id": "missing"})
    with pytest.raises(TypeError):
        append_outcome_event(path, **base, raw_message="must fail")
    for confidence in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            append_outcome_event(path, **{**base, "confidence": confidence})


def test_supersession_cross_trace_preserves_original_and_queries_are_bounded(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)", ("trace-2", "s", "t2", "opaque", 3, 4, "completed", 100, 1, 0, 0))
    first = append_outcome_event(
        path, trace_id="trace-1", event_type="duplicate_proposal", source="user",
        polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability",
    )
    second = append_outcome_event(
        path, trace_id="trace-2", event_type="user_correction", source="user",
        polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability",
        supersedes_event_id=first.event_id,
    )
    rows = query_outcome_events(path, taxonomy_code="duplicate_existing_capability", limit=500)
    assert [row.event_id for row in rows] == [second.event_id, first.event_id]
    assert len(rows) <= 200


def test_summary_is_compact_counts_only_and_empty_db_is_valid(tmp_path):
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    init_outcome_events(path)
    empty = outcome_summary(path)
    assert empty == {"event_count": 0, "event_type_counts": {}, "polarity_counts": {}, "taxonomy_counts": {}}

    # The event store can exist without traces; it still refuses no trace only when traces table exists.
    append_outcome_event(path, trace_id="opaque-trace", event_type="verification_passed", source="verifier", polarity="positive", confidence=1.0)
    summary = outcome_summary(path)
    assert summary["event_count"] == 1
    assert summary["event_type_counts"] == {"verification_passed": 1}
    assert "event_id" not in summary and "trace_id" not in summary


def test_insert_or_replace_cannot_mutate_existing_event(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    event = append_outcome_event(
        path, trace_id="trace-1", event_type="verification_passed", source="verifier",
        polarity="positive", confidence=1.0,
    )
    with sqlite3.connect(path) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT OR REPLACE INTO outcome_events VALUES (?,?,?,?,?,?,?,?,?,?)",
            (event.event_id, "trace-1", "verification_failed", "verifier", "negative", 1.0, None, event.created_at, None, None),
        )
    assert query_outcome_events(path, trace_id="trace-1") == [event]


def test_read_apis_do_not_create_missing_database(tmp_path):
    path = tmp_path / "missing.db"
    assert outcome_summary(path)["event_count"] == 0
    assert query_outcome_events(path) == []
    assert not path.exists()


def test_outcome_summaries_stably_dedupes_and_bounds_unique_trace_ids(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    _trace_db(path)
    append_outcome_event(path, trace_id="trace-1", event_type="verification_passed", source="verifier", polarity="positive", confidence=1.0)
    import agent.outcome_events as module
    real_connect = module._connect_readonly
    connections, statements = [], []

    class ConnectionProxy:
        def __init__(self, con): self.con = con
        def __enter__(self): return self
        def __exit__(self, *args): self.con.close()
        def execute(self, sql, args=()):
            statements.append(sql)
            return self.con.execute(sql, args)

    def tracked_connect(db_path):
        connections.append(db_path)
        return ConnectionProxy(real_connect(db_path))

    monkeypatch.setattr(module, "_connect_readonly", tracked_connect)
    summaries = outcome_summaries(path, ["trace-1", "no-events", "trace-1"])
    assert list(summaries) == ["trace-1", "no-events"]
    assert summaries["trace-1"]["event_count"] == 1
    assert summaries["no-events"]["event_count"] == 0
    assert len(connections) == 1
    event_queries = [sql for sql in statements if "FROM outcome_events" in sql]
    assert len(event_queries) == 1 and "GROUP BY" in event_queries[0]
    with pytest.raises(ValueError, match="200"):
        outcome_summaries(path, [f"trace-{index}" for index in range(201)])


def test_outcome_summaries_missing_db_does_not_create_it(tmp_path):
    path = tmp_path / "missing.db"
    assert outcome_summaries(path, ["a", "b"])["a"]["event_count"] == 0
    assert not path.exists()


def test_event_order_ties_use_event_id_not_rowid(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    for event_id in ("event-z", "event-a"):
        append_outcome_event(path, trace_id="trace-1", event_type="verification_passed", source="verifier", polarity="positive", confidence=1.0, event_id=event_id, created_at=10)
    assert [event.event_id for event in query_outcome_events(path)] == ["event-z", "event-a"]


def test_strict_correction_pair_join_finds_old_original_despite_newer_noise(tmp_path):
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    original = append_outcome_event(path, trace_id="old-original", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability", event_id="original", created_at=1)
    append_outcome_event(path, trace_id="replacement", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=original.event_id, event_id="correction", created_at=2)
    append_outcome_event(path, trace_id="replacement", event_type="verification_passed", source="verifier", polarity="positive", confidence=.9, event_id="verification", created_at=3)
    for index in range(205):
        append_outcome_event(path, trace_id=f"noise-{index}", event_type="verification_failed", source="system", polarity="negative", confidence=0, event_id=f"noise-{index:03}", created_at=100 + index)
    pairs = query_outcome_correction_pairs(path, ["replacement"])
    assert len(pairs) == 1
    assert pairs[0].replacement_trace_id == "replacement"
    assert pairs[0].original_trace_id == "old-original"
    assert pairs[0].taxonomy_code == "duplicate_existing_capability"
    assert not hasattr(pairs[0], "evidence_digest")
    with pytest.raises(ValueError, match="200"):
        query_outcome_correction_pairs(path, [f"trace-{index}" for index in range(201)])


def test_evidence_summaries_are_complete_bounded_and_privacy_safe(tmp_path):
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    append_outcome_event(path, trace_id="starved", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability", evidence_digest="sha256:" + "a" * 64)
    for index in range(205):
        append_outcome_event(path, trace_id="noisy", event_type="verification_failed", source="system", polarity="negative", confidence=0, event_id=f"noise-{index:03}", created_at=100 + index)

    summaries = outcome_evidence_summaries(path, ["starved", "noisy", "starved"])

    assert list(summaries) == ["starved", "noisy"]
    assert isinstance(summaries["starved"], OutcomeEvidenceSummary)
    assert summaries["starved"].event_count == 1
    assert summaries["starved"].has_negative_evidence is True
    assert summaries["starved"].trusted_user_negative_taxonomies == frozenset({"duplicate_existing_capability"})
    assert summaries["noisy"].event_count == 205
    assert not hasattr(summaries["starved"], "trace_id")
    assert not hasattr(summaries["starved"], "evidence_digest")
    with pytest.raises(ValueError, match="200"):
        outcome_evidence_summaries(path, [f"trace-{index}" for index in range(201)])


def _strict_pair_at_times(path, verifier_events):
    original = append_outcome_event(path, trace_id="old", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability", event_id="original", created_at=1)
    append_outcome_event(path, trace_id="new", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=original.event_id, event_id="correction", created_at=10)
    for event_id, event_type, created_at in verifier_events:
        append_outcome_event(path, trace_id="new", event_type=event_type, source="verifier", polarity="positive" if event_type == "verification_passed" else "negative", confidence=.95, event_id=event_id, created_at=created_at)
    return query_outcome_correction_pairs(path, ["new"])


def test_stale_pre_correction_verifier_pass_does_not_authorize_pair(tmp_path):
    assert _strict_pair_at_times(tmp_path / "state.db", [("pass", "verification_passed", 5)]) == []


def test_later_trusted_verifier_failure_invalidates_pair(tmp_path):
    assert _strict_pair_at_times(tmp_path / "state.db", [("pass", "verification_passed", 11), ("failure", "verification_failed", 12)]) == []


def test_latest_trusted_pass_after_correction_authorizes_pair(tmp_path):
    pairs = _strict_pair_at_times(tmp_path / "state.db", [("failure", "verification_failed", 11), ("pass", "verification_passed", 12)])
    assert len(pairs) == 1


def _producer_candidate(event_type="verification_passed", event_id=None):
    candidate = classify_foreground_pytest(
        trace_id="trace-1", tool_call_id="call-1", tool_name="terminal",
        tool_arguments={"command": "pytest tests", "background": False},
        tool_result={"exit_code": 0},
    )
    if event_type != candidate.event_type:
        candidate = replace(candidate, event_type=event_type)
    if event_id is not None:
        candidate = replace(candidate, event_id=event_id)
    return candidate


def test_authorized_producer_append_fixes_event_fields(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    event = append_producer_outcome_event(path, _producer_candidate())
    assert event.event_id.startswith("shadow:")
    assert event.trace_id == "trace-1"
    assert event.event_type == "verification_passed"
    assert event.source == "verifier"
    assert event.polarity == "positive"
    assert event.confidence == 1.0
    assert event.taxonomy_code is None
    assert event.evidence_digest is None
    assert event.supersedes_event_id is None


def test_authorized_producer_exact_retry_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    candidate = _producer_candidate()
    first = append_producer_outcome_event(path, candidate)
    second = append_producer_outcome_event(path, candidate)
    assert second == first
    assert query_outcome_events(path, trace_id="trace-1") == [first]


def test_authorized_producer_conflicting_duplicate_fails_closed(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    append_producer_outcome_event(path, _producer_candidate("verification_passed"))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        append_producer_outcome_event(path, _producer_candidate("verification_failed"))
    assert query_outcome_events(path, trace_id="trace-1")[0].event_type == "verification_passed"


@pytest.mark.parametrize("producer", ["", "foreground_pytest.v2", "user", "foreground_pytest.v1 "])
def test_authorized_producer_rejects_unrecognized_authority(tmp_path, producer):
    path = tmp_path / "state.db"
    _trace_db(path)
    candidate = ShadowOutcomeCandidate(
        event_id="shadow:abc", trace_id="trace-1", tool_call_id="call-1",
        producer=producer, event_type="verification_passed",
    )
    with pytest.raises(ValueError, match="unauthorized producer"):
        append_producer_outcome_event(path, candidate)
    assert query_outcome_events(path) == []


def test_authorized_producer_permits_only_verification_pass_or_fail(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    with pytest.raises(ValueError, match="event type"):
        append_producer_outcome_event(path, _producer_candidate("user_correction"))
    assert query_outcome_events(path) == []


def test_authorized_producer_rejects_forged_structural_id(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    with pytest.raises(ValueError, match="structural"):
        append_producer_outcome_event(path, _producer_candidate(event_id="shadow:forged"))
    assert query_outcome_events(path) == []


def test_low_level_fixture_append_still_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "state.db"
    _trace_db(path)
    kwargs = dict(
        trace_id="trace-1", event_type="verification_passed", source="verifier",
        polarity="positive", confidence=1.0, event_id="fixture-event",
    )
    append_outcome_event(path, **kwargs)
    with pytest.raises(sqlite3.IntegrityError):
        append_outcome_event(path, **kwargs)
