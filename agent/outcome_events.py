"""Append-only, privacy-safe Outcome Events for execution traces.

The table is additive and deliberately has no foreign key: trace rows are replaced by
legacy code.  This module stores enums, counts, opaque identifiers, and an optional
cryptographic digest identifier only; it has no API for raw evidence.
"""
from __future__ import annotations

import re
import sqlite3
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

EVENT_TYPES = frozenset({
    "user_correction", "user_rejection", "user_confirmation",
    "verification_passed", "verification_failed", "assumption_invalidated",
    "wrong_scope", "duplicate_proposal", "unsupported_claim",
    "premature_completion", "rollback_required", "repeated_tool_failure",
})
POLARITIES = frozenset({"positive", "negative", "neutral"})
SOURCES = frozenset({"user", "verifier", "evaluator", "operator", "system"})
TAXONOMY_CODES = frozenset({
    "duplicate_existing_capability", "wrong_live_checkout", "wrong_scope",
    "inspect_before_edit", "prerequisite_not_checked", "premature_completion",
    "verification_missing", "unsupported_claim", "stale_context",
    "repeated_tool_failure",
})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_QUERY_LIMIT = 200
RELATION_KINDS = frozenset({"corrects", "retracts", "replaces"})
# Relations are control-plane records, not inferred user intent. Keeping one
# versioned structural authority prevents text/user-driven retractions.
OPERATOR_RELATION_PRODUCER = "structural_operator.v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcome_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    polarity TEXT NOT NULL,
    confidence REAL NOT NULL,
    taxonomy_code TEXT,
    created_at REAL NOT NULL,
    evidence_digest TEXT,
    supersedes_event_id TEXT,
    CHECK (event_type IN ('user_correction','user_rejection','user_confirmation','verification_passed','verification_failed','assumption_invalidated','wrong_scope','duplicate_proposal','unsupported_claim','premature_completion','rollback_required','repeated_tool_failure')),
    CHECK (source IN ('user','verifier','evaluator','operator','system')),
    CHECK (polarity IN ('positive','negative','neutral')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (taxonomy_code IS NULL OR taxonomy_code IN ('duplicate_existing_capability','wrong_live_checkout','wrong_scope','inspect_before_edit','prerequisite_not_checked','premature_completion','verification_missing','unsupported_claim','stale_context','repeated_tool_failure'))
);
CREATE INDEX IF NOT EXISTS idx_outcome_events_trace ON outcome_events(trace_id, created_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_outcome_events_taxonomy ON outcome_events(taxonomy_code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcome_events_created ON outcome_events(created_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_outcome_events_supersedes ON outcome_events(supersedes_event_id);
CREATE TABLE IF NOT EXISTS outcome_event_relations (
    relation_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL,
    target_event_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL,
    producer TEXT NOT NULL,
    created_at REAL NOT NULL,
    CHECK (relation_kind IN ('corrects','retracts','replaces')),
    CHECK (source_event_id <> target_event_id)
);
CREATE INDEX IF NOT EXISTS idx_outcome_relations_source ON outcome_event_relations(source_event_id);
CREATE INDEX IF NOT EXISTS idx_outcome_relations_target ON outcome_event_relations(target_event_id, relation_kind);
CREATE TRIGGER IF NOT EXISTS outcome_event_relations_validate
BEFORE INSERT ON outcome_event_relations
BEGIN
    SELECT CASE WHEN NEW.producer <> 'structural_operator.v1' THEN RAISE(ABORT, 'unauthorized outcome relation producer') END;
    SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM outcome_events WHERE event_id=NEW.source_event_id)
                      OR NOT EXISTS (SELECT 1 FROM outcome_events WHERE event_id=NEW.target_event_id)
                THEN RAISE(ABORT, 'outcome relation endpoint does not exist') END;
    SELECT CASE WHEN NEW.relation_kind='retracts'
                      AND NOT EXISTS (SELECT 1 FROM outcome_events WHERE event_id=NEW.source_event_id AND source='operator')
                THEN RAISE(ABORT, 'retraction source must be an operator event') END;
    SELECT CASE WHEN EXISTS (
        WITH RECURSIVE reachable(event_id) AS (
            SELECT target_event_id FROM outcome_event_relations WHERE source_event_id=NEW.target_event_id
            UNION SELECT supersedes_event_id FROM outcome_events WHERE event_id=NEW.target_event_id AND supersedes_event_id IS NOT NULL
            UNION SELECT r.target_event_id FROM outcome_event_relations r JOIN reachable x ON r.source_event_id=x.event_id
            UNION SELECT e.supersedes_event_id FROM outcome_events e JOIN reachable x ON e.event_id=x.event_id WHERE e.supersedes_event_id IS NOT NULL
        ) SELECT 1 FROM reachable WHERE event_id=NEW.source_event_id
    ) THEN RAISE(ABORT, 'outcome relation cycle') END;
END;
CREATE TRIGGER IF NOT EXISTS outcome_event_relations_no_replace
BEFORE INSERT ON outcome_event_relations
WHEN EXISTS (SELECT 1 FROM outcome_event_relations WHERE relation_id = NEW.relation_id)
BEGIN SELECT RAISE(ABORT, 'outcome_event_relations relation_id already exists'); END;
CREATE TRIGGER IF NOT EXISTS outcome_event_relations_no_update
BEFORE UPDATE ON outcome_event_relations BEGIN SELECT RAISE(ABORT, 'outcome_event_relations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcome_event_relations_no_delete
BEFORE DELETE ON outcome_event_relations BEGIN SELECT RAISE(ABORT, 'outcome_event_relations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcome_events_no_replace
BEFORE INSERT ON outcome_events
WHEN EXISTS (SELECT 1 FROM outcome_events WHERE event_id = NEW.event_id)
BEGIN SELECT RAISE(ABORT, 'outcome_events event_id already exists'); END;
CREATE TRIGGER IF NOT EXISTS outcome_events_no_update
BEFORE UPDATE ON outcome_events BEGIN SELECT RAISE(ABORT, 'outcome_events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcome_events_no_delete
BEFORE DELETE ON outcome_events BEGIN SELECT RAISE(ABORT, 'outcome_events are append-only'); END;
"""


@dataclass(frozen=True)
class OutcomeEvent:
    event_id: str
    trace_id: str
    event_type: str
    source: str
    polarity: str
    confidence: float
    taxonomy_code: Optional[str]
    created_at: float
    evidence_digest: Optional[str]
    supersedes_event_id: Optional[str]


@dataclass(frozen=True)
class OutcomeEventRelation:
    relation_id: str
    source_event_id: str
    target_event_id: str
    relation_kind: str
    producer: str
    created_at: float


@dataclass(frozen=True)
class OutcomeCorrectionPair:
    """Internal linkage record; opaque IDs must never be placed in output packets."""
    replacement_trace_id: str
    original_trace_id: str
    taxonomy_code: str


@dataclass(frozen=True)
class OutcomeEvidenceSummary:
    """Complete privacy-safe evidence projection for one candidate trace."""
    event_count: int = 0
    event_type_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    polarity_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    taxonomy_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    has_negative_evidence: bool = False
    trusted_user_negative_taxonomies: frozenset[str] = frozenset()
    strict_verifier_status: Optional[str] = None

    @property
    def outcome_names(self) -> frozenset[str]:
        return frozenset(self.event_type_counts)

    @property
    def taxonomy_codes(self) -> frozenset[str]:
        return frozenset(self.taxonomy_counts)


def _connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(str(Path(path)), timeout=3.0)
    con.row_factory = sqlite3.Row
    return con


def _connect_readonly(path: Path | str) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    con = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, timeout=3.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def register_effective_event_resolver(con: sqlite3.Connection) -> None:
    """Register a consistent finite-DAG effective-evidence SQL predicate."""
    create_function = getattr(con, "create_function", None)
    if create_function is None:
        # Preserve compatibility with lightweight connection instrumentation used
        # by callers/tests while registering on the wrapped SQLite connection.
        create_function = con.con.create_function
    if not _table_exists(con, "outcome_event_relations"):
        create_function("outcome_is_effective", 1, lambda _event_id: 1)
        return
    suppressing_rows = con.execute(
        "SELECT source_event_id,target_event_id FROM outcome_event_relations "
        "WHERE relation_kind IN ('retracts','replaces')"
    ).fetchall()
    if not suppressing_rows:
        create_function("outcome_is_effective", 1, lambda _event_id: 1)
        return
    event_ids = {row[0] for row in con.execute("SELECT event_id FROM outcome_events")}
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for row in suppressing_rows:
        source, target = row[0], row[1]
        if source in event_ids and target in event_ids and target not in outgoing[source]:
            outgoing[source].add(target)
            incoming[target].add(source)
    indegree = {event_id: len(incoming[event_id]) for event_id in event_ids}
    ready = deque(sorted(event_id for event_id, degree in indegree.items() if degree == 0))
    effective: dict[str, bool] = {}
    while ready:
        event_id = ready.popleft()
        effective[event_id] = not any(effective.get(source, False) for source in incoming[event_id])
        for target in sorted(outgoing[event_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    # Cyclic legacy/corrupt components remain absent and therefore fail closed.
    effective_ids = frozenset(event_id for event_id, value in effective.items() if value)
    create_function("outcome_is_effective", 1, lambda event_id: int(event_id in effective_ids))


def init_outcome_events(db_path: Path | str) -> None:
    """Create only the outcome-event table, indexes, and append-only triggers."""
    with _connect(db_path) as con:
        con.executescript(_SCHEMA)


def _validate_enum(name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {name}: {value!r}")


def append_outcome_event(
    db_path: Path | str,
    *,
    trace_id: str,
    event_type: str,
    source: str,
    polarity: str,
    confidence: float,
    taxonomy_code: Optional[str] = None,
    evidence_digest: Optional[str] = None,
    supersedes_event_id: Optional[str] = None,
    event_id: Optional[str] = None,
    created_at: Optional[float] = None,
) -> OutcomeEvent:
    """Validate and append one event. There is intentionally no update/delete API."""
    _validate_enum("event_type", event_type, EVENT_TYPES)
    _validate_enum("source", source, SOURCES)
    _validate_enum("polarity", polarity, POLARITIES)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if taxonomy_code is not None:
        _validate_enum("taxonomy_code", taxonomy_code, TAXONOMY_CODES)
    if evidence_digest is not None and not _DIGEST_RE.fullmatch(evidence_digest):
        raise ValueError("evidence_digest must be a sha256:<64 lowercase hex> identifier")
    if not isinstance(trace_id, str) or not trace_id or len(trace_id) > 256:
        raise ValueError("trace_id must be a non-empty opaque identifier")
    event_id = event_id or str(uuid.uuid4())
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", event_id):
        raise ValueError("event_id must be an opaque identifier")
    if created_at is None:
        created_at = time.time()
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)) or not math.isfinite(float(created_at)) or float(created_at) < 0:
        raise ValueError("created_at must be a non-negative finite timestamp")
    created_at = float(created_at)

    init_outcome_events(db_path)
    with _connect(db_path) as con:
        if _table_exists(con, "traces") and con.execute("SELECT 1 FROM traces WHERE trace_id=?", (trace_id,)).fetchone() is None:
            raise ValueError("referenced trace does not exist")
        if supersedes_event_id is not None and con.execute("SELECT 1 FROM outcome_events WHERE event_id=?", (supersedes_event_id,)).fetchone() is None:
            raise ValueError("superseded event does not exist")
        con.execute(
            "INSERT INTO outcome_events(event_id,trace_id,event_type,source,polarity,confidence,taxonomy_code,created_at,evidence_digest,supersedes_event_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (event_id, trace_id, event_type, source, polarity, float(confidence), taxonomy_code, created_at, evidence_digest, supersedes_event_id),
        )
    return OutcomeEvent(event_id, trace_id, event_type, source, polarity, float(confidence), taxonomy_code, created_at, evidence_digest, supersedes_event_id)


def _relation_id(source_event_id: str, target_event_id: str, relation_kind: str, producer: str) -> str:
    payload = "\0".join((source_event_id, target_event_id, relation_kind, producer))
    return "outcome-relation:" + str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def _row_to_relation(row: sqlite3.Row) -> OutcomeEventRelation:
    return OutcomeEventRelation(**dict(row))


def append_outcome_event_relation(
    db_path: Path | str,
    *,
    source_event_id: str,
    target_event_id: str,
    relation_kind: str,
    producer: str,
    relation_id: Optional[str] = None,
    created_at: Optional[float] = None,
) -> OutcomeEventRelation:
    """Append one structural relation, idempotently.

    The source and target are immutable event IDs.  All relation kinds are an
    operator control-plane operation: callers cannot manufacture retractions
    from user prose or an event's claimed ``source`` field.
    """
    _validate_enum("relation kind", relation_kind, RELATION_KINDS)
    if producer != OPERATOR_RELATION_PRODUCER:
        raise ValueError("unauthorized producer")
    for name, value in (("source_event_id", source_event_id), ("target_event_id", target_event_id)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
            raise ValueError(f"{name} must be an opaque event identifier")
    if source_event_id == target_event_id:
        raise ValueError("outcome relation cannot self-link")
    deterministic_id = _relation_id(source_event_id, target_event_id, relation_kind, producer)
    relation_id = relation_id or deterministic_id
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", relation_id):
        raise ValueError("relation_id must be an opaque identifier")
    if created_at is None:
        created_at = time.time()
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)) or not math.isfinite(float(created_at)) or float(created_at) < 0:
        raise ValueError("created_at must be a non-negative finite timestamp")

    init_outcome_events(db_path)
    with _connect(db_path) as con:
        existing_row = con.execute("SELECT * FROM outcome_event_relations WHERE relation_id=?", (relation_id,)).fetchone()
        if existing_row is not None:
            existing = _row_to_relation(existing_row)
            immutable = (source_event_id, target_event_id, relation_kind, producer)
            if (existing.source_event_id, existing.target_event_id, existing.relation_kind, existing.producer) == immutable:
                return existing
            raise ValueError("conflicting duplicate outcome relation")
        endpoints = con.execute(
            "SELECT event_id,source FROM outcome_events WHERE event_id IN (?,?)",
            (source_event_id, target_event_id),
        ).fetchall()
        if len(endpoints) != 2:
            raise ValueError("relation endpoint does not exist")
        endpoint_sources = {row["event_id"]: row["source"] for row in endpoints}
        if relation_kind == "retracts" and endpoint_sources[source_event_id] != "operator":
            raise ValueError("unauthorized producer: retraction source must be an operator event")
        # Edges point source -> target.  A path target -> source means this edge
        # would make effective-evidence resolution cyclic.
        cycle = con.execute(
            "WITH RECURSIVE reachable(event_id) AS ("
            " SELECT target_event_id FROM outcome_event_relations WHERE source_event_id=?"
            " UNION SELECT supersedes_event_id FROM outcome_events WHERE event_id=? AND supersedes_event_id IS NOT NULL"
            " UNION SELECT r.target_event_id FROM outcome_event_relations r JOIN reachable x ON r.source_event_id=x.event_id"
            " UNION SELECT e.supersedes_event_id FROM outcome_events e JOIN reachable x ON e.event_id=x.event_id WHERE e.supersedes_event_id IS NOT NULL"
            ") SELECT 1 FROM reachable WHERE event_id=? LIMIT 1",
            (target_event_id, target_event_id, source_event_id),
        ).fetchone()
        if cycle is not None:
            raise ValueError("outcome relation would create a cycle")
        con.execute(
            "INSERT INTO outcome_event_relations(relation_id,source_event_id,target_event_id,relation_kind,producer,created_at) VALUES (?,?,?,?,?,?)",
            (relation_id, source_event_id, target_event_id, relation_kind, producer, float(created_at)),
        )
    return OutcomeEventRelation(relation_id, source_event_id, target_event_id, relation_kind, producer, float(created_at))


def retract_outcome_event(
    db_path: Path | str,
    *,
    source_event_id: str,
    target_event_id: str,
    producer: str,
    relation_id: Optional[str] = None,
    created_at: Optional[float] = None,
) -> OutcomeEventRelation:
    """Operator-only structural retraction; deliberately accepts no reason/text."""
    return append_outcome_event_relation(
        db_path, source_event_id=source_event_id, target_event_id=target_event_id,
        relation_kind="retracts", producer=producer, relation_id=relation_id,
        created_at=created_at,
    )


def append_producer_outcome_event(
    db_path: Path | str,
    candidate: object,
) -> OutcomeEvent:
    """Append an allowlisted producer candidate, idempotently and fail-closed.

    ``foreground_pytest.v1`` is authorized only for strict verifier pass/fail
    facts.  The producer cannot select source, polarity, confidence, taxonomy,
    evidence, or supersession.  On retry, every candidate-derived and
    producer-fixed immutable field is compared before the existing event is
    returned.  ``created_at`` is store-assigned on the first append and is
    returned unchanged on retries.
    """
    # Local import keeps the low-level fixture API independent and avoids making
    # shadow classification a dependency for readers of this module.
    from agent.shadow_outcome_capture import (
        FOREGROUND_PYTEST_PRODUCER,
        ShadowOutcomeCandidate,
        shadow_candidate_has_valid_structure,
    )

    if not isinstance(candidate, ShadowOutcomeCandidate) or candidate.producer != FOREGROUND_PYTEST_PRODUCER:
        raise ValueError("unauthorized producer")
    if not shadow_candidate_has_valid_structure(candidate):
        raise ValueError("invalid structural candidate")
    fixed = {
        "verification_passed": ("positive", 1.0),
        "verification_failed": ("negative", 1.0),
    }
    if candidate.event_type not in fixed:
        raise ValueError("producer is not authorized for event type")
    polarity, confidence = fixed[candidate.event_type]

    expected = {
        "event_id": candidate.event_id,
        "trace_id": candidate.trace_id,
        "event_type": candidate.event_type,
        "source": "verifier",
        "polarity": polarity,
        "confidence": confidence,
        "taxonomy_code": None,
        "evidence_digest": None,
        "supersedes_event_id": None,
    }

    init_outcome_events(db_path)
    with _connect(db_path) as con:
        row = con.execute("SELECT * FROM outcome_events WHERE event_id=?", (candidate.event_id,)).fetchone()
    if row is not None:
        existing = _row_to_event(row)
        if all(getattr(existing, name) == value for name, value in expected.items()):
            return existing
        raise ValueError("conflicting duplicate outcome event")

    try:
        return append_outcome_event(db_path, **expected)
    except sqlite3.IntegrityError:
        # A concurrent authorized retry can win after the read above.  Re-read and
        # apply the same full semantic comparison; all other collisions fail closed.
        with _connect_readonly(db_path) as con:
            row = con.execute("SELECT * FROM outcome_events WHERE event_id=?", (candidate.event_id,)).fetchone()
        if row is not None:
            existing = _row_to_event(row)
            if all(getattr(existing, name) == value for name, value in expected.items()):
                return existing
        raise ValueError("conflicting duplicate outcome event")


def append_authorized_outcome_event(db_path: Path | str, candidate: object) -> OutcomeEvent:
    """Compatibility spelling for :func:`append_producer_outcome_event`."""
    return append_producer_outcome_event(db_path, candidate)


def _row_to_event(row: sqlite3.Row) -> OutcomeEvent:
    return OutcomeEvent(**dict(row))


def _bounded_unique_trace_ids(trace_ids: Optional[list[str]]) -> list[str]:
    unique = list(dict.fromkeys(trace_ids or []))
    if len(unique) > MAX_QUERY_LIMIT:
        raise ValueError(f"at most {MAX_QUERY_LIMIT} unique trace IDs are allowed")
    return unique


def query_outcome_events(
    db_path: Path | str,
    *,
    trace_id: Optional[str] = None,
    trace_ids: Optional[list[str]] = None,
    taxonomy_code: Optional[str] = None,
    event_type: Optional[str] = None,
    supersedes_event_id: Optional[str] = None,
    limit: int = 100,
    effective_only: bool = False,
) -> list[OutcomeEvent]:
    if trace_id is not None and trace_ids is not None:
        raise ValueError("trace_id and trace_ids are mutually exclusive")
    selected_trace_ids = _bounded_unique_trace_ids(trace_ids) if trace_ids is not None else None
    if selected_trace_ids == []:
        return []
    if taxonomy_code is not None:
        _validate_enum("taxonomy_code", taxonomy_code, TAXONOMY_CODES)
    if event_type is not None:
        _validate_enum("event_type", event_type, EVENT_TYPES)
    bounded = max(1, min(MAX_QUERY_LIMIT, int(limit)))
    if not Path(db_path).exists():
        return []
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return []
        clauses, args = [], []
        if effective_only:
            register_effective_event_resolver(con)
            clauses.append("outcome_is_effective(outcome_events.event_id)=1")
        if selected_trace_ids is not None:
            clauses.append(f"trace_id IN ({','.join('?' for _ in selected_trace_ids)})")
            args.extend(selected_trace_ids)
        for column, value in (("trace_id", trace_id), ("taxonomy_code", taxonomy_code), ("event_type", event_type), ("supersedes_event_id", supersedes_event_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = con.execute(f"SELECT * FROM outcome_events{where} ORDER BY created_at DESC, event_id DESC LIMIT ?", (*args, bounded)).fetchall()
    return [_row_to_event(row) for row in rows]


def query_historical_outcome_events(db_path: Path | str, **kwargs) -> list[OutcomeEvent]:
    """Bounded immutable audit query (includes superseded/retracted evidence)."""
    kwargs.pop("effective_only", None)
    return query_outcome_events(db_path, effective_only=False, **kwargs)


def query_effective_outcome_events(db_path: Path | str, **kwargs) -> list[OutcomeEvent]:
    """Bounded query excluding events targeted by an explicit relation."""
    kwargs.pop("effective_only", None)
    return query_outcome_events(db_path, effective_only=True, **kwargs)


def _empty_summary() -> dict:
    return {"event_count": 0, "event_type_counts": {}, "polarity_counts": {}, "taxonomy_counts": {}}


def outcome_summaries(db_path: Path | str, trace_ids: list[str], *, historical: bool = False) -> dict[str, dict]:
    """Return summaries for at most 200 stably deduplicated traces in one query."""
    unique = _bounded_unique_trace_ids(trace_ids)
    results = {trace_id: _empty_summary() for trace_id in unique}
    if not unique or not Path(db_path).exists():
        return results
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return results
        placeholders = ",".join("?" for _ in unique)
        effective = ""
        if not historical:
            register_effective_event_resolver(con)
            effective = " AND outcome_is_effective(outcome_events.event_id)=1"
        rows = con.execute(
            f"SELECT trace_id,event_type,polarity,taxonomy_code,COUNT(*) AS n FROM outcome_events "
            f"WHERE trace_id IN ({placeholders}){effective} GROUP BY trace_id,event_type,polarity,taxonomy_code",
            unique,
        ).fetchall()
        for row in rows:
            summary = results[row["trace_id"]]
            count = int(row["n"])
            summary["event_count"] += count
            for key, column in (("event_type_counts", "event_type"), ("polarity_counts", "polarity"), ("taxonomy_counts", "taxonomy_code")):
                value = row[column]
                if value is not None:
                    summary[key][value] = summary[key].get(value, 0) + count
    return results


def outcome_evidence_summaries(
    db_path: Path | str, trace_ids: list[str], *, historical: bool = False
) -> dict[str, OutcomeEvidenceSummary]:
    """Aggregate complete evidence for at most 200 candidate traces in one query."""
    unique = _bounded_unique_trace_ids(trace_ids)
    empty = OutcomeEvidenceSummary()
    if not unique or not Path(db_path).exists():
        return {trace_id: empty for trace_id in unique}
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return {trace_id: empty for trace_id in unique}
        placeholders = ",".join("?" for _ in unique)
        relation_filter = ""
        verifier_relation_filter = ""
        if not historical:
            register_effective_event_resolver(con)
            relation_filter = " AND outcome_is_effective(e.event_id)=1"
            verifier_relation_filter = " AND outcome_is_effective(v.event_id)=1"

        rows = con.execute(
            "SELECT e.trace_id,e.event_type,e.source,e.polarity,e.taxonomy_code,"
            "CASE WHEN e.confidence>=.9 THEN 1 ELSE 0 END AS high_confidence,COUNT(*) AS n,"
            "(SELECT v.event_type FROM outcome_events AS v "
            " WHERE v.trace_id=e.trace_id AND v.source='verifier' AND v.confidence>=.9 "
            " AND ((v.event_type='verification_passed' AND v.polarity='positive') "
            "   OR (v.event_type='verification_failed' AND v.polarity='negative')) "
            + verifier_relation_filter +
            " ORDER BY v.created_at DESC,v.event_id DESC LIMIT 1) AS strict_verifier_status "
            f"FROM outcome_events AS e WHERE e.trace_id IN ({placeholders}){relation_filter} "
            "GROUP BY e.trace_id,e.event_type,e.source,e.polarity,e.taxonomy_code,high_confidence",
            unique,
        ).fetchall()

    accumulators = {
        trace_id: {
            "event_count": 0, "event_type_counts": {}, "polarity_counts": {},
            "taxonomy_counts": {}, "has_negative_evidence": False,
            "trusted_user_negative_taxonomies": set(), "strict_verifier_status": None,
        }
        for trace_id in unique
    }
    for row in rows:
        summary = accumulators[row["trace_id"]]
        count = int(row["n"])
        summary["event_count"] += count
        for key, column in (("event_type_counts", "event_type"), ("polarity_counts", "polarity"), ("taxonomy_counts", "taxonomy_code")):
            value = row[column]
            if value is not None:
                summary[key][value] = summary[key].get(value, 0) + count
        if row["polarity"] == "negative":
            summary["has_negative_evidence"] = True
        if row["source"] == "user" and row["polarity"] == "negative" and row["high_confidence"] and row["taxonomy_code"]:
            summary["trusted_user_negative_taxonomies"].add(row["taxonomy_code"])
        summary["strict_verifier_status"] = row["strict_verifier_status"]

    return {
        trace_id: OutcomeEvidenceSummary(
            event_count=value["event_count"],
            event_type_counts=MappingProxyType(dict(sorted(value["event_type_counts"].items()))),
            polarity_counts=MappingProxyType(dict(sorted(value["polarity_counts"].items()))),
            taxonomy_counts=MappingProxyType(dict(sorted(value["taxonomy_counts"].items()))),
            has_negative_evidence=value["has_negative_evidence"],
            trusted_user_negative_taxonomies=frozenset(value["trusted_user_negative_taxonomies"]),
            strict_verifier_status=value["strict_verifier_status"],
        )
        for trace_id, value in accumulators.items()
    }


def query_outcome_correction_pairs(
    db_path: Path | str,
    replacement_trace_ids: list[str],
    limit: int = MAX_QUERY_LIMIT,
) -> list[OutcomeCorrectionPair]:
    """Find strict correction->original pairs, with explicit verifier evidence."""
    trace_ids = _bounded_unique_trace_ids(replacement_trace_ids)
    if not trace_ids or not Path(db_path).exists():
        return []
    bounded = max(1, min(MAX_QUERY_LIMIT, int(limit)))
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return []
        placeholders = ",".join("?" for _ in trace_ids)
        register_effective_event_resolver(con)
        rows = con.execute(
            f"SELECT correction.trace_id AS replacement_trace_id, original.trace_id AS original_trace_id, "
            "correction.taxonomy_code FROM outcome_events AS correction "
            "JOIN outcome_events AS original ON original.event_id=correction.supersedes_event_id "
            f"WHERE correction.trace_id IN ({placeholders}) "
            "AND correction.event_type='user_correction' AND correction.source='user' "
            "AND correction.polarity='positive' AND correction.confidence>=.9 "
            "AND outcome_is_effective(correction.event_id)=1 "
            "AND original.source='user' AND original.polarity='negative' AND original.confidence>=.9 "
            "AND outcome_is_effective(original.event_id)=1 "
            "AND original.trace_id<>correction.trace_id "
            "AND correction.taxonomy_code IS NOT NULL AND correction.taxonomy_code=original.taxonomy_code "
            "AND (SELECT verification.event_type FROM outcome_events AS verification "
            "WHERE verification.trace_id=correction.trace_id AND verification.source='verifier' "
            "AND verification.confidence>=.9 "
            "AND ((verification.event_type='verification_passed' AND verification.polarity='positive') "
            " OR (verification.event_type='verification_failed' AND verification.polarity='negative')) "
            "AND outcome_is_effective(verification.event_id)=1 "
            "AND (verification.created_at>correction.created_at OR "
            " (verification.created_at=correction.created_at AND verification.event_id>=correction.event_id)) "
            "ORDER BY verification.created_at DESC, verification.event_id DESC LIMIT 1)='verification_passed' "
            "ORDER BY correction.created_at DESC, correction.event_id DESC LIMIT ?",
            (*trace_ids, bounded),
        ).fetchall()
    return [OutcomeCorrectionPair(row["replacement_trace_id"], row["original_trace_id"], row["taxonomy_code"]) for row in rows]


def outcome_summary(db_path: Path | str, *, trace_id: Optional[str] = None, historical: bool = False) -> dict:
    """Return effective counts by default; ``historical`` exposes audit counts."""
    if trace_id is not None:
        return outcome_summaries(db_path, [trace_id], historical=historical)[trace_id]
    result = _empty_summary()
    if not Path(db_path).exists():
        return result
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return result
        effective = ""
        if not historical:
            register_effective_event_resolver(con)
            effective = " WHERE outcome_is_effective(outcome_events.event_id)=1"
        result["event_count"] = con.execute(f"SELECT COUNT(*) FROM outcome_events{effective}").fetchone()[0]
        for key, column in (("event_type_counts", "event_type"), ("polarity_counts", "polarity"), ("taxonomy_counts", "taxonomy_code")):
            rows = con.execute(f"SELECT {column}, COUNT(*) FROM outcome_events{effective} GROUP BY {column}").fetchall()
            result[key] = {row[0]: row[1] for row in rows if row[0] is not None}
    return result
