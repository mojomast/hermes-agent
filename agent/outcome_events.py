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


def _empty_summary() -> dict:
    return {"event_count": 0, "event_type_counts": {}, "polarity_counts": {}, "taxonomy_counts": {}}


def outcome_summaries(db_path: Path | str, trace_ids: list[str]) -> dict[str, dict]:
    """Return summaries for at most 200 stably deduplicated traces in one query."""
    unique = _bounded_unique_trace_ids(trace_ids)
    results = {trace_id: _empty_summary() for trace_id in unique}
    if not unique or not Path(db_path).exists():
        return results
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return results
        placeholders = ",".join("?" for _ in unique)
        rows = con.execute(
            f"SELECT trace_id,event_type,polarity,taxonomy_code,COUNT(*) AS n FROM outcome_events "
            f"WHERE trace_id IN ({placeholders}) GROUP BY trace_id,event_type,polarity,taxonomy_code",
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
    db_path: Path | str, trace_ids: list[str]
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
        rows = con.execute(
            "SELECT e.trace_id,e.event_type,e.source,e.polarity,e.taxonomy_code,"
            "CASE WHEN e.confidence>=.9 THEN 1 ELSE 0 END AS high_confidence,COUNT(*) AS n,"
            "(SELECT v.event_type FROM outcome_events AS v "
            " WHERE v.trace_id=e.trace_id AND v.source='verifier' AND v.confidence>=.9 "
            " AND ((v.event_type='verification_passed' AND v.polarity='positive') "
            "   OR (v.event_type='verification_failed' AND v.polarity='negative')) "
            " ORDER BY v.created_at DESC,v.event_id DESC LIMIT 1) AS strict_verifier_status "
            f"FROM outcome_events AS e WHERE e.trace_id IN ({placeholders}) "
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
        rows = con.execute(
            f"SELECT correction.trace_id AS replacement_trace_id, original.trace_id AS original_trace_id, "
            "correction.taxonomy_code FROM outcome_events AS correction "
            "JOIN outcome_events AS original ON original.event_id=correction.supersedes_event_id "
            f"WHERE correction.trace_id IN ({placeholders}) "
            "AND correction.event_type='user_correction' AND correction.source='user' "
            "AND correction.polarity='positive' AND correction.confidence>=.9 "
            "AND original.source='user' AND original.polarity='negative' AND original.confidence>=.9 "
            "AND original.trace_id<>correction.trace_id "
            "AND correction.taxonomy_code IS NOT NULL AND correction.taxonomy_code=original.taxonomy_code "
            "AND (SELECT verification.event_type FROM outcome_events AS verification "
            "WHERE verification.trace_id=correction.trace_id AND verification.source='verifier' "
            "AND verification.confidence>=.9 "
            "AND ((verification.event_type='verification_passed' AND verification.polarity='positive') "
            " OR (verification.event_type='verification_failed' AND verification.polarity='negative')) "
            "AND (verification.created_at>correction.created_at OR "
            " (verification.created_at=correction.created_at AND verification.event_id>=correction.event_id)) "
            "ORDER BY verification.created_at DESC, verification.event_id DESC LIMIT 1)='verification_passed' "
            "ORDER BY correction.created_at DESC, correction.event_id DESC LIMIT ?",
            (*trace_ids, bounded),
        ).fetchall()
    return [OutcomeCorrectionPair(row["replacement_trace_id"], row["original_trace_id"], row["taxonomy_code"]) for row in rows]


def outcome_summary(db_path: Path | str, *, trace_id: Optional[str] = None) -> dict:
    """Return counts and allowlisted taxonomy values only (never IDs/evidence)."""
    if trace_id is not None:
        return outcome_summaries(db_path, [trace_id])[trace_id]
    result = _empty_summary()
    if not Path(db_path).exists():
        return result
    with _connect_readonly(db_path) as con:
        if not _table_exists(con, "outcome_events"):
            return result
        result["event_count"] = con.execute("SELECT COUNT(*) FROM outcome_events").fetchone()[0]
        for key, column in (("event_type_counts", "event_type"), ("polarity_counts", "polarity"), ("taxonomy_counts", "taxonomy_code")):
            rows = con.execute(f"SELECT {column}, COUNT(*) FROM outcome_events GROUP BY {column}").fetchall()
            result[key] = {row[0]: row[1] for row in rows if row[0] is not None}
    return result
