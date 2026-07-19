"""Canonical, privacy-safe recurrence metrics for shadow lesson evaluation.

This module is intentionally a reporting boundary.  It reads only structural,
allowlisted Outcome Event fields and trace/session linkage, never span metadata,
prompts, user text, evidence digests, or tool payloads.  Evaluation cannot alter
prompts or activate a lesson.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from agent.outcome_events import MAX_QUERY_LIMIT, TAXONOMY_CODES, register_effective_event_resolver

SHADOW_RECURRENCE_SCHEMA_VERSION = "shadow_recurrence_report.v1"
LESSON_POLICY_VERSION = "shadow_lesson_lifecycle.v1"
_LESSON_NAMESPACE = uuid.UUID("fb89fe52-8894-59ca-9f84-568982895cca")


def _empty_report() -> dict[str, Any]:
    lessons = [_lesson(code, set(), set(), set(), set(), 0) for code in sorted(TAXONOMY_CODES)]
    return _report(lessons, 0, 0, _empty_verifier_counts())


def _empty_verifier_counts() -> dict[str, int]:
    return {
        "historical_pass": 0,
        "historical_fail": 0,
        "effective_pass": 0,
        "effective_fail": 0,
    }


def _lesson_id(taxonomy_code: str) -> str:
    # The source taxonomy is already exported beside this identifier.  UUID5 gives
    # stable linkage across reports without disclosing a store identifier.
    value = f"{LESSON_POLICY_VERSION}\0{taxonomy_code}"
    return "lesson:" + str(uuid.uuid5(_LESSON_NAMESPACE, value))


def _state(root_count: int, eligible: bool) -> tuple[int, str]:
    if eligible:
        return 3, "activation_eligible"
    if root_count >= 2:
        return 2, "candidate"
    if root_count == 1:
        return 1, "observed"
    return 0, "absent"


def _lesson(
    taxonomy_code: str,
    historical_roots: set[str],
    effective_roots: set[str],
    sessions: set[str],
    strict_pairs: set[tuple[str, str]],
    effective_event_count: int,
    historical_event_count: int | None = None,
) -> dict[str, Any]:
    root_count = len(effective_roots)
    eligible = root_count >= 3 and len(sessions) >= 2 and bool(strict_pairs)
    lifecycle_state, state = _state(root_count, eligible)
    return {
        "lesson_id": _lesson_id(taxonomy_code),
        "taxonomy_code": taxonomy_code,
        "lifecycle_state": lifecycle_state,
        "state": state,
        "historical_event_count": effective_event_count if historical_event_count is None else historical_event_count,
        "effective_event_count": effective_event_count,
        "historical_root_count": len(historical_roots),
        "effective_root_count": root_count,
        "session_count": len(sessions),
        "valid_strict_corrected_pair_count": len(strict_pairs),
        "activation_eligible": eligible,
        "active": False,
        "rejected": False,
        "shadow_only": True,
        "activation_allowed": False,
        "prompt_modified": False,
    }


def _report(
    lessons: list[dict[str, Any]],
    historical_event_count: int,
    effective_event_count: int,
    verifier_counts: dict[str, int],
    *,
    aggregate_lessons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_lessons if aggregate_lessons is not None else lessons
    taxonomy_counts = {
        item["taxonomy_code"]: item["effective_root_count"] for item in aggregate
    }
    taxonomy_states = {
        item["taxonomy_code"]: item["state"] for item in aggregate
    }
    return {
        "schema_version": SHADOW_RECURRENCE_SCHEMA_VERSION,
        "policy_version": LESSON_POLICY_VERSION,
        "policy": {
            "observed_minimum_roots": 1,
            "candidate_minimum_roots": 2,
            "eligibility_minimum_roots": 3,
            "eligibility_minimum_sessions": 2,
            "strict_corrected_pair_required": True,
        },
        "shadow_only": True,
        "activation_allowed": False,
        "prompt_modified": False,
        "historical_event_count": historical_event_count,
        "effective_event_count": effective_event_count,
        "event_counts": {
            "historical": historical_event_count,
            "effective": effective_event_count,
        },
        "captured_verifier_counts": verifier_counts,
        "captured_verifier_event_counts": {
            "historical": {
                "verification_passed": verifier_counts["historical_pass"],
                "verification_failed": verifier_counts["historical_fail"],
            },
            "effective": {
                "verification_passed": verifier_counts["effective_pass"],
                "verification_failed": verifier_counts["effective_fail"],
            },
        },
        "taxonomy_counts": taxonomy_counts,
        "recurrence_counts": dict(taxonomy_counts),
        "taxonomy_states": taxonomy_states,
        "candidate_lesson_count": sum(item["lifecycle_state"] >= 2 for item in aggregate),
        "activation_eligible_lesson_count": sum(item["activation_eligible"] for item in aggregate),
        "active_lesson_count": 0,
        "lessons": lessons,
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
    }


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _read_connection(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=3.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def evaluate_shadow_recurrence(db_path: Path | str, *, limit: int = MAX_QUERY_LIMIT) -> dict[str, Any]:
    """Evaluate versioned shadow lesson state from all stored structural evidence.

    ``limit`` bounds the deterministic taxonomy projection, not evidence scanned;
    aggregate-before-limit prevents newer/noisy rows from starving older evidence.
    The function is read-only and a missing database produces an empty report
    without creating a file.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    bounded = min(limit, MAX_QUERY_LIMIT)
    path = Path(db_path)
    if not path.exists():
        report = _empty_report()
        report["lessons"] = report["lessons"][:bounded]
        return report

    with _read_connection(path) as con:
        if not _table_exists(con, "outcome_events"):
            report = _empty_report()
            report["lessons"] = report["lessons"][:bounded]
            return report
        has_relations = _table_exists(con, "outcome_event_relations")
        register_effective_event_resolver(con)
        effective_sql = "outcome_is_effective(e.event_id)=1"

        counts = con.execute(
            f"SELECT COUNT(*) AS historical, SUM(CASE WHEN {effective_sql} THEN 1 ELSE 0 END) AS effective "
            "FROM outcome_events e"
        ).fetchone()
        historical_event_count = int(counts["historical"] or 0)
        effective_event_count = int(counts["effective"] or 0)

        verifier_counts = _empty_verifier_counts()
        verifier_rows = con.execute(
            f"SELECT event_type, {effective_sql} AS effective, COUNT(*) AS n "
            "FROM outcome_events e WHERE source='verifier' "
            "AND event_type IN ('verification_passed','verification_failed') "
            "GROUP BY event_type,effective"
        ).fetchall()
        for row in verifier_rows:
            suffix = "pass" if row["event_type"] == "verification_passed" else "fail"
            verifier_counts[f"historical_{suffix}"] += int(row["n"])
            if row["effective"]:
                verifier_counts[f"effective_{suffix}"] += int(row["n"])

        taxonomy_rows = con.execute(
            f"SELECT e.taxonomy_code,e.trace_id,{effective_sql} AS effective,COUNT(*) AS n "
            "FROM outcome_events e WHERE e.taxonomy_code IS NOT NULL "
            "AND e.source='user' AND e.polarity='negative' AND e.confidence>=.9 "
            "GROUP BY e.taxonomy_code,e.trace_id,effective"
        ).fetchall()

        historical_roots = {code: set() for code in TAXONOMY_CODES}
        effective_roots = {code: set() for code in TAXONOMY_CODES}
        historical_taxonomy_events = {code: 0 for code in TAXONOMY_CODES}
        effective_taxonomy_events = {code: 0 for code in TAXONOMY_CODES}
        for row in taxonomy_rows:
            code = row["taxonomy_code"]
            # Fail closed if a legacy/corrupt database bypassed current CHECKs.
            if code not in TAXONOMY_CODES:
                continue
            historical_roots[code].add(row["trace_id"])
            historical_taxonomy_events[code] += int(row["n"])
            if row["effective"]:
                effective_roots[code].add(row["trace_id"])
                effective_taxonomy_events[code] += int(row["n"])

        strict_pairs = {code: set() for code in TAXONOMY_CODES}
        correction_effective = "AND outcome_is_effective(c.event_id)=1 "
        original_valid = "AND outcome_is_effective(o.event_id)=1 "
        verifier_effective = "AND outcome_is_effective(v.event_id)=1 "
        pair_rows = con.execute(
            "SELECT c.taxonomy_code,c.trace_id AS replacement_root,o.trace_id AS original_root "
            "FROM outcome_events c JOIN outcome_events o ON o.event_id=c.supersedes_event_id "
            "WHERE c.event_type='user_correction' AND c.source='user' "
            "AND c.polarity='positive' AND c.confidence>=.9 "
            + correction_effective +
            "AND o.source='user' AND o.polarity='negative' AND o.confidence>=.9 "
            + original_valid +
            "AND o.trace_id<>c.trace_id AND c.taxonomy_code IS NOT NULL "
            "AND c.taxonomy_code=o.taxonomy_code AND (SELECT v.event_type FROM outcome_events v "
            "WHERE v.trace_id=c.trace_id AND v.source='verifier' AND v.confidence>=.9 "
            "AND ((v.event_type='verification_passed' AND v.polarity='positive') "
            "OR (v.event_type='verification_failed' AND v.polarity='negative')) "
            + verifier_effective +
            "AND (v.created_at>c.created_at OR (v.created_at=c.created_at AND v.event_id>=c.event_id)) "
            "ORDER BY v.created_at DESC,v.event_id DESC LIMIT 1)='verification_passed'"
        ).fetchall()
        for row in pair_rows:
            code = row["taxonomy_code"]
            if code in TAXONOMY_CODES:
                pair = (row["replacement_root"], row["original_root"])
                strict_pairs[code].add(pair)

        sessions = {code: set() for code in TAXONOMY_CODES}
        if _table_exists(con, "traces"):
            all_roots = sorted(set().union(*effective_roots.values()))
            # SQLite commonly permits 999 bind variables; chunk independently of
            # report limits so large evidence sets remain complete.
            root_sessions: dict[str, str] = {}
            for start in range(0, len(all_roots), MAX_QUERY_LIMIT):
                chunk = all_roots[start:start + MAX_QUERY_LIMIT]
                placeholders = ",".join("?" for _ in chunk)
                rows = con.execute(
                    f"SELECT trace_id,session_id FROM traces WHERE trace_id IN ({placeholders}) "
                    "AND session_id IS NOT NULL",
                    chunk,
                ).fetchall()
                root_sessions.update({row["trace_id"]: row["session_id"] for row in rows})
            for code in TAXONOMY_CODES:
                sessions[code] = {
                    root_sessions[root] for root in effective_roots[code] if root in root_sessions
                }

    all_lessons = [
        _lesson(
            code, historical_roots[code], effective_roots[code], sessions[code],
            strict_pairs[code], effective_taxonomy_events[code], historical_taxonomy_events[code],
        )
        for code in sorted(TAXONOMY_CODES)
    ]
    return _report(
        all_lessons[:bounded], historical_event_count, effective_event_count,
        verifier_counts, aggregate_lessons=all_lessons,
    )


# Clear compatibility spellings for callers that think in report or lifecycle terms.
evaluate_shadow_lessons = evaluate_shadow_recurrence
shadow_recurrence_report = evaluate_shadow_recurrence
evaluate_lesson_lifecycle = evaluate_shadow_recurrence
