"""Primary-runtime adapter for correction learning in immutable shadow mode.

The adapter accepts only structural, allowlisted facts.  It never returns lesson
content to the model, never modifies a prompt, and has no activation API.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from agent.outcome_events import (
    OPERATOR_RELATION_PRODUCER,
    TAXONOMY_CODES,
    OutcomeEvent,
    append_outcome_event,
    init_outcome_events,
    query_effective_outcome_events,
    query_historical_outcome_events,
)
from agent.shadow_recurrence import evaluate_shadow_recurrence

EXPLICIT_CORRECTION_PRODUCER = "explicit_correct_command.v1"
OPERATOR_CONTROL_PRODUCER = OPERATOR_RELATION_PRODUCER
_ID_NAMESPACE = uuid.UUID("61fbc1d3-a04d-5ae4-852f-a680266f498c")


def validate_taxonomy(value: str) -> str:
    """Validate the complete `/correct` payload as one exact taxonomy token."""
    if not isinstance(value, str) or value not in TAXONOMY_CODES:
        raise ValueError("taxonomy must be one exact allowlisted code")
    return value


def _correction_event_id(trace_id: str, taxonomy_code: str) -> str:
    framed = f"{EXPLICIT_CORRECTION_PRODUCER}\0{len(trace_id)}:{trace_id}\0{taxonomy_code}"
    return "explicit-correction:" + str(uuid.uuid5(_ID_NAMESPACE, framed))


def record_explicit_correction(
    db_path: Path | str, *, trace_id: str, taxonomy_code: str
) -> OutcomeEvent:
    """Append an idempotent negative correction linked to one assistant trace.

    No free-form user text is accepted or stored.
    """
    code = validate_taxonomy(taxonomy_code)
    event_id = _correction_event_id(trace_id, code)
    try:
        return append_outcome_event(
            db_path,
            event_id=event_id,
            trace_id=trace_id,
            event_type="user_correction",
            source="user",
            polarity="negative",
            confidence=1.0,
            taxonomy_code=code,
        )
    except sqlite3.IntegrityError:
        existing = query_historical_outcome_events(
            db_path, trace_id=trace_id, taxonomy_code=code, event_type="user_correction", limit=200
        )
        for event in existing:
            if (
                event.event_id == event_id
                and event.source == "user"
                and event.polarity == "negative"
                and event.confidence == 1.0
                and event.evidence_digest is None
                and event.supersedes_event_id is None
            ):
                return event
        raise ValueError("conflicting duplicate explicit correction")


def record_session_correction(session_db: Any, *, session_id: str, taxonomy_code: str) -> OutcomeEvent:
    """Attach an explicit correction to the latest completed assistant trace."""
    code = validate_taxonomy(taxonomy_code)
    trace_id = session_db.latest_assistant_trace_id(session_id)
    if not trace_id:
        raise ValueError("no previous assistant trace is available in this session")
    return record_explicit_correction(session_db.db_path, trace_id=trace_id, taxonomy_code=code)


def inspect_lessons(db_path: Path | str) -> list[dict[str, Any]]:
    """Operator projection: bounded allowlisted lesson state, with no raw evidence."""
    report = evaluate_shadow_recurrence(db_path)
    allowed = (
        "lesson_id", "taxonomy_code", "state", "historical_event_count",
        "effective_event_count", "historical_root_count", "effective_root_count",
        "session_count", "valid_strict_corrected_pair_count",
    )
    return [
        {key: lesson[key] for key in allowed}
        for lesson in report["lessons"]
        if lesson.get("state") != "absent"
    ]


def inspect_lesson_events(db_path: Path | str, lesson_id: str) -> dict[str, Any]:
    """Operator-only evidence handles for one pseudonymous lesson."""
    lessons = evaluate_shadow_recurrence(db_path)["lessons"]
    lesson = next((item for item in lessons if item["lesson_id"] == lesson_id), None)
    if lesson is None:
        raise ValueError("unknown lesson identifier")
    events = query_effective_outcome_events(
        db_path, taxonomy_code=lesson["taxonomy_code"], limit=200
    )
    return {
        "lesson": {key: lesson[key] for key in ("lesson_id", "taxonomy_code", "state")},
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "source": event.source,
                "polarity": event.polarity,
                "confidence": event.confidence,
                "created_at": event.created_at,
            }
            for event in events
        ],
    }


def retract_lesson_event(db_path: Path | str, target_event_id: str) -> str:
    """Atomically append an operator control event and structural retraction.

    This function is intentionally not registered as a model tool or gateway
    command.  Its only runtime caller is the local CLI operator command.
    """
    if not isinstance(target_event_id, str) or not target_event_id or len(target_event_id) > 128:
        raise ValueError("invalid event identifier")
    init_outcome_events(db_path)
    control_id = "operator-retraction:" + str(uuid.uuid4())
    relation_id = "outcome-relation:" + str(uuid.uuid4())
    now = time.time()
    with sqlite3.connect(str(Path(db_path)), timeout=3.0) as con:
        row = con.execute(
            "SELECT trace_id,taxonomy_code FROM outcome_events WHERE event_id=?", (target_event_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown event identifier")
        con.execute(
            "INSERT INTO outcome_events(event_id,trace_id,event_type,source,polarity,confidence,taxonomy_code,created_at,evidence_digest,supersedes_event_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (control_id, row[0], "assumption_invalidated", "operator", "neutral", 1.0, row[1], now, None, None),
        )
        con.execute(
            "INSERT INTO outcome_event_relations(relation_id,source_event_id,target_event_id,relation_kind,producer,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (relation_id, control_id, target_event_id, "retracts", OPERATOR_CONTROL_PRODUCER, now),
        )
    return relation_id
