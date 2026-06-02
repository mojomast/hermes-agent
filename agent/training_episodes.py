"""Trace-to-episode projections for Hermes self-improvement.

Bounded contexts:
- Execution Trace: append-only operational trace rows + spans from SessionDB.
- Training Episode: privacy-minimized, replay/eval projection of one trace.
- Outcome Signal: measurable evidence extracted from structural trace metadata.

This module intentionally does not store raw prompts, tool args/results, env vars,
headers, or model payloads. It consumes the existing structural traces/spans.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from hermes_state import DEFAULT_DB_PATH

SCHEMA_VERSION = "training_episode.v1"
MAX_METADATA_CHARS = 2_000
SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer|sk-[a-z0-9])")
RAW_CONTENT_KEYS = {
    "result",
    "output",
    "prompt",
    "message",
    "content",
    "stdout",
    "stderr",
    "body",
    "payload",
    "headers",
    "args",
    "arguments",
    "command",
    "input",
    "env",
}
RAW_PAYLOAD_REDACTION = "[REDACTED_RAW_PAYLOAD]"
GENERATED_ARTIFACT_PATTERNS = (
    ".jsonl", ".db", ".sqlite", ".sqlite3", ".npy", ".pt", ".pth",
    ".safetensors", ".onnx", ".parquet", ".feather", ".arrow", ".log",
)


@dataclass(frozen=True)
class EpisodeStep:
    step_index: int
    span_type: str
    name: str
    status: str
    duration_ms: Optional[int]
    error_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeSignal:
    name: str
    value: float | int | bool | str
    weight: float = 1.0
    source: str = "trace"


@dataclass(frozen=True)
class TrainingEpisode:
    schema_version: str
    episode_id: str
    trace_id: str
    session_id: Optional[str]
    turn_id: Optional[str]
    user_message_hash: Optional[str]
    started_at: float
    ended_at: Optional[float]
    elapsed_ms: Optional[int]
    trace_status: str
    steps: List[EpisodeStep]
    outcome_signals: List[OutcomeSignal]
    reward: float
    ready_for_training: bool
    privacy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"state DB not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _safe_json(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    if len(value) > MAX_METADATA_CHARS:
        return {"truncated": True, "length": len(value)}
    try:
        data = json.loads(value)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return minimize_payload(data)


def _raw_payload_digest(value: Any) -> Dict[str, Any]:
    """Describe a raw payload without exporting any of its contents."""
    try:
        encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8", "replace")
    except Exception:
        encoded = str(type(value).__name__).encode("utf-8", "replace")
    return {
        "redacted": RAW_PAYLOAD_REDACTION,
        "length": len(encoded),
        "sha256_prefix": hashlib.sha256(encoded).hexdigest()[:16],
    }


def minimize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe metadata with secrets and raw-content payloads stripped."""
    safe: Dict[str, Any] = {}
    for key, value in list(data.items())[:40]:
        k = str(key)
        normalized_key = k.lower()
        if normalized_key in RAW_CONTENT_KEYS:
            safe[k] = _raw_payload_digest(value)
            continue
        if SECRET_RE.search(k):
            safe[k] = "[REDACTED]"
            continue
        if isinstance(value, (bool, int, float)) or value is None:
            safe[k] = value
        elif isinstance(value, str):
            if SECRET_RE.search(value):
                safe[k] = "[REDACTED]"
            elif len(value) > 160:
                safe[k] = {"truncated": True, "length": len(value), "sha256_prefix": hashlib.sha256(value.encode('utf-8', 'replace')).hexdigest()[:16]}
            else:
                safe[k] = value
        elif isinstance(value, (list, tuple)):
            safe[k] = [str(v)[:80] for v in value[:10]]
        elif isinstance(value, dict):
            safe[k] = minimize_payload({str(a): b for a, b in list(value.items())[:10]})
        else:
            safe[k] = str(value)[:120]
    return safe


def span_duration_ms(row: sqlite3.Row) -> Optional[int]:
    if row["start_time"] is None or row["end_time"] is None:
        return None
    return max(0, int((float(row["end_time"]) - float(row["start_time"])) * 1000))


def outcome_signals_for(trace: sqlite3.Row, span_rows: Iterable[sqlite3.Row]) -> List[OutcomeSignal]:
    spans = list(span_rows)
    errors = [s for s in spans if str(s["status"] or "").lower() in {"error", "failed"} or s["error_class"]]
    tool_spans = [s for s in spans if s["span_type"] in {"tool_call", "subagent"}]
    final = next((s for s in spans if s["span_type"] == "final_answer"), None)
    final_meta = _safe_json(final["metadata_json"] if final else None)
    completed = bool(final_meta.get("completed")) if final_meta else str(trace["status"] or "") == "completed"
    interrupted = bool(final_meta.get("interrupted")) if final_meta else False
    model_calls = int(trace["total_model_calls"] or 0)
    retry_pressure = max(0, model_calls - 1)
    signals: List[OutcomeSignal] = [
        OutcomeSignal("trace_completed", completed and not interrupted, 3.0),
        OutcomeSignal("trace_error_count", len(errors), -1.5),
        OutcomeSignal("tool_call_count", len(tool_spans), -0.03),
        OutcomeSignal("model_retry_pressure", retry_pressure, -0.2),
        OutcomeSignal("elapsed_ms", int(trace["total_wall_ms"] or 0), -0.00001),
        OutcomeSignal("has_final_answer", final is not None, 1.0),
    ]
    if interrupted:
        signals.append(OutcomeSignal("interrupted", True, -2.0))
    if any(s["name"] in {"terminal", "execute_code"} and s["status"] == "completed" for s in spans):
        signals.append(OutcomeSignal("verification_attempted", True, 0.5))
    return signals


def reward_from_signals(signals: Iterable[OutcomeSignal]) -> float:
    score = 0.0
    for sig in signals:
        value = sig.value
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            numeric = 1.0 if value else 0.0
        score += numeric * sig.weight
    return round(score, 4)


def trace_to_episode(trace: sqlite3.Row | Dict[str, Any], spans: Iterable[sqlite3.Row | Dict[str, Any]]) -> TrainingEpisode:
    # sqlite rows and dicts both support [] lookup; normalize missing via helper.
    trace_d = dict(trace)
    span_ds = [dict(s) for s in spans]
    steps = [
        EpisodeStep(
            step_index=i,
            span_type=str(row.get("span_type") or "unknown"),
            name=str(row.get("name") or "unknown"),
            status=str(row.get("status") or "unknown"),
            duration_ms=(None if row.get("start_time") is None or row.get("end_time") is None else max(0, int((float(row["end_time"]) - float(row["start_time"])) * 1000))),
            error_class=row.get("error_class"),
            metadata=_safe_json(row.get("metadata_json")),
        )
        for i, row in enumerate(span_ds)
    ]
    signals = outcome_signals_for(trace_d, span_ds)
    reward = reward_from_signals(signals)
    ready = bool(trace_d.get("status") == "completed" and any(s.name == "trace_completed" and bool(s.value) for s in signals))
    return TrainingEpisode(
        schema_version=SCHEMA_VERSION,
        episode_id=f"episode:{trace_d.get('trace_id')}",
        trace_id=str(trace_d.get("trace_id")),
        session_id=trace_d.get("session_id"),
        turn_id=trace_d.get("turn_id"),
        user_message_hash=trace_d.get("user_message_hash"),
        started_at=float(trace_d.get("start_time") or 0.0),
        ended_at=trace_d.get("end_time"),
        elapsed_ms=trace_d.get("total_wall_ms"),
        trace_status=str(trace_d.get("status") or "unknown"),
        steps=steps,
        outcome_signals=signals,
        reward=reward,
        ready_for_training=ready,
        privacy={
            "raw_content_exported": False,
            "secret_redaction": "metadata keys/values matching token/key/password patterns redacted",
            "raw_payload_redaction": "metadata under denylisted raw-content keys is replaced with length/hash descriptors",
            "raw_content_keys": sorted(RAW_CONTENT_KEYS),
            "generated_artifact_patterns": list(GENERATED_ARTIFACT_PATTERNS),
        },
    )


def iter_episodes(db_path: Path = DEFAULT_DB_PATH, limit: int = 100, min_reward: float | None = None, ready_only: bool = False) -> Iterable[TrainingEpisode]:
    with _connect_readonly(Path(db_path)) as con:
        traces = con.execute("SELECT * FROM traces ORDER BY start_time DESC LIMIT ?", (int(limit),)).fetchall()
        for trace in traces:
            spans = con.execute("SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC", (trace["trace_id"],)).fetchall()
            episode = trace_to_episode(trace, spans)
            if min_reward is not None and episode.reward < min_reward:
                continue
            if ready_only and not episode.ready_for_training:
                continue
            yield episode


def export_episodes_jsonl(output_path: Path, db_path: Path = DEFAULT_DB_PATH, limit: int = 100, ready_only: bool = False) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    count = 0
    rewards: List[float] = []
    with output_path.open("w", encoding="utf-8") as fh:
        for episode in iter_episodes(db_path=db_path, limit=limit, ready_only=ready_only):
            fh.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")
            count += 1
            rewards.append(episode.reward)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {"ok": True, "path": str(output_path), "episode_count": count, "elapsed_ms": elapsed_ms, "avg_reward": (sum(rewards) / count if count else 0.0)}


def episode_summary(db_path: Path = DEFAULT_DB_PATH, limit: int = 100) -> Dict[str, Any]:
    start = time.perf_counter()
    episodes = list(iter_episodes(db_path=db_path, limit=limit))
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    rewards = [e.reward for e in episodes]
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(episodes),
        "ready_for_training_count": sum(1 for e in episodes if e.ready_for_training),
        "recent_rewards": rewards[:10],
        "avg_reward": (sum(rewards) / len(rewards) if rewards else 0.0),
        "conversion_latency_ms": elapsed_ms,
    }


def default_episode_export_path() -> Path:
    return get_hermes_home() / "exports" / "training_episodes.jsonl"
