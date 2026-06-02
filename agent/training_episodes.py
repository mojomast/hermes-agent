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
REPLAY_EVAL_SCHEMA_VERSION = "training_episode_replay_eval.v1"
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
RAW_CONTENT_KEY_MARKERS = (
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
    "response",
    "transcript",
    "completion",
    "observation",
)
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


def _is_raw_content_key(key: str) -> bool:
    """Return True for raw-payload fields, including compound/camel aliases."""
    normalized = key.lower()
    if normalized in RAW_CONTENT_KEYS:
        return True
    key_parts = [part for part in re.split(r"[^a-z0-9]+", normalized) if part]
    if any(part in RAW_CONTENT_KEY_MARKERS for part in key_parts):
        return True
    compacted = "".join(key_parts) or normalized
    return any(marker in compacted for marker in RAW_CONTENT_KEY_MARKERS)


def _minimize_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        if SECRET_RE.search(value):
            return "[REDACTED]"
        if len(value) > 160:
            return {"truncated": True, "length": len(value), "sha256_prefix": hashlib.sha256(value.encode('utf-8', 'replace')).hexdigest()[:16]}
        return value
    if isinstance(value, (list, tuple)):
        return [_minimize_value(v) for v in list(value)[:10]]
    if isinstance(value, dict):
        return minimize_payload({str(a): b for a, b in list(value.items())[:10]})
    return str(value)[:120]


def minimize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe metadata with secrets and raw-content payloads stripped."""
    safe: Dict[str, Any] = {}
    for key, value in list(data.items())[:40]:
        k = str(key)
        if _is_raw_content_key(k):
            safe[k] = _raw_payload_digest(value)
            continue
        if SECRET_RE.search(k):
            safe[k] = "[REDACTED]"
            continue
        safe[k] = _minimize_value(value)
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


def _is_raw_payload_descriptor(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("redacted") == RAW_PAYLOAD_REDACTION
        and isinstance(value.get("length"), int)
        and isinstance(value.get("sha256_prefix"), str)
        and len(value.get("sha256_prefix", "")) >= 8
    )


def _walk_metadata_raw_keys(value: Any, path: str = "metadata") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_raw_content_key(str(key)):
                yield child_path, child
            yield from _walk_metadata_raw_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value[:50]):
            yield from _walk_metadata_raw_keys(child, f"{path}[{index}]")


def privacy_findings_for_episode(episode: TrainingEpisode | Dict[str, Any], *, forbidden_substrings: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Return privacy regression findings for a minimized TrainingEpisode.

    Findings never echo raw forbidden content; canaries are represented by hash
    prefixes so eval output remains safe to store/share.
    """
    data = episode.to_dict() if isinstance(episode, TrainingEpisode) else dict(episode)
    findings: List[Dict[str, Any]] = []
    blob = json.dumps(data, sort_keys=True, default=str)

    if data.get("privacy", {}).get("raw_content_exported") is True:
        findings.append({"code": "raw_content_exported_true", "message": "Episode privacy metadata reports raw content export."})

    for substring in forbidden_substrings:
        if substring and substring in blob:
            digest = hashlib.sha256(str(substring).encode("utf-8", "replace")).hexdigest()[:16]
            findings.append({"code": "forbidden_substring_present", "message": "Forbidden substring appeared in episode JSON.", "substring_sha256_prefix": digest})

    for step_index, step in enumerate(data.get("steps", [])):
        metadata = step.get("metadata", {}) if isinstance(step, dict) else {}
        for path, value in _walk_metadata_raw_keys(metadata, f"steps[{step_index}].metadata"):
            if not _is_raw_payload_descriptor(value):
                findings.append({"code": "raw_payload_key_without_digest", "message": "Raw-payload metadata key did not contain a redacted digest descriptor.", "path": path})

    return findings


def replay_episode_eval(episode: TrainingEpisode, *, min_reward: float = 1.0, require_ready: bool = True, forbidden_substrings: Iterable[str] = ()) -> Dict[str, Any]:
    """Deterministically evaluate one minimized TrainingEpisode projection."""
    privacy_findings = privacy_findings_for_episode(episode, forbidden_substrings=forbidden_substrings)
    reasons: List[str] = []
    if episode.reward < min_reward:
        reasons.append("reward_below_min")
    if require_ready and not episode.ready_for_training:
        reasons.append("not_ready_for_training")
    if not episode.steps:
        reasons.append("no_steps")
    if not any(sig.name == "has_final_answer" and bool(sig.value) for sig in episode.outcome_signals):
        reasons.append("missing_final_answer_signal")
    if privacy_findings:
        reasons.append("privacy_findings_present")

    return {
        "episode_id": episode.episode_id,
        "trace_id": episode.trace_id,
        "trace_status": episode.trace_status,
        "ready_for_training": episode.ready_for_training,
        "reward": episode.reward,
        "step_count": len(episode.steps),
        "outcome_signal_count": len(episode.outcome_signals),
        "passed": not reasons,
        "reasons": reasons,
        "privacy_finding_count": len(privacy_findings),
    }


def replay_eval_episodes(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    limit: int = 100,
    ready_only: bool = False,
    min_reward: float = 1.0,
    require_ready: bool = True,
    include_episodes: bool = False,
    max_failures: int = 20,
    forbidden_substrings: Iterable[str] = (),
) -> Dict[str, Any]:
    """Run deterministic replay/eval over recent TrainingEpisode projections."""
    start = time.perf_counter()
    episodes = list(iter_episodes(db_path=db_path, limit=limit, ready_only=ready_only))
    rows = [replay_episode_eval(e, min_reward=min_reward, require_ready=require_ready, forbidden_substrings=forbidden_substrings) for e in episodes]
    rewards = [e.reward for e in episodes]
    privacy_findings: List[Dict[str, Any]] = []
    outcome_signal_totals: Dict[str, float] = {}
    for episode in episodes:
        privacy_findings.extend(privacy_findings_for_episode(episode, forbidden_substrings=forbidden_substrings))
        for signal in episode.outcome_signals:
            value = signal.value
            if isinstance(value, bool):
                numeric = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                numeric = float(value)
            else:
                numeric = 1.0 if value else 0.0
            outcome_signal_totals[signal.name] = outcome_signal_totals.get(signal.name, 0.0) + numeric

    failed_rows = [row for row in rows if not row["passed"]]
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result: Dict[str, Any] = {
        "schema_version": REPLAY_EVAL_SCHEMA_VERSION,
        "ok": not failed_rows and not privacy_findings,
        "db_path": str(db_path),
        "limit": int(limit),
        "ready_only": bool(ready_only),
        "thresholds": {"min_reward": min_reward, "require_ready": require_ready},
        "episode_count": len(episodes),
        "evaluated_count": len(rows),
        "passed_count": len(rows) - len(failed_rows),
        "failed_count": len(failed_rows),
        "ready_for_training_count": sum(1 for e in episodes if e.ready_for_training),
        "avg_reward": (sum(rewards) / len(rewards) if rewards else 0.0),
        "min_observed_reward": (min(rewards) if rewards else 0.0),
        "max_observed_reward": (max(rewards) if rewards else 0.0),
        "outcome_signal_totals": outcome_signal_totals,
        "privacy": {
            "raw_content_exported": False,
            "checked_episode_count": len(episodes),
            "finding_count": len(privacy_findings),
            "findings": privacy_findings[:max_failures],
        },
        "failures": failed_rows[:max_failures],
        "elapsed_ms": elapsed_ms,
    }
    if include_episodes:
        result["episodes"] = rows
    return result


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
