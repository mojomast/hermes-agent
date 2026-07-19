"""Privacy-conscious, per-turn structural trace recording.

Only operational structure belongs here: hashes, names, timestamps, counts,
statuses, and bounded JSON metadata. Callers must not pass prompts, tool
arguments/results, credentials, or provider payloads as metadata.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def hash_user_message(message: Any) -> str:
    """Return a stable SHA-256 digest without retaining message content."""
    if isinstance(message, str):
        payload = message.encode("utf-8", errors="replace")
    else:
        try:
            payload = json.dumps(message, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            payload = repr(message).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Bound metadata and make it JSON serializable."""
    if not metadata:
        return {}
    safe: Dict[str, Any] = {}
    for key, value in list(metadata.items())[:50]:
        key = str(key)
        if value is None or isinstance(value, (bool, int, float, str)):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [str(item) for item in value[:20]]
        elif isinstance(value, dict):
            safe[key] = {str(k): str(v) for k, v in list(value.items())[:20]}
        else:
            safe[key] = str(value)
    return safe


@dataclass
class SpanRecord:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    span_type: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    error_class: Optional[str] = None
    metadata_json: Optional[str] = None


@dataclass
class TraceRecorder:
    session_id: str
    turn_id: str
    user_message_hash: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    total_wall_ms: Optional[int] = None
    total_model_calls: int = 0
    total_tool_calls: int = 0
    total_subagents: int = 0

    def __post_init__(self) -> None:
        self._spans: list[SpanRecord] = []
        self._lock = threading.RLock()
        self._local = threading.local()
        self._persisted = False

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "span_stack", None)
        if stack is None:
            stack = []
            self._local.span_stack = stack
        return stack

    @contextlib.contextmanager
    def span(self, span_type: str, name: str,
             metadata: Optional[Dict[str, Any]] = None) -> Iterator[SpanRecord]:
        """Record a nested span and preserve explicit statuses set by callers."""
        stack = self._stack()
        span = SpanRecord(
            span_id=str(uuid.uuid4()), trace_id=self.trace_id,
            parent_span_id=stack[-1] if stack else None,
            span_type=span_type, name=name, start_time=time.time(),
            metadata_json=json.dumps(_safe_metadata(metadata), sort_keys=True),
        )
        with self._lock:
            self._spans.append(span)
            if span_type == "model_call":
                self.total_model_calls += 1
            elif span_type == "tool_call":
                self.total_tool_calls += 1
            elif span_type == "subagent":
                self.total_tool_calls += 1
                self.total_subagents += 1
        stack.append(span.span_id)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.error_class = exc.__class__.__name__
            raise
        else:
            if span.status == "running":
                span.status = "completed"
        finally:
            span.end_time = time.time()
            if stack and stack[-1] == span.span_id:
                stack.pop()
            elif span.span_id in stack:
                stack.remove(span.span_id)

    def finish(self, status: str = "completed", error_class: Optional[str] = None) -> None:
        with self._lock:
            if self._persisted:
                return
            self._close_unfinished_spans_locked()
            self.status = status
            self.end_time = time.time()
            self.total_wall_ms = int((self.end_time - self.start_time) * 1000)
            if error_class:
                self._spans.append(SpanRecord(
                    span_id=str(uuid.uuid4()), trace_id=self.trace_id,
                    parent_span_id=None, span_type="turn_error", name="turn_error",
                    start_time=self.end_time, end_time=self.end_time, status="error",
                    error_class=error_class, metadata_json="{}",
                ))

    def _close_unfinished_spans_locked(self) -> int:
        now = time.time()
        unfinished = [span for span in self._spans if span.end_time is None]
        if unfinished:
            logger.debug("Closing %d unfinished structural trace span(s)", len(unfinished))
        for span in unfinished:
            span.end_time = now
            if span.status == "running":
                span.status = "abandoned"
        self._stack().clear()
        return len(unfinished)

    def unfinished_span_count(self) -> int:
        with self._lock:
            return sum(span.end_time is None for span in self._spans)

    def mark_persisted(self) -> None:
        with self._lock:
            self._persisted = True

    def to_trace_row(self) -> Dict[str, Any]:
        self.finish(self.status if self.status != "running" else "completed")
        return {
            "trace_id": self.trace_id, "session_id": self.session_id,
            "turn_id": self.turn_id, "user_message_hash": self.user_message_hash,
            "start_time": self.start_time, "end_time": self.end_time,
            "status": self.status, "total_wall_ms": self.total_wall_ms,
            "total_model_calls": self.total_model_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_subagents": self.total_subagents,
        }

    def to_span_rows(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "span_id": span.span_id, "trace_id": span.trace_id,
                    "parent_span_id": span.parent_span_id,
                    "span_type": span.span_type, "name": span.name,
                    "start_time": span.start_time, "end_time": span.end_time,
                    "status": span.status, "error_class": span.error_class,
                    "metadata_json": span.metadata_json or "{}",
                }
                for span in self._spans
            ]


def active_tracer(obj: Any) -> Optional[TraceRecorder]:
    tracer = getattr(obj, "_turn_tracer", None)
    return tracer if isinstance(tracer, TraceRecorder) else None
