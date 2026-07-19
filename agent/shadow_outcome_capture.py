"""Pure, privacy-safe classification of foreground pytest tool outcomes.

This module deliberately does not persist anything.  It accepts the transient tool
arguments/result needed to classify a call, then projects only structural opaque
identifiers and an allowlisted outcome into a frozen candidate.
"""
from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass
from typing import Any, Mapping, Optional

FOREGROUND_PYTEST_PRODUCER = "foreground_pytest.v1"
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_SHELL_SYNTAX = frozenset(";&|<>`$\n\r\x00")
_PYTHON_BASENAME_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_NON_EXECUTION_OPTIONS = frozenset({
    "--help", "-h", "--version", "--collect-only", "--co", "--fixtures",
    "--funcargs", "--markers", "--setup-plan", "--setup-only", "--setup-show",
    "--trace-config",
})


@dataclass(frozen=True)
class ShadowOutcomeCandidate:
    """Minimal persistence request; contains no command or result content."""

    event_id: str
    trace_id: str
    tool_call_id: str
    producer: str
    event_type: str


def _is_direct_pytest(command: str) -> bool:
    """Recognize a deliberately narrow, non-shell pytest command grammar."""
    if not isinstance(command, str) or not command.strip():
        return False
    # Even quoted metacharacters are rejected.  Shadow evidence should be lost in
    # ambiguous cases rather than accidentally treating shell execution as direct.
    if any(character in command for character in _SHELL_SYNTAX):
        return False
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not words:
        return False
    executable = words[0].rsplit("/", 1)[-1]
    if executable in {"pytest", "py.test"}:
        pytest_arguments = words[1:]
    elif (
        len(words) >= 3
        and _PYTHON_BASENAME_RE.fullmatch(executable)
        and words[1:3] == ["-m", "pytest"]
    ):
        pytest_arguments = words[3:]
    else:
        return False

    # These modes inspect pytest itself or collect/setup tests without executing
    # them, and can report status 0 without providing verification evidence.
    return not any(
        word in _NON_EXECUTION_OPTIONS
        or any(
            word.startswith(option + "=")
            for option in _NON_EXECUTION_OPTIONS
            if option.startswith("--")
        )
        for word in pytest_arguments
    )


def _structural_event_id(trace_id: str, tool_call_id: str, producer: str) -> str:
    # Length framing avoids delimiter ambiguity.  Command, result, and outcome are
    # intentionally absent so contradictory replays collide and fail closed.
    payload = "".join(f"{len(value)}:{value}" for value in (trace_id, tool_call_id, producer))
    return "shadow:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shadow_candidate_has_valid_structure(candidate: ShadowOutcomeCandidate) -> bool:
    """Check that a candidate's opaque IDs and deterministic ID agree."""
    return bool(
        isinstance(candidate.trace_id, str)
        and _OPAQUE_ID_RE.fullmatch(candidate.trace_id)
        and isinstance(candidate.tool_call_id, str)
        and _OPAQUE_ID_RE.fullmatch(candidate.tool_call_id)
        and candidate.event_id
        == _structural_event_id(candidate.trace_id, candidate.tool_call_id, candidate.producer)
    )


def classify_foreground_pytest(
    *,
    trace_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> Optional[ShadowOutcomeCandidate]:
    """Return a pass/fail candidate for an unambiguous direct foreground pytest call.

    Exit codes 0 and 1 are pytest's pass and test-failure statuses.  Collection,
    usage, interruption, internal-error, and unknown statuses yield no evidence.
    Invalid inputs fail closed by returning ``None``.
    """
    if (
        tool_name != "terminal"
        or not isinstance(trace_id, str)
        or not _OPAQUE_ID_RE.fullmatch(trace_id)
        or not isinstance(tool_call_id, str)
        or not _OPAQUE_ID_RE.fullmatch(tool_call_id)
        or not isinstance(tool_arguments, Mapping)
        or not isinstance(tool_result, Mapping)
        or tool_arguments.get("background", False) is not False
        or not _is_direct_pytest(tool_arguments.get("command"))
    ):
        return None

    exit_code = tool_result.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return None
    if exit_code == 0:
        event_type = "verification_passed"
    elif exit_code == 1:
        event_type = "verification_failed"
    else:
        return None

    return ShadowOutcomeCandidate(
        event_id=_structural_event_id(trace_id, tool_call_id, FOREGROUND_PYTEST_PRODUCER),
        trace_id=trace_id,
        tool_call_id=tool_call_id,
        producer=FOREGROUND_PYTEST_PRODUCER,
        event_type=event_type,
    )
