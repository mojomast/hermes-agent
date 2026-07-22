"""Safe, correction-aware Behavioral Hint Adaptation Layer.

Reads only the canonical structural recurrence report, selects at most one
source-owned hint, and returns an ephemeral suffix plus bounded telemetry.
Every ambiguous condition fails closed.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.contrastive_episode_retrieval import CATEGORY_TAXONOMIES, HINT_TEMPLATES, task_frame_from_text
from agent.shadow_recurrence import (
    LESSON_POLICY_VERSION,
    SHADOW_RECURRENCE_SCHEMA_VERSION,
    evaluate_shadow_recurrence,
)

POLICY_VERSION = "behavioral_adaptation.v1"
DECISION_SCHEMA_VERSION = "behavioral_adaptation_decision.v1"
MAX_SUFFIX_CHARS = 512
MAX_EXPOSURE_COUNT = 1000
_ASSIGNMENT_SALT = b"hermes-behavioral-adaptation-v1\0"
_HINT_PREFIX = "<behavioral-hint>"
_HINT_SUFFIX = "</behavioral-hint>"

_EXPECTED_POLICY = {
    "observed_minimum_roots": 1,
    "candidate_minimum_roots": 2,
    "eligibility_minimum_roots": 3,
    "eligibility_minimum_sessions": 2,
    "strict_corrected_pair_required": True,
}
_EXPECTED_PRIVACY = {
    "raw_content_exported": False,
    "raw_user_text_read": False,
    "raw_prompt_read": False,
    "raw_tool_payloads_read": False,
    "evidence_digests_exported": False,
    "store_identifiers_exported": False,
    "taxonomy_codes_allowlisted": True,
    "deterministic_pseudonymous_lesson_ids": True,
    "pseudonymous_lesson_ids_linkable_across_reports": True,
}
_REPORT_FIELDS = {
    "schema_version", "policy_version", "policy", "shadow_only",
    "activation_allowed", "prompt_modified", "historical_event_count",
    "effective_event_count", "event_counts", "captured_verifier_counts",
    "captured_verifier_event_counts", "taxonomy_counts", "recurrence_counts",
    "taxonomy_states", "candidate_lesson_count",
    "activation_eligible_lesson_count", "active_lesson_count", "lessons", "privacy",
}
_LESSON_FIELDS = {
    "lesson_id", "taxonomy_code", "lifecycle_state", "state",
    "historical_event_count", "effective_event_count", "historical_root_count",
    "effective_root_count", "session_count", "valid_strict_corrected_pair_count",
    "activation_eligible", "active", "rejected", "shadow_only",
    "activation_allowed", "prompt_modified",
}


@dataclass(frozen=True)
class AdaptationConfig:
    enabled: bool = False
    treatment_percent: int = 50
    valid: bool = True


@dataclass(frozen=True)
class AdaptationDecision:
    suffix: str | None
    telemetry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"suffix": self.suffix, "telemetry": dict(self.telemetry)}


def parse_behavioral_adaptation_config(value: Any) -> AdaptationConfig:
    """Parse config with literal-boolean opt-in and no coercion."""
    if value is None:
        return AdaptationConfig()
    if isinstance(value, AdaptationConfig):
        return value
    if not isinstance(value, Mapping) or any(
        key not in {"enabled", "treatment_percent"} for key in value
    ):
        return AdaptationConfig(valid=False)
    enabled = value.get("enabled", False)
    percent = value.get("treatment_percent", 50)
    if type(enabled) is not bool or type(percent) is not int or not 0 <= percent <= 100:
        return AdaptationConfig(valid=False)
    return AdaptationConfig(enabled=enabled, treatment_percent=percent, valid=True)


def effective_behavioral_adaptation_config(
    loaded_config: Any, *, ignore_user_config: bool,
) -> AdaptationConfig:
    """Resolve the effective section, honoring the CLI config-bypass seam."""
    if ignore_user_config:
        return AdaptationConfig()
    section = loaded_config.get("behavioral_adaptation") if isinstance(loaded_config, Mapping) else None
    return parse_behavioral_adaptation_config(section)


def treatment_for_session(session_id: str, treatment_percent: int) -> bool:
    """Return stable treatment assignment without exporting the session key."""
    if type(treatment_percent) is not int or not 0 <= treatment_percent <= 100:
        return False
    if not isinstance(session_id, str) or not session_id:
        return False
    if treatment_percent == 0:
        return False
    if treatment_percent == 100:
        return True
    digest = hashlib.sha256(
        _ASSIGNMENT_SALT + session_id.encode("utf-8", errors="replace")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 100 < treatment_percent


def _telemetry(
    *, config_state: str, cohort: str, reason: str, eligible: bool = False,
    selected: bool = False, taxonomy_code: str | None = None,
    hint_chars: int = 0, privacy_valid: bool = False,
) -> dict[str, Any]:
    # Full allowlist: no task text, hashes, session keys, errors, or evidence IDs.
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "config_state": config_state,
        "cohort": cohort,
        "eligible": eligible,
        "selected_for_treatment": selected,
        "applied": False,
        "activated": False,  # compatibility: actual application only
        "request_count": 0,
        "application_count": 0,
        "reason": reason,
        "taxonomy_code": taxonomy_code,
        "hint_char_count": hint_chars,
        "privacy_contract_valid": privacy_valid,
        "raw_content_stored": False,
        "identifiers_stored": False,
    }


def _decision(
    reason: str, *, config_state: str, cohort: str = "disabled",
    eligible: bool = False, selected: bool = False,
    taxonomy_code: str | None = None, privacy_valid: bool = False,
) -> AdaptationDecision:
    return AdaptationDecision(None, _telemetry(
        config_state=config_state, cohort=cohort, reason=reason,
        eligible=eligible, selected=selected, taxonomy_code=taxonomy_code,
        privacy_valid=privacy_valid,
    ))


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _count_map(value: Any, keys: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == keys
        and all(_count(item) for item in value.values())
    )


def _taxonomy_map(value: Any, *, states: bool = False) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if not isinstance(key, str) or key not in HINT_TEMPLATES:
            return False
        if states:
            if item not in {"absent", "observed", "candidate", "activation_eligible"}:
                return False
        elif not _count(item):
            return False
    return True


def _pseudonymous_lesson_id(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("lesson:"):
        return False
    try:
        return uuid.UUID(value.removeprefix("lesson:")).version == 5
    except (ValueError, AttributeError):
        return False


def _validate_report(report: Any) -> bool:
    """Validate the complete canonical projection without propagating bad types."""
    try:
        if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
            return False
        if report.get("schema_version") != SHADOW_RECURRENCE_SCHEMA_VERSION:
            return False
        if report.get("policy_version") != LESSON_POLICY_VERSION:
            return False
        if report.get("policy") != _EXPECTED_POLICY or report.get("privacy") != _EXPECTED_PRIVACY:
            return False
        if (
            report.get("shadow_only") is not True
            or report.get("activation_allowed") is not False
            or report.get("prompt_modified") is not False
            or report.get("active_lesson_count") != 0
        ):
            return False
        if not all(_count(report.get(key)) for key in (
            "historical_event_count", "effective_event_count",
            "candidate_lesson_count", "activation_eligible_lesson_count",
            "active_lesson_count",
        )):
            return False
        if not _count_map(report.get("event_counts"), {"historical", "effective"}):
            return False
        if not _count_map(report.get("captured_verifier_counts"), {
            "historical_pass", "historical_fail", "effective_pass", "effective_fail",
        }):
            return False
        verifier_events = report.get("captured_verifier_event_counts")
        if (
            not isinstance(verifier_events, dict)
            or set(verifier_events) != {"historical", "effective"}
            or not all(_count_map(value, {"verification_passed", "verification_failed"})
                       for value in verifier_events.values())
        ):
            return False
        if (
            not _taxonomy_map(report.get("taxonomy_counts"))
            or not _taxonomy_map(report.get("recurrence_counts"))
            or report.get("recurrence_counts") != report.get("taxonomy_counts")
            or not _taxonomy_map(report.get("taxonomy_states"), states=True)
        ):
            return False
        lessons = report.get("lessons")
        if not isinstance(lessons, list):
            return False
        for lesson in lessons:
            if not isinstance(lesson, dict) or set(lesson) != _LESSON_FIELDS:
                return False
            taxonomy = lesson.get("taxonomy_code")
            eligible = lesson.get("activation_eligible")
            if (
                not isinstance(taxonomy, str)
                or taxonomy not in HINT_TEMPLATES
                or not _pseudonymous_lesson_id(lesson.get("lesson_id"))
                or type(eligible) is not bool
                or lesson.get("active") is not False
                or lesson.get("shadow_only") is not True
                or lesson.get("activation_allowed") is not False
                or lesson.get("prompt_modified") is not False
            ):
                return False
            if eligible and not (
                lesson.get("state") == "activation_eligible"
                and lesson.get("lifecycle_state") == 3
                and type(lesson.get("effective_root_count")) is int
                and lesson["effective_root_count"] >= 3
                and type(lesson.get("session_count")) is int
                and lesson["session_count"] >= 2
                and type(lesson.get("valid_strict_corrected_pair_count")) is int
                and lesson["valid_strict_corrected_pair_count"] >= 1
            ):
                return False
        return True
    except Exception:
        return False


def _relevant_taxonomies(task_text: str) -> set[str]:
    text = task_text.lower() if isinstance(task_text, str) else ""
    categories = task_frame_from_text(text).get("categories", [])
    return set().union(*(
        CATEGORY_TAXONOMIES.get(category, set()) for category in categories
    )) if categories else set()


def decide_behavioral_adaptation(
    *, config: Any, session_id: str, task_text: str,
    db_path: Path | str | None, foreground: bool,
) -> AdaptationDecision:
    """Make one fail-closed, matched treatment/control decision for a turn."""
    cfg = parse_behavioral_adaptation_config(config)
    if not cfg.valid:
        return _decision("malformed_config", config_state="malformed")
    if not cfg.enabled:
        return _decision("disabled", config_state="disabled")
    if not foreground:
        return _decision("non_foreground", config_state="enabled")
    if not isinstance(session_id, str) or not session_id or db_path is None:
        return _decision("missing_session_context", config_state="enabled")

    treatment = treatment_for_session(session_id, cfg.treatment_percent)
    cohort = "treatment" if treatment else "control"
    try:
        report = evaluate_shadow_recurrence(Path(db_path))
    except Exception:
        return _decision("evidence_unavailable", config_state="enabled", cohort=cohort)
    if not _validate_report(report):
        return _decision("privacy_contract_invalid", config_state="enabled", cohort=cohort)

    try:
        relevant = _relevant_taxonomies(task_text)
        eligible = sorted(
            lesson["taxonomy_code"] for lesson in report["lessons"]
            if lesson["activation_eligible"] and lesson["taxonomy_code"] in relevant
        )
    except Exception:
        return _decision("classification_unavailable", config_state="enabled", cohort=cohort)
    if not eligible:
        return _decision(
            "no_relevant_eligible_pair", config_state="enabled", cohort=cohort,
            selected=treatment, privacy_valid=True,
        )

    taxonomy = eligible[0]
    suffix = f"{_HINT_PREFIX}{HINT_TEMPLATES[taxonomy]}{_HINT_SUFFIX}"
    if len(suffix) > MAX_SUFFIX_CHARS:
        return _decision(
            "hint_budget_exceeded", config_state="enabled", cohort=cohort,
            eligible=True, selected=treatment, taxonomy_code=taxonomy,
            privacy_valid=True,
        )
    telemetry = _telemetry(
        config_state="enabled", cohort=cohort, reason="eligible", eligible=True,
        selected=treatment, taxonomy_code=taxonomy,
        hint_chars=len(suffix) if treatment else 0, privacy_valid=True,
    )
    return AdaptationDecision(suffix if treatment else None, telemetry)
