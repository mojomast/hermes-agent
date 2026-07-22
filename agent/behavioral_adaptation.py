"""Safe, correction-aware Behavioral Hint Adaptation Layer.

The layer is deliberately narrow: it reads the canonical structural recurrence
report, selects at most one source-owned hint, and returns an ephemeral prompt
suffix plus bounded telemetry. It never returns evidence/store identifiers or
stored content, and every ambiguous condition fails closed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.contrastive_episode_retrieval import HINT_TEMPLATES, task_frame_from_text, CATEGORY_TAXONOMIES
from agent.shadow_recurrence import (
    LESSON_POLICY_VERSION,
    SHADOW_RECURRENCE_SCHEMA_VERSION,
    evaluate_shadow_recurrence,
)

POLICY_VERSION = "behavioral_adaptation.v1"
DECISION_SCHEMA_VERSION = "behavioral_adaptation_decision.v1"
MAX_SUFFIX_CHARS = 512
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
    """Parse config with literal-boolean opt-in and no coercion.

    Missing config is valid and disabled. Once a section is present, only the
    exact keys and types in this schema are accepted; malformed sections are
    marked invalid and disabled.
    """
    if value is None:
        return AdaptationConfig()
    if isinstance(value, AdaptationConfig):
        return value
    if not isinstance(value, Mapping) or any(k not in {"enabled", "treatment_percent"} for k in value):
        return AdaptationConfig(valid=False)
    enabled = value.get("enabled", False)
    percent = value.get("treatment_percent", 50)
    if type(enabled) is not bool or type(percent) is not int or not 0 <= percent <= 100:
        return AdaptationConfig(valid=False)
    return AdaptationConfig(enabled=enabled, treatment_percent=percent, valid=True)


def treatment_for_session(session_id: str, treatment_percent: int) -> bool:
    """Return stable treatment assignment without exporting the session key."""
    if type(treatment_percent) is not int or not 0 <= treatment_percent <= 100:
        return False
    if treatment_percent == 0:
        return False
    if treatment_percent == 100:
        return True
    if not isinstance(session_id, str) or not session_id:
        return False
    digest = hashlib.sha256(_ASSIGNMENT_SALT + session_id.encode("utf-8", errors="replace")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return bucket < treatment_percent


def _telemetry(*, config_state: str, cohort: str, reason: str,
               eligible: bool = False, activated: bool = False,
               taxonomy_code: str | None = None, hint_chars: int = 0,
               privacy_valid: bool = False) -> dict[str, Any]:
    # This is the full telemetry allowlist. In particular there is no free-form
    # error field, task text, hash, session key, or evidence identifier.
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "config_state": config_state,
        "cohort": cohort,
        "eligible": eligible,
        "activated": activated,
        "reason": reason,
        "taxonomy_code": taxonomy_code,
        "hint_char_count": hint_chars,
        "privacy_contract_valid": privacy_valid,
        "raw_content_stored": False,
        "identifiers_stored": False,
    }


def _decision(reason: str, *, config_state: str, cohort: str = "disabled",
              eligible: bool = False, privacy_valid: bool = False) -> AdaptationDecision:
    return AdaptationDecision(None, _telemetry(
        config_state=config_state, cohort=cohort, reason=reason,
        eligible=eligible, privacy_valid=privacy_valid,
    ))


def _validate_report(report: Any) -> bool:
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") != SHADOW_RECURRENCE_SCHEMA_VERSION:
        return False
    if report.get("policy_version") != LESSON_POLICY_VERSION:
        return False
    if report.get("policy") != _EXPECTED_POLICY or report.get("privacy") != _EXPECTED_PRIVACY:
        return False
    lessons = report.get("lessons")
    if not isinstance(lessons, list):
        return False
    for lesson in lessons:
        if not isinstance(lesson, dict):
            return False
        taxonomy = lesson.get("taxonomy_code")
        eligible = lesson.get("activation_eligible")
        if taxonomy not in HINT_TEMPLATES or type(eligible) is not bool:
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


def _relevant_taxonomies(task_text: str) -> set[str]:
    text = task_text.lower() if isinstance(task_text, str) else ""
    categories = task_frame_from_text(text).get("categories", [])
    # Reuse the existing source-owned allowlisted task frame rather than
    # introducing a generic episode/text similarity path.
    return set().union(
        *(CATEGORY_TAXONOMIES.get(category, set()) for category in categories)
    ) if categories else set()


def decide_behavioral_adaptation(
    *, config: Any, session_id: str, task_text: str, db_path: Path | str,
    foreground: bool,
) -> AdaptationDecision:
    """Make one fail-closed treatment decision for a foreground turn."""
    cfg = parse_behavioral_adaptation_config(config)
    if not cfg.valid:
        return _decision("malformed_config", config_state="malformed")
    if not cfg.enabled:
        return _decision("disabled", config_state="disabled")
    if not foreground:
        return _decision("non_foreground", config_state="enabled")

    treatment = treatment_for_session(session_id, cfg.treatment_percent)
    cohort = "treatment" if treatment else "control"
    if not treatment:
        return _decision("control", config_state="enabled", cohort=cohort)

    try:
        report = evaluate_shadow_recurrence(Path(db_path))
    except Exception:
        return _decision("evidence_unavailable", config_state="enabled", cohort=cohort)
    if not _validate_report(report):
        return _decision("privacy_contract_invalid", config_state="enabled", cohort=cohort)

    relevant = _relevant_taxonomies(task_text)
    eligible = sorted(
        lesson["taxonomy_code"] for lesson in report["lessons"]
        if lesson["activation_eligible"] and lesson["taxonomy_code"] in relevant
    )
    if not eligible:
        return _decision(
            "no_relevant_eligible_pair", config_state="enabled", cohort=cohort,
            privacy_valid=True,
        )

    taxonomy = eligible[0]
    suffix = f"{_HINT_PREFIX}{HINT_TEMPLATES[taxonomy]}{_HINT_SUFFIX}"
    if len(suffix) > MAX_SUFFIX_CHARS:
        return _decision("hint_budget_exceeded", config_state="enabled", cohort=cohort,
                         eligible=True, privacy_valid=True)
    return AdaptationDecision(suffix, _telemetry(
        config_state="enabled", cohort=cohort, reason="activated", eligible=True,
        activated=True, taxonomy_code=taxonomy, hint_chars=len(suffix), privacy_valid=True,
    ))
