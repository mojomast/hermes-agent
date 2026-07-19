"""Deterministic, privacy-safe contrastive retrieval in shadow mode.

No output from this module is wired into prompt assembly. Matching uses only
allowlisted task categories, structural episode fields, and explicit Outcome Events.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any

from hermes_state import DEFAULT_DB_PATH
from agent.episode_retrieval import SAFE_TOOL_NAMES, episode_features
from agent.outcome_events import OutcomeEvidenceSummary, outcome_evidence_summaries, query_outcome_correction_pairs
from agent.training_episodes import TrainingEpisode, iter_episodes

SCHEMA_VERSION = "contrastive_episode_retrieval.v1"
EVAL_SCHEMA_VERSION = "contrastive_replay_eval.v1"
MAX_SCAN = 200
CATEGORIES = (
    "capability_audit", "capability_extension_ideation", "project_ideation",
    "duplicate_risk", "cross_session_recall", "system_state_audit", "debugging",
    "user_correction", "verification", "deployment",
)
CATEGORY_KEYWORDS = {
    "capability_audit": ("capability audit", "capability inventory", "existing capability", "learns from mistakes", "learn from its mistakes", "mistake-learning", "failed episodes"),
    "capability_extension_ideation": ("capability extension", "capability subsystem", "new capability", "extend capability"),
    "project_ideation": ("project idea", "project ideation", "new project"),
    "duplicate_risk": ("duplicate", "already exists", "existing"),
    "cross_session_recall": ("cross-session", "prior work", "continuity", "memory"),
    "system_state_audit": ("system state", "os", "port", "process", "disk"),
    "debugging": ("debug", "failure", "error", "fix"),
    "user_correction": ("correction", "wrong", "reject"),
    "verification": ("verify", "verification", "test"),
    "deployment": ("deploy", "deployment", "release"),
}
CATEGORY_TOOLS = {
    "capability_audit": {"search_files", "read_file", "semantic_code", "training_episodes"},
    "capability_extension_ideation": {"search_files", "read_file", "semantic_code", "training_episodes"},
    "project_ideation": {"search_files", "read_file"},
    "cross_session_recall": {"search_files", "training_episodes"},
    "system_state_audit": {"terminal", "execute_code"},
    "debugging": {"terminal", "execute_code", "read_file", "search_files"},
    "verification": {"terminal", "execute_code", "pytest"},
    "deployment": {"terminal", "execute_code"},
}
CATEGORY_TAXONOMIES = {
    "capability_audit": {"duplicate_existing_capability"},
    "capability_extension_ideation": {"duplicate_existing_capability"},
    "project_ideation": {"duplicate_existing_capability"},
    "duplicate_risk": {"duplicate_existing_capability"},
    "cross_session_recall": {"stale_context"},
    "system_state_audit": {"prerequisite_not_checked", "wrong_live_checkout"},
    "debugging": {"inspect_before_edit", "repeated_tool_failure", "prerequisite_not_checked"},
    "user_correction": {"wrong_scope", "unsupported_claim", "premature_completion"},
    "verification": {"verification_missing", "premature_completion"},
    "deployment": {"wrong_live_checkout", "verification_missing"},
}
SAFE_OUTCOME_NAMES = {
    "user_correction", "user_rejection", "user_confirmation", "verification_passed",
    "verification_failed", "assumption_invalidated", "wrong_scope", "duplicate_proposal",
    "unsupported_claim", "premature_completion", "rollback_required", "repeated_tool_failure",
}
HINT_TEMPLATES = {
    "duplicate_existing_capability": (
        "Avoid proposing a new capability subsystem before checking the existing capability inventory; "
        "prefer continuity, skill, source, and project lookup first."
    ),
    "wrong_live_checkout": "Confirm the live checkout structurally before changing or evaluating deployment state.",
    "wrong_scope": "Confirm the requested scope before making changes outside the smallest relevant surface.",
    "inspect_before_edit": "Inspect the relevant implementation surface before editing it.",
    "prerequisite_not_checked": "Check structural prerequisites before taking the dependent action.",
    "premature_completion": "Require explicit completion evidence before marking the work complete.",
    "verification_missing": "Require task-specific verification evidence before treating a result as successful.",
    "unsupported_claim": "Keep claims bounded to evidence established by the current work.",
    "stale_context": "Refresh continuity and source context before relying on prior assumptions.",
    "repeated_tool_failure": "After repeated tool failure, inspect the failure structure and change approach.",
}

FORBIDDEN_RAW_CONTENT_KEYS = {
    "prompt", "user_message", "userMessage", "tool_args", "toolArgs",
    "result", "stdout", "stderr", "response", "transcript",
    "model_output", "modelOutput", "metadata", "steps",
}


def _validate_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric non-bool value")
    return float(value)


def _validate_reward_thresholds(minimum: Any, maximum: Any) -> tuple[float, float]:
    minimum_value = _validate_finite_number("min_positive_reward", minimum)
    maximum_value = _validate_finite_number("max_negative_reward", maximum)
    if maximum_value >= minimum_value:
        raise ValueError("max_negative_reward must be less than min_positive_reward")
    return minimum_value, maximum_value


def _expected_privacy_contract(task_text_used: bool) -> dict[str, bool]:
    return {
        "raw_content_exported": False,
        "raw_task_text_exported": False,
        "raw_task_text_used_for_category_classification": task_text_used,
        "stored_raw_prompts_or_messages_used_for_matching": False,
        "raw_tool_args_or_results_used_for_matching": False,
        "deterministic_pseudonymous_episode_ids": True,
        "pseudonymous_ids_linkable_across_packets": True,
        "hashes_used_for_matching": False,
        "embeddings_used_for_matching": False,
    }


def _exported_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _exported_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _exported_string_values(child)


def _forbidden_raw_key_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(
            int(key in FORBIDDEN_RAW_CONTENT_KEYS) + _forbidden_raw_key_count(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_forbidden_raw_key_count(child) for child in value)
    return 0


def _structural_count(value: Any) -> int:
    """Normalize optional fixture provenance without accepting bools or coercions."""
    return value if type(value) is int and value >= 0 else 0


def _contains(text: str, keyword: str) -> bool:
    return keyword in text if " " in keyword or "-" in keyword else re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def task_frame_from_text(task_text: str) -> dict[str, Any]:
    lowered = (task_text or "").lower()
    categories = [category for category in CATEGORIES if any(_contains(lowered, word) for word in CATEGORY_KEYWORDS[category])]
    return {"categories": categories, "source": "allowlisted_structural_categories"}


def _relevant(
    categories: list[str], episode: TrainingEpisode, evidence: OutcomeEvidenceSummary,
    extra_taxonomies: frozenset[str] = frozenset(),
) -> tuple[float, list[str], list[str]]:
    features = episode_features(episode)
    tools = sorted(set(features["tool_names"]) & SAFE_TOOL_NAMES)
    taxonomies = evidence.taxonomy_codes | extra_taxonomies
    desired_tools = set().union(*(CATEGORY_TOOLS.get(category, set()) for category in categories)) if categories else set()
    desired_taxonomies = set().union(*(CATEGORY_TAXONOMIES.get(category, set()) for category in categories)) if categories else set()
    matched = [f"tool:{name}" for name in sorted(set(tools) & desired_tools)]
    matched += [f"taxonomy:{code}" for code in sorted(taxonomies & desired_taxonomies)]
    if "user_correction" in categories and evidence.outcome_names & {"user_correction", "user_rejection"}:
        matched.append("outcome:user_correction")
    if "verification" in categories and evidence.outcome_names & {"verification_passed", "verification_failed"}:
        matched.append("outcome:verification")
    score = round(len(matched) + min(max(episode.reward, 0), 5) / 10, 4) if matched else 0.0
    return score, matched, tools


def _compact_match(
    episode: TrainingEpisode, evidence: OutcomeEvidenceSummary, categories: list[str],
    extra_taxonomies: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    score, matched, tools = _relevant(categories, episode, evidence, extra_taxonomies)
    return {
        "episode_id": "episode:" + hashlib.sha256(episode.episode_id.encode("utf-8")).hexdigest()[:20],
        "score": score,
        "reward": episode.reward,
        "matched_features": matched,
        "safe_tool_names": tools,
        "outcome_names": sorted(evidence.outcome_names & SAFE_OUTCOME_NAMES),
        "taxonomy_codes": sorted(evidence.taxonomy_codes | extra_taxonomies),
        "outcome_event_count": evidence.event_count,
    }


def retrieve_contrastive_episodes(
    db_path: Path = DEFAULT_DB_PATH,
    task_text: str = "",
    positive_limit: int = 3,
    negative_limit: int = 3,
    corrected_limit: int = 3,
    min_positive_reward: float = 1.0,
    max_negative_reward: float = 0.0,
) -> dict[str, Any]:
    min_positive_reward, max_negative_reward = _validate_reward_thresholds(
        min_positive_reward, max_negative_reward,
    )
    start = time.perf_counter()
    frame = task_frame_from_text(task_text)
    # Deliberately includes failures and bypasses the legacy success-only defaults.
    episodes = list(iter_episodes(db_path=Path(db_path), limit=MAX_SCAN, min_reward=None, ready_only=False))
    candidate_trace_ids = [episode.trace_id for episode in episodes]
    evidence_by_trace = outcome_evidence_summaries(db_path, candidate_trace_ids)
    pairs = query_outcome_correction_pairs(db_path, candidate_trace_ids, limit=MAX_SCAN)
    pair_taxonomies_by_trace: dict[str, set[str]] = {}
    for pair in pairs:
        pair_taxonomies_by_trace.setdefault(pair.replacement_trace_id, set()).add(pair.taxonomy_code)
    corrected_traces = set(pair_taxonomies_by_trace)
    corrected_taxonomies = {pair.taxonomy_code for pair in pairs}

    positive, negative, corrected = [], [], []
    for episode in episodes:
        evidence = evidence_by_trace[episode.trace_id]
        pair_taxonomies = frozenset(pair_taxonomies_by_trace.get(episode.trace_id, ()))
        match = _compact_match(episode, evidence, frame["categories"], pair_taxonomies)
        if match["score"] <= 0:
            continue
        trusted_positive = evidence.strict_verifier_status == "verification_passed"
        has_unpaired_correction = "user_correction" in evidence.outcome_names and episode.trace_id not in corrected_traces
        if episode.trace_id in corrected_traces:
            corrected.append(match)
            if episode.ready_for_training and episode.reward >= min_positive_reward and trusted_positive:
                positive.append(match)
        elif evidence.has_negative_evidence or not episode.ready_for_training or episode.reward <= max_negative_reward:
            negative.append(match)
        elif episode.ready_for_training and episode.reward >= min_positive_reward and trusted_positive and not has_unpaired_correction:
            positive.append(match)

    sorter = lambda row: (row["score"], row["reward"], row["episode_id"])
    positive.sort(key=sorter, reverse=True)
    negative.sort(key=sorter, reverse=True)
    corrected.sort(key=sorter, reverse=True)

    relevant_taxonomies = set().union(*(CATEGORY_TAXONOMIES.get(category, set()) for category in frame["categories"])) if frame["categories"] else set()
    hints = []
    for taxonomy in sorted(relevant_taxonomies & set(HINT_TEMPLATES)):
        supporting_traces = {
            trace_id for trace_id, evidence in evidence_by_trace.items()
            if taxonomy in evidence.trusted_user_negative_taxonomies
        }
        corrected_support = taxonomy in corrected_taxonomies
        supported = len(supporting_traces) >= 2 or corrected_support
        # One explicit, high-confidence user correction may be evaluated as an
        # unsupported shadow candidate; weak/inferred events never become hints.
        if supported or len(supporting_traces) == 1:
            hints.append({
                "taxonomy_code": taxonomy,
                "text": HINT_TEMPLATES[taxonomy],
                "support_count": len(supporting_traces),
                "corrected_pair_support": corrected_support,
                "supported": supported,
                "shadow_only": True,
                "activation": False,
            })

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_frame": frame,
        "positive_matches": positive[:max(0, min(200, int(positive_limit)))],
        "negative_matches": negative[:max(0, min(200, int(negative_limit)))],
        "corrected_matches": corrected[:max(0, min(200, int(corrected_limit)))],
        "contrastive_hints": hints,
        "eligible_episode_count": len(episodes),
        "privacy": _expected_privacy_contract(bool(task_text)),
        "shadow_only": True,
        "prompt_modified": False,
        "elapsed_ms": elapsed_ms,
    }


def contrastive_replay_eval(
    db_path: Path = DEFAULT_DB_PATH,
    task_text: str = "",
    *,
    forbidden_substrings: tuple[str, ...] = (),
    expected_canary_count: int | None = None,
    planted_canary_count: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    packet = retrieve_contrastive_episodes(db_path=db_path, task_text=task_text, **kwargs)
    hints = packet["contrastive_hints"]
    episode_taxonomies = {
        (row["episode_id"], taxonomy)
        for pool in (packet["positive_matches"], packet["negative_matches"], packet["corrected_matches"])
        for row in pool for taxonomy in row["taxonomy_codes"]
    }
    taxonomy_counts = Counter(taxonomy for _, taxonomy in episode_taxonomies)
    recurrence_counts = {key: value for key, value in taxonomy_counts.items() if value > 1}
    canaries = tuple(dict.fromkeys(value for value in forbidden_substrings if value))
    expected_canary_coverage_count = _structural_count(expected_canary_count)
    planted_count = _structural_count(planted_canary_count)
    canary_coverage_ok = (
        expected_canary_coverage_count > 0
        and expected_canary_coverage_count == planted_count == len(canaries)
    )
    privacy_eval_valid = canary_coverage_ok
    string_values = tuple(_exported_string_values(packet))
    canary_findings = sum(any(canary in value for value in string_values) for canary in canaries)
    raw_key_finding_count = _forbidden_raw_key_count(packet)
    privacy_finding_count = canary_findings + raw_key_finding_count
    privacy_declaration_valid = packet.get("privacy") == _expected_privacy_contract(bool(task_text))
    unsupported_hint_count = sum(not hint.get("supported", False) for hint in hints)
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "eligible_episode_count": packet["eligible_episode_count"],
        "positive_match_count": len(packet["positive_matches"]),
        "negative_match_count": len(packet["negative_matches"]),
        "corrected_match_count": len(packet["corrected_matches"]),
        "generated_hint_count": len(hints),
        "unsupported_hint_count": unsupported_hint_count,
        "privacy_finding_count": privacy_finding_count,
        "privacy_canary_coverage_count": len(canaries),
        "expected_canary_coverage_count": expected_canary_coverage_count,
        "planted_canary_count": planted_count,
        "canary_coverage_ok": canary_coverage_ok,
        "privacy_eval_valid": privacy_eval_valid,
        "privacy_declaration_valid": privacy_declaration_valid,
        "forbidden_raw_key_finding_count": raw_key_finding_count,
        "task_frame": packet["task_frame"],
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "recurrence_counts": dict(sorted(recurrence_counts.items())),
        "latency_ms": packet["elapsed_ms"],
        "latency": {"retrieval_ms": packet["elapsed_ms"]},
        "shadow_only": True,
        "prompt_modified": False,
        "ok": (
            privacy_eval_valid
            and privacy_finding_count == 0
            and privacy_declaration_valid
            and unsupported_hint_count == 0
        ),
    }
