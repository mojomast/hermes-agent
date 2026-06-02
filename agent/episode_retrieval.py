"""Privacy-safe retrieval of Training Episodes into bounded Behavioral Hints.

Bounded contexts:
- Episode Retrieval: structural matching over privacy-minimized TrainingEpisode projections.
- Behavioral Hint: compact, non-commanding guidance derived from successful similar episodes.

This module intentionally avoids raw prompts, tool args/results, metadata values, and
hash-derived prompt matching. It only uses task-text keyword categories and structural
episode metadata such as tool/span names, statuses, outcome signal names, and reward.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from hermes_state import DEFAULT_DB_PATH

from agent.training_episodes import TrainingEpisode, iter_episodes

SCHEMA_VERSION = "episode_retrieval.v1"
HINT_SCHEMA_VERSION = "behavioral_hints.v1"
MAX_LIMIT = 20
MAX_SCAN_LIMIT = 500
MAX_HINTS = 3

SAFE_TOOL_NAMES = {
    "terminal",
    "execute_code",
    "read_file",
    "search_files",
    "patch",
    "write_file",
    "web_search",
    "web_extract",
    "fetch_url",
    "semantic_code",
    "training_episodes",
    "delegate_task",
    "todo",
    "process",
    "git",
    "pytest",
}
SAFE_SPAN_TYPES = {"turn", "model_call", "tool_call", "subagent", "final_answer", "prompt_assembly"}

TASK_FEATURE_KEYWORDS = {
    "git_dirty_tree": ("git", "commit", "stage", "staging", "dirty tree", "status", "push", "branch", "diverg"),
    "focused_tests": ("pytest", "test", "tests", "verification", "regression", "py_compile", "compile"),
    "code_edit": ("edit", "patch", "implement", "module", "function", "dataclass", "file", "code"),
    "semantic_navigation": ("semantic", "definition", "references", "symbol", "large module", "navigation"),
    "web_dashboard_api": ("web", "dashboard", "api", "route", "frontend", "backend", "server"),
    "privacy_redaction": ("privacy", "redact", "redaction", "secret", "private", "canary", "payload"),
    "training_episode": ("training episode", "episode", "trace", "replay", "eval", "outcome signal", "behavioral hint"),
    "concurrent_delegation": ("subagent", "delegate", "parallel", "concurrent"),
}

TASK_TO_STRUCTURAL_FEATURES = {
    "git_dirty_tree": {"tool:terminal", "tool:execute_code", "outcome:verification_attempted"},
    "focused_tests": {"tool:terminal", "tool:execute_code", "outcome:verification_attempted"},
    "code_edit": {"tool:patch", "tool:write_file", "tool:read_file", "tool:search_files", "span:tool_call"},
    "semantic_navigation": {"tool:semantic_code", "tool:search_files", "tool:read_file"},
    "web_dashboard_api": {"tool:web_search", "tool:web_extract", "tool:fetch_url", "tool:terminal"},
    "privacy_redaction": {"outcome:trace_completed", "status:completed"},
    "training_episode": {"tool:training_episodes", "outcome:trace_completed", "outcome:has_final_answer"},
    "concurrent_delegation": {"tool:delegate_task", "span:subagent"},
}


@dataclass(frozen=True)
class TaskFrame:
    """Privacy-safe structural summary of the current task."""

    summary: str
    categories: List[str] = field(default_factory=list)
    structural_features: List[str] = field(default_factory=list)
    source: str = "keyword_structural"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeMatch:
    episode_id: str
    score: float
    reward: float
    ready_for_training: bool
    matched_features: List[str]
    tool_names: List[str]
    span_names: List[str]
    outcome_signal_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehavioralHint:
    code: str
    text: str
    confidence: float
    support_count: int
    matched_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except Exception:
        parsed = 5
    return max(1, min(MAX_LIMIT, parsed))


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def task_frame_from_text(text: str) -> TaskFrame:
    """Convert raw current-task text into non-sensitive structural categories.

    The raw text is not returned, embedded, hashed, or otherwise exported.
    """
    lowered = (text or "").lower()
    categories: List[str] = []
    features: Set[str] = set()
    for category, keywords in TASK_FEATURE_KEYWORDS.items():
        if any(_contains_keyword(lowered, kw) for kw in keywords):
            categories.append(category)
            features.update(TASK_TO_STRUCTURAL_FEATURES.get(category, set()))
    if not categories:
        categories.append("general_task")
        features.add("outcome:has_final_answer")
    summary = ", ".join(categories[:8])
    return TaskFrame(summary=summary, categories=categories[:8], structural_features=sorted(features))


def _safe_span_name(span_type: str, name: str) -> str:
    normalized_type = span_type if span_type in SAFE_SPAN_TYPES else "other"
    if normalized_type == "tool_call" and name in SAFE_TOOL_NAMES:
        return name
    if name in {"turn.final_answer", "llm_api_call", "root"}:
        return name
    if normalized_type != "tool_call":
        return normalized_type
    return "other_tool"


def episode_features(episode: TrainingEpisode) -> Dict[str, Any]:
    """Return structural, privacy-safe features for a TrainingEpisode."""
    tool_names: List[str] = []
    span_names: List[str] = []
    statuses: List[str] = []
    error_classes: List[str] = []
    features: Set[str] = set()

    for step in episode.steps:
        span_type = step.span_type if step.span_type in SAFE_SPAN_TYPES else "other"
        safe_name = _safe_span_name(span_type, step.name)
        span_names.append(safe_name)
        features.add(f"span:{span_type}")
        status = str(step.status or "unknown").lower()
        statuses.append(status)
        features.add(f"status:{status}")
        if span_type == "tool_call":
            tool_names.append(safe_name)
            features.add(f"tool:{safe_name}")
        if step.error_class:
            error_classes.append("error")
            features.add("error_present")

    outcome_signal_names = sorted({signal.name for signal in episode.outcome_signals})
    for name in outcome_signal_names:
        features.add(f"outcome:{name}")
    if episode.ready_for_training:
        features.add("ready_for_training")
    if episode.trace_status:
        features.add(f"trace_status:{episode.trace_status}")

    return {
        "features": sorted(features),
        "tool_names": sorted(set(tool_names)),
        "span_names": sorted(set(span_names)),
        "status_names": sorted(set(statuses)),
        "error_count": len(error_classes),
        "step_count": len(episode.steps),
        "outcome_signal_names": outcome_signal_names,
        "ready_for_training": episode.ready_for_training,
        "reward": episode.reward,
    }


def score_episode_match(task_frame: TaskFrame, episode: TrainingEpisode) -> float:
    """Score a structural match without consulting raw episode payloads."""
    features = set(episode_features(episode)["features"])
    task_features = set(task_frame.structural_features)
    if not task_features:
        return 0.0
    overlap = task_features & features
    if not overlap:
        return 0.0
    jaccard = len(overlap) / max(1, len(task_features | features))
    quality_bonus = min(max(episode.reward, 0.0), 5.0) / 25.0
    ready_bonus = 0.1 if episode.ready_for_training else 0.0
    return round(jaccard + quality_bonus + ready_bonus, 4)


def _match_to_result(task_frame: TaskFrame, episode: TrainingEpisode) -> EpisodeMatch:
    feats = episode_features(episode)
    matched = sorted(set(task_frame.structural_features) & set(feats["features"]))
    return EpisodeMatch(
        episode_id=episode.episode_id,
        score=score_episode_match(task_frame, episode),
        reward=episode.reward,
        ready_for_training=episode.ready_for_training,
        matched_features=matched,
        tool_names=feats["tool_names"],
        span_names=feats["span_names"],
        outcome_signal_names=feats["outcome_signal_names"],
    )


def retrieve_similar_episodes(
    db_path: Path = DEFAULT_DB_PATH,
    task_text: str = "",
    limit: int = 5,
    min_reward: float = 1.0,
) -> Dict[str, Any]:
    """Retrieve compact structural matches from recent successful Training Episodes."""
    top_limit = _clamp_limit(limit)
    task_frame = task_frame_from_text(task_text)
    start = time.perf_counter()
    scan_limit = max(50, min(MAX_SCAN_LIMIT, top_limit * 50))
    candidates = list(iter_episodes(db_path=Path(db_path), limit=scan_limit, min_reward=min_reward, ready_only=True))
    matches = [_match_to_result(task_frame, episode) for episode in candidates]
    matches = [match for match in matches if match.score > 0.0]
    matches.sort(key=lambda match: (match.score, match.reward, match.episode_id), reverse=True)
    selected = matches[:top_limit]
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_frame": task_frame.to_dict(),
        "matches_count": len(selected),
        "candidate_count": len(candidates),
        "matches": [match.to_dict() for match in selected],
        "privacy": {
            "raw_content_exported": False,
            "includes_raw_episode_payloads": False,
            "matching_uses_raw_prompts": False,
            "matching_uses_tool_outputs": False,
        },
        "elapsed_ms": elapsed_ms,
    }


def _supported(match: EpisodeMatch, feature: str) -> bool:
    return feature in set(match.matched_features) or feature in {f"tool:{name}" for name in match.tool_names} or feature in {f"outcome:{name}" for name in match.outcome_signal_names}


def _hint_candidates(task_frame: TaskFrame, matches: Sequence[EpisodeMatch]) -> List[BehavioralHint]:
    hints: List[BehavioralHint] = []
    support_features = sorted({feature for match in matches for feature in match.matched_features})
    support_count = len(matches)
    avg_score = sum(match.score for match in matches) / support_count if support_count else 0.0

    if {"git_dirty_tree", "focused_tests"} & set(task_frame.categories) and any(_supported(m, "tool:terminal") or _supported(m, "tool:execute_code") for m in matches):
        hints.append(BehavioralHint(
            code="scoped_git_and_tests",
            text="Similar successful tasks used targeted status checks, explicit scoped changes, and focused tests before finalizing.",
            confidence=round(min(0.95, 0.55 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    if "code_edit" in task_frame.categories and any(tool in m.tool_names for m in matches for tool in ("read_file", "search_files", "patch", "write_file")):
        hints.append(BehavioralHint(
            code="inspect_before_edit",
            text="Comparable successful edits inspected the smallest relevant code surface, applied scoped file changes, then verified the touched behavior.",
            confidence=round(min(0.95, 0.5 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    if "semantic_navigation" in task_frame.categories and any(tool in {"semantic_code", "search_files", "read_file"} for m in matches for tool in m.tool_names):
        hints.append(BehavioralHint(
            code="semantic_navigation_first",
            text="Similar successful tasks used structural code navigation before editing large or ambiguous modules.",
            confidence=round(min(0.95, 0.5 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    if "privacy_redaction" in task_frame.categories:
        hints.append(BehavioralHint(
            code="privacy_compact_outputs",
            text="Comparable successful work kept adaptation artifacts compact and avoided exporting raw prompts, tool outputs, or private details.",
            confidence=round(min(0.95, 0.5 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    if "training_episode" in task_frame.categories:
        hints.append(BehavioralHint(
            code="structural_episode_evidence",
            text="Similar successful episode work relied on deterministic structural signals and privacy checks rather than replaying raw trace content.",
            confidence=round(min(0.95, 0.5 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    if not hints and matches:
        hints.append(BehavioralHint(
            code="verify_before_final",
            text="Similar successful tasks completed tool work and verification before returning the final answer.",
            confidence=round(min(0.9, 0.45 + avg_score), 3),
            support_count=support_count,
            matched_features=support_features[:8],
        ))
    return hints[:MAX_HINTS]


def behavioral_hints_for_task(
    db_path: Path = DEFAULT_DB_PATH,
    task_text: str = "",
    limit: int = 5,
    min_reward: float = 1.0,
) -> Dict[str, Any]:
    """Return compact, privacy-safe Behavioral Hints for the current task frame."""
    retrieval = retrieve_similar_episodes(db_path=db_path, task_text=task_text, limit=limit, min_reward=min_reward)
    matches = [EpisodeMatch(**row) for row in retrieval["matches"]]
    hints = _hint_candidates(TaskFrame(**retrieval["task_frame"]), matches)
    return {
        "schema_version": HINT_SCHEMA_VERSION,
        "task_frame": retrieval["task_frame"],
        "matches_count": retrieval["matches_count"],
        "hints": [hint.to_dict() for hint in hints],
        "evidence_metrics": retrieval["matches"],
        "privacy": {
            "raw_content_exported": False,
            "includes_raw_episode_payloads": False,
            "matching_uses_raw_prompts": False,
            "matching_uses_tool_outputs": False,
        },
        "elapsed_ms": retrieval["elapsed_ms"],
    }


def format_behavioral_hints_for_prompt(result: Dict[str, Any], *, max_hints: int = 3, max_chars: int = 1200) -> str:
    """Format retrieved hints for an optional disabled-by-default prompt seam.

    The caller controls whether this gets injected. This helper keeps a hard
    character budget and emits no raw episode payloads.
    """
    hints = result.get("hints", []) if isinstance(result, dict) else []
    lines = [
        '<behavioral_hints source="training_episode_structural" privacy="no_raw_prompts_or_tool_outputs">'
    ]
    for hint in hints[: max(0, min(MAX_HINTS, int(max_hints)))]:
        text = str(hint.get("text", "")).strip()
        if text:
            lines.append(f"- {text[:300]}")
    lines.append("</behavioral_hints>")
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        return rendered[: max(0, max_chars - 32)].rstrip() + "\n</behavioral_hints>"
    return rendered
