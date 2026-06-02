"""Pure heuristic routing for Hermes task execution efficiency.

This module is deliberately side-effect free: no config reads, no tool imports,
no model calls, and no filesystem access.  It provides a small deterministic
surface that later integration code can use to decide whether a request should
stay serial, be handled as a single-agent batch, or fan out to parallel
subagents.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class TaskRoute(enum.Enum):
    """Execution route categories for a user/task prompt."""

    SERIAL = "serial"
    BATCHED_SINGLE_AGENT = "batched_single_agent"
    PARALLEL_SUBAGENTS = "parallel_subagents"


@dataclass(frozen=True)
class TaskRouterConfig:
    """Tunable thresholds for the pure task router."""

    enable_parallel_subagents: bool = True
    min_parallel_subtasks: int = 3
    max_parallel_subagents: int = 4


@dataclass(frozen=True)
class RoutedSubtask:
    """A best-effort suggested subtask for future delegate_task fanout."""

    goal: str
    context: str = ""
    toolsets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"goal": self.goal, "context": self.context}
        if self.toolsets:
            payload["toolsets"] = list(self.toolsets)
        return payload


@dataclass(frozen=True)
class TaskRouteDecision:
    """Serializable routing decision with stable reason codes and metrics."""

    route: TaskRoute
    reasons: tuple[str, ...]
    metrics: Mapping[str, int | bool]
    subtasks: tuple[RoutedSubtask, ...] = field(default_factory=tuple)

    @property
    def should_delegate(self) -> bool:
        return self.route is TaskRoute.PARALLEL_SUBAGENTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "should_delegate": self.should_delegate,
        }


_SIMPLE_PATTERNS = (
    r"^\s*what\s+is\b",
    r"^\s*what\s+are\b",
    r"^\s*explain\b",
    r"^\s*summarize\s+this\b",
    r"^\s*calculate\b",
    r"^\s*convert\b",
    r"^\s*format\b",
)
_DEPENDENCY_PATTERNS = (
    r"\bfirst\b.*\bthen\b",
    r"\bthen\b",
    r"\bafter\s+that\b",
    r"\bonce\s+(?:that|the|you|we|i)\b",
    r"\bbased on (that|the|what)\b",
    r"\bstep by step\b",
)
_SAFETY_PATTERNS = (
    r"\bdelete\b",
    r"\bdeleting\b",
    r"\bforce\s+push\b",
    r"\bproduction\s+secrets?\b",
    r"\bdeployment\s+credentials?\b",
    r"\bdatabase\s+migration\b",
    r"\bmigrate\b",
    r"\bdestroy\w*\b",
    r"\bdestructive\b",
    r"\bapi\s+keys?\b",
    r"\bpasswords?\b",
    r"\boauth\s+tokens?\b",
    r"\bprivate\s+keys?\b",
    r"\bssh\s+keys?\b",
    r"\baccess\s+tokens?\b",
)
_WRITE_PATTERNS = (
    r"\bedit\b",
    r"\bmodify\b",
    r"\brewrite\b",
    r"\brefactor\b",
    r"\bfix\b",
    r"\bimplement\b",
    r"\bupdate\b",
    r"\bapply\b",
    r"\bcreate\b",
)
_PARALLEL_PATTERNS = (
    r"\bindependently\b",
    r"\bin parallel\b",
    r"\bsimultaneously\b",
    r"\bcompare\b",
    r"\bcontrast\b",
)
_EXPLICIT_DELEGATION_PATTERNS = (
    r"\buse\s+(?:\w+\s+)?(?:parallel\s+)?subagents?\b",
    r"\bparallel\s+subagents?\b",
    r"\bdelegate\b.*\bagents?\b",
    r"\bseparate\s+agents?\b",
    r"\bfan\s+out\b",
    r"\bspawn\b.*\bagents?\b",
    r"\bsplit\b.*\bacross\s+agents?\b",
)
_FORBID_DELEGATION_PATTERNS = (
    r"\bdo\s+not\s+use\s+(?:any\s+)?(?:parallel\s+)?subagents?\b",
    r"\bdon't\s+use\s+(?:any\s+)?(?:parallel\s+)?subagents?\b",
    r"\bno\s+(?:parallel\s+)?subagents?\b",
    r"\bwithout\s+(?:using\s+)?(?:any\s+)?(?:parallel\s+)?subagents?\b",
    r"\bsingle\s+agent\s+only\b",
    r"\bkeep\s+it\s+serial\b",
)
_READONLY_PATTERNS = (
    r"\bresearch\b",
    r"\binvestigate\b",
    r"\banaly[sz]e\b",
    r"\baudit\b",
    r"\breview\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\binspect\b",
    r"\bcheck\b",
    r"\btriage\b",
    r"\bread[- ]only\b",
)
_READONLY_GUARD_CLAUSE_RE = re.compile(
    r"(?:^|[.;,])\s*(?:"
    r"do\s+not\s+(?:modify|edit|update|change|write|apply|create|rewrite|refactor)(?:\s+(?:files?|changes?))?"
    r"|don't\s+(?:modify|edit|update|change|write|apply|create|rewrite|refactor)(?:\s+(?:files?|changes?))?"
    r"|no\s+(?:file\s+)?(?:changes?|modifications?|edits?|writes?)"
    r"|without\s+(?:modifying|editing|updating|changing|writing|applying|creating|rewriting|refactoring)(?:\s+(?:files?|changes?))?"
    r"|read[- ]only(?:\s+(?:review|audit|analysis|mode|only))?"
    r"|report\s+only"
    r")\.?\s*(?=$|[.;,])",
    re.IGNORECASE,
)
_HOMOGENEOUS_PATTERNS = (
    r"\bsame\b",
    r"\beach\b",
    r"\bevery\b",
    r"\bformatting cleanup\b",
    r"\bmatching imports\b",
)


def route_task(
    task: str,
    *,
    config: TaskRouterConfig | None = None,
) -> TaskRouteDecision:
    """Classify a task prompt into serial, batched, or parallel execution.

    The router is conservative by design: when signals conflict, it prefers
    serial/batched execution over spawning subagents.  It returns machine-stable
    reason codes and simple metrics so behavior can be benchmarked over time.
    """

    cfg = config or TaskRouterConfig()
    raw = task or ""
    normalized = _normalize(raw)
    reasons: list[str] = []

    if not normalized:
        return _decision(
            TaskRoute.SERIAL,
            ["empty_task"],
            estimated_subtasks=0,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=False,
            has_dependencies=False,
            safety_risk=False,
        )

    routing_text = _strip_readonly_guard_phrases(normalized)
    safety_risk = _safety_risk(routing_text)
    has_dependencies = _has_any(routing_text, _DEPENDENCY_PATTERNS)
    simple = _has_any(routing_text, _SIMPLE_PATTERNS)
    write_heavy = _write_heavy(routing_text)
    explicit_delegation = _has_any(normalized, _EXPLICIT_DELEGATION_PATTERNS)
    delegation_forbidden = _has_any(normalized, _FORBID_DELEGATION_PATTERNS)
    parallel_signal = _has_any(normalized, _PARALLEL_PATTERNS) or explicit_delegation
    read_only_signal = _has_any(normalized, _READONLY_PATTERNS) or explicit_delegation
    homogeneous = _has_any(normalized, _HOMOGENEOUS_PATTERNS)

    items = _candidate_items(raw)
    estimated_subtasks = len(items) if items else (1 if normalized else 0)
    requires_shared_context = bool(write_heavy or homogeneous or has_dependencies)
    parallelizable_subtasks = (
        estimated_subtasks
        if parallel_signal and read_only_signal and not requires_shared_context and not safety_risk
        else 0
    )

    if safety_risk:
        reasons.append("safety_guard")
        return _decision(
            TaskRoute.SERIAL,
            reasons,
            estimated_subtasks=estimated_subtasks,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=requires_shared_context,
            has_dependencies=has_dependencies,
            safety_risk=True,
        )

    if has_dependencies:
        reasons.append("dependent_steps")
        return _decision(
            TaskRoute.SERIAL,
            reasons,
            estimated_subtasks=estimated_subtasks,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=True,
            has_dependencies=True,
            safety_risk=False,
        )

    if simple and estimated_subtasks <= 1:
        reasons.append("simple")
        return _decision(
            TaskRoute.SERIAL,
            reasons,
            estimated_subtasks=1,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=False,
            has_dependencies=False,
            safety_risk=False,
        )

    if delegation_forbidden:
        reasons.append("delegation_forbidden_by_user")
        route = TaskRoute.BATCHED_SINGLE_AGENT if estimated_subtasks > 1 else TaskRoute.SERIAL
        return _decision(
            route,
            reasons,
            estimated_subtasks=estimated_subtasks,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=requires_shared_context,
            has_dependencies=False,
            safety_risk=False,
        )

    if not cfg.enable_parallel_subagents:
        reasons.append("parallel_disabled")
        route = TaskRoute.BATCHED_SINGLE_AGENT if estimated_subtasks > 1 else TaskRoute.SERIAL
        if estimated_subtasks <= 1:
            reasons.append("ambiguous")
        return _decision(
            route,
            reasons,
            estimated_subtasks=estimated_subtasks,
            parallelizable_subtasks=0,
            recommended_subagents=0,
            requires_shared_context=requires_shared_context,
            has_dependencies=False,
            safety_risk=False,
        )

    min_parallel = 2 if explicit_delegation else max(2, int(cfg.min_parallel_subtasks))
    max_parallel = max(0, int(cfg.max_parallel_subagents))
    if parallel_signal and read_only_signal and not requires_shared_context:
        if parallelizable_subtasks >= min_parallel and max_parallel >= 2:
            recommended = min(parallelizable_subtasks, max_parallel)
            reasons.append("independent_workstreams")
            if parallelizable_subtasks > recommended:
                reasons.append("capped_parallelism")
            subtasks = tuple(_make_subtasks(items[:recommended]))
            return TaskRouteDecision(
                route=TaskRoute.PARALLEL_SUBAGENTS,
                reasons=tuple(reasons),
                metrics={
                    "estimated_subtasks": estimated_subtasks,
                    "parallelizable_subtasks": parallelizable_subtasks,
                    "recommended_subagents": recommended,
                    "requires_shared_context": False,
                    "has_dependencies": False,
                    "safety_risk": False,
                },
                subtasks=subtasks,
            )
        reasons.append("below_parallel_threshold")

    if estimated_subtasks > 1:
        if homogeneous:
            reasons.append("homogeneous_batch")
        elif write_heavy:
            reasons.append("small_related_batch")
        else:
            reasons.append("multiple_items")
        return _decision(
            TaskRoute.BATCHED_SINGLE_AGENT,
            reasons,
            estimated_subtasks=estimated_subtasks,
            parallelizable_subtasks=parallelizable_subtasks,
            recommended_subagents=0,
            requires_shared_context=requires_shared_context or homogeneous,
            has_dependencies=False,
            safety_risk=False,
        )

    reasons.append("ambiguous")
    return _decision(
        TaskRoute.SERIAL,
        reasons,
        estimated_subtasks=1,
        parallelizable_subtasks=0,
        recommended_subagents=0,
        requires_shared_context=requires_shared_context,
        has_dependencies=False,
        safety_risk=False,
    )


def to_delegate_tasks(decision: TaskRouteDecision) -> list[dict[str, Any]]:
    """Convert a parallel decision into a delegate_task-compatible tasks array."""

    if decision.route is not TaskRoute.PARALLEL_SUBAGENTS:
        return []
    return [subtask.to_dict() for subtask in decision.subtasks]


def _decision(
    route: TaskRoute,
    reasons: list[str],
    *,
    estimated_subtasks: int,
    parallelizable_subtasks: int,
    recommended_subagents: int,
    requires_shared_context: bool,
    has_dependencies: bool,
    safety_risk: bool,
) -> TaskRouteDecision:
    return TaskRouteDecision(
        route=route,
        reasons=tuple(reasons),
        metrics={
            "estimated_subtasks": int(estimated_subtasks),
            "parallelizable_subtasks": int(parallelizable_subtasks),
            "recommended_subagents": int(recommended_subagents),
            "requires_shared_context": bool(requires_shared_context),
            "has_dependencies": bool(has_dependencies),
            "safety_risk": bool(safety_risk),
        },
    )


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _strip_readonly_guard_phrases(text: str) -> str:
    """Remove negated write clauses that mean "read-only", not "write-heavy".

    Examples: "Do not modify files", "No file changes", and
    "without editing" should not make an otherwise read-only review look like a
    shared-context mutation task. Positive write requests still remain intact.
    """

    previous = text or ""
    while True:
        stripped = _READONLY_GUARD_CLAUSE_RE.sub(" ", previous)
        stripped = re.sub(
            r"\s+without\s+(?:modifying|editing|updating|changing|writing|applying|creating|rewriting|refactoring)(?:\s+(?:files?|changes?))?\.?\s*$",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(
            r"^\s*read[- ]only\s+(?=(?:review|audit|analysis|analy[sz]e|inspect|check|compare|evaluate|triage)\b)",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = " ".join(stripped.split())
        if stripped == previous:
            return stripped
        previous = stripped


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _safety_risk(text: str) -> bool:
    if not _has_any(text, _SAFETY_PATTERNS):
        return False
    if re.search(r"\b(delete\s+command\s+behavior|rm\s+safety)\b", text):
        return False
    if re.search(r"\b(database\s+migration\s+guide|migration\s+docs?|deployment\s+checklist|rollback\s+docs?)\b", text):
        return False
    return True


def _write_heavy(text: str) -> bool:
    if not _has_any(text, _WRITE_PATTERNS):
        return False
    # Noun phrases such as "after the refactor" should not turn read-only audits into shared-context writes.
    if re.search(r"\bafter\s+the\s+refactor\b", text):
        without_temporal = re.sub(r"\bafter\s+the\s+refactor\b", "", text)
        return _has_any(without_temporal, _WRITE_PATTERNS)
    return True


def _candidate_items(task: str) -> list[str]:
    """Extract coarse independent item candidates while preserving order."""

    stripped = (task or "").strip()
    if not stripped:
        return []

    bullet_items: list[str] = []
    for line in stripped.splitlines():
        match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", line)
        if match:
            bullet_items.append(_clean_item(match.group(1)))
    if len(bullet_items) > 1:
        return [item for item in bullet_items if item]

    lowered = _strip_readonly_guard_phrases(stripped.lower())
    # Remove common framing verbs so comma/and splitting yields item names.
    lowered = re.sub(
        r"^\s*(?:use\s+(?:\w+\s+)?(?:parallel\s+)?subagents?\s+to|spawn\s+(?:parallel\s+)?subagents?\s+for|please\s+delegate|delegate|research|investigate|analyze|analyse|audit|review|compare|inspect|check|triage)\s+",
        "",
        lowered,
    )
    lowered = re.sub(r"\s+independently\.?$", "", lowered)
    lowered = re.sub(r"\s+and\s+compare(?:\s+their\s+tradeoffs)?\.?$", "", lowered)
    lowered = re.sub(r"\s+their\s+tradeoffs\.?$", "", lowered)

    if ";" in lowered:
        parts = [_clean_item(part) for part in lowered.split(";")]
        return [part for part in parts if part]

    # Split comma lists and final "and". Avoid treating plain two-word phrases
    # as multiple tasks unless there is a comma or clear list structure.
    if "," in lowered:
        lowered = re.sub(r",?\s+and\s+", ", ", lowered)
        parts = [_clean_item(part) for part in lowered.split(",")]
        return [part for part in parts if part]

    # For no-comma multi-item patterns, split only when the sentence contains a
    # strong workstream signal and a small number of "and" separators.
    if re.search(r"\b(independently|compare|research|investigate|analyze|analyse|audit|review)\b", stripped, re.I):
        parts = [_clean_item(part) for part in re.split(r"\s+and\s+", lowered)]
        if len(parts) > 1:
            return [part for part in parts if part]

    return []


def _clean_item(item: str) -> str:
    item = re.sub(r"\s+", " ", item.strip(" .:-"))
    item = re.sub(r"^(?:the|a|an)\s+", "", item)
    return item


def _make_subtasks(items: list[str]) -> list[RoutedSubtask]:
    subtasks: list[RoutedSubtask] = []
    for item in items:
        goal = item.strip()
        if not goal:
            continue
        subtasks.append(
            RoutedSubtask(
                goal=f"Investigate {goal}",
                context="Return concise findings, evidence, risks, and recommended next action.",
            )
        )
    return subtasks
