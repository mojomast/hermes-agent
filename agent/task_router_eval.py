"""Deterministic evaluation helpers for Hermes task routing.

This module is intentionally offline and side-effect free except for the explicit
``load_eval_cases`` helper.  It gives future router changes a small measurable
regression surface before any runtime delegation integration changes behavior.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from agent.task_router import TaskRoute, TaskRouteDecision, TaskRouterConfig, route_task


_UNSAFE_TAGS = frozenset({"safety", "destructive", "dependent", "write_heavy", "secrets"})
_METRIC_KEYS = frozenset(
    {
        "estimated_subtasks",
        "parallelizable_subtasks",
        "recommended_subagents",
        "requires_shared_context",
        "has_dependencies",
        "safety_risk",
    }
)
_CONFIG_KEYS = frozenset({"enable_parallel_subagents", "min_parallel_subtasks", "max_parallel_subagents"})


@dataclass(frozen=True)
class TaskRouterEvalCase:
    """A golden routing case for deterministic router evaluation."""

    id: str
    task: str
    expected_route: TaskRoute
    tags: tuple[str, ...] = ()
    required_reasons: tuple[str, ...] = ()
    forbidden_reasons: tuple[str, ...] = ()
    forbidden_routes: tuple[TaskRoute, ...] = ()
    expected_metrics: Mapping[str, int | bool] = field(default_factory=dict)
    config: TaskRouterConfig | None = None
    min_recommended_subagents: int | None = None
    max_recommended_subagents: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRouterEvalCase":
        try:
            case_id = str(payload["id"])
            task = str(payload["task"])
            expected_route = TaskRoute(str(payload["expected_route"]))
        except KeyError as exc:  # pragma: no cover - exercised via ValueError path
            raise ValueError(f"missing required eval case field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise ValueError(
                f"invalid expected_route for eval case {payload.get('id', '<unknown>')!r}: "
                f"{payload.get('expected_route')!r}"
            ) from exc

        min_recommended = _optional_int(payload.get("min_recommended_subagents"))
        max_recommended = _optional_int(payload.get("max_recommended_subagents"))
        if min_recommended is not None and min_recommended < 0:
            raise ValueError("min_recommended_subagents must be >= 0")
        if max_recommended is not None and max_recommended < 0:
            raise ValueError("max_recommended_subagents must be >= 0")
        if (
            min_recommended is not None
            and max_recommended is not None
            and min_recommended > max_recommended
        ):
            raise ValueError("min_recommended_subagents cannot exceed max_recommended_subagents")

        return cls(
            id=case_id,
            task=task,
            expected_route=expected_route,
            tags=tuple(str(tag) for tag in payload.get("tags", ())),
            required_reasons=tuple(
                str(reason) for reason in payload.get("required_reasons", ())
            ),
            forbidden_reasons=tuple(
                str(reason) for reason in payload.get("forbidden_reasons", ())
            ),
            forbidden_routes=_coerce_forbidden_routes(payload.get("forbidden_routes", ())),
            min_recommended_subagents=min_recommended,
            max_recommended_subagents=max_recommended,
            expected_metrics=_coerce_expected_metrics(payload.get("expected_metrics", {})),
            config=_coerce_config(payload.get("config")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            "expected_route": self.expected_route.value,
            "tags": list(self.tags),
            "required_reasons": list(self.required_reasons),
            "forbidden_reasons": list(self.forbidden_reasons),
            "forbidden_routes": [route.value for route in self.forbidden_routes],
        }
        if self.min_recommended_subagents is not None:
            payload["min_recommended_subagents"] = self.min_recommended_subagents
        if self.max_recommended_subagents is not None:
            payload["max_recommended_subagents"] = self.max_recommended_subagents
        if self.expected_metrics:
            payload["expected_metrics"] = dict(self.expected_metrics)
        if self.config:
            payload["config"] = {
                "enable_parallel_subagents": self.config.enable_parallel_subagents,
                "min_parallel_subtasks": self.config.min_parallel_subtasks,
                "max_parallel_subagents": self.config.max_parallel_subagents,
            }
        return payload


@dataclass(frozen=True)
class TaskRouterCaseResult:
    """Evaluation result for a single golden case."""

    case: TaskRouterEvalCase
    decision: TaskRouteDecision
    passed: bool
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "decision": self.decision.to_dict(),
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class TaskRouterEvalSummary:
    """Aggregate deterministic metrics for a router evaluation corpus."""

    total: int
    passed: int
    failed: int
    accuracy: float
    over_delegate_count: int
    under_delegate_count: int
    unsafe_delegate_count: int
    route_confusion: Mapping[str, Mapping[str, int]]
    reason_counts: Mapping[str, int]
    average_recommended_subagents: float
    expected_parallel_count: int
    actual_parallel_count: int
    parallel_true_positive_count: int
    parallel_precision: float
    parallel_recall: float
    parallel_f1: float
    total_recommended_subagents: int
    tag_counts: Mapping[str, int]
    tag_failures: Mapping[str, int]
    tag_accuracy: Mapping[str, float]
    results: tuple[TaskRouterCaseResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "over_delegate_count": self.over_delegate_count,
            "under_delegate_count": self.under_delegate_count,
            "unsafe_delegate_count": self.unsafe_delegate_count,
            "route_confusion": {
                expected: dict(actual)
                for expected, actual in self.route_confusion.items()
            },
            "reason_counts": dict(self.reason_counts),
            "average_recommended_subagents": self.average_recommended_subagents,
            "expected_parallel_count": self.expected_parallel_count,
            "actual_parallel_count": self.actual_parallel_count,
            "parallel_true_positive_count": self.parallel_true_positive_count,
            "parallel_precision": self.parallel_precision,
            "parallel_recall": self.parallel_recall,
            "parallel_f1": self.parallel_f1,
            "total_recommended_subagents": self.total_recommended_subagents,
            "tag_counts": dict(self.tag_counts),
            "tag_failures": dict(self.tag_failures),
            "tag_accuracy": dict(self.tag_accuracy),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class TaskRouterBenchmarkVariant:
    """A named router configuration variant for offline benchmarking."""

    name: str
    config: TaskRouterConfig


@dataclass(frozen=True)
class TaskRouterBenchmarkDelta:
    """Metric delta for a benchmark variant relative to a baseline variant."""

    variant: str
    baseline_variant: str
    accuracy_delta: float
    over_delegate_delta: int
    under_delegate_delta: int
    unsafe_delegate_delta: int
    parallel_precision_delta: float
    parallel_recall_delta: float
    parallel_f1_delta: float
    total_recommended_subagents_delta: int
    average_recommended_subagents_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "baseline_variant": self.baseline_variant,
            "accuracy_delta": self.accuracy_delta,
            "over_delegate_delta": self.over_delegate_delta,
            "under_delegate_delta": self.under_delegate_delta,
            "unsafe_delegate_delta": self.unsafe_delegate_delta,
            "parallel_precision_delta": self.parallel_precision_delta,
            "parallel_recall_delta": self.parallel_recall_delta,
            "parallel_f1_delta": self.parallel_f1_delta,
            "total_recommended_subagents_delta": self.total_recommended_subagents_delta,
            "average_recommended_subagents_delta": self.average_recommended_subagents_delta,
        }


@dataclass(frozen=True)
class TaskRouterBenchmarkSummary:
    """Comparative benchmark summary for multiple router configs."""

    baseline_variant: str
    variants: Mapping[str, TaskRouterEvalSummary]
    deltas: Mapping[str, TaskRouterBenchmarkDelta]
    best_by_parallel_f1: str
    best_safe_variant: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_variant": self.baseline_variant,
            "variants": {
                name: self.variants[name].to_dict()
                for name in sorted(self.variants)
            },
            "deltas": {
                name: self.deltas[name].to_dict()
                for name in sorted(self.deltas)
            },
            "best_by_parallel_f1": self.best_by_parallel_f1,
            "best_safe_variant": self.best_safe_variant,
        }


RouterFn = Callable[..., TaskRouteDecision]


def load_eval_cases(path: str | Path) -> list[TaskRouterEvalCase]:
    """Load newline-delimited JSON eval cases with basic validation."""

    cases: list[TaskRouterEvalCase] = []
    case_ids: set[str] = set()
    case_path = Path(path)
    for line_number, raw_line in enumerate(case_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
            case = TaskRouterEvalCase.from_dict(payload)
        except Exception as exc:  # noqa: BLE001 - include line number for fixture debugging
            raise ValueError(f"invalid task router eval case at {case_path}:{line_number}") from exc
        if case.id in case_ids:
            raise ValueError(f"duplicate task router eval case id: {case.id}")
        case_ids.add(case.id)
        cases.append(case)
    return cases


def evaluate_task_router(
    cases: Iterable[TaskRouterEvalCase],
    *,
    config: TaskRouterConfig | None = None,
    router: RouterFn = route_task,
) -> TaskRouterEvalSummary:
    """Evaluate a ``route_task``-style callable against deterministic cases."""

    results: list[TaskRouterCaseResult] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    over_delegate_count = 0
    under_delegate_count = 0
    unsafe_delegate_count = 0
    recommended_total = 0
    expected_parallel_count = 0
    actual_parallel_count = 0
    parallel_true_positive_count = 0
    tag_counts: Counter[str] = Counter()
    tag_failures: Counter[str] = Counter()

    for case in cases:
        case_config = _case_config(config, case.config)
        decision = router(case.task, config=case_config)
        failures = _case_failures(case, decision)
        result = TaskRouterCaseResult(
            case=case,
            decision=decision,
            passed=not failures,
            failures=tuple(failures),
        )
        results.append(result)

        expected = case.expected_route.value
        actual = decision.route.value
        confusion[expected][actual] += 1
        reason_counts.update(decision.reasons)
        recommended = int(decision.metrics.get("recommended_subagents", 0))
        recommended_total += recommended
        expected_parallel = case.expected_route is TaskRoute.PARALLEL_SUBAGENTS
        actual_parallel = decision.route is TaskRoute.PARALLEL_SUBAGENTS
        expected_parallel_count += int(expected_parallel)
        actual_parallel_count += int(actual_parallel)
        parallel_true_positive_count += int(expected_parallel and actual_parallel)
        tag_counts.update(case.tags)
        if failures:
            tag_failures.update(case.tags)

        if decision.route is TaskRoute.PARALLEL_SUBAGENTS:
            if case.expected_route is not TaskRoute.PARALLEL_SUBAGENTS:
                over_delegate_count += 1
            if _UNSAFE_TAGS.intersection(case.tags):
                unsafe_delegate_count += 1
        elif case.expected_route is TaskRoute.PARALLEL_SUBAGENTS:
            under_delegate_count += 1

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    parallel_precision = (
        parallel_true_positive_count / actual_parallel_count
        if actual_parallel_count
        else (1.0 if expected_parallel_count == 0 else 0.0)
    )
    parallel_recall = (
        parallel_true_positive_count / expected_parallel_count
        if expected_parallel_count
        else 1.0
    )
    parallel_f1 = (
        2 * parallel_precision * parallel_recall / (parallel_precision + parallel_recall)
        if (parallel_precision + parallel_recall)
        else 0.0
    )
    tag_accuracy = {
        tag: ((tag_counts[tag] - tag_failures.get(tag, 0)) / tag_counts[tag])
        for tag in sorted(tag_counts)
    }
    return TaskRouterEvalSummary(
        total=total,
        passed=passed,
        failed=failed,
        accuracy=(passed / total) if total else 1.0,
        over_delegate_count=over_delegate_count,
        under_delegate_count=under_delegate_count,
        unsafe_delegate_count=unsafe_delegate_count,
        route_confusion=_complete_confusion(confusion),
        reason_counts=dict(sorted(reason_counts.items())),
        average_recommended_subagents=(recommended_total / total) if total else 0.0,
        expected_parallel_count=expected_parallel_count,
        actual_parallel_count=actual_parallel_count,
        parallel_true_positive_count=parallel_true_positive_count,
        parallel_precision=parallel_precision,
        parallel_recall=parallel_recall,
        parallel_f1=parallel_f1,
        total_recommended_subagents=recommended_total,
        tag_counts=dict(sorted(tag_counts.items())),
        tag_failures=dict(sorted(tag_failures.items())),
        tag_accuracy=tag_accuracy,
        results=tuple(results),
    )


def benchmark_task_router_configs(
    cases: Iterable[TaskRouterEvalCase],
    variants: Iterable[TaskRouterBenchmarkVariant],
    *,
    baseline_variant: str = "default",
    router: RouterFn = route_task,
    include_case_config_cases: bool = False,
) -> TaskRouterBenchmarkSummary:
    """Evaluate the same corpus across named router configuration variants.

    By default, cases with per-case router config overrides are excluded so a
    variant name like ``disabled`` or ``cap_2`` means the variant config is
    applied uniformly to every benchmarked case. Use ``evaluate_task_router``
    for exact golden-corpus validation of per-case config behavior.
    """

    case_list = list(cases)
    if not include_case_config_cases:
        case_list = [case for case in case_list if case.config is None]
    if not case_list:
        raise ValueError("benchmark corpus is empty")

    variant_list = list(variants)
    if not variant_list:
        raise ValueError("at least one benchmark variant is required")

    summaries: dict[str, TaskRouterEvalSummary] = {}
    for variant in variant_list:
        name = variant.name.strip()
        if not name:
            raise ValueError("benchmark variant name must not be empty")
        if name in summaries:
            raise ValueError(f"duplicate benchmark variant: {name}")
        summaries[name] = evaluate_task_router(case_list, config=variant.config, router=router)

    if baseline_variant not in summaries:
        raise ValueError(f"baseline variant not found: {baseline_variant}")

    baseline = summaries[baseline_variant]
    deltas = {
        name: _benchmark_delta(name, baseline_variant, summary, baseline)
        for name, summary in summaries.items()
    }
    best_by_parallel_f1 = min(summaries, key=lambda name: _benchmark_rank_key(name, summaries[name]))
    safe_candidates = [
        name
        for name, summary in summaries.items()
        if summary.unsafe_delegate_count == 0
        and summary.over_delegate_count == 0
        and summary.accuracy >= baseline.accuracy
    ]
    best_safe_variant = (
        min(safe_candidates, key=lambda name: _benchmark_rank_key(name, summaries[name]))
        if safe_candidates
        else None
    )

    return TaskRouterBenchmarkSummary(
        baseline_variant=baseline_variant,
        variants=dict(sorted(summaries.items())),
        deltas=dict(sorted(deltas.items())),
        best_by_parallel_f1=best_by_parallel_f1,
        best_safe_variant=best_safe_variant,
    )


def _benchmark_delta(
    variant: str,
    baseline_variant: str,
    summary: TaskRouterEvalSummary,
    baseline: TaskRouterEvalSummary,
) -> TaskRouterBenchmarkDelta:
    return TaskRouterBenchmarkDelta(
        variant=variant,
        baseline_variant=baseline_variant,
        accuracy_delta=summary.accuracy - baseline.accuracy,
        over_delegate_delta=summary.over_delegate_count - baseline.over_delegate_count,
        under_delegate_delta=summary.under_delegate_count - baseline.under_delegate_count,
        unsafe_delegate_delta=summary.unsafe_delegate_count - baseline.unsafe_delegate_count,
        parallel_precision_delta=summary.parallel_precision - baseline.parallel_precision,
        parallel_recall_delta=summary.parallel_recall - baseline.parallel_recall,
        parallel_f1_delta=summary.parallel_f1 - baseline.parallel_f1,
        total_recommended_subagents_delta=(
            summary.total_recommended_subagents - baseline.total_recommended_subagents
        ),
        average_recommended_subagents_delta=(
            summary.average_recommended_subagents - baseline.average_recommended_subagents
        ),
    )


def _benchmark_rank_key(name: str, summary: TaskRouterEvalSummary) -> tuple[float, int, int, float, int, str]:
    return (
        -summary.parallel_f1,
        summary.unsafe_delegate_count,
        summary.over_delegate_count,
        -summary.accuracy,
        summary.total_recommended_subagents,
        name,
    )


def _case_failures(
    case: TaskRouterEvalCase,
    decision: TaskRouteDecision,
) -> list[str]:
    failures: list[str] = []
    if decision.route is not case.expected_route:
        failures.append(
            f"route expected {case.expected_route.value}, got {decision.route.value}"
        )
    if decision.route in case.forbidden_routes:
        failures.append(f"forbidden route selected: {decision.route.value}")

    reasons = set(decision.reasons)
    for reason in case.required_reasons:
        if reason not in reasons:
            failures.append(f"missing required reason: {reason}")
    for reason in case.forbidden_reasons:
        if reason in reasons:
            failures.append(f"forbidden reason present: {reason}")

    for key, expected_value in case.expected_metrics.items():
        actual_value = decision.metrics.get(key)
        if actual_value != expected_value:
            failures.append(
                f"metric {key} expected {expected_value!r}, got {actual_value!r}"
            )

    recommended = int(decision.metrics.get("recommended_subagents", 0))
    if (
        case.min_recommended_subagents is not None
        and recommended < case.min_recommended_subagents
    ):
        failures.append(
            "recommended_subagents below minimum: "
            f"{recommended} < {case.min_recommended_subagents}"
        )
    if (
        case.max_recommended_subagents is not None
        and recommended > case.max_recommended_subagents
    ):
        failures.append(
            "recommended_subagents above maximum: "
            f"{recommended} > {case.max_recommended_subagents}"
        )
    return failures


def _complete_confusion(
    confusion: Mapping[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    routes = [route.value for route in TaskRoute]
    return {
        expected: {actual: int(confusion.get(expected, {}).get(actual, 0)) for actual in routes}
        for expected in routes
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_expected_metrics(value: Any) -> dict[str, int | bool]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected_metrics must be an object")
    metrics: dict[str, int | bool] = {}
    for key, metric_value in value.items():
        metric_key = str(key)
        if metric_key not in _METRIC_KEYS:
            raise ValueError(f"unknown expected metric key: {metric_key}")
        if isinstance(metric_value, bool):
            metrics[metric_key] = metric_value
        elif isinstance(metric_value, int):
            metrics[metric_key] = metric_value
        else:
            raise ValueError(
                f"expected metric {key!r} must be an int or bool, got {type(metric_value).__name__}"
            )
    return metrics


def _coerce_forbidden_routes(value: Any) -> tuple[TaskRoute, ...]:
    if not value:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("forbidden_routes must be a list")
    routes: list[TaskRoute] = []
    for route_value in value:
        try:
            routes.append(TaskRoute(str(route_value)))
        except ValueError as exc:
            raise ValueError(f"invalid forbidden route: {route_value!r}") from exc
    return tuple(routes)


def _coerce_config(value: Any) -> TaskRouterConfig | None:
    if not value:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("config must be an object")
    unknown = set(str(key) for key in value) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")
    return TaskRouterConfig(**dict(value))


def _case_config(
    global_config: TaskRouterConfig | None,
    case_config: TaskRouterConfig | None,
) -> TaskRouterConfig | None:
    if case_config is None:
        return global_config
    return case_config
