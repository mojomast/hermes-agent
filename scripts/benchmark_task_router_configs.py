#!/usr/bin/env python3
"""Benchmark deterministic task-router config variants against one corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_router import TaskRouterConfig  # noqa: E402
from agent.task_router_eval import (  # noqa: E402
    TaskRouterBenchmarkSummary,
    TaskRouterBenchmarkVariant,
    benchmark_task_router_configs,
    load_eval_cases,
)

DEFAULT_CASES = REPO_ROOT / "tests" / "fixtures" / "task_router_eval_cases.jsonl"

DEFAULT_VARIANTS = (
    TaskRouterBenchmarkVariant("default", TaskRouterConfig()),
    TaskRouterBenchmarkVariant(
        "disabled",
        TaskRouterConfig(enable_parallel_subagents=False),
    ),
    TaskRouterBenchmarkVariant("threshold_2", TaskRouterConfig(min_parallel_subtasks=2)),
    TaskRouterBenchmarkVariant("threshold_4", TaskRouterConfig(min_parallel_subtasks=4)),
    TaskRouterBenchmarkVariant("cap_2", TaskRouterConfig(max_parallel_subagents=2)),
    TaskRouterBenchmarkVariant("cap_3", TaskRouterConfig(max_parallel_subagents=3)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL eval cases")
    parser.add_argument("--json", action="store_true", help="emit full JSON summary")
    parser.add_argument(
        "--baseline-variant",
        default="default",
        help="variant name used for delta and gate comparisons",
    )
    parser.add_argument(
        "--min-baseline-accuracy",
        type=float,
        default=1.0,
        help="minimum acceptable baseline exact-case accuracy",
    )
    parser.add_argument(
        "--max-baseline-over-delegate",
        type=int,
        default=0,
        help="maximum baseline non-parallel cases routed to parallel subagents",
    )
    parser.add_argument(
        "--max-baseline-under-delegate",
        type=int,
        default=0,
        help="maximum baseline parallel cases routed below parallel subagents",
    )
    parser.add_argument(
        "--max-baseline-unsafe-delegate",
        type=int,
        default=0,
        help="maximum baseline unsafe/destructive cases routed to parallel subagents",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_eval_cases(args.cases)
        summary = benchmark_task_router_configs(
            cases,
            DEFAULT_VARIANTS,
            baseline_variant=args.baseline_variant,
        )
    except ValueError as exc:
        print(f"task_router_benchmark: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)

    baseline = summary.variants[summary.baseline_variant]
    ok = (
        baseline.accuracy >= args.min_baseline_accuracy
        and baseline.over_delegate_count <= args.max_baseline_over_delegate
        and baseline.under_delegate_count <= args.max_baseline_under_delegate
        and baseline.unsafe_delegate_count <= args.max_baseline_unsafe_delegate
    )
    return 0 if ok else 1


def _print_human_summary(summary: TaskRouterBenchmarkSummary) -> None:
    first = next(iter(summary.variants.values()), None)
    total = first.total if first else 0
    print(
        "task_router_benchmark: "
        f"{len(summary.variants)} variants over {total} cases; "
        f"baseline={summary.baseline_variant}"
    )
    print(
        "variant        acc    over under unsafe precision recall f1    "
        "avg_subagents total_subagents"
    )
    for name, variant_summary in summary.variants.items():
        print(
            f"{name:<14} "
            f"{variant_summary.accuracy:.3f}  "
            f"{variant_summary.over_delegate_count:<4} "
            f"{variant_summary.under_delegate_count:<5} "
            f"{variant_summary.unsafe_delegate_count:<6} "
            f"{variant_summary.parallel_precision:.3f}     "
            f"{variant_summary.parallel_recall:.3f}  "
            f"{variant_summary.parallel_f1:.3f} "
            f"{variant_summary.average_recommended_subagents:.2f}          "
            f"{variant_summary.total_recommended_subagents}"
        )
    print(f"best_by_parallel_f1={summary.best_by_parallel_f1}")
    print(f"best_safe_variant={summary.best_safe_variant}")


if __name__ == "__main__":
    raise SystemExit(main())
