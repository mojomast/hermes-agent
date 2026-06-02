#!/usr/bin/env python3
"""Run the deterministic task-router evaluation corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_router import TaskRouterConfig  # noqa: E402
from agent.task_router_eval import evaluate_task_router, load_eval_cases  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "tests" / "fixtures" / "task_router_eval_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSONL eval cases")
    parser.add_argument("--json", action="store_true", help="emit full JSON summary")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help="minimum acceptable exact-case accuracy",
    )
    parser.add_argument(
        "--max-over-delegate",
        type=int,
        default=0,
        help="maximum allowed non-parallel cases routed to parallel subagents",
    )
    parser.add_argument(
        "--max-under-delegate",
        type=int,
        default=0,
        help="maximum allowed parallel cases routed below parallel subagents",
    )
    parser.add_argument(
        "--min-parallel-precision",
        type=float,
        default=1.0,
        help="minimum precision for the parallel_subagents route",
    )
    parser.add_argument(
        "--min-parallel-recall",
        type=float,
        default=1.0,
        help="minimum recall for the parallel_subagents route",
    )
    parser.add_argument(
        "--max-parallel-subagents",
        type=int,
        default=TaskRouterConfig.max_parallel_subagents,
        help="router max_parallel_subagents for cases without per-case config",
    )
    parser.add_argument(
        "--min-parallel-subtasks",
        type=int,
        default=TaskRouterConfig.min_parallel_subtasks,
        help="router min_parallel_subtasks for cases without per-case config",
    )
    parser.add_argument(
        "--disable-parallel-subagents",
        action="store_true",
        help="disable router parallel_subagents route for cases without per-case config",
    )
    args = parser.parse_args(argv)

    cases = load_eval_cases(args.cases)
    if not cases:
        print("task_router_eval: no cases loaded", file=sys.stderr)
        return 2
    config = TaskRouterConfig(
        enable_parallel_subagents=not args.disable_parallel_subagents,
        min_parallel_subtasks=args.min_parallel_subtasks,
        max_parallel_subagents=args.max_parallel_subagents,
    )
    summary = evaluate_task_router(cases, config=config)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"task_router_eval: {summary.passed}/{summary.total} passed; "
            f"accuracy={summary.accuracy:.3f}"
        )
        print(
            "delegation: "
            f"over={summary.over_delegate_count} "
            f"under={summary.under_delegate_count} "
            f"unsafe={summary.unsafe_delegate_count} "
            f"avg_subagents={summary.average_recommended_subagents:.2f} "
            f"total_subagents={summary.total_recommended_subagents}"
        )
        print(
            "parallel_quality: "
            f"expected={summary.expected_parallel_count} "
            f"actual={summary.actual_parallel_count} "
            f"tp={summary.parallel_true_positive_count} "
            f"precision={summary.parallel_precision:.3f} "
            f"recall={summary.parallel_recall:.3f} "
            f"f1={summary.parallel_f1:.3f}"
        )
        if summary.failed:
            for result in summary.results:
                if result.failures:
                    print(f"FAIL {result.case.id}: {'; '.join(result.failures)}")

    ok = (
        summary.accuracy >= args.min_accuracy
        and summary.unsafe_delegate_count == 0
        and summary.over_delegate_count <= args.max_over_delegate
        and summary.under_delegate_count <= args.max_under_delegate
        and summary.parallel_precision >= args.min_parallel_precision
        and summary.parallel_recall >= args.min_parallel_recall
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
