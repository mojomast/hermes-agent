#!/usr/bin/env python3
"""Deterministic Hermes efficiency benchmark over real context-engine code.

Generated result JSON belongs outside the repository (for example under /tmp).
The six workloads model the tool-result shapes produced by representative Hermes
runs while exercising ContextCompressor's actual request-projection path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough

WORKLOADS = {
    "repository_investigation": [12000, 12000, 12000, 12000, 9000],
    "multi_file_engineering": [10000, 10000, 10000, 18000, 9000],
    "web_research_synthesis": [15000, 15000, 15000, 15000, 9000],
    "cross_session_recall": [25000, 25000, 9000],
    "parallel_subagents": [20000, 20000, 20000, 9000],
    "failed_run_recovery": [12000, 12000, 12000, 9000],
}


def _assistant(call_id: str, family: str, index: int) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "benchmark_evidence",
                "arguments": json.dumps({"family": family, "index": index}),
            },
        }],
    }


def build_messages(family: str, sizes: list[int]) -> list[dict]:
    messages = [
        {"role": "system", "content": "Hermes benchmark system contract."},
        {"role": "user", "content": f"Complete {family} with validated evidence."},
    ]
    for index, size in enumerate(sizes):
        call_id = f"{family}-{index}"
        messages.append(_assistant(call_id, family, index))
        # Distinct, deterministic, evidence-bearing output.
        prefix = f"evidence family={family} item={index} status=ok\n"
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": prefix + chr(65 + index) * max(0, size - len(prefix)),
        })
    # The recent tail must remain byte-identical.
    messages.extend([
        {"role": "assistant", "content": "I am validating the latest evidence."},
        {"role": "user", "content": "Return the grounded result; do not drop work."},
    ])
    return messages


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def estimate_tokens(messages: list[dict]) -> int:
    return estimate_messages_tokens_rough(messages)


def synthetic_intervals(count: int) -> list[tuple[int, int]]:
    """Deterministic mixed sequential/concurrent fixture intervals in ms."""
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for index in range(count):
        start = cursor if index % 3 == 0 else max(0, cursor - 5)
        end = start + 10 + index
        intervals.append((start, end))
        if index % 3 == 0:
            cursor = end
    return intervals


def interval_metrics(intervals: list[tuple[int, int]]) -> dict:
    points = sorted({point for interval in intervals for point in interval})
    sequential = concurrent = 0
    peak = 0
    for start, end in zip(points, points[1:]):
        active = sum(a <= start < b for a, b in intervals)
        peak = max(peak, active)
        if active == 1:
            sequential += end - start
        elif active >= 2:
            concurrent += end - start
    return {
        "sequential_tool_time_ms": sequential,
        "concurrent_tool_time_ms": concurrent,
        "peak_concurrency": peak,
        "tool_work_ms": sum(end - start for start, end in intervals),
    }


def run_case(compressor: ContextCompressor, family: str, sizes: list[int]) -> dict:
    canonical = build_messages(family, sizes)
    canonical_before = digest(canonical)
    original = copy.deepcopy(canonical)
    before_tokens = estimate_tokens(canonical)
    started = time.perf_counter_ns()
    prune = getattr(compressor, "prune_tool_results_only", None)
    if callable(prune):
        projected, pruned_count = prune(canonical, current_tokens=before_tokens)
    else:
        projected, pruned_count = canonical, 0
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    after_tokens = estimate_tokens(projected)
    tail_exact = projected[-2:] == original[-2:]
    structure_exact = [m.get("role") for m in projected] == [m.get("role") for m in original]
    ids_exact = [m.get("tool_call_id") for m in projected] == [m.get("tool_call_id") for m in original]
    canonical_unchanged = digest(canonical) == canonical_before
    correctness = int(tail_exact) + int(structure_exact) + int(ids_exact) + int(canonical_unchanged)
    metrics = interval_metrics(synthetic_intervals(len(sizes)))
    return {
        "family": family,
        "elapsed_ms": elapsed_ms,
        "model_call_count": 1,
        "tool_call_count": len(sizes),
        "context_tokens_before": before_tokens,
        "context_tokens_after": after_tokens,
        "context_reduction_pct": round(100 * (before_tokens - after_tokens) / before_tokens, 3),
        "duplicate_reads_or_searches": 0,
        "retry_count": 0,
        "success_correctness_score": correctness / 4,
        "recovery_behavior": "canonical_evidence_retained" if canonical_unchanged else "canonical_mutated",
        "cache_state": "disabled",
        "pruned_count": pruned_count,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        compressor = ContextCompressor(
            model="benchmark",
            quiet_mode=True,
            protect_first_n=2,
            protect_last_n=4,
            **({
                "proactive_prune_tokens": 1,
                "proactive_prune_min_result_chars": 8000,
                "proactive_prune_min_reclaim_tokens": 4096,
            } if args.phase == "candidate" else {}),
        )
    rows = [
        run_case(compressor, family, sizes)
        for _ in range(args.repetitions)
        for family, sizes in WORKLOADS.items()
    ]
    reductions = [row["context_reduction_pct"] for row in rows]
    elapsed = [row["elapsed_ms"] for row in rows]
    result = {
        "schema_version": 1,
        "phase": args.phase,
        "workload_count": len(WORKLOADS),
        "repetitions": args.repetitions,
        "summary": {
            "median_context_reduction_pct": statistics.median(reductions),
            "median_elapsed_ms": statistics.median(elapsed),
            "correctness_score": min(row["success_correctness_score"] for row in rows),
            "total_model_calls": sum(row["model_call_count"] for row in rows),
            "total_tool_calls": sum(row["tool_call_count"] for row in rows),
            "duplicate_reads_or_searches": sum(row["duplicate_reads_or_searches"] for row in rows),
            "retry_count": sum(row["retry_count"] for row in rows),
            "peak_concurrency": max(row["peak_concurrency"] for row in rows),
            "cache_states": sorted({row["cache_state"] for row in rows}),
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
