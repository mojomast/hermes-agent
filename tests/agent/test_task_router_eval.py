"""Tests for deterministic task-router evaluation metrics."""

import json
from pathlib import Path

import pytest

from agent.task_router import TaskRoute, TaskRouteDecision, TaskRouterConfig
from scripts.benchmark_task_router_configs import main as benchmark_task_router_main
from scripts.evaluate_task_router import main as evaluate_task_router_main
from agent.task_router_eval import (
    TaskRouterBenchmarkVariant,
    TaskRouterEvalCase,
    benchmark_task_router_configs,
    evaluate_task_router,
    load_eval_cases,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "task_router_eval_cases.jsonl"
)


def test_load_eval_cases_from_jsonl():
    cases = load_eval_cases(FIXTURE_PATH)

    assert len(cases) >= 12
    assert cases[0].id == "empty_prompt"
    assert cases[0].expected_route is TaskRoute.SERIAL
    assert all(case.id for case in cases)


def test_eval_summary_is_deterministic_and_serializable():
    cases = load_eval_cases(FIXTURE_PATH)

    first = evaluate_task_router(cases).to_dict()
    second = evaluate_task_router(cases).to_dict()

    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["total"] == len(cases)
    assert set(decoded["route_confusion"]) == {
        "serial",
        "batched_single_agent",
        "parallel_subagents",
    }
    assert "reason_counts" in decoded


def test_eval_counts_over_and_under_delegation_with_fake_routers():
    cases = [
        TaskRouterEvalCase(
            id="expected_serial",
            task="What is Redis?",
            expected_route=TaskRoute.SERIAL,
        ),
        TaskRouterEvalCase(
            id="expected_parallel",
            task="Investigate a, b, and c independently.",
            expected_route=TaskRoute.PARALLEL_SUBAGENTS,
        ),
    ]

    def always_parallel(task: str, *, config=None):
        return TaskRouteDecision(
            route=TaskRoute.PARALLEL_SUBAGENTS,
            reasons=("fake",),
            metrics={
                "estimated_subtasks": 3,
                "parallelizable_subtasks": 3,
                "recommended_subagents": 3,
                "requires_shared_context": False,
                "has_dependencies": False,
                "safety_risk": False,
            },
        )

    def always_serial(task: str, *, config=None):
        return TaskRouteDecision(
            route=TaskRoute.SERIAL,
            reasons=("fake",),
            metrics={
                "estimated_subtasks": 1,
                "parallelizable_subtasks": 0,
                "recommended_subagents": 0,
                "requires_shared_context": False,
                "has_dependencies": False,
                "safety_risk": False,
            },
        )

    over_summary = evaluate_task_router(cases, router=always_parallel)
    under_summary = evaluate_task_router(cases, router=always_serial)

    assert over_summary.over_delegate_count == 1
    assert over_summary.under_delegate_count == 0
    assert under_summary.over_delegate_count == 0
    assert under_summary.under_delegate_count == 1


def test_eval_flags_unsafe_delegation_with_fake_router():
    cases = [
        TaskRouterEvalCase(
            id="destructive",
            task="Delete files and force push.",
            expected_route=TaskRoute.SERIAL,
            tags=("safety", "destructive"),
        )
    ]

    def unsafe_parallel(task: str, *, config=None):
        return TaskRouteDecision(
            route=TaskRoute.PARALLEL_SUBAGENTS,
            reasons=("fake",),
            metrics={
                "estimated_subtasks": 3,
                "parallelizable_subtasks": 3,
                "recommended_subagents": 3,
                "requires_shared_context": False,
                "has_dependencies": False,
                "safety_risk": False,
            },
        )

    summary = evaluate_task_router(cases, router=unsafe_parallel)

    assert summary.unsafe_delegate_count == 1
    assert summary.over_delegate_count == 1
    assert summary.failed == 1
    assert "route expected serial" in summary.results[0].failures[0]


def test_current_router_passes_baseline_fixture():
    cases = load_eval_cases(FIXTURE_PATH)
    summary = evaluate_task_router(cases)

    assert summary.total == len(cases)
    assert summary.accuracy == 1.0
    assert summary.failed == 0
    assert summary.over_delegate_count == 0
    assert summary.under_delegate_count == 0
    assert summary.unsafe_delegate_count == 0
    assert summary.route_confusion["parallel_subagents"]["parallel_subagents"] >= 2


def test_config_can_disable_parallelism_in_eval():
    cases = [
        TaskRouterEvalCase(
            id="disabled_parallel",
            task="Investigate parser, cache, and retry logic independently.",
            expected_route=TaskRoute.BATCHED_SINGLE_AGENT,
            required_reasons=("parallel_disabled",),
        )
    ]

    summary = evaluate_task_router(
        cases,
        config=TaskRouterConfig(enable_parallel_subagents=False),
    )

    assert summary.passed == 1
    assert summary.results[0].decision.metrics["recommended_subagents"] == 0


def test_eval_case_supports_expected_metrics_and_case_config():
    case = TaskRouterEvalCase.from_dict(
        {
            "id": "case_config_cap",
            "task": "Investigate auth, billing, telemetry, and search independently.",
            "expected_route": "parallel_subagents",
            "required_reasons": ["capped_parallelism"],
            "expected_metrics": {"recommended_subagents": 2, "safety_risk": False},
            "config": {"max_parallel_subagents": 2},
            "tags": ["parallel", "config"],
        }
    )

    summary = evaluate_task_router([case])

    assert summary.passed == 1
    assert summary.results[0].decision.metrics["recommended_subagents"] == 2
    assert case.to_dict()["expected_metrics"] == {
        "recommended_subagents": 2,
        "safety_risk": False,
    }
    assert case.to_dict()["config"] == {
        "enable_parallel_subagents": True,
        "min_parallel_subtasks": 3,
        "max_parallel_subagents": 2,
    }


def test_eval_case_expected_metric_mismatch_fails_clearly():
    case = TaskRouterEvalCase(
        id="metric_mismatch",
        task="What is Redis?",
        expected_route=TaskRoute.SERIAL,
        expected_metrics={"estimated_subtasks": 99},
    )

    summary = evaluate_task_router([case])

    assert summary.failed == 1
    assert "metric estimated_subtasks expected 99" in summary.results[0].failures[0]


def test_eval_summary_parallel_quality_and_tag_metrics():
    cases = [
        TaskRouterEvalCase(
            id="serial",
            task="What is Redis?",
            expected_route=TaskRoute.SERIAL,
            tags=("serial",),
        ),
        TaskRouterEvalCase(
            id="parallel",
            task="Investigate auth, billing, and telemetry independently.",
            expected_route=TaskRoute.PARALLEL_SUBAGENTS,
            tags=("parallel",),
        ),
    ]

    def always_parallel(task: str, *, config=None):
        return TaskRouteDecision(
            route=TaskRoute.PARALLEL_SUBAGENTS,
            reasons=("fake",),
            metrics={
                "estimated_subtasks": 3,
                "parallelizable_subtasks": 3,
                "recommended_subagents": 3,
                "requires_shared_context": False,
                "has_dependencies": False,
                "safety_risk": False,
            },
        )

    summary = evaluate_task_router(cases, router=always_parallel)

    assert summary.expected_parallel_count == 1
    assert summary.actual_parallel_count == 2
    assert summary.parallel_true_positive_count == 1
    assert summary.parallel_precision == 0.5
    assert summary.parallel_recall == 1.0
    assert round(summary.parallel_f1, 3) == 0.667
    assert summary.total_recommended_subagents == 6
    assert summary.tag_counts == {"parallel": 1, "serial": 1}
    assert summary.tag_failures == {"serial": 1}
    assert summary.tag_accuracy == {"parallel": 1.0, "serial": 0.0}


def test_load_eval_cases_rejects_duplicate_ids(tmp_path):
    cases_file = tmp_path / "cases.jsonl"
    row = {"id": "dup", "task": "What is Redis?", "expected_route": "serial"}
    cases_file.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    try:
        load_eval_cases(cases_file)
    except ValueError as exc:
        assert "duplicate task router eval case id: dup" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate IDs should fail")


def test_load_eval_cases_reports_line_number_for_bad_json(tmp_path):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps({"id": "ok", "task": "What is Redis?", "expected_route": "serial"})
        + "\n{bad json}\n"
    )

    try:
        load_eval_cases(cases_file)
    except ValueError as exc:
        assert f"{cases_file}:2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("malformed JSON should fail")


def test_eval_corpus_has_single_canonical_fixture_location():
    fixture_files = sorted(FIXTURE_PATH.parents[1].glob("**/task_router_eval_cases.jsonl"))

    assert fixture_files == [FIXTURE_PATH]


def test_evaluate_task_router_cli_default_passes(capsys):
    exit_code = evaluate_task_router_main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "task_router_eval:" in captured.out
    assert "accuracy=1.000" in captured.out
    assert "parallel_quality:" in captured.out
    assert captured.err == ""


def test_evaluate_task_router_cli_json_output_is_parseable(capsys):
    exit_code = evaluate_task_router_main(["--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["total"] >= 25
    assert payload["failed"] == 0
    assert payload["over_delegate_count"] == 0
    assert payload["under_delegate_count"] == 0
    assert payload["parallel_precision"] == 1.0
    assert payload["parallel_recall"] == 1.0


def test_evaluate_task_router_cli_reports_failures_and_nonzero_exit(tmp_path, capsys):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps(
            {
                "id": "intentionally_wrong",
                "task": "What is Redis?",
                "expected_route": "parallel_subagents",
            }
        )
        + "\n"
    )

    exit_code = evaluate_task_router_main(["--cases", str(cases_file)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "task_router_eval: 0/1 passed" in captured.out
    assert "FAIL intentionally_wrong:" in captured.out
    assert "route expected parallel_subagents, got serial" in captured.out


def test_evaluate_task_router_cli_threshold_flags_affect_exit_status(capsys):
    assert evaluate_task_router_main(["--min-accuracy", "1.01"]) == 1
    capsys.readouterr()

    assert evaluate_task_router_main(["--disable-parallel-subagents"]) == 1
    captured = capsys.readouterr()
    assert "under=" in captured.out


def test_evaluate_task_router_cli_empty_corpus_returns_usage_error(tmp_path, capsys):
    cases_file = tmp_path / "empty.jsonl"
    cases_file.write_text("\n")

    exit_code = evaluate_task_router_main(["--cases", str(cases_file)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "task_router_eval: no cases loaded" in captured.err


def test_benchmark_task_router_configs_compares_variants():
    cases = load_eval_cases(FIXTURE_PATH)

    summary = benchmark_task_router_configs(
        iter(cases),
        [
            TaskRouterBenchmarkVariant("default", TaskRouterConfig()),
            TaskRouterBenchmarkVariant(
                "disabled",
                TaskRouterConfig(enable_parallel_subagents=False),
            ),
        ],
    )

    assert set(summary.variants) == {"default", "disabled"}
    assert summary.baseline_variant == "default"
    assert summary.variants["default"].under_delegate_count == 0
    assert summary.variants["disabled"].under_delegate_count > 0
    assert summary.variants["disabled"].actual_parallel_count == 0
    assert summary.variants["disabled"].total_recommended_subagents == 0
    assert summary.deltas["disabled"].parallel_recall_delta < 0
    assert summary.best_safe_variant == "default"


def test_benchmark_can_opt_into_case_config_overrides():
    cases = load_eval_cases(FIXTURE_PATH)

    summary = benchmark_task_router_configs(
        cases,
        [
            TaskRouterBenchmarkVariant("disabled", TaskRouterConfig(enable_parallel_subagents=False)),
        ],
        baseline_variant="disabled",
        include_case_config_cases=True,
    )

    assert summary.variants["disabled"].actual_parallel_count == 1
    assert summary.variants["disabled"].total_recommended_subagents == 3


def test_benchmark_rejects_duplicate_variant_names():
    cases = load_eval_cases(FIXTURE_PATH)

    with pytest.raises(ValueError, match="duplicate benchmark variant"):
        benchmark_task_router_configs(
            cases,
            [
                TaskRouterBenchmarkVariant("default", TaskRouterConfig()),
                TaskRouterBenchmarkVariant("default", TaskRouterConfig()),
            ],
        )


def test_benchmark_rejects_missing_baseline_variant():
    cases = load_eval_cases(FIXTURE_PATH)

    with pytest.raises(ValueError, match="baseline variant not found"):
        benchmark_task_router_configs(
            cases,
            [TaskRouterBenchmarkVariant("candidate", TaskRouterConfig())],
            baseline_variant="default",
        )


def test_benchmark_rejects_empty_cases_or_variants():
    with pytest.raises(ValueError, match="benchmark corpus is empty"):
        benchmark_task_router_configs(
            [],
            [TaskRouterBenchmarkVariant("default", TaskRouterConfig())],
        )

    with pytest.raises(ValueError, match="at least one benchmark variant"):
        benchmark_task_router_configs(
            load_eval_cases(FIXTURE_PATH),
            [],
        )

    with pytest.raises(ValueError, match="variant name must not be empty"):
        benchmark_task_router_configs(
            load_eval_cases(FIXTURE_PATH),
            [TaskRouterBenchmarkVariant("  ", TaskRouterConfig())],
        )


def test_benchmark_summary_is_json_serializable_and_deterministic():
    cases = load_eval_cases(FIXTURE_PATH)
    variants = [
        TaskRouterBenchmarkVariant("default", TaskRouterConfig()),
        TaskRouterBenchmarkVariant("cap_2", TaskRouterConfig(max_parallel_subagents=2)),
    ]

    first = benchmark_task_router_configs(cases, variants).to_dict()
    second = benchmark_task_router_configs(cases, variants).to_dict()

    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["baseline_variant"] == "default"
    assert set(decoded) == {
        "baseline_variant",
        "variants",
        "deltas",
        "best_by_parallel_f1",
        "best_safe_variant",
    }
    assert set(decoded["variants"]) == {"cap_2", "default"}


def test_benchmark_cli_default_passes(capsys):
    exit_code = benchmark_task_router_main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "task_router_benchmark:" in captured.out
    assert "baseline=default" in captured.out
    assert "best_by_parallel_f1=" in captured.out
    assert captured.err == ""


def test_benchmark_cli_json_output_is_parseable(capsys):
    exit_code = benchmark_task_router_main(["--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["baseline_variant"] == "default"
    assert "variants" in payload
    assert "default" in payload["variants"]
    assert "disabled" in payload["variants"]
    assert "deltas" in payload


def test_benchmark_cli_baseline_gate_failure_exits_nonzero(capsys):
    exit_code = benchmark_task_router_main(["--min-baseline-accuracy", "1.01"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "task_router_benchmark:" in captured.out


def test_benchmark_cli_empty_corpus_returns_usage_error(tmp_path, capsys):
    cases_file = tmp_path / "empty.jsonl"
    cases_file.write_text("\n")

    exit_code = benchmark_task_router_main(["--cases", str(cases_file)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "benchmark corpus is empty" in captured.err
