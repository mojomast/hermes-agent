"""Tests for the pure task router efficiency helper."""

import json

from agent.task_router import TaskRoute, TaskRouterConfig, route_task, to_delegate_tasks


def test_empty_prompt_routes_serial():
    decision = route_task("   ")

    assert decision.route is TaskRoute.SERIAL
    assert "empty_task" in decision.reasons
    assert decision.metrics["estimated_subtasks"] == 0
    assert decision.metrics["recommended_subagents"] == 0


def test_simple_question_routes_serial():
    decision = route_task("What is a Python virtualenv?")

    assert decision.route is TaskRoute.SERIAL
    assert "simple" in decision.reasons
    assert decision.metrics["parallelizable_subtasks"] == 0


def test_dependency_chain_blocks_parallel():
    decision = route_task(
        "First inspect the API, then based on that result update the caller."
    )

    assert decision.route is TaskRoute.SERIAL
    assert "dependent_steps" in decision.reasons
    assert decision.metrics["has_dependencies"] is True
    assert decision.metrics["recommended_subagents"] == 0


def test_destructive_task_routes_serial():
    decision = route_task("Delete unused files and force push the cleaned branch.")

    assert decision.route is TaskRoute.SERIAL
    assert "safety_guard" in decision.reasons
    assert decision.metrics["safety_risk"] is True
    assert decision.metrics["recommended_subagents"] == 0


def test_homogeneous_batch_routes_batched_single_agent():
    decision = route_task(
        "Apply the same formatting cleanup to README.md, docs/usage.md, and docs/config.md."
    )

    assert decision.route is TaskRoute.BATCHED_SINGLE_AGENT
    assert "homogeneous_batch" in decision.reasons
    assert decision.metrics["estimated_subtasks"] >= 3
    assert decision.metrics["requires_shared_context"] is True


def test_write_heavy_multi_step_not_parallel():
    decision = route_task("Edit foo.py and update the tests and README.")

    assert decision.route in {TaskRoute.SERIAL, TaskRoute.BATCHED_SINGLE_AGENT}
    assert decision.route is not TaskRoute.PARALLEL_SUBAGENTS
    assert decision.metrics["recommended_subagents"] == 0


def test_independent_research_routes_parallel_with_subtasks():
    decision = route_task(
        "Investigate the CLI startup path, gateway batching, and tool sandbox independently."
    )

    assert decision.route is TaskRoute.PARALLEL_SUBAGENTS
    assert "independent_workstreams" in decision.reasons
    assert decision.metrics["parallelizable_subtasks"] >= 3
    assert decision.metrics["recommended_subagents"] >= 2
    assert len(decision.subtasks) == decision.metrics["recommended_subagents"]


def test_parallelism_is_capped_by_config():
    decision = route_task(
        "Analyze auth, billing, notifications, search, indexing, telemetry, and permissions independently.",
        config=TaskRouterConfig(max_parallel_subagents=3),
    )

    assert decision.route is TaskRoute.PARALLEL_SUBAGENTS
    assert decision.metrics["recommended_subagents"] == 3
    assert "capped_parallelism" in decision.reasons


def test_parallel_disabled_forces_non_parallel():
    decision = route_task(
        "Investigate CLI, gateway, and memory independently.",
        config=TaskRouterConfig(enable_parallel_subagents=False),
    )

    assert decision.route is not TaskRoute.PARALLEL_SUBAGENTS
    assert "parallel_disabled" in decision.reasons
    assert decision.metrics["recommended_subagents"] == 0


def test_min_parallel_threshold_is_respected():
    decision = route_task(
        "Compare parser and renderer independently.",
        config=TaskRouterConfig(min_parallel_subtasks=3),
    )

    assert decision.route is not TaskRoute.PARALLEL_SUBAGENTS
    assert "below_parallel_threshold" in decision.reasons
    assert decision.metrics["recommended_subagents"] == 0


def test_to_delegate_tasks_only_for_parallel():
    serial = route_task("What is Redis?")
    assert to_delegate_tasks(serial) == []

    parallel = route_task("Compare Redis, Postgres, and SQLite independently.")
    tasks = to_delegate_tasks(parallel)
    assert len(tasks) == parallel.metrics["recommended_subagents"]
    assert all(set(task) >= {"goal", "context"} for task in tasks)


def test_decision_payload_is_json_serializable():
    decision = route_task("Investigate parser, cache, and retry logic independently.")
    payload = decision.to_dict()

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["route"] == "parallel_subagents"
    assert isinstance(decoded["metrics"]["estimated_subtasks"], int)



def test_explicit_subagents_lower_safe_readonly_threshold():
    decision = route_task("Use two subagents to compare Redis and SQLite independently.")

    assert decision.route is TaskRoute.PARALLEL_SUBAGENTS
    assert "independent_workstreams" in decision.reasons
    assert decision.metrics["recommended_subagents"] == 2


def test_user_can_forbid_subagents_even_with_independent_signal():
    prompts = [
        "Do not use subagents; review auth, billing, and search independently.",
        "Please do not use any subagents; review auth, billing, and search independently.",
        "Review auth, billing, and search independently without using subagents.",
        "No parallel subagents please; review auth, billing, and search independently.",
    ]

    for prompt in prompts:
        decision = route_task(prompt)
        assert decision.route is TaskRoute.BATCHED_SINGLE_AGENT
        assert "delegation_forbidden_by_user" in decision.reasons
        assert "independent_workstreams" not in decision.reasons
        assert decision.metrics["recommended_subagents"] == 0


def test_secret_terms_block_parallel_delegation():
    decision = route_task("Independently audit production API keys, database passwords, and OAuth tokens.")

    assert decision.route is TaskRoute.SERIAL
    assert "safety_guard" in decision.reasons
    assert decision.metrics["safety_risk"] is True


def test_readonly_qualifier_after_semicolon_is_not_fake_subtask():
    decision = route_task("Analyze auth, billing, and search independently; do not write changes.")

    assert decision.route is TaskRoute.PARALLEL_SUBAGENTS
    assert decision.metrics["estimated_subtasks"] == 3
    assert decision.metrics["recommended_subagents"] == 3


def test_negated_write_guard_does_not_block_readonly_parallel_review():
    prompts = [
        "Review auth, billing, and search independently. Do not modify files.",
        "Use parallel subagents to review auth, billing, and search. Do not edit files.",
        "Read-only review auth, billing, and search independently. No file changes.",
        "Review auth, billing, and search independently without modifying files.",
    ]

    for prompt in prompts:
        decision = route_task(prompt)
        assert decision.route is TaskRoute.PARALLEL_SUBAGENTS
        assert "independent_workstreams" in decision.reasons
        assert "small_related_batch" not in decision.reasons
        assert decision.metrics["requires_shared_context"] is False
        assert decision.metrics["recommended_subagents"] == 3
        subtask_goals = "\n".join(subtask.goal for subtask in decision.subtasks)
        assert "do not" not in subtask_goals
        assert "without modifying" not in subtask_goals
        assert "no file changes" not in subtask_goals


def test_negated_write_guard_keeps_positive_write_requests_conservative():
    decision = route_task("Review auth, billing, and search independently, then update docs. Do not edit tests.")

    assert decision.route is TaskRoute.SERIAL
    assert "dependent_steps" in decision.reasons
    assert decision.metrics["recommended_subagents"] == 0


def test_forbidden_subagents_still_override_readonly_guard():
    decision = route_task("Do not use subagents; read-only review auth, billing, and search independently. No file changes.")

    assert decision.route is TaskRoute.BATCHED_SINGLE_AGENT
    assert "delegation_forbidden_by_user" in decision.reasons
    assert decision.metrics["recommended_subagents"] == 0
