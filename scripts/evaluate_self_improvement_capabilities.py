#!/usr/bin/env python3
"""Small offline eval harness for Hermes self-improvement substrates.

Measures:
- fixture execution trace -> TrainingEpisode conversion/export latency + reward
- fixture codebase -> semantic symbols/definitions/references/diagnostics latency

Writes no generated artifacts unless --output-json is supplied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import sys
import tempfile
import time
from pathlib import Path

# Allow direct execution by absolute path without relying on the caller's cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.semantic_code_index import get_diagnostics, go_to_definition, list_references, semantic_lookup
from agent.contrastive_episode_retrieval import contrastive_replay_eval
from agent.outcome_events import (
    OPERATOR_RELATION_PRODUCER, append_outcome_event, append_producer_outcome_event,
    retract_outcome_event,
)
from agent.shadow_outcome_capture import classify_foreground_pytest
from agent.shadow_recurrence import evaluate_shadow_recurrence
from agent.tracing import TraceRecorder, hash_user_message
from agent.training_episodes import export_episodes_jsonl, iter_episodes, replay_eval_episodes
from hermes_state import SessionDB

DEFAULT_CONTRASTIVE_TASK_TEXT = "audit whether Hermes learns from mistakes and uses prior failed episodes"
FIXTURE_CONTRASTIVE_TASK_TEXT = "ideate a new capability subsystem extension"
FIXTURE_PRIVACY_CANARIES = (
    "FIXTURE_PROMPT_VELVET", "FIXTURE_USER_MANGO", "FIXTURE_ARGS_OTTER",
    "FIXTURE_RESULT_FOX", "FIXTURE_STDOUT_BEAR", "FIXTURE_STDERR_WOLF",
    "FIXTURE_RESPONSE_LYNX", "FIXTURE_TRANSCRIPT_HARE",
    "FIXTURE_MODEL_RAVEN", "FIXTURE_NESTED_ELK",
)


def _finite_float(value: str | float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_limit(value: str | int) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("limit must be an integer from 1 to 200")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer from 1 to 200") from exc
    if not 1 <= result <= 200:
        raise ValueError("limit must be an integer from 1 to 200")
    return result


def _fixture_raw_metadata() -> dict:
    return {
        "prompt": FIXTURE_PRIVACY_CANARIES[0],
        "user_message": FIXTURE_PRIVACY_CANARIES[1],
        "tool_args": FIXTURE_PRIVACY_CANARIES[2],
        "result": FIXTURE_PRIVACY_CANARIES[3],
        "stdout": FIXTURE_PRIVACY_CANARIES[4],
        "stderr": FIXTURE_PRIVACY_CANARIES[5],
        "response": FIXTURE_PRIVACY_CANARIES[6],
        "transcript": FIXTURE_PRIVACY_CANARIES[7],
        "modelOutput": FIXTURE_PRIVACY_CANARIES[8],
        "nested": {"payload": FIXTURE_PRIVACY_CANARIES[9]},
    }


def _record_fixture_trace(
    db: SessionDB, *, trace_id: str, successful: bool, tools: tuple[str, ...],
    plant_canaries: bool = False, session_id: str = "eval-session",
) -> None:
    recorder = TraceRecorder(
        session_id=session_id, turn_id=f"eval-{trace_id}",
        user_message_hash=hash_user_message("eval task"), trace_id=trace_id,
    )
    with recorder.span("turn", "root"):
        with recorder.span("model_call", "llm_api_call"):
            pass
        for index, tool in enumerate(tools):
            metadata = _fixture_raw_metadata() if plant_canaries and index == 0 else {}
            with recorder.span("tool_call", tool, metadata) as span:
                if not successful:
                    span.status = "error"
                    span.error_class = "DuplicateCapabilityProposal"
        if successful:
            with recorder.span("final_answer", "turn.final_answer", {"completed": True, "response_len": 64}):
                pass
    recorder.finish("completed" if successful else "failed")
    db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())


def make_trace(db_path: Path) -> int:
    """Create a failed duplicate proposal and its verified structural replacement."""
    db = SessionDB(db_path)
    try:
        _record_fixture_trace(
            db, trace_id="fixture-duplicate-mistake", successful=False,
            tools=("write_file",), plant_canaries=True,
        )
        _record_fixture_trace(
            db, trace_id="fixture-inventory-replacement", successful=True,
            tools=("search_files", "read_file", "training_episodes"),
        )
    finally:
        db.close()

    mistake = append_outcome_event(
        db_path,
        trace_id="fixture-duplicate-mistake",
        event_type="duplicate_proposal",
        source="user",
        polarity="negative",
        confidence=0.99,
        taxonomy_code="duplicate_existing_capability",
    )
    append_outcome_event(
        db_path,
        trace_id="fixture-inventory-replacement",
        event_type="user_correction",
        source="user",
        polarity="positive",
        confidence=0.99,
        taxonomy_code="duplicate_existing_capability",
        supersedes_event_id=mistake.event_id,
    )
    append_outcome_event(
        db_path,
        trace_id="fixture-inventory-replacement",
        event_type="verification_passed",
        source="verifier",
        polarity="positive",
        confidence=1.0,
    )
    # Count-only provenance: fixture construction owns this fixed list and
    # planted every deduplicated canary in the first trace's metadata.
    return len(set(FIXTURE_PRIVACY_CANARIES))


def make_code_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(
        "class Greeter:\n"
        "    def hello(self, name):\n"
        "        return format_name(name)\n"
        "\n"
        "def format_name(value):\n"
        "    return value.title()\n",
        encoding="utf-8",
    )
    (root / "index.ts").write_text("export function loadUser() { return { name: 'Ada' } }\n", encoding="utf-8")


def _lesson_for(report: dict, taxonomy_code: str) -> dict:
    return next(item for item in report["lessons"] if item["taxonomy_code"] == taxonomy_code)


def _count_only_recurrence(report: dict) -> dict:
    """Project the canonical report without lesson/store identifiers."""
    if (
        report.get("shadow_only") is not True
        or report.get("activation_allowed") is not False
        or report.get("prompt_modified") is not False
    ):
        raise RuntimeError("shadow recurrence safety declarations failed closed")
    return {
        "schema_version": "shadow_recurrence_eval.v1",
        "policy_version": report.get("policy_version", ""),
        "shadow_only": report.get("shadow_only") is True,
        "activation_allowed": False,
        "prompt_modified": False,
        "historical_event_count": int(report.get("historical_event_count", 0)),
        "effective_event_count": int(report.get("effective_event_count", 0)),
        "candidate_lesson_count": int(report.get("candidate_lesson_count", 0)),
        "activation_eligible_lesson_count": int(report.get("activation_eligible_lesson_count", 0)),
        "active_lesson_count": int(report.get("active_lesson_count", 0)),
    }


def _validate_recurrence_privacy(report: dict, *, expected_canary_count: int, planted_canary_count: int) -> tuple[int, int]:
    privacy = report.get("privacy")
    false_keys = (
        "raw_content_exported", "raw_user_text_read", "raw_prompt_read",
        "raw_tool_payloads_read", "evidence_digests_exported", "store_identifiers_exported",
    )
    malformed = not isinstance(privacy, dict)
    malformed = malformed or any(privacy.get(key) is not False for key in false_keys)
    malformed = malformed or privacy.get("taxonomy_codes_allowlisted") is not True
    if malformed:
        raise RuntimeError("malformed shadow recurrence privacy declarations")
    blob = json.dumps(report, sort_keys=True)
    coverage = sum(canary not in blob for canary in set(FIXTURE_PRIVACY_CANARIES))
    if expected_canary_count != planted_canary_count or coverage != expected_canary_count:
        raise RuntimeError("shadow recurrence privacy canary coverage mismatch")
    return coverage, sum(canary in blob for canary in set(FIXTURE_PRIVACY_CANARIES))


_SYNTHETIC_MANUAL_USER_MISTAKE_ROOTS = (
    ("recurrence-a", "recurrence-negative-a", "recurrence-session-1"),
    ("recurrence-b", "recurrence-negative-b", "recurrence-session-2"),
    ("recurrence-c", "recurrence-negative-c", "recurrence-session-3"),
    ("recurrence-d", "recurrence-negative-d", "recurrence-session-4"),
)


def _append_synthetic_manual_user_mistake(
    db_path: Path, *, trace_id: str, event_id: str, taxonomy_code: str, created_at: int,
):
    """Plant synthetic-only manual user evidence; never an active adaptation."""
    return append_outcome_event(
        db_path, trace_id=trace_id, event_id=event_id,
        event_type="duplicate_proposal", source="user", polarity="negative",
        confidence=.99, taxonomy_code=taxonomy_code, created_at=created_at,
    )


def evaluate_synthetic_shadow_recurrence(db_path: Path, *, limit: int, planted_canary_count: int) -> dict:
    """Exercise capture, persistence, lifecycle, privacy, and parity end to end."""
    db = SessionDB(db_path)
    try:
        for trace_id, session_id, canaries in (
            *((trace_id, session_id, trace_id == "recurrence-a")
              for trace_id, _event_id, session_id in _SYNTHETIC_MANUAL_USER_MISTAKE_ROOTS),
            ("recurrence-replacement", "recurrence-replacement-session", False),
            ("recurrence-operator", "recurrence-operator-session", False),
        ):
            _record_fixture_trace(
                db, trace_id=trace_id, successful=True, tools=("terminal",),
                plant_canaries=canaries, session_id=session_id,
            )
    finally:
        db.close()

    taxonomy = "duplicate_existing_capability"
    first = _append_synthetic_manual_user_mistake(
        db_path, trace_id="recurrence-a", event_id="recurrence-negative-a",
        taxonomy_code=taxonomy, created_at=1,
    )
    observed = evaluate_shadow_recurrence(db_path, limit=limit)
    _append_synthetic_manual_user_mistake(
        db_path, trace_id="recurrence-b", event_id="recurrence-negative-b",
        taxonomy_code=taxonomy, created_at=2,
    )
    candidate = evaluate_shadow_recurrence(db_path, limit=limit)
    for created_at, (trace_id, event_id, _session_id) in enumerate(
        _SYNTHETIC_MANUAL_USER_MISTAKE_ROOTS[2:], start=3,
    ):
        _append_synthetic_manual_user_mistake(
            db_path, trace_id=trace_id, event_id=event_id,
            taxonomy_code=taxonomy, created_at=created_at,
        )
    append_outcome_event(
        db_path, trace_id="recurrence-replacement", event_id="recurrence-correction",
        event_type="user_correction", source="user", polarity="positive",
        confidence=.99, taxonomy_code=taxonomy, supersedes_event_id=first.event_id, created_at=5,
    )

    classifier_inputs = (
        dict(trace_id="recurrence-replacement", tool_call_id="pytest-pass", tool_name="terminal",
             tool_arguments={"command": "pytest tests", "background": False}, tool_result={"exit_code": 0}),
        dict(trace_id="recurrence-replacement", tool_call_id="pytest-fail", tool_name="terminal",
             tool_arguments={"command": "python -m pytest tests", "background": False}, tool_result={"exit_code": 1}),
    )
    classified = [classify_foreground_pytest(**item) for item in classifier_inputs]
    authorization_findings = 0
    idempotency_findings = 0
    persisted = append_producer_outcome_event(db_path, classified[0])
    retried = append_producer_outcome_event(db_path, classified[0])
    if retried != persisted:
        idempotency_findings += 1
    try:
        append_producer_outcome_event(db_path, replace(classified[0], producer="foreground_pytest.v2"))
        authorization_findings += 1
    except ValueError:
        pass

    sequential = [classify_foreground_pytest(**item) for item in classifier_inputs]
    with ThreadPoolExecutor(max_workers=len(classifier_inputs)) as pool:
        concurrent = list(pool.map(lambda item: classify_foreground_pytest(**item), classifier_inputs))
    parity_findings = int(sequential != concurrent)

    eligible = evaluate_shadow_recurrence(db_path, limit=limit)
    append_outcome_event(
        db_path, trace_id="recurrence-replacement", event_id="recurrence-later-fail",
        event_type="verification_failed", source="verifier", polarity="negative",
        confidence=1, created_at=persisted.created_at + 1,
    )
    downgraded = evaluate_shadow_recurrence(db_path, limit=limit)
    append_outcome_event(
        db_path, trace_id="recurrence-replacement", event_id="recurrence-later-pass",
        event_type="verification_passed", source="verifier", polarity="positive",
        confidence=1, created_at=persisted.created_at + 2,
    )
    recovered = evaluate_shadow_recurrence(db_path, limit=limit)
    operator = append_outcome_event(
        db_path, trace_id="recurrence-operator", event_id="recurrence-operator-event",
        event_type="assumption_invalidated", source="operator", polarity="neutral",
        confidence=1, created_at=persisted.created_at + 3,
    )
    retract_outcome_event(
        db_path, source_event_id=operator.event_id, target_event_id="recurrence-negative-b",
        producer=OPERATOR_RELATION_PRODUCER, created_at=persisted.created_at + 4,
    )
    final = evaluate_shadow_recurrence(db_path, limit=limit)
    coverage, privacy_findings = _validate_recurrence_privacy(
        final, expected_canary_count=len(set(FIXTURE_PRIVACY_CANARIES)),
        planted_canary_count=planted_canary_count,
    )
    observed_item = _lesson_for(observed, taxonomy)
    candidate_item = _lesson_for(candidate, taxonomy)
    eligible_item = _lesson_for(eligible, taxonomy)
    downgraded_item = _lesson_for(downgraded, taxonomy)
    recovered_item = _lesson_for(recovered, taxonomy)
    final_item = _lesson_for(final, taxonomy)
    section = _count_only_recurrence(final)
    section.update({
        "classified_candidate_count": sum(item is not None for item in classified),
        "authorized_persist_count": 1,
        "idempotent_retry_count": 1,
        "authorization_finding_count": authorization_findings,
        "idempotency_finding_count": idempotency_findings,
        "parity_finding_count": parity_findings,
        "privacy_finding_count": privacy_findings,
        "effective_retraction_count": final["historical_event_count"] - final["effective_event_count"],
        "observed_transition_count": int(observed_item["state"] == "observed"),
        "candidate_transition_count": int(candidate_item["state"] == "candidate"),
        "activation_eligible_transition_count": int(eligible_item["activation_eligible"]),
        "verified_pair_count": eligible_item["valid_strict_corrected_pair_count"],
        "downgrade_transition_count": int(not downgraded_item["activation_eligible"]),
        "recovery_transition_count": int(recovered_item["activation_eligible"]),
        "privacy_canary_coverage_count": coverage,
        "expected_canary_coverage_count": len(set(FIXTURE_PRIVACY_CANARIES)),
        "canary_coverage_ok": coverage == planted_canary_count,
    })
    section["ok"] = all((
        section["authorization_finding_count"] == 0,
        section["idempotency_finding_count"] == 0,
        section["parity_finding_count"] == 0,
        section["privacy_finding_count"] == 0,
        section["canary_coverage_ok"],
        eligible_item["activation_eligible"],
        not downgraded_item["activation_eligible"],
        recovered_item["activation_eligible"],
        final_item["activation_eligible"],
        final["active_lesson_count"] == 0,
    ))
    if not section["ok"]:
        raise RuntimeError("synthetic shadow recurrence evaluation failed closed")
    return section


def run_eval(*, db_path: Path | None = None, limit: int = 100, ready_only: bool = False, min_reward: float = 1.0, max_negative_reward: float = 0.0, task_text: str | None = None, require_ready: bool = True, include_episodes: bool = False) -> dict:
    limit = _bounded_limit(limit)
    min_reward = _finite_float(min_reward, "min_reward")
    max_negative_reward = _finite_float(max_negative_reward, "max_negative_reward")
    if max_negative_reward >= min_reward:
        raise ValueError("max_negative_reward must be less than min_reward")
    contrastive_task_text = task_text if task_text is not None else DEFAULT_CONTRASTIVE_TASK_TEXT
    with tempfile.TemporaryDirectory(prefix="hermes-self-improve-eval-") as tmp:
        tmp_path = Path(tmp)
        fixture_db_path = tmp_path / "state.db"
        recurrence_db_path = tmp_path / "shadow-recurrence-state.db"
        planted_canary_count = make_trace(fixture_db_path)
        fixture_recurrence = evaluate_synthetic_shadow_recurrence(
            recurrence_db_path, limit=200, planted_canary_count=planted_canary_count,
        )
        trace_start = time.perf_counter()
        episodes = list(iter_episodes(db_path=fixture_db_path, limit=10))
        export = export_episodes_jsonl(tmp_path / "episodes.jsonl", db_path=fixture_db_path)
        fixture_replay = replay_eval_episodes(
            db_path=fixture_db_path,
            limit=10,
            min_reward=min_reward,
            require_ready=require_ready,
            include_episodes=include_episodes,
        )
        fixture_contrastive = contrastive_replay_eval(
            db_path=fixture_db_path,
            task_text=FIXTURE_CONTRASTIVE_TASK_TEXT,
            positive_limit=limit,
            negative_limit=limit,
            corrected_limit=limit,
            min_positive_reward=min_reward,
            max_negative_reward=max_negative_reward,
            forbidden_substrings=FIXTURE_PRIVACY_CANARIES,
            expected_canary_count=len(set(FIXTURE_PRIVACY_CANARIES)),
            planted_canary_count=planted_canary_count,
        )
        fixture_requirements_ok = (
            fixture_contrastive["negative_match_count"] >= 1
            and fixture_contrastive["corrected_match_count"] >= 1
            and fixture_contrastive["positive_match_count"] >= 1
            and fixture_contrastive["generated_hint_count"] >= 1
            and fixture_contrastive["unsupported_hint_count"] == 0
            and bool(fixture_contrastive["recurrence_counts"])
            and "capability_extension_ideation" in fixture_contrastive["task_frame"]["categories"]
            and fixture_contrastive["shadow_only"] is True
            and fixture_contrastive["prompt_modified"] is False
        )
        fixture_contrastive["activation_allowed"] = False
        fixture_contrastive["fixture_requirements_ok"] = fixture_requirements_ok
        fixture_contrastive["ok"] = (
            fixture_contrastive["ok"]
            and fixture_contrastive["privacy_eval_valid"]
            and fixture_requirements_ok
        )
        if not fixture_contrastive["ok"]:
            raise RuntimeError("synthetic contrastive shadow evaluation failed closed")
        trace_ms = int((time.perf_counter() - trace_start) * 1000)

        code_root = tmp_path / "code"
        make_code_fixture(code_root)
        code_start = time.perf_counter()
        summary = semantic_lookup(code_root, "summary")
        defs = go_to_definition(code_root, "format_name")
        refs = list_references(code_root, "format_name")
        diags = get_diagnostics(code_root)
        code_ms = int((time.perf_counter() - code_start) * 1000)

        result = {
            "trace_to_rl": {
                "episode_count": len(episodes),
                "ready_for_training": sum(1 for e in episodes if e.ready_for_training),
                "reward": episodes[0].reward if episodes else 0.0,
                "jsonl_bytes": (tmp_path / "episodes.jsonl").stat().st_size,
                "conversion_and_export_latency_ms": trace_ms,
                "export": export,
            },
            "fixture_replay_eval": fixture_replay,
            "fixture_contrastive_replay_eval": fixture_contrastive,
            "fixture_shadow_recurrence_eval": fixture_recurrence,
            "semantic_coding": {
                "file_count": summary["file_count"],
                "languages": summary["languages"],
                "symbol_count": summary["symbol_count"],
                "reference_count": summary["reference_count"],
                "diagnostic_count": diags["total"],
                "definition_hits": defs["total"],
                "reference_hits": refs["total"],
                "lookup_latency_ms": code_ms,
            },
        }
        if db_path is not None:
            result["training_episode_replay_eval"] = replay_eval_episodes(
                db_path=db_path,
                limit=limit,
                ready_only=ready_only,
                min_reward=min_reward,
                require_ready=require_ready,
                include_episodes=include_episodes,
            )
            result["contrastive_replay_eval"] = contrastive_replay_eval(
                db_path=db_path,
                task_text=contrastive_task_text,
                positive_limit=limit,
                negative_limit=limit,
                corrected_limit=limit,
                min_positive_reward=min_reward,
                max_negative_reward=max_negative_reward,
            )
            result["contrastive_replay_eval"]["shadow_only"] = True
            result["contrastive_replay_eval"]["activation_allowed"] = False
            result["contrastive_replay_eval"]["prompt_modified"] = False
            result["shadow_recurrence"] = _count_only_recurrence(
                evaluate_shadow_recurrence(db_path, limit=limit)
            )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--db-path", type=Path, help="Optional Hermes state DB to replay/eval in addition to the synthetic fixture.")
    parser.add_argument("--limit", type=_bounded_limit, default=100)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--min-reward", type=lambda value: _finite_float(value, "min_reward"), default=1.0)
    parser.add_argument("--max-negative-reward", type=lambda value: _finite_float(value, "max_negative_reward"), default=0.0)
    parser.add_argument("--task-text", help="Optional task text for operator-invoked contrastive shadow evaluation.")
    parser.add_argument("--no-require-ready", action="store_true")
    parser.add_argument("--include-episodes", action="store_true")
    args = parser.parse_args()
    result = run_eval(
        db_path=args.db_path,
        limit=args.limit,
        ready_only=args.ready_only,
        min_reward=args.min_reward,
        max_negative_reward=args.max_negative_reward,
        task_text=args.task_text,
        require_ready=not args.no_require_ready,
        include_episodes=args.include_episodes,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
