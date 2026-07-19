#!/usr/bin/env python3
"""Small offline eval harness for Hermes self-improvement substrates.

Measures:
- fixture execution trace -> TrainingEpisode conversion/export latency + reward
- fixture codebase -> semantic symbols/definitions/references/diagnostics latency

Writes no generated artifacts unless --output-json is supplied.
"""
from __future__ import annotations

import argparse
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
from agent.outcome_events import append_outcome_event
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
    plant_canaries: bool = False,
) -> None:
    recorder = TraceRecorder(
        session_id="eval-session", turn_id=f"eval-{trace_id}",
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
        planted_canary_count = make_trace(fixture_db_path)
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
        fixture_contrastive["fixture_requirements_ok"] = fixture_requirements_ok
        fixture_contrastive["ok"] = (
            fixture_contrastive["ok"]
            and fixture_contrastive["privacy_eval_valid"]
            and fixture_requirements_ok
        )
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
