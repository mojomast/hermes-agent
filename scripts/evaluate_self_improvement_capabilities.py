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
import tempfile
import time
from pathlib import Path

from agent.semantic_code_index import get_diagnostics, go_to_definition, list_references, semantic_lookup
from agent.tracing import TraceRecorder, hash_user_message
from agent.training_episodes import export_episodes_jsonl, iter_episodes, replay_eval_episodes
from hermes_state import SessionDB


def make_trace(db_path: Path) -> None:
    db = SessionDB(db_path)
    try:
        recorder = TraceRecorder(session_id="eval-session", turn_id="eval-turn", user_message_hash=hash_user_message("eval task"))
        with recorder.span("turn", "root"):
            with recorder.span("model_call", "llm_api_call"):
                pass
            with recorder.span("tool_call", "execute_code", {"arg_count": 1}):
                pass
            with recorder.span("final_answer", "turn.final_answer", {"completed": True, "response_len": 64}):
                pass
        recorder.finish("completed")
        db.record_trace(recorder.to_trace_row(), recorder.to_span_rows())
    finally:
        db.close()


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


def run_eval(*, db_path: Path | None = None, limit: int = 100, ready_only: bool = False, min_reward: float = 1.0, require_ready: bool = True, include_episodes: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix="hermes-self-improve-eval-") as tmp:
        tmp_path = Path(tmp)
        fixture_db_path = tmp_path / "state.db"
        make_trace(fixture_db_path)
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
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--db-path", type=Path, help="Optional Hermes state DB to replay/eval in addition to the synthetic fixture.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--no-require-ready", action="store_true")
    parser.add_argument("--include-episodes", action="store_true")
    args = parser.parse_args()
    result = run_eval(
        db_path=args.db_path,
        limit=args.limit,
        ready_only=args.ready_only,
        min_reward=args.min_reward,
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
