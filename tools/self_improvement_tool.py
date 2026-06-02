"""Hermes self-improvement tools: trace episodes and semantic coding primitives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agent.semantic_code_index import semantic_lookup
from agent.training_episodes import default_episode_export_path, episode_summary, export_episodes_jsonl, iter_episodes, replay_eval_episodes
from hermes_state import DEFAULT_DB_PATH
from tools.registry import registry, tool_error


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


TRAINING_EPISODES_SCHEMA = {
    "name": "training_episodes",
    "description": "Convert Hermes execution traces into privacy-minimized TrainingEpisode projections, summarize rewards, or export JSONL for eval/training.",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["summary", "export", "preview", "replay_eval"], "description": "Operation to run."},
            "db_path": {"type": "string", "description": "Optional Hermes state.db path. Defaults to active HERMES_HOME state.db."},
            "output_path": {"type": "string", "description": "JSONL export path. Defaults to ~/.hermes/exports/training_episodes.jsonl."},
            "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 5000},
            "ready_only": {"type": "boolean", "default": False},
            "min_reward": {"type": "number", "default": 1.0, "description": "Minimum acceptable reward for replay_eval."},
            "require_ready": {"type": "boolean", "default": True, "description": "Fail replay_eval rows that are not ready_for_training."},
            "include_episodes": {"type": "boolean", "default": False, "description": "Include compact per-episode replay/eval rows."},
        },
        "required": ["operation"],
    },
}


def training_episodes(operation: str, db_path: str = "", output_path: str = "", limit: int = 100, ready_only: bool = False, min_reward: float = 1.0, require_ready: bool = True, include_episodes: bool = False) -> str:
    try:
        db = Path(db_path).expanduser() if db_path else DEFAULT_DB_PATH
        if operation == "summary":
            return _json({"success": True, "data": episode_summary(db_path=db, limit=limit)})
        if operation == "export":
            out = Path(output_path).expanduser() if output_path else default_episode_export_path()
            return _json({"success": True, "data": export_episodes_jsonl(output_path=out, db_path=db, limit=limit, ready_only=ready_only)})
        if operation == "preview":
            episodes = [e.to_dict() for e in iter_episodes(db_path=db, limit=min(limit, 10), ready_only=ready_only)]
            return _json({"success": True, "data": {"episodes": episodes, "count": len(episodes)}})
        if operation == "replay_eval":
            return _json({
                "success": True,
                "data": replay_eval_episodes(
                    db_path=db,
                    limit=limit,
                    ready_only=ready_only,
                    min_reward=min_reward,
                    require_ready=require_ready,
                    include_episodes=include_episodes,
                ),
            })
        return tool_error(f"unsupported operation: {operation}")
    except Exception as exc:
        return tool_error(str(exc))


SEMANTIC_CODE_SCHEMA = {
    "name": "semantic_code",
    "description": "IDE-grade semantic coding primitive: index project symbols, find symbols, go to definitions, list references, and parser diagnostics for Python/JS/TS.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_path": {"type": "string", "description": "Project root to index."},
            "operation": {"type": "string", "enum": ["summary", "index", "find_symbol", "definition", "references", "diagnostics"]},
            "query": {"type": "string", "description": "Symbol name or substring depending on operation."},
            "kind": {"type": "string", "description": "Optional symbol kind filter: class/function/variable/interface/type."},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
            "include_context": {"type": "boolean", "default": False, "description": "Opt in to include raw source-line context in reference results. Defaults false for privacy/cache safety."},
        },
        "required": ["project_path", "operation"],
    },
}


def semantic_code(project_path: str, operation: str, query: str = "", kind: str = "", limit: int = 50, include_context: bool = False) -> str:
    try:
        result = semantic_lookup(project_path=project_path, operation=operation, query=query or "", kind=(kind or None), limit=limit, include_context=include_context)
        return _json({"success": True, "data": result})
    except Exception as exc:
        return tool_error(str(exc))


registry.register(
    name="training_episodes",
    emoji="🧬",
    toolset="self_improvement",
    schema=TRAINING_EPISODES_SCHEMA,
    handler=lambda args, **kw: training_episodes(
        operation=args.get("operation", "summary"),
        db_path=args.get("db_path", ""),
        output_path=args.get("output_path", ""),
        limit=int(args.get("limit") or 100),
        ready_only=bool(args.get("ready_only") or False),
        min_reward=float(args.get("min_reward") if args.get("min_reward") is not None else 1.0),
        require_ready=bool(args.get("require_ready") if args.get("require_ready") is not None else True),
        include_episodes=bool(args.get("include_episodes") or False),
    ),
    description="Trace-to-RL TrainingEpisode projection/export tool",
)

registry.register(
    name="semantic_code",
    emoji="🧭",
    toolset="coding",
    schema=SEMANTIC_CODE_SCHEMA,
    handler=lambda args, **kw: semantic_code(
        project_path=args.get("project_path", ""),
        operation=args.get("operation", "summary"),
        query=args.get("query", ""),
        kind=args.get("kind", ""),
        limit=int(args.get("limit") or 50),
        include_context=bool(args.get("include_context") or False),
    ),
    description="Semantic code index and coding primitives",
)
