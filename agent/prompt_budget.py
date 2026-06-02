"""Prompt budget accounting helpers.

This module records token *counts only* for prompt components. It intentionally
never persists prompt text, tool schemas, tool arguments/results, API keys, or
other secret-bearing content.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_token_encoder() -> Any | None:
    """Return a cached tiktoken encoder when available.

    Prompt budgeting calls this helper many times per turn. Caching both the
    encoder and the import-miss path avoids repeated import/lookup overhead in
    environments where ``tiktoken`` is unavailable while preserving the existing
    chars/4 fallback behavior.
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-4")
        except Exception:
            return None


def estimate_token_count(value: Any) -> int:
    """Estimate tokens using tiktoken when available, falling back to chars/4."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(value)
    if not text:
        return 0

    enc = _get_token_encoder()
    if enc is not None:
        try:
            return int(len(enc.encode(text)))
        except Exception:
            pass
    return max(1, int((len(text) + 3) // 4))


def _sum_message_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for msg in messages or []:
        total += estimate_token_count(msg)
    return total


def build_prompt_budget_record(
    *,
    trace_id: Optional[str],
    session_id: Optional[str],
    turn_id: Optional[str],
    api_messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    system_components: Optional[Dict[str, str]] = None,
    current_user_message_index: Optional[int] = None,
    memory_injection: str = "",
    plugin_user_context: str = "",
    available_output_budget: Optional[int] = None,
) -> Dict[str, int | str | None]:
    """Return a content-free prompt budget row for ``prompt_budgets``.

    ``current_user_message_index`` is the index in ``api_messages`` after any
    system/prefill insertion and sanitizer pass. If unknown, the last user
    message is treated as the current user message.
    """
    components = system_components or {}
    total_system_text_tokens = estimate_token_count(components.get("system_prompt", ""))
    memory_tokens = estimate_token_count(components.get("memory", "")) + estimate_token_count(memory_injection)
    user_profile_tokens = estimate_token_count(components.get("user_profile", ""))
    context_file_tokens = estimate_token_count(components.get("context_files", ""))
    # If system_prompt is the full assembled prompt, subtract known sub-buckets
    # so totals do not double count memory/profile/context file content.
    system_prompt_tokens = max(
        0,
        total_system_text_tokens
        - estimate_token_count(components.get("memory", ""))
        - user_profile_tokens
        - context_file_tokens,
    )

    tool_schema_tokens = estimate_token_count(tools or [])

    developer_prompt_tokens = 0
    conversation_history_tokens = 0
    tool_result_tokens = 0
    current_user_message_tokens = 0

    if current_user_message_index is None or not (0 <= current_user_message_index < len(api_messages)):
        for idx in range(len(api_messages) - 1, -1, -1):
            if api_messages[idx].get("role") == "user":
                current_user_message_index = idx
                break

    for idx, msg in enumerate(api_messages or []):
        role = msg.get("role")
        tok = estimate_token_count(msg)
        if role == "system":
            # System prompt is broken down from components above. If the prompt
            # came from a stored session and no components are available, count
            # the whole system message as system_prompt_tokens.
            if not components:
                system_prompt_tokens += tok
            continue
        if role == "developer":
            developer_prompt_tokens += tok
            continue
        if idx == current_user_message_index and role == "user":
            # Current user message may include ephemeral memory context appended
            # to the API copy; that context is counted in memory_tokens above.
            tok = max(0, tok - estimate_token_count(memory_injection))
            current_user_message_tokens += tok
            # Plugin context has no dedicated DB column; subtracting it from the
            # current user message would hide input tokens, so leave it included.
            _ = plugin_user_context
            continue
        if role == "tool":
            tool_result_tokens += tok
            continue
        conversation_history_tokens += tok

    total_input_tokens = (
        system_prompt_tokens
        + developer_prompt_tokens
        + tool_schema_tokens
        + memory_tokens
        + user_profile_tokens
        + conversation_history_tokens
        + context_file_tokens
        + tool_result_tokens
        + current_user_message_tokens
    )

    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "system_prompt_tokens": int(system_prompt_tokens),
        "developer_prompt_tokens": int(developer_prompt_tokens),
        "tool_schema_tokens": int(tool_schema_tokens),
        "memory_tokens": int(memory_tokens),
        "user_profile_tokens": int(user_profile_tokens),
        "conversation_history_tokens": int(conversation_history_tokens),
        "context_file_tokens": int(context_file_tokens),
        "tool_result_tokens": int(tool_result_tokens),
        "current_user_message_tokens": int(current_user_message_tokens),
        "total_input_tokens": int(total_input_tokens),
        "available_output_budget": None if available_output_budget is None else int(available_output_budget),
    }


def emit_prompt_budget_warnings(record: Mapping[str, Any]) -> None:
    """Log threshold warnings using counts only (no prompt content)."""
    total = int(record.get("total_input_tokens") or 0)
    if total <= 0:
        return

    def pct(name: str) -> float:
        return float(record.get(name) or 0) / float(total)

    warnings: list[str] = []
    # Heuristic for trivial no-tool requests: short current message, no tools,
    # little prior history/tool output, yet a large input prompt.
    if (
        total > 5000
        and int(record.get("tool_schema_tokens") or 0) == 0
        and int(record.get("current_user_message_tokens") or 0) <= 80
        and int(record.get("tool_result_tokens") or 0) == 0
    ):
        warnings.append("total_input_tokens_gt_5000_for_trivial_no_tool_request")
    if pct("memory_tokens") > 0.20:
        warnings.append("memory_tokens_gt_20_percent")
    if pct("tool_schema_tokens") > 0.25:
        warnings.append("tool_schema_tokens_gt_25_percent")
    if pct("conversation_history_tokens") > 0.50:
        warnings.append("conversation_history_tokens_gt_50_percent")

    if warnings:
        logger.warning(
            "Prompt budget threshold warning session=%s turn=%s total=%d warnings=%s "
            "memory=%d tool_schema=%d history=%d",
            record.get("session_id") or "",
            record.get("turn_id") or "",
            total,
            ",".join(warnings),
            int(record.get("memory_tokens") or 0),
            int(record.get("tool_schema_tokens") or 0),
            int(record.get("conversation_history_tokens") or 0),
        )
