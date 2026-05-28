"""Persisted session summary helpers."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

SUMMARY_PROVIDER = None  # None = auto (resolves via auxiliary chain: OpenRouter → Nous → …)
SUMMARY_MODEL = None  # Use provider default

_SYSTEM_PROMPT = (
    "Write a short factual resume of this Hermes session in 2-4 concise sentences. "
    "Cover what the user needed, what Hermes did, important files/tools/URLs/commands, "
    "and any unresolved outcome. Return plain text only."
)

_TITLE_PROMPT = (
    "Write a short factual title for this Hermes session in 3-7 words. "
    "Use the same core topic and outcome reflected in the session transcript. "
    "Return plain text only. No quotes, no bullet, no prefix, and no punctuation at the end."
)


def _format_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown").upper()
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content.strip()}")
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool in tool_calls:
                if not isinstance(tool, dict):
                    continue
                fn = (
                    tool.get("function")
                    if isinstance(tool.get("function"), dict)
                    else tool
                )
                name = str(fn.get("name") or tool.get("name") or "tool").strip()
                args = fn.get("arguments") or tool.get("arguments") or ""
                if name:
                    parts.append(f"ASSISTANT_TOOL: {name} {args}".strip())
        if (
            role == "TOOL"
            and msg.get("tool_name")
            and isinstance(content, str)
            and content.strip()
        ):
            parts.append(f"TOOL_RESULT {msg.get('tool_name')}: {content.strip()}")
    transcript = "\n\n".join(parts).strip()
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "\n\n...[truncated]"
    return transcript


def _generate_session_text(
    messages: List[Dict[str, Any]],
    session_meta: Optional[Dict[str, Any]],
    *,
    system_prompt: str,
    max_tokens: int,
    max_length: int,
) -> Optional[str]:
    session_meta = session_meta or {}
    transcript = _format_messages(messages)
    if not transcript:
        return None
    user_prompt = (
        f"Session title: {session_meta.get('title') or '(untitled)'}\n"
        f"Source: {session_meta.get('source') or 'unknown'}\n"
        f"Model: {session_meta.get('model') or 'unknown'}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    try:
        response = call_llm(
            task="summary",
            provider=SUMMARY_PROVIDER,
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=45.0,
        )
        text = " ".join(extract_content_or_reasoning(response).split()).strip()
        if not text:
            return None
        text = text.strip("\"'")
        if text.lower().startswith("title:"):
            text = text[6:].strip()
        if len(text) > max_length:
            text = text[: max_length - 1].rstrip() + "..."
        return text or None
    except Exception as exc:
        logger.warning("Session text generation failed: %s", exc)
        return None


def generate_session_summary(
    messages: List[Dict[str, Any]], session_meta: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    return _generate_session_text(
        messages,
        session_meta,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=512,
        max_length=280,
    )


def generate_session_title(
    messages: List[Dict[str, Any]], session_meta: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    return _generate_session_text(
        messages,
        session_meta,
        system_prompt=_TITLE_PROMPT,
        max_tokens=256,
        max_length=80,
    )


def refresh_session_summary(session_db, session_id: str) -> Optional[str]:
    if not session_db or not session_id:
        return None
    try:
        messages = session_db.get_messages_as_conversation(session_id)
        if not messages:
            return None
        meta = session_db.get_session(session_id) or {}
        summary = generate_session_summary(messages, meta)
        if not summary:
            return None
        session_db.set_session_summary(session_id, summary)
        return summary
    except Exception as exc:
        logger.warning("Failed to refresh session summary for %s: %s", session_id, exc)
        return None


def ensure_session_metadata(session_db, session_id: str) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {"title": None, "summary": None}
    if not session_db or not session_id:
        return result
    try:
        meta = session_db.get_session(session_id) or {}
        messages = session_db.get_messages_as_conversation(session_id)
        if not messages:
            return result

        title = str(meta.get("title") or "").strip()
        if not title:
            generated_title = generate_session_title(messages, meta)
            if generated_title:
                try:
                    if session_db.set_session_title(session_id, generated_title):
                        result["title"] = generated_title
                        meta["title"] = generated_title
                except ValueError:
                    # Duplicate title — skip but continue to summary
                    logger.debug(
                        "Duplicate title '%s' for session %s — skipping",
                        generated_title, session_id,
                    )

        summary = str(meta.get("summary") or "").strip()
        if not summary:
            generated_summary = generate_session_summary(messages, meta)
            if generated_summary:
                try:
                    if session_db.set_session_summary(session_id, generated_summary):
                        result["summary"] = generated_summary
                except Exception as exc:
                    logger.warning(
                        "Failed to set summary for %s: %s", session_id, exc
                    )
    except Exception as exc:
        logger.warning("Failed to ensure session metadata for %s: %s", session_id, exc)
    return result


def maybe_auto_summarize(
    session_db, session_id: str, conversation_history: List[Dict[str, Any]]
) -> None:
    if not session_db or not session_id:
        return
    try:
        existing = session_db.get_session(session_id) or {}
        if (
            str(existing.get("summary") or "").strip()
            and str(existing.get("title") or "").strip()
        ):
            return
    except Exception:
        return
    if not any(msg.get("role") == "assistant" for msg in (conversation_history or [])):
        return
    thread = threading.Thread(
        target=ensure_session_metadata,
        args=(session_db, session_id),
        daemon=True,
        name="auto-session-metadata",
    )
    thread.start()


def backfill_session_metadata(
    session_db, limit: int = 50, force: bool = False
) -> Dict[str, int]:
    rows = session_db.list_sessions_missing_summary(limit=limit, include_existing=force)
    processed = 0
    updated = 0
    failed = 0
    for row in rows:
        processed += 1
        try:
            result = ensure_session_metadata(session_db, row["id"])
            if result.get("title") or result.get("summary"):
                updated += 1
        except Exception:
            failed += 1
    return {"processed": processed, "updated": updated, "failed": failed}


def backfill_session_summaries(
    session_db, limit: int = 50, force: bool = False
) -> Dict[str, int]:
    return backfill_session_metadata(session_db, limit=limit, force=force)
