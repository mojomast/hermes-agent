from unittest.mock import MagicMock, patch

from agent.session_summarizer import ensure_session_metadata, maybe_auto_summarize


def test_ensure_session_metadata_fills_missing_title_and_summary():
    db = MagicMock()
    db.get_session.return_value = {
        "id": "sess-1",
        "title": None,
        "summary": None,
        "source": "cli",
        "model": "glm-5",
    }
    db.get_messages_as_conversation.return_value = [
        {"role": "user", "content": "Help me fix the dashboard graph labels"},
        {
            "role": "assistant",
            "content": "I will inspect the graph renderer and session data",
        },
    ]
    db.set_session_title.return_value = True
    db.set_session_summary.return_value = True

    with (
        patch(
            "agent.session_summarizer.generate_session_title",
            return_value="Fix Dashboard Graph Labels",
        ),
        patch(
            "agent.session_summarizer.generate_session_summary",
            return_value="The session focused on restoring graph session labels and inspecting renderer fallbacks.",
        ),
    ):
        result = ensure_session_metadata(db, "sess-1")

    assert result == {
        "title": "Fix Dashboard Graph Labels",
        "summary": "The session focused on restoring graph session labels and inspecting renderer fallbacks.",
    }
    db.set_session_title.assert_called_once_with("sess-1", "Fix Dashboard Graph Labels")
    db.set_session_summary.assert_called_once()


def test_ensure_session_metadata_skips_existing_fields():
    db = MagicMock()
    db.get_session.return_value = {
        "id": "sess-1",
        "title": "Keep Existing",
        "summary": "Existing summary",
    }
    db.get_messages_as_conversation.return_value = [
        {"role": "user", "content": "hello"}
    ]

    result = ensure_session_metadata(db, "sess-1")

    assert result == {"title": None, "summary": None}
    db.set_session_title.assert_not_called()
    db.set_session_summary.assert_not_called()


def test_maybe_auto_summarize_runs_when_assistant_message_exists():
    db = MagicMock()
    db.get_session.return_value = {"id": "sess-1", "title": None, "summary": None}
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "tool", "content": "tool output"},
    ]

    with patch("agent.session_summarizer.ensure_session_metadata") as ensure_meta:
        maybe_auto_summarize(db, "sess-1", history)
        import time

        time.sleep(0.3)
        ensure_meta.assert_called_once_with(db, "sess-1")
