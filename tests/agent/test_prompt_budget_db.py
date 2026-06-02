import sqlite3

from hermes_state import SessionDB


def _index_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table_name})").fetchall()
    return {row[1] for row in rows}


def test_sessiondb_prompt_budgets_insert_and_indexes(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        budget_id = db.record_prompt_budget({
            "trace_id": "trace-3",
            "session_id": "session-3",
            "turn_id": "turn-3",
            "system_prompt_tokens": 10,
            "developer_prompt_tokens": 2,
            "tool_schema_tokens": 5,
            "memory_tokens": 3,
            "user_profile_tokens": 4,
            "conversation_history_tokens": 6,
            "context_file_tokens": 7,
            "tool_result_tokens": 8,
            "current_user_message_tokens": 9,
            "total_input_tokens": 45,
            "available_output_budget": 1000,
        })

        row = db._conn.execute(
            "SELECT * FROM prompt_budgets WHERE budget_id = ?", (budget_id,)
        ).fetchone()
        assert row is not None
        assert row["trace_id"] == "trace-3"
        assert row["session_id"] == "session-3"
        assert row["turn_id"] == "turn-3"
        assert row["system_prompt_tokens"] == 10
        assert row["developer_prompt_tokens"] == 2
        assert row["tool_schema_tokens"] == 5
        assert row["memory_tokens"] == 3
        assert row["user_profile_tokens"] == 4
        assert row["conversation_history_tokens"] == 6
        assert row["context_file_tokens"] == 7
        assert row["tool_result_tokens"] == 8
        assert row["current_user_message_tokens"] == 9
        assert row["total_input_tokens"] == 45
        assert row["available_output_budget"] == 1000
        assert _index_names(db._conn, "prompt_budgets") >= {
            "idx_prompt_budgets_session_time",
            "idx_prompt_budgets_trace",
        }
    finally:
        db.close()
