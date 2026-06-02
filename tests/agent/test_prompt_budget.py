import builtins

import pytest

from agent import prompt_budget
from agent.prompt_budget import build_prompt_budget_record, estimate_token_count


@pytest.fixture(autouse=True)
def clear_encoder_cache():
    cache_clear = getattr(prompt_budget._get_token_encoder, "cache_clear", None)
    if cache_clear:
        cache_clear()
    yield
    cache_clear = getattr(prompt_budget._get_token_encoder, "cache_clear", None)
    if cache_clear:
        cache_clear()


def test_estimate_token_count_caches_missing_tiktoken_import(monkeypatch):
    original_import = builtins.__import__
    attempted_tiktoken_imports = 0

    def counting_import(name, *args, **kwargs):
        nonlocal attempted_tiktoken_imports
        if name == "tiktoken":
            attempted_tiktoken_imports += 1
            raise ModuleNotFoundError("No module named 'tiktoken'")
        return original_import(name, *args, **kwargs)

    prompt_budget._get_token_encoder.cache_clear()
    monkeypatch.setattr(builtins, "__import__", counting_import)

    assert [estimate_token_count("abcdefgh") for _ in range(5)] == [2, 2, 2, 2, 2]
    assert attempted_tiktoken_imports == 1


def test_estimate_token_count_fallback_still_matches_existing_behavior(monkeypatch):
    monkeypatch.setattr(prompt_budget, "_get_token_encoder", lambda: None)

    assert estimate_token_count(None) == 0
    assert estimate_token_count("") == 0
    assert estimate_token_count("abcd") == 1
    assert estimate_token_count("abcde") == 2
    assert estimate_token_count([1, 2]) == 2  # json text is "[1, 2]"


def test_estimate_token_count_uses_cached_encoder_when_available(monkeypatch):
    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return text.split()

    encoder = FakeEncoder()
    monkeypatch.setattr(prompt_budget, "_get_token_encoder", lambda: encoder)

    assert estimate_token_count("one two three") == 3
    assert encoder.calls == ["one two three"]


def test_build_prompt_budget_record_bucket_total_is_consistent(monkeypatch):
    monkeypatch.setattr(prompt_budget, "_get_token_encoder", lambda: None)

    record = build_prompt_budget_record(
        trace_id="trace-1",
        session_id="session-1",
        turn_id="turn-1",
        api_messages=[
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "tool", "content": "tool output"},
            {"role": "user", "content": "current question"},
        ],
        tools=[{"type": "function", "function": {"name": "tool"}}],
        system_components={
            "system_prompt": "system text memory text profile text context text",
            "memory": "memory text",
            "user_profile": "profile text",
            "context_files": "context text",
        },
        current_user_message_index=4,
        memory_injection="retrieved memory",
        available_output_budget=123,
    )

    bucket_names = [
        "system_prompt_tokens",
        "developer_prompt_tokens",
        "tool_schema_tokens",
        "memory_tokens",
        "user_profile_tokens",
        "conversation_history_tokens",
        "context_file_tokens",
        "tool_result_tokens",
        "current_user_message_tokens",
    ]

    assert record["trace_id"] == "trace-1"
    assert record["available_output_budget"] == 123
    assert record["total_input_tokens"] == sum(int(record[name]) for name in bucket_names)
    assert record["memory_tokens"] >= estimate_token_count("memory text")
    assert record["tool_schema_tokens"] > 0
