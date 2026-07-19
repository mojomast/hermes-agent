from dataclasses import FrozenInstanceError

import pytest

from agent.shadow_outcome_capture import (
    FOREGROUND_PYTEST_PRODUCER,
    ShadowOutcomeCandidate,
    classify_foreground_pytest,
)


def _classify(command="pytest tests/agent/test_example.py -q", exit_code=0, **overrides):
    values = {
        "trace_id": "trace-123",
        "tool_call_id": "call-456",
        "tool_name": "terminal",
        "tool_arguments": {"command": command, "background": False},
        "tool_result": {"exit_code": exit_code, "output": "raw secret output"},
    }
    values.update(overrides)
    return classify_foreground_pytest(**values)


def test_classifies_only_direct_foreground_pytest_pass_and_fail():
    passed = _classify()
    failed = _classify(command="python -m pytest tests/agent -x", exit_code=1)

    assert isinstance(passed, ShadowOutcomeCandidate)
    assert passed.event_type == "verification_passed"
    assert failed.event_type == "verification_failed"
    assert passed.producer == FOREGROUND_PYTEST_PRODUCER
    assert passed.trace_id == "trace-123"
    assert passed.tool_call_id == "call-456"


def test_omitted_background_is_foreground_but_explicit_true_is_rejected():
    assert _classify(tool_arguments={"command": "pytest tests"}) is not None
    assert _classify(tool_arguments={"command": "pytest tests", "background": True}) is None


@pytest.mark.parametrize(
    "command",
    [
        "pytest tests",
        "py.test tests",
        "/usr/local/bin/pytest tests",
        "./.venv/bin/py.test tests",
        "python -m pytest tests",
        "python3 -m pytest tests",
        "python3.12 -m pytest tests",
        "/usr/bin/python3.11 -m pytest tests",
        ".venv/bin/python -m pytest tests",
    ],
)
def test_accepts_supported_pytest_and_python_executable_path_forms(command):
    assert _classify(command=command) is not None


@pytest.mark.parametrize(
    "mode",
    [
        "--help", "-h", "--version", "--collect-only", "--co",
        "--fixtures", "--funcargs", "--markers", "--setup-plan",
        "--setup-only", "--setup-show", "--trace-config",
    ],
)
def test_rejects_pytest_non_execution_modes_even_when_exit_zero(mode):
    assert _classify(command=f"pytest tests {mode}", exit_code=0) is None
    assert _classify(command=f"python3 -m pytest {mode} tests", exit_code=0) is None


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5, 6, -1, None, "0", True, False])
def test_non_test_outcomes_and_malformed_exit_codes_produce_no_candidate(exit_code):
    assert _classify(exit_code=exit_code) is None


@pytest.mark.parametrize(
    "command",
    [
        "pytest tests && echo done",
        "pytest tests; echo done",
        "pytest tests | tee result.txt",
        "pytest tests > result.txt",
        "pytest tests < input.txt",
        "pytest tests &",
        "pytest $(touch leaked)",
        "pytest `touch leaked`",
        "PYTHONPATH=. pytest tests",
        "bash -c 'pytest tests'",
        "python pytest tests",
        "python -m pytestx tests",
        "python -c 'import pytest'",
        "pytest 'unterminated",
        "",
    ],
)
def test_rejects_shell_operators_wrappers_and_malformed_commands(command):
    assert _classify(command=command) is None


def test_rejects_background_non_terminal_and_malformed_shapes():
    assert _classify(tool_arguments={"command": "pytest", "background": True}) is None
    assert _classify(tool_arguments={"command": "pytest", "background": 1}) is None
    assert _classify(tool_arguments={"command": "pytest", "background": 0}) is None
    assert _classify(tool_name="process") is None
    assert _classify(tool_arguments="pytest") is None
    assert _classify(tool_result="exit 0") is None


def test_candidate_is_frozen_privacy_safe_and_id_is_structural_only():
    first = _classify(tool_result={"exit_code": 0, "output": "secret A"})
    second = _classify(
        tool_arguments={"command": "pytest totally/different/path.py", "background": False},
        tool_result={"exit_code": 1, "output": "secret B"},
    )

    # A contradictory replay of one structural tool call intentionally collides so
    # persistence can fail closed rather than recording mutually inconsistent facts.
    assert first.event_id == second.event_id
    assert set(first.__dataclass_fields__) == {
        "event_id", "trace_id", "tool_call_id", "producer", "event_type"
    }
    assert "secret" not in repr(first)
    assert "totally/different/path.py" not in repr(first)
    with pytest.raises(FrozenInstanceError):
        first.event_type = "verification_failed"


def test_invalid_structural_identifiers_fail_closed():
    assert _classify(trace_id="") is None
    assert _classify(tool_call_id="") is None
    assert _classify(trace_id="trace with raw text") is None
