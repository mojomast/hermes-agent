import inspect
import json

import pytest

import agent.contrastive_episode_retrieval as contrastive_module
from agent.contrastive_episode_retrieval import (
    contrastive_replay_eval,
    retrieve_contrastive_episodes,
    task_frame_from_text,
)
from agent.outcome_events import append_outcome_event, outcome_summary
from agent.training_episodes import iter_episodes
from agent.tracing import TraceRecorder
from hermes_state import SessionDB


CANARIES = {
    "prompt": "CANARY_PROMPT_VELVET", "user_message": "CANARY_USER_MANGO",
    "tool_args": "CANARY_ARGS_OTTER", "result": "CANARY_RESULT_FOX",
    "stdout": "CANARY_STDOUT_BEAR", "stderr": "CANARY_STDERR_WOLF",
    "response": "CANARY_RESPONSE_LYNX", "transcript": "CANARY_TRANSCRIPT_HARE",
    "modelOutput": "CANARY_MODEL_RAVEN",
}


def _record(path, trace_id, *, successful, tools=("terminal",), start=1.0):
    db = SessionDB(path)
    recorder = TraceRecorder(session_id="private-session", turn_id="private-turn", user_message_hash="private-hash", trace_id=trace_id)
    with recorder.span("turn", "root"):
        for tool in tools:
            with recorder.span("tool_call", tool, {**CANARIES, "nested": {"payload": "CANARY_NESTED_ELK"}}) as span:
                if not successful:
                    span.status = "error"
                    span.error_class = "Failure"
        if successful:
            with recorder.span("final_answer", "turn.final_answer", {"completed": True, **CANARIES}):
                pass
    recorder.finish("completed" if successful else "failed")
    row = recorder.to_trace_row()
    row["start_time"] = start
    db.record_trace(row, recorder.to_span_rows())
    db.close()


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key.lower().replace("_", "")
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_required_corrected_pair_generates_shadow_duplicate_inventory_hint(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    _record(path, "replacement", successful=True, tools=("search_files", "read_file"), start=2)
    mistake = append_outcome_event(
        path, trace_id="mistake", event_type="duplicate_proposal", source="user",
        polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability",
    )
    append_outcome_event(
        path, trace_id="replacement", event_type="user_correction", source="user",
        polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability",
        supersedes_event_id=mistake.event_id,
    )
    append_outcome_event(
        path, trace_id="replacement", event_type="verification_passed", source="verifier",
        polarity="positive", confidence=1.0,
    )

    packet = retrieve_contrastive_episodes(path, "Ideate a new capability subsystem extension")

    assert packet["schema_version"] == "contrastive_episode_retrieval.v1"
    assert "capability_extension_ideation" in packet["task_frame"]["categories"]
    assert len(packet["negative_matches"]) == 1
    assert len(packet["corrected_matches"]) == 1
    assert len(packet["positive_matches"]) == 1
    assert packet["positive_matches"][0]["episode_id"] == packet["corrected_matches"][0]["episode_id"]
    assert "replacement" not in packet["corrected_matches"][0]["episode_id"]
    hint = packet["contrastive_hints"][0]
    assert hint["taxonomy_code"] == "duplicate_existing_capability"
    assert "existing capability inventory" in hint["text"]
    assert all(word in hint["text"] for word in ("continuity", "skill", "source", "project"))
    assert hint["shadow_only"] is True
    assert hint["activation"] is False


def test_single_weak_or_unverified_correction_does_not_generate_supported_hint(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "weak", successful=False)
    append_outcome_event(
        path, trace_id="weak", event_type="user_correction", source="evaluator",
        polarity="negative", confidence=.4, taxonomy_code="duplicate_existing_capability",
    )
    packet = retrieve_contrastive_episodes(path, "propose a capability extension")
    assert packet["contrastive_hints"] == []


def test_multiple_independent_high_confidence_user_corrections_can_be_shadow_candidate(tmp_path):
    path = tmp_path / "state.db"
    for index in range(2):
        trace = f"bad-{index}"
        _record(path, trace, successful=False, start=index + 1)
        append_outcome_event(
            path, trace_id=trace, event_type="duplicate_proposal", source="user",
            polarity="negative", confidence=.95, taxonomy_code="duplicate_existing_capability",
        )
    packet = retrieve_contrastive_episodes(path, "new capability subsystem idea")
    assert packet["contrastive_hints"]
    assert packet["contrastive_hints"][0]["shadow_only"] is True
    assert packet["contrastive_hints"][0]["activation"] is False


def test_packet_and_episode_projection_never_return_raw_payloads_or_identifiers(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True)
    append_outcome_event(path, trace_id="safe", event_type="verification_passed", source="verifier", polarity="positive", confidence=1)
    episode = next(iter_episodes(path, min_reward=None, ready_only=False))
    packet = retrieve_contrastive_episodes(path, "debug and verify deployment")
    blob = json.dumps(packet)

    assert episode.outcome_event_count == 1
    assert episode.outcome_event_type_counts == {"verification_passed": 1}
    assert episode.outcome_taxonomy_counts == {}
    forbidden_keys = {"prompt", "usermessage", "toolargs", "result", "stdout", "stderr", "response", "transcript", "modeloutput", "metadata", "steps", "sessionid", "turnid", "traceid"}
    assert forbidden_keys.isdisjoint(set(_all_keys(packet)))
    assert "private-session" not in blob and "private-turn" not in blob and "private-hash" not in blob
    for value in [*CANARIES.values(), "CANARY_NESTED_ELK"]:
        assert value not in blob


def test_task_frame_classifies_mistake_learning_capability_audit():
    packet = task_frame_from_text(
        "audit whether Hermes learns from mistakes and uses prior failed episodes"
    )
    assert "capability_audit" in packet["categories"]


def test_empty_db_and_limits_are_valid(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    packet = retrieve_contrastive_episodes(path, "")
    assert packet["positive_matches"] == []
    assert packet["negative_matches"] == []
    assert packet["corrected_matches"] == []
    assert packet["contrastive_hints"] == []
    assert outcome_summary(path)["event_count"] == 0


def test_positive_negative_and_unrelated_structural_pools(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "verified-positive-private", successful=True, tools=("terminal",), start=1)
    append_outcome_event(
        path, trace_id="verified-positive-private", event_type="verification_passed",
        source="verifier", polarity="positive", confidence=1.0,
    )
    _record(path, "failed-negative-private", successful=False, tools=("terminal",), start=2)

    packet = retrieve_contrastive_episodes(path, "debug and verify the failure")
    assert len(packet["positive_matches"]) == 1
    assert len(packet["negative_matches"]) == 1
    blob = json.dumps(packet)
    assert "verified-positive-private" not in blob
    assert "failed-negative-private" not in blob

    unrelated = retrieve_contrastive_episodes(path, "new project ideation", max_negative_reward=0)
    assert unrelated["positive_matches"] == []
    assert unrelated["negative_matches"] == []


def test_shadow_eval_detects_privacy_canary_without_echoing_it(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))
    append_outcome_event(path, trace_id="safe", event_type="verification_passed", source="verifier", polarity="positive", confidence=1)
    report = contrastive_replay_eval(
        path,
        "debug and verify",
        forbidden_substrings=("terminal",),
        expected_canary_count=1,
        planted_canary_count=1,
    )
    assert report["privacy_finding_count"] > 0
    assert report["ok"] is False
    assert "terminal" not in json.dumps(report)


def test_weak_system_supersession_cannot_support_hint(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    _record(path, "replacement", successful=True, tools=("search_files",), start=2)
    mistake = append_outcome_event(
        path, trace_id="mistake", event_type="duplicate_proposal", source="evaluator",
        polarity="negative", confidence=.1, taxonomy_code="duplicate_existing_capability",
    )
    append_outcome_event(
        path, trace_id="replacement", event_type="rollback_required", source="system",
        polarity="neutral", confidence=0.0, taxonomy_code="duplicate_existing_capability",
        supersedes_event_id=mistake.event_id,
    )
    append_outcome_event(
        path, trace_id="replacement", event_type="verification_passed", source="system",
        polarity="positive", confidence=0.0,
    )
    packet = retrieve_contrastive_episodes(path, "new capability subsystem")
    assert packet["corrected_matches"] == []
    assert packet["contrastive_hints"] == []


def test_failed_high_reward_episode_is_not_positive(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "failed-with-answer", successful=True, tools=("terminal",))
    import sqlite3
    with sqlite3.connect(path) as con:
        con.execute("UPDATE traces SET status='failed' WHERE trace_id='failed-with-answer'")
    packet = retrieve_contrastive_episodes(path, "debug failure")
    assert packet["positive_matches"] == []
    assert len(packet["negative_matches"]) == 1


def test_task_frame_handles_common_mistake_learning_variants():
    for text in ("Does Hermes learn from its mistakes?", "mistake-learning capability audit"):
        assert "capability_audit" in task_frame_from_text(text)["categories"]


def test_corrected_pair_survives_more_than_200_newer_noise_events(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    _record(path, "replacement", successful=True, tools=("search_files",), start=2)
    mistake = append_outcome_event(path, trace_id="mistake", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability", created_at=1)
    append_outcome_event(path, trace_id="replacement", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=mistake.event_id, created_at=2)
    append_outcome_event(path, trace_id="replacement", event_type="verification_passed", source="verifier", polarity="positive", confidence=1, created_at=3)
    for index in range(205):
        append_outcome_event(path, trace_id="replacement", event_type="verification_failed", source="system", polarity="negative", confidence=0, created_at=100 + index)
    packet = retrieve_contrastive_episodes(path, "new capability subsystem")
    assert len(packet["corrected_matches"]) == 1
    assert packet["contrastive_hints"][0]["corrected_pair_support"] is True


def test_taxonomy_only_corrected_pair_survives_same_trace_noise(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    # terminal is irrelevant to capability ideation: pair taxonomy is the only match.
    _record(path, "replacement", successful=True, tools=("terminal",), start=2)
    mistake = append_outcome_event(path, trace_id="mistake", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability", event_id="original", created_at=1)
    append_outcome_event(path, trace_id="replacement", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=mistake.event_id, event_id="correction", created_at=2)
    append_outcome_event(path, trace_id="replacement", event_type="verification_passed", source="verifier", polarity="positive", confidence=1, event_id="pass", created_at=3)
    for index in range(205):
        append_outcome_event(path, trace_id="replacement", event_type="verification_failed", source="system", polarity="negative", confidence=0, event_id=f"noise-{index:03}", created_at=100 + index)

    packet = retrieve_contrastive_episodes(path, "new capability subsystem")

    assert len(packet["corrected_matches"]) == 1
    assert packet["corrected_matches"][0]["matched_features"] == ["taxonomy:duplicate_existing_capability"]


def test_noise_on_one_candidate_cannot_starve_another_candidates_negative_evidence(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "starved", successful=True, tools=("terminal",), start=1)
    _record(path, "noisy", successful=True, tools=("terminal",), start=2)
    append_outcome_event(path, trace_id="starved", event_type="verification_passed", source="verifier", polarity="positive", confidence=1, created_at=1)
    append_outcome_event(path, trace_id="starved", event_type="user_rejection", source="user", polarity="negative", confidence=.99, created_at=2)
    for index in range(205):
        append_outcome_event(path, trace_id="noisy", event_type="verification_failed", source="system", polarity="negative", confidence=0, event_id=f"noise-{index:03}", created_at=100 + index)

    packet = retrieve_contrastive_episodes(path, "debug failure")

    assert len(packet["negative_matches"]) == 2
    assert packet["positive_matches"] == []


def test_completed_high_reward_without_trusted_verifier_is_not_positive(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "unverified", successful=True, tools=("terminal",))
    packet = retrieve_contrastive_episodes(path, "debug failure")
    assert packet["positive_matches"] == []


@pytest.mark.parametrize("source,confidence", [("system", 1.0), ("evaluator", 1.0), ("verifier", .89)])
def test_untrusted_verification_pass_is_not_positive(tmp_path, source, confidence):
    path = tmp_path / "state.db"
    _record(path, "untrusted", successful=True, tools=("terminal",))
    append_outcome_event(path, trace_id="untrusted", event_type="verification_passed", source=source, polarity="positive", confidence=confidence)
    packet = retrieve_contrastive_episodes(path, "debug failure")
    assert packet["positive_matches"] == []


def test_ready_replacement_with_verification_attempted_is_not_a_corrected_pair(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    _record(path, "replacement", successful=True, tools=("terminal", "search_files"), start=2)
    mistake = append_outcome_event(path, trace_id="mistake", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability")
    append_outcome_event(path, trace_id="replacement", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=mistake.event_id)
    packet = retrieve_contrastive_episodes(path, "new capability subsystem")
    assert packet["corrected_matches"] == []
    assert packet["contrastive_hints"][0]["corrected_pair_support"] is False


def test_recurrence_counts_unique_public_episode_ids_across_overlapping_pools(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "mistake", successful=False, tools=("write_file",), start=1)
    _record(path, "replacement", successful=True, tools=("search_files",), start=2)
    mistake = append_outcome_event(path, trace_id="mistake", event_type="duplicate_proposal", source="user", polarity="negative", confidence=.99, taxonomy_code="duplicate_existing_capability")
    append_outcome_event(path, trace_id="replacement", event_type="user_correction", source="user", polarity="positive", confidence=.99, taxonomy_code="duplicate_existing_capability", supersedes_event_id=mistake.event_id)
    append_outcome_event(path, trace_id="replacement", event_type="verification_passed", source="verifier", polarity="positive", confidence=1)

    report = contrastive_replay_eval(path, "new capability subsystem")

    assert report["taxonomy_counts"] == {"duplicate_existing_capability": 2}
    assert report["recurrence_counts"] == {"duplicate_existing_capability": 2}



def test_eval_without_planted_canaries_is_explicitly_privacy_invalid(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))

    report = contrastive_replay_eval(path, "debug")

    assert report["expected_canary_coverage_count"] == 0
    assert report["planted_canary_count"] == 0
    assert report["canary_coverage_ok"] is False
    assert report["privacy_eval_valid"] is False
    assert report["ok"] is False



def test_eval_requires_all_declared_canaries_to_be_planted_in_raw_span_metadata(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))
    report = contrastive_replay_eval(
        path, "debug", forbidden_substrings=(CANARIES["prompt"], "NOT_PLANTED"),
        expected_canary_count=2, planted_canary_count=1,
    )
    assert report["expected_canary_coverage_count"] == 2
    assert report["planted_canary_count"] == 1
    assert report["canary_coverage_ok"] is False
    assert report["privacy_eval_valid"] is False
    assert report["ok"] is False


@pytest.mark.parametrize("name,value", [
    ("min_positive_reward", True),
    ("min_positive_reward", float("nan")),
    ("min_positive_reward", float("inf")),
    ("max_negative_reward", False),
    ("max_negative_reward", float("-inf")),
])
def test_reward_thresholds_require_finite_non_bool_numbers(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        retrieve_contrastive_episodes(tmp_path / "unused.db", "debug", **{name: value})


@pytest.mark.parametrize("minimum,maximum", [(1.0, 1.0), (1.0, 2.0)])
def test_reward_threshold_ranges_must_not_overlap(tmp_path, minimum, maximum):
    with pytest.raises(ValueError, match="max_negative_reward"):
        retrieve_contrastive_episodes(
            tmp_path / "unused.db", "debug",
            min_positive_reward=minimum, max_negative_reward=maximum,
        )


def test_privacy_contract_is_precise_and_task_frame_is_reported(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))
    packet = retrieve_contrastive_episodes(path, "debug")
    assert packet["privacy"] == {
        "raw_content_exported": False,
        "raw_task_text_exported": False,
        "raw_task_text_used_for_category_classification": True,
        "stored_raw_prompts_or_messages_used_for_matching": False,
        "raw_tool_args_or_results_used_for_matching": False,
        "deterministic_pseudonymous_episode_ids": True,
        "pseudonymous_ids_linkable_across_packets": True,
        "hashes_used_for_matching": False,
        "embeddings_used_for_matching": False,
    }
    report = contrastive_replay_eval(path, "debug")
    assert report["task_frame"] == packet["task_frame"]


def test_privacy_scanner_checks_only_exported_string_values_and_dedupes_canaries(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))
    append_outcome_event(path, trace_id="safe", event_type="verification_passed", source="verifier", polarity="positive", confidence=1)
    clean = contrastive_replay_eval(
        path, "debug", forbidden_substrings=("safe_tool_names",),
        expected_canary_count=1, planted_canary_count=1,
    )
    assert clean["privacy_finding_count"] == 0
    report = contrastive_replay_eval(
        path, "debug", forbidden_substrings=("terminal", "terminal", ""),
        expected_canary_count=1, planted_canary_count=1,
    )
    assert report["privacy_canary_coverage_count"] == 1
    assert report["privacy_finding_count"] == 1
    assert "terminal" not in json.dumps(report)


@pytest.mark.parametrize("mutation", ["raw_key", "privacy", "unsupported"])
def test_eval_fails_structural_privacy_and_hint_integrity_findings(monkeypatch, mutation):
    packet = {
        "schema_version": "contrastive_episode_retrieval.v1",
        "task_frame": {"categories": [], "source": "allowlisted_structural_categories"},
        "positive_matches": [], "negative_matches": [], "corrected_matches": [],
        "contrastive_hints": [], "eligible_episode_count": 0,
        "privacy": contrastive_module._expected_privacy_contract(False),
        "elapsed_ms": 0,
    }
    if mutation == "raw_key":
        packet["nested"] = {"prompt": "redacted"}
    elif mutation == "privacy":
        packet["privacy"].pop("raw_content_exported")
    else:
        packet["contrastive_hints"] = [{"supported": False}]
    monkeypatch.setattr(contrastive_module, "retrieve_contrastive_episodes", lambda **kwargs: packet)

    report = contrastive_module.contrastive_replay_eval(task_text="debug")

    assert report["ok"] is False


def test_eval_provenance_never_selects_or_materializes_raw_span_metadata():
    source = inspect.getsource(contrastive_module)
    assert "SELECT metadata_json FROM spans" not in source
    assert "_observed_metadata_canary_count" not in source


def test_eval_rejects_mismatched_structural_canary_provenance(tmp_path):
    path = tmp_path / "state.db"
    _record(path, "safe", successful=True, tools=("terminal",))

    report = contrastive_replay_eval(
        path, "debug", forbidden_substrings=("one", "two"),
        expected_canary_count=2, planted_canary_count=1,
    )

    assert report["expected_canary_coverage_count"] == 2
    assert report["planted_canary_count"] == 1
    assert report["canary_coverage_ok"] is False
    assert report["privacy_eval_valid"] is False
    assert report["ok"] is False
