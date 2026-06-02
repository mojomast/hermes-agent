from scripts.evaluate_self_improvement_capabilities import run_eval


def test_evaluate_self_improvement_capabilities_includes_fixture_replay_eval():
    result = run_eval(limit=10, min_reward=1.0)

    assert "fixture_replay_eval" in result
    replay = result["fixture_replay_eval"]
    assert replay["schema_version"] == "training_episode_replay_eval.v1"
    assert replay["episode_count"] == 1
    assert replay["passed_count"] == 1
    assert replay["privacy"]["finding_count"] == 0
