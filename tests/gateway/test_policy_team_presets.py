"""v0.23 launch-policy sessions (CLI session.create, kanban cards) take the profile's team preset."""

import pytest

from gateway.run_model_presets import policy_preset_params
from hermes_cli.team_model_presets import PresetRoutingError
from tests.preset_fixtures import build_catalog


def test_unrouted_request_takes_preset_route_and_effort(tmp_path):
    config, home, _ = build_catalog(tmp_path, effort="max")
    merged = policy_preset_params({"source": "cli", "cwd": "/w"}, config, home)
    assert (merged["model"], merged["provider"], merged["reasoning"]) == ("claude-test", "claude-test", "max")
    assert merged["source"] == "cli" and merged["cwd"] == "/w"


def test_explicit_route_wins(tmp_path):
    config, home, _ = build_catalog(tmp_path)
    for params in ({"model": "gpt-x"}, {"provider": "openai-codex"}):
        assert policy_preset_params(params, config, home) is params


def test_card_effort_is_kept_over_preset_effort(tmp_path):
    config, home, _ = build_catalog(tmp_path, effort="max")
    merged = policy_preset_params({"reasoning": "low"}, config, home)
    assert merged["model"] == "claude-test" and merged["reasoning"] == "low"


def test_unbound_profile_is_unchanged(tmp_path):
    params = {"source": "cli"}
    assert policy_preset_params(params, {}, tmp_path) is params


def test_invalid_selection_fails_closed(tmp_path):
    config, home, root = build_catalog(tmp_path)
    (root / "estado/seleccion/team.json").write_text("corrupt")
    with pytest.raises(PresetRoutingError):
        policy_preset_params({}, config, home)


def test_other_profile_home_is_refused(tmp_path):
    config, _, _ = build_catalog(tmp_path)
    with pytest.raises(PresetRoutingError):
        policy_preset_params({}, config, tmp_path / "profiles" / "mayky-reviewer")


def test_build_policy_freezes_preset_route(tmp_path):
    from gateway.session_policy import build_policy
    config, home, _ = build_catalog(tmp_path, effort="max")
    config = {**config, "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"}}
    policy = build_policy(policy_preset_params({"source": "cli", "cwd": str(tmp_path)}, config, home), config,
                          private_secrets={})
    assert policy.model == "claude-test"
    assert policy.config()["model"]["provider"] == "claude-test"
    assert policy.config()["agent"]["reasoning_effort"] == "max"
