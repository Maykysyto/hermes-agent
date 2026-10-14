"""Preset defaults preserve native CLI overrides and conversation boundaries."""
import json
from types import SimpleNamespace

import pytest

from tests.preset_fixtures import build_catalog


@pytest.fixture
def configured(tmp_path, monkeypatch):
    import cli
    config, home, root = build_catalog(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli, "_hermes_home", home)
    monkeypatch.setattr(cli, "CLI_CONFIG", {**cli.CLI_CONFIG, **config,
        "model": {"default": "native-model", "provider": "native-provider"},
        "agent": {**cli.CLI_CONFIG["agent"], "reasoning_effort": "low",
                  "reasoning_overrides": {"claude-test": "medium"}}})
    monkeypatch.setattr(cli.HermesCLI, "_init_ui_state", lambda self: None)
    return cli, config, home, root


def test_new_startup_uses_preset_without_explicit_flags(configured):
    cli, _, _, _ = configured
    instance = cli.HermesCLI()
    assert (instance.model, instance.provider) == ("claude-test", "claude-test")
    assert instance.reasoning_config["effort"] == "high"
    assert not instance._explicit_model_override
    assert instance._explicit_reasoning_config is None
    assert instance._team_model_preset["preset"] == "balanced"


@pytest.mark.parametrize("args", [{"model": "manual-model"}, {"provider": "manual-provider"}])
def test_explicit_native_route_bypasses_corrupt_selection(configured, args):
    cli, _, _, root = configured
    (root / "estado/seleccion/team.json").write_text("corrupt")
    instance = cli.HermesCLI(**args)
    assert instance.model == args.get("model", "native-model")
    assert instance.provider == args.get("provider", "native-provider")
    assert instance._team_model_preset is None


def test_explicit_reasoning_wins_but_does_not_disable_preset(configured):
    cli, _, _, _ = configured
    instance = cli.HermesCLI(reasoning="max")
    assert instance.model == "claude-test"
    assert instance.reasoning_config["effort"] == "max"


def test_resume_preserves_snapshot_effort_without_reading_selection(configured):
    cli, config, home, root = configured
    from hermes_cli.team_model_presets import resolve_profile_preset
    from hermes_state_registry import acquire
    snapshot = resolve_profile_preset(config, home)
    db = acquire()
    db.create_session(session_id="stored-preset", source="cli", model="claude-test",
                      model_config={"team_model_preset": snapshot, "provider": "claude-test",
                                    "reasoning_config": {"enabled": True, "effort": "max"}})
    (root / "estado/seleccion/team.json").write_text("corrupt")
    instance = cli.HermesCLI(resume="stored-preset")
    instance._restore_session_model(db.get_session("stored-preset"), quiet=True)
    assert (instance.model, instance.provider) == ("claude-test", "claude-test")
    assert instance.reasoning_config["effort"] == "max"
    assert instance._team_model_preset == snapshot


def test_new_refreshes_preset_even_if_model_unchanged(configured):
    cli, _, _, root = configured
    from hermes_cli.cli_session_mixin import _reset_model_to_config_default
    from hermes_cli.cli_model_switch_mixin import _resolve_cli_reasoning
    instance = cli.HermesCLI()
    instance._team_model_preset = {**instance._team_model_preset, "reasoning_effort": "low"}
    _reset_model_to_config_default(instance, silent=True)
    _resolve_cli_reasoning(instance)
    assert instance.reasoning_config["effort"] == "high"
    (root / "estado/seleccion/team.json").write_text("corrupt")
    from hermes_cli.team_model_presets import PresetRoutingError
    with pytest.raises(PresetRoutingError):
        _reset_model_to_config_default(instance, silent=True)


def test_snapshot_and_route_persist_to_model_config(configured):
    cli, _, _, _ = configured
    from hermes_cli.cli_agent_setup_mixin import _persist_team_preset
    instance = cli.HermesCLI(reasoning="max")
    instance._session_db.create_session(session_id=instance.session_id, source="cli", model=instance.model)
    _persist_team_preset(instance)
    metadata = json.loads(instance._session_db.get_session(instance.session_id)["model_config"])
    assert metadata["team_model_preset"] == instance._team_model_preset
    assert metadata["provider"] == "claude-test"
    assert metadata["reasoning_config"]["effort"] == "max"
    assert "api_key" not in metadata


def test_explicit_switch_same_pair_uses_native_reasoning(configured):
    cli, _, _, _ = configured
    instance = cli.HermesCLI()
    result = SimpleNamespace(new_model="claude-test", target_provider="claude-test", api_key="",
                             base_url="", api_mode="chat_completions", runtime_capabilities=None)
    assert instance._stage_and_swap_model(result, instance.model)
    assert instance.reasoning_config["effort"] == "medium"


def test_resume_same_route_restores_effective_effort(configured):
    cli, config, home, _ = configured
    from hermes_cli.team_model_presets import resolve_profile_preset
    instance = cli.HermesCLI()
    snapshot = resolve_profile_preset(config, home)
    instance.reasoning_config = {"enabled": True, "effort": "low"}
    instance._restore_session_model({"model": "claude-test", "model_config": {
        "provider": "claude-test", "team_model_preset": snapshot,
        "reasoning_config": {"enabled": True, "effort": "max"}}}, quiet=True)
    assert instance.reasoning_config["effort"] == "max"


def test_corrupt_new_does_not_rotate_current_session(configured):
    cli, _, _, root = configured
    instance = cli.HermesCLI()
    original = instance.session_id
    (root / "estado/seleccion/team.json").write_text("corrupt")
    from hermes_cli.team_model_presets import PresetRoutingError
    with pytest.raises(PresetRoutingError):
        instance.new_session(silent=True)
    assert instance.session_id == original


def test_resume_updates_live_agent_effort_on_same_route(configured):
    cli, config, home, _ = configured
    from hermes_cli.team_model_presets import resolve_profile_preset
    instance = cli.HermesCLI()
    instance.agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": "low"})
    instance._restore_session_model({"model": "claude-test", "model_config": {
        "provider": "claude-test", "team_model_preset": resolve_profile_preset(config, home),
        "reasoning_config": {"enabled": True, "effort": "max"}}}, quiet=True)
    assert instance.agent.reasoning_config["effort"] == "max"


@pytest.mark.parametrize("failure", ["rejected", "exception"])
def test_new_provider_failure_keeps_previous_session_and_default(configured, monkeypatch, failure):
    cli, _, _, _ = configured
    from hermes_cli.team_model_presets import PresetRoutingError
    instance = cli.HermesCLI(model="manual-sol", provider="manual-provider")
    original = (instance.session_id, instance.model, instance.provider, instance._team_model_preset)

    def reject_route(**kwargs):
        if failure == "exception":
            raise RuntimeError("provider unavailable")
        return SimpleNamespace(success=False, error="provider unavailable")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", reject_route)
    with pytest.raises(PresetRoutingError, match="provider unavailable"):
        instance.new_session(silent=True)
    assert (instance.session_id, instance.model, instance.provider, instance._team_model_preset) == original
    assert not instance._team_preset_route_active
