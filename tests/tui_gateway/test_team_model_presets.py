"""Team defaults are selected once per new TUI/Desktop/dashboard conversation."""
import copy
import json

import pytest
import hermes_yaml as yaml


@pytest.fixture
def preset_chat(tmp_path, monkeypatch):
    from tests.preset_fixtures import build_catalog
    from tui_gateway import server

    config, home, root = build_catalog(tmp_path)
    config.update(model={"default": "profile-model", "provider": "openrouter"},
                  agent={"reasoning_effort": "low"})
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_profile_home", lambda *a: home)
    for name in ("_enable_gateway_prompts", "_schedule_agent_build", "_schedule_session_cap_enforcement",
                 "_register_session_cwd", "_project_info_for_cwd"):
        monkeypatch.setattr(server, name, lambda *a: None)
    def create(**params):
        response = server._methods["session.create"]("r", {"cols": 80, **params})
        assert "error" not in response, response
        return server._sessions[response["result"]["session_id"]], response["result"]
    return server, config, home, root, create


def test_new_chat_fixes_preset_without_turn_or_profile_write(preset_chat):
    from hermes_cli.team_model_presets import resolve_profile_preset
    server, config, home, _, create = preset_chat
    original = (home / "config.yaml").read_bytes()
    session, result = create()
    snapshot = resolve_profile_preset(config, home=home)
    assert session["team_model_preset"] == snapshot
    assert session["model_override"] is None
    assert result["info"]["model"] == snapshot["model"]
    assert result["info"]["provider"] == snapshot["provider"]
    kwargs = server._deferred_build_agent_kwargs(session, None)
    assert kwargs["team_model_preset"] == snapshot
    model, provider, reasoning, selected = server._preset_agent_defaults(snapshot, None, None, None)
    assert selected == snapshot
    assert (model, provider, reasoning) == (
        {"model": snapshot["model"], "provider": snapshot["provider"]}, None, None)
    assert server._preset_agent_reasoning(snapshot, snapshot["model"], {"provider": snapshot["provider"]}, reasoning) == {
        "enabled": True, "effort": "high"}
    assert (home / "config.yaml").read_bytes() == original


@pytest.mark.parametrize("params", [{"model": "explicit-model", "provider": "openrouter"},
                                     {"provider": "openrouter"}])
def test_explicit_route_bypasses_even_corrupt_selection(preset_chat, params):
    server, _, _, root, create = preset_chat
    (root / "estado" / "seleccion" / "team.json").write_text("broken", encoding="utf-8")
    session, result = create(**params)
    assert result["info"]["provider"] == params["provider"]
    assert session.get("team_model_preset") is None
    kwargs = server._deferred_build_agent_kwargs(session, None)
    if params.get("model"):
        assert kwargs["model_override"] == params
    else:
        assert kwargs["provider_override"] == params["provider"]


def test_explicit_effort_beats_preset_and_row_restores_snapshot(preset_chat):
    server, _, _, _, create = preset_chat
    session, _ = create(reasoning_effort="medium")
    snapshot = session["team_model_preset"]
    row_model, model_config = server._workdir_row_model_config(session)
    assert row_model == snapshot["model"]
    assert model_config["provider"] == snapshot["provider"]
    assert model_config["team_model_preset"] == snapshot
    assert model_config["reasoning_config"] == {"enabled": True, "effort": "medium"}
    restored = server._stored_session_runtime_overrides(
        {"model": row_model, "model_config": json.dumps(model_config)})
    assert restored["team_model_preset"] == snapshot
    assert restored["reasoning_config_override"] == model_config["reasoning_config"]


def test_rebuild_uses_snapshot_after_selection_corruption_but_new_chat_errors(preset_chat):
    server, _, _, root, create = preset_chat
    session, _ = create()
    snapshot = copy.deepcopy(session["team_model_preset"])
    (root / "estado" / "seleccion" / "team.json").write_text("broken", encoding="utf-8")
    assert server._deferred_build_agent_kwargs(session, None)["team_model_preset"] == snapshot
    response = server._methods["session.create"]("r", {"cols": 80})
    assert "error" in response
    assert len(server._sessions) == 1


def test_legacy_resume_does_not_select_preset(preset_chat):
    server, _, _, _, _ = preset_chat
    restored = server._stored_session_runtime_overrides({"model": "old-model", "model_config": {"provider": "openrouter"}})
    assert "team_model_preset" not in restored
    assert restored["model_override"]["model"] == "old-model"


def test_tools_reset_keeps_snapshot_and_new_chat_reselects(preset_chat, monkeypatch):
    server, _, _, root, create = preset_chat
    session, result = create()
    snapshot = copy.deepcopy(session["team_model_preset"])
    selection = root / "estado" / "seleccion" / "team.json"
    selection.unlink()
    captured = []
    monkeypatch.setattr(server, "_rebuild_session_agent", lambda sid, session, **kw: captured.append(copy.deepcopy(session.get("team_model_preset"))))
    monkeypatch.setattr(server, "_session_info", lambda *a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *a: None)
    server._reset_session_agent(result["session_id"], session)
    assert captured == [snapshot]
    assert create()[0].get("team_model_preset") is None
    selection.write_text("broken", encoding="utf-8")
    server._reset_session_agent(result["session_id"], session)
    assert captured == [snapshot, snapshot]


def test_new_bot_chat_resume_keeps_preset_and_does_not_sync_profile_model(preset_chat, monkeypatch):
    import types
    server, _, _, _, create = preset_chat
    session, _ = create(follow_profile_config=True)
    model, config = server._workdir_row_model_config(session)
    overrides = server._stored_session_runtime_overrides({"model": model, "model_config": config, "title": "Bot Chat"})
    assert overrides["team_model_preset"] == session["team_model_preset"]
    assert overrides["model_override"]["model"] == model
    session["agent"] = types.SimpleNamespace(model=model)
    switched = []
    monkeypatch.setattr(server, "_config_model_target", lambda: ("changed-profile", "openrouter"))
    monkeypatch.setattr(server, "_apply_model_switch", lambda *a, **kw: switched.append(a))
    server._sync_agent_model_with_config("sid", session)
    assert switched == []


def test_preset_reaches_real_agent_factory_without_inference(preset_chat, monkeypatch):
    import types
    server, _, _, _, create = preset_chat
    session, result = create()
    captured = {}
    routes = []
    def runtime(model, provider):
        routes.append((model, provider))
        return model["model"], {"provider": model["provider"]}
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", runtime)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *a, **kw: [])
    def agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs["model"])
    monkeypatch.setattr("run_agent.AIAgent", agent)
    server._make_agent(result["session_id"], session["session_key"],
                       **server._deferred_build_agent_kwargs(session, None))
    assert captured["model"] == "claude-test"
    assert captured["provider"] == "claude-test"
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert routes == [({"model": "claude-test", "provider": "claude-test"}, None)]


def test_creation_binds_preset_to_owning_profile(preset_chat, monkeypatch, tmp_path):
    from tests.preset_fixtures import build_catalog
    server, _, launch, _, create = preset_chat
    config, secondary, _ = build_catalog(tmp_path / "secondary", profile="secondary-profile", model="secondary-model")
    config["model"] = {"default": "other-profile-model", "provider": "openrouter"}
    (secondary / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(server, "_profile_home", lambda profile=None: secondary if profile == "secondary-profile" else launch)
    session, result = create(profile="secondary-profile")
    assert session["team_model_preset"]["model"] == "secondary-model"
    assert session["profile_home"] == str(secondary)
    assert result["info"]["model"] == "secondary-model"
    assert create()[0]["team_model_preset"]["model"] == "claude-test"



def test_lazy_resume_record_restores_saved_default(preset_chat):
    server, _, home, root, create = preset_chat
    session, _ = create()
    model, config = server._workdir_row_model_config(session)
    (root / "estado" / "seleccion" / "team.json").write_text("broken", encoding="utf-8")
    ctx = server._Resume("r", {}, "stored-key")
    ctx.target = session["session_key"]
    ctx.found = {"model": model, "model_config": config}
    restored = ctx.record("tui", str(home), [], lazy=True)
    assert restored["team_model_preset"] == session["team_model_preset"]
    assert server._deferred_build_agent_kwargs(restored, None)["team_model_preset"] == session["team_model_preset"]


def test_branch_selects_new_default_but_inherits_explicit_native_route(preset_chat):
    server, _, _, _, create = preset_chat
    session, _ = create()
    defaults = server._branch_model_defaults(session)
    assert defaults["team_model_preset"] == session["team_model_preset"]
    assert "model_override" not in defaults
    session["model_override"] = {"model": "explicit-model", "provider": "openrouter"}
    defaults = server._branch_model_defaults(session)
    assert defaults["team_model_preset"] is None
    assert defaults["model_override"] == session["model_override"]


@pytest.mark.parametrize("lazy", [False, True])
def test_sqlite_restart_resume_keeps_snapshot_without_catalog(preset_chat, monkeypatch, lazy):
    from hermes_state import SessionDB
    server, _, home, root, create = preset_chat
    session, _ = create(reasoning_effort="medium")
    model, config = server._workdir_row_model_config(session)
    key, snapshot = session["session_key"], session["team_model_preset"]
    with SessionDB(db_path=home / "state.db") as db:
        db.create_session(key, source="tui", model=model, model_config=config)
    server._sessions.clear()
    (root / "estado" / "seleccion" / "team.json").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a: None)
    response = server._methods["session.resume"]("resume", {"session_id": key, "lazy": lazy})
    assert "error" not in response, response
    restored = server._sessions[response["result"]["session_id"]]
    assert restored["team_model_preset"] == snapshot
    assert server._deferred_build_agent_kwargs(restored, None)["team_model_preset"] == snapshot


@pytest.mark.parametrize("explicit_effort,expected", [(None, "low"), ("medium", "medium")])
@pytest.mark.parametrize("fallback_model", ["fallback-model", "claude-test"])
def test_fallback_recomputes_inherited_effort_but_keeps_explicit_effort(
        preset_chat, monkeypatch, explicit_effort, expected, fallback_model):
    import types
    server, _, _, _, create = preset_chat
    session, result = create(**({"reasoning_effort": explicit_effort} if explicit_effort else {}))
    captured = {}
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *a: (
        fallback_model, {"provider": "openrouter", "requested_provider": "openrouter"}))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *a, **kw: [])
    def agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs["model"])
    monkeypatch.setattr("run_agent.AIAgent", agent)
    with server._profile_build_scope(session["profile_home"]):
        assert server._load_reasoning_config(fallback_model) == {"enabled": True, "effort": "low"}
        server._make_agent(result["session_id"], session["session_key"],
                           **server._deferred_build_agent_kwargs(session, None))
    assert captured["model"] == fallback_model
    assert captured["provider"] == "openrouter"
    assert captured["reasoning_config"] == {"enabled": True, "effort": expected}
