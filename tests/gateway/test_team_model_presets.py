"""Team defaults are pinned per conversation without replacing explicit /model choices."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _select(root, name, provider, model):
    restriction = {
        "roles": {"coding": {"perfil": "mayky-coding"}},
        "proveedores": {
            "openai-codex": {"modelos": ["gpt-6.1-sol"], "esfuerzos": "runtime:codex_supported_efforts"},
            "claude-subscription-directsdk-experimental": {
                "modelos": ["claude-opus-5-5[1m]"], "esfuerzos": ["high"],
            },
        },
    }
    catalog = root / "catalogo"
    catalog.mkdir(exist_ok=True, parents=True)
    restriction_bytes = _canonical(restriction)
    (catalog / "restricciones.json").write_bytes(restriction_bytes)
    document = {
        "preset": name, "version": 1, "estado": "editada", "padre_sha256": None,
        "restricciones_sha256": hashlib.sha256(restriction_bytes).hexdigest(),
        "roles": {"coding": {"proveedor": provider, "modelo": model, "esfuerzo": "high"}},
    }
    data = _canonical(document)
    digest = hashlib.sha256(data).hexdigest()
    path = catalog / "presets" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "v1.json").write_bytes(data)
    with (catalog / "indice.jsonl").open("a") as stream:
        stream.write(json.dumps({"preset": name, "version": 1, "sha256": digest}) + "\n")
    selection = root / "estado" / "seleccion"
    selection.mkdir(parents=True, exist_ok=True)
    (selection / "mayky-team.json").write_bytes(_canonical({
        "ambito": "mayky-team", "preset": name, "version": 1, "sha256": digest,
    }))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    home = tmp_path / "profiles" / "mayky-coding"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = tmp_path / "presets"
    _select(root, "equilibrado", "claude-subscription-directsdk-experimental", "claude-opus-5-5[1m]")
    config = {
        "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
        "agent": {"reasoning_effort": "low"},
        "team_model_preset": {
            "catalog_root": str(root), "scope": "mayky-team", "role": "coding",
            "enabled_after": "2020-01-01T00:00:00+00:00",
        },
    }
    calls = []

    def resolve(provider, target_model=None):
        calls.append((provider, target_model))
        return {"provider": provider, "api_key": "runtime-secret", "capabilities": {}}

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", resolve)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: resolve("openai-codex", "gpt-6.1-sol"))
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_with_fallback",
                        lambda config, requested=None, target_model=None: (resolve(requested, target_model), None))

    def make():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.session_store = store
        runner.config = SimpleNamespace(platforms={})
        return store, runner

    store, runner = make()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm")
    entry = store.get_or_create_session(source)
    return SimpleNamespace(root=root, config=config, store=store, runner=runner,
                           source=source, entry=entry, make=make, calls=calls)


def _route(setup):
    return setup.runner._resolve_session_agent_runtime(session_key=setup.entry.session_key, user_config=setup.config)


def test_new_conversation_uses_preset_route_and_effort(setup):
    model, runtime = _route(setup)
    assert model == "claude-opus-5-5[1m]"
    assert runtime["provider"] == "claude-subscription-directsdk-experimental"
    assert setup.calls == [(runtime["provider"], model)]
    assert setup.runner._resolve_session_reasoning_config(session_key=setup.entry.session_key, model=model) == {"enabled": True, "effort": "high"}
    assert setup.store.get_model_override(setup.entry.session_key) is None


def test_preset_snapshot_survives_selection_change_and_restart(setup):
    first = _route(setup)
    _select(setup.root, "codex", "openai-codex", "gpt-6.1-sol")
    setup.store, setup.runner = setup.make()
    assert _route(setup) == first
    assert setup.runner._resolve_session_reasoning_config(session_key=setup.entry.session_key) == {"enabled": True, "effort": "high"}


def test_new_session_takes_new_selection(setup):
    assert _route(setup)[0] == "claude-opus-5-5[1m]"
    _select(setup.root, "codex", "openai-codex", "gpt-6.1-sol")
    setup.entry = setup.store.reset_session(setup.entry.session_key)
    assert _route(setup)[0] == "gpt-6.1-sol"


def test_legacy_conversation_keeps_profile_default(setup):
    setup.config["team_model_preset"]["enabled_after"] = "2999-01-01T00:00:00+00:00"
    assert _route(setup)[0] == "gpt-6.1-sol"


def test_explicit_session_model_and_reasoning_win(setup):
    setup.store.set_model_override(setup.entry.session_key, {"model": "gpt-6.1-sol", "provider": "openai-codex"})
    setup.runner._set_session_reasoning_override(setup.entry.session_key, {"effort": "medium"})
    assert _route(setup)[0] == "gpt-6.1-sol"
    assert setup.runner._resolve_session_reasoning_config(session_key=setup.entry.session_key) == {"effort": "medium"}


def test_corrupt_selection_fails_before_provider_resolution(setup):
    (setup.root / "estado" / "seleccion" / "mayky-team.json").write_text("broken")
    with pytest.raises(ValueError, match="preset|selecci|JSON"):
        _route(setup)
    assert setup.calls == []


def test_pinned_session_does_not_reread_corrupt_catalog(setup):
    first = _route(setup)
    (setup.root / "estado" / "seleccion" / "mayky-team.json").write_text("broken")
    assert _route(setup) == first


def test_snapshot_initialization_uses_one_winner(setup):
    from gateway.run_model_presets import session_preset_defaults

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: session_preset_defaults(setup.runner, setup.entry.session_key, setup.config), range(2)))
    assert values[0] == values[1]
    assert values[0]["model"] == "claude-opus-5-5[1m]"


def test_failed_snapshot_write_does_not_publish(setup, monkeypatch):
    from gateway.run_model_presets import session_preset_defaults

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(setup.store, "_save_entry", fail)
    with pytest.raises(OSError, match="disk failure"):
        session_preset_defaults(setup.runner, setup.entry.session_key, setup.config)
    assert "team_model_preset" not in setup.entry.metadata


def test_resume_uses_original_session_snapshot(setup):
    assert _route(setup)[0] == "claude-opus-5-5[1m]"
    original_id = setup.entry.session_id
    _select(setup.root, "codex", "openai-codex", "gpt-6.1-sol")
    setup.entry = setup.store.reset_session(setup.entry.session_key)
    assert _route(setup)[0] == "gpt-6.1-sol"
    setup.entry = setup.store.switch_session(setup.entry.session_key, original_id, preserve_prompt_pin=False)
    assert _route(setup)[0] == "claude-opus-5-5[1m]"


def test_explicit_channel_route_bypasses_invalid_selection(setup):
    import gateway.run as gateway_run
    from unittest.mock import patch

    (setup.root / "estado" / "seleccion" / "mayky-team.json").write_text("broken")
    channel = SimpleNamespace(model="gpt-6.1-sol", provider="openai-codex")
    with patch.object(gateway_run, "_get_channel_override", return_value=channel):
        model, runtime = setup.runner._resolve_session_agent_runtime(source=setup.source, session_key=setup.entry.session_key,
                                                                    user_config=setup.config)
        assert model == "gpt-6.1-sol"
        assert runtime["provider"] == "openai-codex"
        assert setup.runner._resolve_session_reasoning_config(source=setup.source, session_key=setup.entry.session_key)["effort"] == "low"


def test_preset_route_preserves_native_configured_fallback(setup, monkeypatch):
    def fallback(config, requested=None, target_model=None):
        assert requested == "claude-subscription-directsdk-experimental"
        assert target_model == "claude-opus-5-5[1m]"
        return {"provider": "openai-codex", "api_key": "fallback-secret"}, {"provider": "openai-codex", "model": "gpt-6.1-sol"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_with_fallback", fallback)
    model, runtime = _route(setup)
    assert model == "gpt-6.1-sol"
    assert runtime["provider"] == "openai-codex"
    assert setup.runner._pre_agent_fallback_notice is not None
    assert setup.runner._resolve_session_reasoning_config(session_key=setup.entry.session_key, model=model)["effort"] == "low"


@pytest.mark.parametrize("model,provider", [
    ("gpt-6.1-sol", "openai-codex"),
    ("claude-opus-5-5[1m]", "claude-subscription-directsdk-experimental"),
])
def test_explicit_model_after_preset_uses_native_effort(setup, model, provider):
    assert _route(setup)[0] == "claude-opus-5-5[1m]"
    setup.store.set_model_override(setup.entry.session_key, {"model": model, "provider": provider})
    assert _route(setup)[0] == model
    assert setup.runner._resolve_session_reasoning_config(session_key=setup.entry.session_key, model=model)["effort"] == "low"


def test_snapshot_is_inherited_by_compression_child(setup):
    assert _route(setup)[0] == "claude-opus-5-5[1m]"
    db = setup.store._db_for_key(setup.entry.session_key)
    db.publish_compression_child(parent_session_id=setup.entry.session_id, child_session_id="compressed-child",
                                 source="telegram", messages=[{"role": "user", "content": "summary"}], require_compression_lease=False)
    snapshot = db.get_session_model_config_value("compressed-child", "team_model_preset")
    assert snapshot["model"] == "claude-opus-5-5[1m]"
