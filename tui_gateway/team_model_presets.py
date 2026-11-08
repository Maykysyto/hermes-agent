"""Conversation-owned team defaults; never an overlay on profile configuration."""
from .method_ctx import bind_module


def _resolve_new_session_preset(params: dict, profile_home=None) -> dict | None:
    if any(str(params.get(key) or "").strip() for key in ("model", "provider")):
        return None
    from hermes_cli.team_model_presets import resolve_profile_preset
    with _session_profile_runtime_scope({"profile_home": str(profile_home) if profile_home else None},
                                        hydrate_secrets=False):
        return resolve_profile_preset(_load_cfg(), home=Path(profile_home) if profile_home else None)


def _preset_agent_defaults(snapshot, model_override, provider_override, reasoning_override):
    """A saved default supplies the route only while no native route override is present."""
    selected_snapshot = None
    if snapshot is not None and not model_override and not provider_override:
        from hermes_cli.team_model_presets import validate_preset_snapshot
        snapshot = validate_preset_snapshot(snapshot)
        model_override = {"model": snapshot["model"], "provider": snapshot["provider"]}
        selected_snapshot = snapshot
    return model_override, provider_override, reasoning_override, selected_snapshot


def _preset_agent_reasoning(snapshot, model: str, runtime: dict, reasoning_override):
    """Choose inherited effort after resolution; explicit effort follows a fallback unchanged."""
    if reasoning_override is not None:
        return reasoning_override
    provider = runtime.get("requested_provider") or runtime.get("provider")
    if (snapshot is not None and model == snapshot["model"] and provider == snapshot["provider"]
            and not runtime.get("_fallback_notice")):
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort(snapshot["reasoning_effort"])
    return None


def _session_preset_snapshot(model_config: dict) -> dict | None:
    if "team_model_preset" not in model_config:
        return None
    from hermes_cli.team_model_presets import validate_preset_snapshot
    return validate_preset_snapshot(model_config["team_model_preset"])


def _deferred_build_agent_kwargs(current: dict, session_db) -> dict:
    """_make_agent kwargs for a deferred (first-prompt) build. A lazy-resumed (watch) session carries the
    stored conversation id so the upgrade continues it; a cold deferred resume restores the full persisted
    runtime identity (like the eager resume's overrides splat) so the build can't drop the provider. No
    stored runtime, or an unroutable provider → this session's picked model/effort/tier, else the default."""
    kw = {"session_db": session_db, "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(current),
          "platform_override": _session_source(current), "cwd_override": _session_cwd(current)}
    if resume_sid := current.get("resume_session_id"):
        kw["session_id"] = resume_sid
    resume_overrides = current.get("resume_runtime_overrides")
    if isinstance(resume_overrides, dict) and resume_overrides and _overrides_have_routable_provider(resume_overrides):
        kw.update(resume_overrides)
    else:
        if override := current.get("model_override"):
            kw["model_override"] = override
        kw.update({k: v for k, v in (("reasoning_config_override", current.get("create_reasoning_override")),
                                     ("service_tier_override", current.get("create_service_tier_override")))
                   if v is not None})
    if current.get("team_model_preset") is not None:
        kw["team_model_preset"] = current["team_model_preset"]
    if current.get("provider_override"):
        kw["provider_override"] = current["provider_override"]
    return kw


def _profile_build_scope(profile_home):
    """Bind HERMES_HOME + secret + terminal scope for an agent build: the same composition a turn
    binds (``_session_profile_runtime_scope``). Home alone leaves ``get_secret()`` on the LAUNCH
    ``.env``; home + secrets alone leaves ``_make_agent``'s terminal probing on the launch process's
    ambient ``TERMINAL_*`` (a ``terminal.backend: docker`` secondary built a ``local`` agent)."""
    return _session_profile_runtime_scope({"profile_home": str(profile_home) if profile_home else None})


def _make_agent_in_context(sid: str, key: str, **kwargs):
    """``_make_agent`` with the session context bound for the build and cleared after."""
    tokens = _set_session_context(key, cwd=kwargs.get("cwd_override"))
    try:
        return _make_agent(sid, key, session_id=key, **kwargs)
    finally:
        _clear_session_context(tokens)


def _branch_model_defaults(session: dict) -> dict:
    override = session.get("model_override")
    params = override if isinstance(override, dict) else {}
    params = {**params, "provider": params.get("provider") or session.get("provider_override")}
    defaults = {"team_model_preset": _resolve_new_session_preset(params, session.get("profile_home"))}
    if override:
        defaults["model_override"] = override
    for key, value in (("provider_override", session.get("provider_override")),
                       ("reasoning_config_override", session.get("create_reasoning_override"))):
        if value is not None:
            defaults[key] = value
    return defaults


def register(server) -> None:
    bind_module(globals(), server)
