"""Profile-bound team defaults at the gateway's model and reasoning seams."""

from __future__ import annotations


def session_preset_defaults(runner, session_key: str | None, config: dict | None = None,
                            *, initialize: bool = True) -> dict | None:
    """Existing sessions retain their snapshot; legacy sessions retain their profile route."""
    store = getattr(runner, "session_store", None)
    if not session_key or store is None:
        return None
    if config is None:
        from gateway.run import _load_gateway_config
        config = _load_gateway_config()
    return store.resolve_model_preset(session_key, config, initialize=initialize)


def preset_reasoning_config(runner, session_key: str | None, source=None, model: str = "") -> dict | None:
    """A session's preset effort wins over profile reasoning overrides."""
    channel = channel_override(runner, source)
    explicit = runner._session_model_override(session_key) or (channel and (channel.model or channel.provider))
    if explicit or getattr(runner, "_pre_agent_fallback_notice", None):
        return None
    snapshot = session_preset_defaults(runner, session_key)
    if snapshot is None or (model and model != snapshot["model"]):
        return None
    from hermes_constants import parse_reasoning_effort
    return parse_reasoning_effort(snapshot["reasoning_effort"])


def channel_override(runner, source):
    """The explicit channel route must not depend on a valid default catalog."""
    from gateway.run import _get_channel_override
    config = getattr(runner, "config", None)
    if not config or source is None:
        return None
    return _get_channel_override(
        config, source.platform, str(source.chat_id) if source.chat_id else "",
        thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
        parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
    )


def session_default_model(runner, session_key, user_config, source, model, override):
    """Pin a default only when no explicit session/channel route was requested."""
    channel = channel_override(runner, source)
    explicit = override or (channel and (channel.model or channel.provider))
    preset = session_preset_defaults(runner, session_key, user_config) if not explicit else None
    return (preset["model"] if preset else model), preset


def default_runtime_kwargs(preset, config):
    """Resolve the selected pair together, preserving the native fallback policy."""
    from gateway.run import _load_gateway_config, _resolve_runtime_agent_kwargs, _runtime_agent_kwargs
    if not preset:
        return _resolve_runtime_agent_kwargs()
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    runtime, fallback = resolve_runtime_with_fallback(
        config if config is not None else _load_gateway_config(),
        requested=preset["provider"], target_model=preset["model"],
    )
    result = {**_runtime_agent_kwargs(runtime), "capabilities": dict(runtime.get("capabilities") or {})}
    if fallback:
        from hermes_cli.fallback_config import pre_agent_fallback_notice
        result["model"] = fallback["model"]
        result["_fallback_notice"] = pre_agent_fallback_notice(
            preset["provider"], preset["model"], runtime.get("provider"), fallback["model"],
        )
    return result


def apply_channel_runtime(runner, source, model, runtime_kwargs):
    """Apply the existing channel tier after defaults and before the session override."""
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    channel = channel_override(runner, source)
    if channel:
        if channel.model:
            model = channel.model
        if channel.provider:
            runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(channel.provider, target_model=model or None)
            runtime_model = runtime_kwargs.pop("model", None)
            if runtime_model and not channel.model:
                model = runtime_model
    return model, runtime_kwargs
