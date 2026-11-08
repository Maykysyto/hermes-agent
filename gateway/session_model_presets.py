"""Durable defaults, distinct from a conversation's explicit /model choice."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

_KEY = "team_model_preset"
_CHECKED = "team_model_preset_checked"


class SessionModelPresetMixin:
    """Resolve once under the routing lock, publishing only durable snapshots."""

    def resolve_model_preset(self, session_key: str, config: dict, *, initialize: bool = True) -> dict | None:
        from hermes_cli.team_model_presets import resolve_profile_preset, validate_preset_snapshot

        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            if _KEY in entry.metadata:
                value = entry.metadata[_KEY]
                return validate_preset_snapshot(value) if value is not None else None
            db = self._db_for_key(session_key)
            stored = db.get_session_model_config_value(entry.session_id, _KEY) if db else None
            checked = db.get_session_model_config_value(entry.session_id, _CHECKED, False) if db else False
            if stored is not None:
                snapshot = validate_preset_snapshot(stored)
            elif checked:
                snapshot = None
            else:
                if not initialize:
                    return None
                binding = config.get(_KEY)
                if not binding:
                    return None
                enabled_after = datetime.fromisoformat(binding["enabled_after"])
                if enabled_after.utcoffset() is None:
                    raise ValueError("team_model_preset enabled_after must include UTC timezone")
                row = db.get_session(entry.session_id) if db else None
                created = (datetime.fromtimestamp(row["started_at"], UTC)
                           if row and row.get("started_at") is not None else entry.created_at.astimezone(UTC))
                if created < enabled_after:
                    return None
                snapshot = resolve_profile_preset(config)
            metadata = {**entry.metadata, _KEY: snapshot}
            if db:
                db.patch_session_model_config(entry.session_id, {_KEY: snapshot, _CHECKED: True})
            self._save_entry(session_key, entry_data=replace(entry, metadata=metadata).to_dict(), lock_held=True)
            entry.metadata = metadata
            return dict(snapshot) if snapshot is not None else None
