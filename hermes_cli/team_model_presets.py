"""Read-only MYT-95 defaults. Identity and credentials remain owned by the profile."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,40}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_KEYS = {"model", "provider", "reasoning_effort", "preset", "version", "sha256", "scope", "role"}


class PresetRoutingError(ValueError):
    """A configured preset cannot be used safely; never silently fall back."""


def _require(condition, message):
    if not condition:
        raise PresetRoutingError(message)


def _slug(value):
    return isinstance(value, str) and bool(_SLUG.fullmatch(value))


def validate_preset_snapshot(value) -> dict:
    """Validate stored provenance without consulting a potentially changed catalog."""
    _require(isinstance(value, dict) and set(value) == _SNAPSHOT_KEYS, "Invalid team preset snapshot fields")
    for key in ("model", "provider", "reasoning_effort"):
        item = value[key]
        _require(isinstance(item, str) and bool(item.strip()) and item == item.strip(), f"Invalid preset {key}")
    for key in ("preset", "scope", "role"):
        _require(_slug(value[key]), f"Invalid preset {key}")
    _require(type(value["version"]) is int and value["version"] > 0, "Invalid preset version")
    _require(isinstance(value["sha256"], str) and bool(_HASH.fullmatch(value["sha256"])), "Invalid preset sha256")
    return dict(value)


def _json(path):
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _binding(config):
    binding = config.get("team_model_preset")
    if binding is None or binding == {}:
        return None
    _require(isinstance(binding, dict), "team_model_preset must be a mapping")
    root = binding.get("catalog_root")
    _require(isinstance(root, str) and Path(root).is_absolute(), "Preset catalog_root must be absolute")
    for key in ("scope", "role"):
        _require(_slug(binding.get(key)), f"Invalid team_model_preset {key}")
    enabled = binding.get("enabled_after")
    _require(isinstance(enabled, str), "Preset enabled_after must be UTC")
    instant = datetime.fromisoformat(enabled)
    _require(instant.utcoffset() == timedelta(0), "Preset enabled_after must be UTC")
    return binding


def _validate_package(restrictions, role, package):
    _require(isinstance(package, dict) and set(package) == {"proveedor", "modelo", "esfuerzo"}, f"Invalid package for {role}")
    for item in package.values():
        _require(isinstance(item, str) and bool(item.strip()) and item == item.strip(), f"Invalid package text for {role}")
    provider, model, effort = (package[key] for key in ("proveedor", "modelo", "esfuerzo"))
    providers = restrictions["proveedores"]
    _require(provider in providers, f"Preset provider not allowed: {provider}")
    provider_config = providers[provider]
    _require(model in provider_config["modelos"], f"Preset model not allowed: {model}")
    levels = provider_config.get("esfuerzos")
    if levels == "runtime:codex_supported_efforts":
        _require(provider == "openai-codex", "Codex effort source requires openai-codex")
        from agent.reasoning_effort import codex_supported_efforts
        levels = codex_supported_efforts(model)
    _require(isinstance(levels, (list, tuple)), f"Invalid effort source for {provider}")
    from hermes_constants import VALID_REASONING_EFFORTS
    _require(effort in levels and effort in {"none", *VALID_REASONING_EFFORTS}
             and effort not in restrictions.get("esfuerzos_excluidos", []), f"Preset effort not allowed: {effort}")
    fixed = restrictions["roles"][role].get("fijo")
    _require(not fixed or (provider, model) == (fixed["proveedor"], fixed["modelo"]), f"Protected preset role: {role}")


def _index(root):
    result = {}
    for line in (root / "catalogo/indice.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        name, version, digest = entry["preset"], entry["version"], entry["sha256"]
        _require(_slug(name) and type(version) is int and version > 0, "Invalid preset index identity")
        _require(isinstance(digest, str) and bool(_HASH.fullmatch(digest)), "Invalid preset index hash")
        key = name, version
        _require(key not in result or result[key] == digest, "Conflicting preset index hashes")
        result[key] = digest
    return result


def _version(root, selection, restrictions):
    name, version = selection["preset"], selection["version"]
    _require(_slug(name) and type(version) is int and version > 0, "Invalid selected preset identity")
    directory = root / "catalogo/presets" / name
    versions = sorted(int(match[1]) for path in directory.iterdir()
                      if (match := re.fullmatch(r"v(\d+)\.json", path.name)))
    _require(bool(versions) and versions == list(range(1, max(versions) + 1)), "Preset versions are not consecutive")
    index = _index(root)
    previous = None
    document = None
    for number in range(1, version + 1):
        data = (directory / f"v{number}.json").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        _require(index.get((name, number)) == digest, "Preset hash does not match index")
        document = json.loads(data)
        canonical = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _require(data == canonical, "Preset bytes are not canonical")
        _require(document["preset"] == name and document["version"] == number, "Preset identity mismatch")
        _require(document.get("padre_sha256") == previous, "Preset parent hash mismatch")
        _require(document.get("estado") in {"snapshot-vigente", "propuesta", "editada"}, "Invalid preset state")
        roles = document["roles"]
        _require(isinstance(roles, dict) and set(roles) == set(restrictions["roles"]), "Invalid preset roles")
        for role, package in roles.items():
            _validate_package(restrictions, role, package)
        previous = digest
    _require(previous == selection["sha256"], "Preset hash does not match selection")
    return document


def resolve_profile_preset(config: dict, home: Path | None = None) -> dict | None:
    """Resolve the selected package, or None for an unbound/unselected profile.

    No external helper is imported, no catalog state is written, and invalid selected
    state always raises PresetRoutingError. Explicit CLI choices bypass this reader.
    """
    try:
        binding = _binding(config)
        if binding is None:
            return None
        root, scope, role = Path(binding["catalog_root"]), binding["scope"], binding["role"]
        selection_path = root / "estado/seleccion" / f"{scope}.json"
        if not selection_path.exists() and not selection_path.is_symlink():
            return None
        selection = _json(selection_path)
        _require(selection["ambito"] == scope, "Preset selection scope mismatch")
        restrictions = _json(root / "catalogo/restricciones.json")
        _require(isinstance(restrictions.get("roles"), dict) and bool(restrictions["roles"]), "Invalid preset role restrictions")
        _require(isinstance(restrictions.get("proveedores"), dict) and bool(restrictions["proveedores"]), "Invalid preset provider restrictions")
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        _require(role in restrictions["roles"] and restrictions["roles"][role]["perfil"] == Path(home).name,
                 "Preset role does not match profile")
        document = _version(root, selection, restrictions)
        package = document["roles"][role]
        return validate_preset_snapshot({"model": package["modelo"], "provider": package["proveedor"],
            "reasoning_effort": package["esfuerzo"], "preset": selection["preset"], "version": selection["version"],
            "sha256": selection["sha256"], "scope": scope, "role": role})
    except PresetRoutingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PresetRoutingError(f"Cannot resolve team model preset: {exc}") from exc
