"""Read-only runtime resolution of MYT-95 selection and integrity contracts."""
import hashlib
import json

import pytest

from tests.preset_fixtures import build_catalog, canonical


def resolve(config, home):
    from hermes_cli.team_model_presets import resolve_profile_preset
    return resolve_profile_preset(config, home)


def test_selected_package_and_provenance_are_read_only(tmp_path):
    config, home, root = build_catalog(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    snapshot = resolve(config, home)
    assert {k: snapshot[k] for k in ("model", "provider", "reasoning_effort", "scope", "role")} == {
        "model": "claude-test", "provider": "claude-test", "reasoning_effort": "high", "scope": "team", "role": "coding"}
    assert snapshot["sha256"] == hashlib.sha256(before[root / "catalogo/presets/balanced/v1.json"]).hexdigest()
    assert snapshot["preset"] == "balanced" and snapshot["version"] == 1
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_absent_mapping_or_selection_keeps_native_default(tmp_path):
    assert resolve({}, tmp_path) is None
    config, home, root = build_catalog(tmp_path)
    (root / "estado/seleccion/team.json").unlink()
    assert resolve(config, home) is None


@pytest.mark.parametrize("fault", ["selection", "hash", "canonical", "parent", "profile", "effort", "index", "root", "utc"])
def test_present_invalid_selection_fails_closed(tmp_path, fault):
    from hermes_cli.team_model_presets import PresetRoutingError
    config, home, root = build_catalog(tmp_path)
    version = root / "catalogo/presets/balanced/v1.json"
    selection = root / "estado/seleccion/team.json"
    def altered_document():
        document = json.loads(version.read_bytes())
        if fault == "parent":
            document["padre_sha256"] = "0" * 64
        if fault == "effort":
            document["roles"]["coding"]["esfuerzo"] = "ultra"
        version.write_bytes(json.dumps(document).encode() if fault == "canonical" else canonical(document))
        digest = hashlib.sha256(version.read_bytes()).hexdigest()
        (root / "catalogo/indice.jsonl").write_text(json.dumps({"preset": "balanced", "version": 1, "sha256": digest}))
        selection.write_bytes(canonical({"ambito": "team", "preset": "balanced", "version": 1, "sha256": digest}))

    mutations = {
        "selection": lambda: selection.write_text("{"),
        "hash": lambda: version.write_bytes(version.read_bytes() + b" "),
        "canonical": altered_document,
        "parent": altered_document,
        "effort": altered_document,
        "profile": lambda: None,
        "index": lambda: (root / "catalogo/indice.jsonl").write_text("{}"),
        "root": lambda: config["team_model_preset"].update(catalog_root="relative"),
        "utc": lambda: config["team_model_preset"].update(enabled_after="2026-10-15"),
    }
    mutations[fault]()
    home = home.with_name("foreign") if fault == "profile" else home
    with pytest.raises(PresetRoutingError):
        resolve(config, home)


def test_snapshot_validation_rejects_secret_or_extra_fields(tmp_path):
    from hermes_cli.team_model_presets import validate_preset_snapshot, PresetRoutingError
    config, home, _ = build_catalog(tmp_path)
    snapshot = resolve(config, home)
    assert validate_preset_snapshot(snapshot) == snapshot
    with pytest.raises(PresetRoutingError):
        validate_preset_snapshot({**snapshot, "api_key": "secret"})


def test_dangling_selection_is_present_corruption(tmp_path):
    from hermes_cli.team_model_presets import PresetRoutingError
    config, home, root = build_catalog(tmp_path)
    selection = root / "estado/seleccion/team.json"
    selection.unlink()
    selection.symlink_to("missing-target.json")
    with pytest.raises(PresetRoutingError):
        resolve(config, home)


def test_codex_efforts_follow_runtime_model_support(tmp_path):
    from hermes_cli.team_model_presets import PresetRoutingError
    config, home, root = build_catalog(tmp_path, model="gpt-6.1-sol", provider="openai-codex", effort="none")
    restrictions = root / "catalogo/restricciones.json"
    value = json.loads(restrictions.read_bytes())
    value["proveedores"]["openai-codex"]["esfuerzos"] = "runtime:codex_supported_efforts"
    restrictions.write_bytes(canonical(value))
    with pytest.raises(PresetRoutingError, match="effort"):
        resolve(config, home)


def test_version_two_validates_parent_and_current_selection(tmp_path):
    config, home, root = build_catalog(tmp_path)
    original = resolve(config, home)
    path = root / "catalogo/presets/balanced/v2.json"
    document = json.loads(path.with_name("v1.json").read_bytes())
    document.update(version=2, padre_sha256=original["sha256"])
    document["roles"]["coding"]["esfuerzo"] = "low"
    path.write_bytes(canonical(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with (root / "catalogo/indice.jsonl").open("a") as handle:
        handle.write(json.dumps({"preset": "balanced", "version": 2, "sha256": digest}) + "\n")
    (root / "estado/seleccion/team.json").write_bytes(canonical({"ambito": "team", "preset": "balanced", "version": 2, "sha256": digest}))
    snapshot = resolve(config, home)
    assert snapshot["version"] == 2 and snapshot["reasoning_effort"] == "low"
    assert original["version"] == 1 and original["reasoning_effort"] == "high"
