"""Temporary MYT-95 catalogs shared by runtime routing contract tests."""
import hashlib
import json


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def build_catalog(tmp_path, *, profile="mayky-coding", model="claude-test", provider="claude-test", effort="high"):
    root = tmp_path / "catalog"
    home = tmp_path / "profiles" / profile
    home.mkdir(parents=True)
    restrictions = {"roles": {"coding": {"perfil": profile}}, "proveedores": {
        provider: {"modelos": [model], "esfuerzos": ["none", "low", "medium", "high", "max"]}}}
    catalog = root / "catalogo"
    path = catalog / "presets" / "balanced" / "v1.json"
    path.parent.mkdir(parents=True)
    (catalog / "restricciones.json").write_bytes(canonical(restrictions))
    document = {"preset": "balanced", "version": 1, "estado": "editada", "padre_sha256": None,
                "roles": {"coding": {"proveedor": provider, "modelo": model, "esfuerzo": effort}}}
    path.write_bytes(canonical(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (catalog / "indice.jsonl").write_text(json.dumps({"preset": "balanced", "version": 1, "sha256": digest}) + "\n")
    selection = root / "estado" / "seleccion" / "team.json"
    selection.parent.mkdir(parents=True)
    selection.write_bytes(canonical({"ambito": "team", "preset": "balanced", "version": 1, "sha256": digest}))
    config = {"team_model_preset": {"catalog_root": str(root), "scope": "team", "role": "coding", "enabled_after": "2026-10-15T00:00:00Z"}}
    return config, home, root
