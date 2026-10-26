"""A review-lane run must not inherit the implementer's per-card model override.

``model_override``/``provider_override`` pin the backend of the profile that
implements the card. ``request_review(reviewer=...)`` hands the SAME card to a
different profile, and ``_worker_argv`` used to pass the card's ``-m/--provider``
to whichever worker claimed it — so the reviewer silently ran on the producer's
model and "independent review" became the producer grading itself.

Contract pinned here:

* review-lane spawn for a reviewer that is not the implementer: no ``-m`` /
  ``--provider`` (the reviewer resolves its own profile/preset model);
* the card row keeps its override, so the next implementation round (after
  ``request_changes``) runs on the implementer's model again;
* a card assigned to the reviewer from the start (its own pinned model) and a
  review by the implementing profile itself keep the override.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

OPUS = ("claude-opus-5-5[1m]", "claude-subscription-directsdk-experimental")


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}})
    monkeypatch.setattr(kbd, "check_respawn_guard", lambda _c, _t, **_k: None)
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    return home


def _spawned_argv(conn):
    argvs: list[tuple[str, list[str]]] = []

    def spawn(task, workspace):
        argvs.append((task.assignee, kbd._worker_argv(task, task.assignee, None)))

    kbd.dispatch_once(conn, spawn_fn=spawn)
    return argvs


def _route(argv: list[str]):
    argv = argv[argv.index("-p"):]
    model = argv[argv.index("-m") + 1] if "-m" in argv else None
    provider = argv[argv.index("--provider") + 1] if "--provider" in argv else None
    return model, provider


def test_reviewer_does_not_inherit_implementer_override(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="coder",
                             model_override=OPUS[0], provider_override=OPUS[1])
        [(who, argv)] = _spawned_argv(conn)
        assert (who, _route(argv)) == ("coder", OPUS)
        run = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(conn, tid, summary="done", reviewer="reviewer", expected_run_id=run)

        [(who, argv)] = _spawned_argv(conn)
        assert (who, _route(argv)) == ("reviewer", (None, None))
        task = kb.get_task(conn, tid)
        assert (task.model_override, task.provider_override) == OPUS   # row untouched

        ok, implementer = kb.request_changes(conn, tid, reason="fix R1", expected_run_id=task.current_run_id)
        assert (ok, implementer) == (True, "coder")
        [(who, argv)] = _spawned_argv(conn)
        assert (who, _route(argv)) == ("coder", OPUS)


def test_own_pinned_reviewer_card_and_self_review_keep_override(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        own = kb.create_task(conn, title="review gate", assignee="reviewer",
                             model_override="gpt-6-astra", provider_override="openai-codex")
        [(who, argv)] = _spawned_argv(conn)
        assert (who, _route(argv)) == ("reviewer", ("gpt-6-astra", "openai-codex"))
        kb.complete_task(conn, own, result="ok", expected_run_id=kb.get_task(conn, own).current_run_id)

        selfrev = kb.create_task(conn, title="impl", assignee="coder",
                                 model_override=OPUS[0], provider_override=OPUS[1])
        _spawned_argv(conn)
        run = kb.get_task(conn, selfrev).current_run_id
        assert kb.request_review(conn, selfrev, summary="done", expected_run_id=run)   # no reassign
        [(who, argv)] = _spawned_argv(conn)
        assert (who, _route(argv)) == ("coder", OPUS)
