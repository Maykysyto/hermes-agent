"""Respawn guard vs a rate-limit ``last_failure_error`` stamp that a later run superseded."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

STAMP = "pid 1 exited rate-limited (quota wall) — requeued without counting a failure"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and a frozen clock."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    kb.init_db()
    return home


@pytest.fixture
def now(monkeypatch):
    clock = [5_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    return clock


def _rate_limited_then_handoff(conn, now):
    """A rate_limited run requeued the way ``_reap_dead_workers`` does it, then a
    retry that hands off to review. Returns the task id, left in ``review``."""
    tid = kb.create_task(conn, title="rl-then-handoff", assignee="impl")
    kb.claim_task(conn, tid)
    kb._end_run(conn, tid, outcome="rate_limited", status="rate_limited", error=STAMP)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, last_failure_error=? WHERE id=?",
        (STAMP, tid),
    )
    conn.commit()
    now[0] += 600
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="ready", reviewer="rev",
                             expected_run_id=claimed.current_run_id)
    assert kb.get_task(conn, tid).last_failure_error == STAMP
    return tid


def test_review_handoff_after_rate_limit_reaches_the_reviewer(kanban_home, now):
    """#119070: the stamp nothing clears must not park the review lane as
    ``blocker_auth`` once a later run superseded the rate-limited one."""
    with kbc.connect() as conn:
        tid = _rate_limited_then_handoff(conn, now)
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_changes_requested_after_rate_limit_reaches_the_implementer(kanban_home, now):
    with kbc.connect() as conn:
        tid = _rate_limited_then_handoff(conn, now)
        now[0] += 60
        assert kb.claim_review_task(conn, tid) is not None
        ok, _ = kb.request_changes(conn, tid, reason="fix the edge case")
        assert ok and kb.get_task(conn, tid).status == "ready"
        now[0] += 7200  # past the recent_success window
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


@pytest.mark.parametrize("delta", [0, 1])
def test_auth_spawn_failure_after_review_handoff_still_blocks(kanban_home, now, delta):
    """An auth failure after the handoff is a live blocker, also when both runs
    end in the same second (reviewer probe H1, t_b1258251)."""
    with kbc.connect() as conn:
        tid = _rate_limited_then_handoff(conn, now)
        now[0] += delta
        assert kb.claim_review_task(conn, tid) is not None
        assert not kbd._record_task_failure(
            conn, tid, "provider authentication failed", outcome="spawn_failed",
            failure_limit=3, release_claim=True, end_run=True)
        assert kb.get_task(conn, tid).status == "review"
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"


def test_acceptance_auth_error_without_a_new_run_still_blocks(kanban_home, now, monkeypatch):
    """``record_acceptance`` writes a fresh error without closing a run: it is
    newer than the good handoff, not a stale stamp (reviewer probe H2, t_b1258251)."""
    from hermes_cli import kanban_pr_acceptance_store as store

    receipt = {"ok": False, "classification": "auth", "head_sha": None, "checks": [],
               "detail": "GitHub refused the acceptance read (HTTP 401)",
               "recovery": "Fix the assigned profile credentials and retry completion."}
    monkeypatch.setattr(store, "collect_acceptance", lambda *a, **kw: dict(receipt))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="acceptance auth", assignee="impl",
                             completion_contract="example/project")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary="ready", reviewer="rev",
                                 expected_run_id=claimed.current_run_id)
        now[0] += 60
        assert not kb.complete_task(
            conn, tid, summary="human approval",
            metadata={"published_pr": "https://github.com/example/project/pull/1"})
        assert kb.get_task(conn, tid).status == "review"
        assert len(kb.list_runs(conn, tid)) == 1
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"
