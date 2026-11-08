"""Review hand-off framing for the goal judge (Mayky MYT-248)."""

import hermes_cli.goals as goals
import tools.kanban_tools as kt


def test_handoff_note_reaches_judge_and_keeps_goal(monkeypatch):
    seen = {}

    def fake_judge(**kwargs):
        seen.update(kwargs)
        return "done", "ok", False, None, False

    monkeypatch.setattr(kt, "judge_goal", fake_judge)
    verdict = kt._judge_handoff("goal text", "evidence", kt._REVIEW_HANDOFF_GOAL_NOTE)
    assert verdict[0] == "done"
    assert seen["goal"] == "goal text"
    assert seen["handoff_note"] == kt._REVIEW_HANDOFF_GOAL_NOTE


def test_empty_note_calls_judge_without_it(monkeypatch):
    seen = {}

    def fake_judge(**kwargs):
        seen.update(kwargs)
        return "continue", "", False, None, False

    monkeypatch.setattr(kt, "judge_goal", fake_judge)
    kt._judge_handoff("goal", "evidence", "")
    assert "handoff_note" not in seen


def test_old_judge_signature_falls_back_without_note(monkeypatch):
    calls = []

    def old_judge(goal, last_response):
        calls.append((goal, last_response))
        return "blocked", "missing", False, None, False

    monkeypatch.setattr(kt, "judge_goal", old_judge)
    assert kt._judge_handoff("goal", "evidence", "note")[0] == "blocked"
    assert calls == [("goal", "evidence")]


def test_judge_prompt_prepends_note_and_preserves_goal(monkeypatch):
    prompts = []

    def fake_llm(_call_llm, _system, prompt, _timeout):
        prompts.append(prompt)
        return '{"done": false, "reason": "x"}'

    monkeypatch.setattr(goals, "_call_goal_judge_llm", fake_llm)
    goal = "G" * 1900
    goals.judge_goal(goal=goal, last_response="resp", handoff_note="[NOTE]")
    goals.judge_goal(goal=goal, last_response="resp")
    with_note, without = prompts
    assert with_note.startswith("[NOTE]\n\n")
    assert with_note[len("[NOTE]\n\n"):] == without
    assert goal in with_note
