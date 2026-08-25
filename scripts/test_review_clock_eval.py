"""FSRS vs always-include vs never-review three-cell protocol."""

from __future__ import annotations

import json
from pathlib import Path

import inside_memory
import inside_recall
import review_clock_eval as eval_mod


def test_fixture_is_over_budget() -> None:
    pack = eval_mod.fixture()
    assert len(pack) == eval_mod.N_LESSONS + eval_mod.N_DISTRACTORS
    assert len(pack) > eval_mod.LIMIT
    assert eval_mod.N_DISTRACTORS > eval_mod.LIMIT


def test_recall_now_selects_due_at_that_instant() -> None:
    early = "2026-08-17T00:00:00.000Z"
    later = "2026-08-24T00:00:00.000Z"
    lesson = {
        "id": "due-later",
        "workspace": eval_mod.WS,
        "kind": "lesson",
        "text": "Review this lesson on the due clock.",
        "trust": 0.01,
        "ts": early,
        "tombstone": False,
        "links": [],
        "valid_to": None,
        "due_at": later,
    }
    crowd = eval_mod.fixture(early)
    atoms = crowd + [lesson]
    before = inside_recall.recall(eval_mod.WS, atoms=atoms, limit=8, now=early)
    after = inside_recall.recall(eval_mod.WS, atoms=atoms, limit=8, now=later)
    assert "due-later" not in {row["id"] for row in before}
    assert after[0]["id"] == "due-later"


def test_three_cells_and_fsrs_beats_baselines() -> None:
    report = eval_mod.run_protocol(sha="test")
    assert report["protocol"] == "review-clock-three-cell"
    assert set(report["cells"]) == {"fsrs", "always_include", "never_review"}
    fsrs = report["cells"]["fsrs"]
    always = report["cells"]["always_include"]
    never = report["cells"]["never_review"]
    assert always["keep_test_hits"] == 0
    assert never["keep_test_hits"] == 0
    assert always["keep_test_rate"] == 0.0
    assert never["keep_test_rate"] == 0.0
    assert fsrs["keep_test_hits"] >= eval_mod.N_LESSONS
    assert fsrs["keep_test_rate"] > always["keep_test_rate"]
    assert fsrs["keep_test_rate"] > never["keep_test_rate"]
    assert fsrs["week_later_R"] > always["week_later_R"]
    assert fsrs["week_later_R"] > never["week_later_R"]
    assert report["fsrs_beats_always_include"] is True
    assert report["fsrs_beats_never_review"] is True
    assert report["ranking"][0] == "fsrs"
    assert fsrs["keep_test_opportunities"] == eval_mod.N_LESSONS * eval_mod.KEEP_DAYS


def test_measured_line_names_all_three_rates() -> None:
    report = eval_mod.run_protocol(sha="test")
    line = eval_mod.measured_line(report)
    assert line.startswith("keep-test ")
    assert "(FSRS/always/never)" in line
    assert "week-later R" in line
    fsrs = report["cells"]["fsrs"]["keep_test_rate"]
    assert f"{fsrs:.3f}" in line
    week_r = report["cells"]["fsrs"]["week_later_R"]
    assert f"{week_r:.3f}" in line


def test_write_report_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "review-clock-three-cell.json"
    report = eval_mod.write_report(path, eval_mod.run_protocol(sha="test"))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["protocol"] == report["protocol"]
    assert loaded["cells"]["fsrs"]["keep_test_hits"] == report["cells"]["fsrs"]["keep_test_hits"]
    assert loaded["packset_sha"] == "test"


def test_first_test_does_not_close_live() -> None:
    pack = eval_mod.first_test(eval_mod.fixture(), eval_mod.DAY0)
    for aid in eval_mod.lesson_ids():
        atom = next(row for row in pack if row["id"] == aid)
        assert atom.get("valid_to") is None
        assert inside_memory.is_live(atom, eval_mod.DAY0)
        assert atom.get("due_at")
        assert int((atom.get("review") or {}).get("reps") or 0) == 1
