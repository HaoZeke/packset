"""review-clock-3cell: FSRS vs always-include vs never-review."""

from __future__ import annotations

import inside_memory
import inside_recall
import review_clock_3cell as proto
from test_inside_recall import atom, crowd


def test_recall_now_controls_due_queue() -> None:
    atoms = crowd(70)
    due = atom(
        "due1",
        text="Review this lesson on the due clock.",
        trust=0.01,
        due_at="2026-08-08T00:00:00.000Z",
    )
    hot = atom("hot", text="High trust not due.", trust=9.0)
    atoms.extend([due, hot])
    before = inside_recall.recall(
        proto.WS,
        seeds=["hot"],
        atoms=atoms,
        limit=8,
        now="2026-08-07T00:00:00.000Z",
    )
    after = inside_recall.recall(
        proto.WS,
        seeds=["hot"],
        atoms=atoms,
        limit=8,
        now="2026-08-08T00:00:00.000Z",
    )
    assert "due1" not in {row["id"] for row in before}
    assert after[0]["id"] == "due1"


def test_retrievability_matches_schedule_formula() -> None:
    atom_rec = atom("r1", text="Keep-testing claim.")
    first = inside_memory.schedule_review(atom_rec, now="2026-08-01T00:00:00.000Z")
    assert inside_memory.retrievability(first, "2026-08-01T00:00:00.000Z") == 0.99
    week = inside_memory.retrievability(first, "2026-08-08T00:00:00.000Z")
    assert abs(week - (0.9**7)) < 1e-9


def test_three_cell_protocol_measures_all_arms() -> None:
    row = proto.run()
    assert row["protocol"] == "review-clock-3cell"
    assert set(row["cells"]) == {"fsrs", "always_include", "never_review"}
    fsrs = row["cells"]["fsrs"]
    always = row["cells"]["always_include"]
    never = row["cells"]["never_review"]
    for cell in (fsrs, always, never):
        assert isinstance(cell["week_later_mean_retrievability"], float)
        assert isinstance(cell["seat_later_target_splice_hit_rate"], float)
        assert isinstance(cell["target_test_trials"], int)
        assert len(cell["per_seat_hit_rate"]) == proto.SEAT_DAYS
    assert fsrs["week_later_mean_retrievability"] > always["week_later_mean_retrievability"]
    assert fsrs["week_later_mean_retrievability"] > never["week_later_mean_retrievability"]
    assert fsrs["seat_later_target_splice_hit_rate"] > always["seat_later_target_splice_hit_rate"]
    assert fsrs["seat_later_target_splice_hit_rate"] > never["seat_later_target_splice_hit_rate"]
    assert fsrs["target_test_trials"] > always["target_test_trials"]
    assert fsrs["target_test_trials"] > never["target_test_trials"]
    assert row["fsrs_beats_always_include"] is True
    assert row["fsrs_beats_never_review"] is True
    assert row["delta"]["retrievability_fsrs_minus_always_include"] > 0
    assert row["delta"]["retrievability_fsrs_minus_never_review"] > 0
    assert row["n_targets"] == 40
    assert row["seat_days"] == 7
    line = proto.measured_line(row)
    assert "R " in line
    assert "(FSRS/always/never)" in line
    assert f"{fsrs['week_later_mean_retrievability']:.6f}" in line
