"""Named keep-testing vs LRU protocol. Numbers come from a real run."""

from __future__ import annotations

import inside_memory
import keep_testing_vs_lru as proto


def test_lru_evicts_oldest_last_access() -> None:
    atoms = proto.make_pack()
    for i, atom in enumerate(atoms):
        atom["lru_last"] = proto._shift_hours(proto.ENCODE_AT, i)
    held = proto.lru_window(atoms, budget=3)
    ids = [atom["id"] for atom in held]
    assert ids == ["a063", "a062", "a061"]
    incoming = dict(atoms[0])
    incoming["lru_last"] = proto._shift_hours(proto.ENCODE_AT, 100)
    after = proto.evict_lru(held, incoming, budget=3)
    after_ids = {atom["id"] for atom in after}
    assert "a000" in after_ids
    assert "a061" not in after_ids


def test_keep_testing_pins_queried_due_atom() -> None:
    atoms = proto.encode_keep_testing(proto.make_pack())
    clock = proto._shift_hours(proto.ENCODE_AT, 48)
    assert inside_memory.is_due(atoms[40], clock)
    window = proto.keep_testing_window(atoms, clock, query_id="a040", budget=8)
    assert window[0]["id"] == "a040"
    assert inside_memory.is_due(window[0], clock)


def test_schedule_review_is_the_keep_testing_clock() -> None:
    atom = proto.make_pack()[0]
    first = inside_memory.schedule_review(atom, now=proto.ENCODE_AT)
    later = proto._shift_hours(proto.ENCODE_AT, 24)
    assert inside_memory.is_due(first, later)
    stretched = inside_memory.schedule_review(first, now=later, recalled=True)
    assert stretched["due_at"] > first["due_at"]
    assert int(stretched["review"]["reps"]) == 1
    assert stretched.get("valid_to") is None


def test_protocol_keep_testing_beats_lru_on_long_tail() -> None:
    result = proto.run(seed=proto.SEED)
    assert result["protocol"] == proto.PROTOCOL
    assert result["n_tail"] == proto.N_ATOMS - proto.BUDGET
    assert result["lru_tail_hits"] < result["keep_testing_tail_hits"]
    assert result["delta_tail_hit_rate"] > 0.0
    assert 0.0 <= result["lru_tail_hit_rate"] <= 1.0
    assert 0.0 <= result["keep_testing_tail_hit_rate"] <= 1.0
    assert result["keep_testing_queue_retention"] == 1.0
    assert result["sha"]
    assert result["encode_at"] == proto.ENCODE_AT
    assert result["test_at"] == proto.test_at()
