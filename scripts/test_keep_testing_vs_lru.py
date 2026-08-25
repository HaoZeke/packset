"""Named keep-testing vs LRU protocol. Numbers come from a real run."""

from __future__ import annotations

import inside_memory
import keep_testing_vs_lru as proto


def test_protocol_name_and_sha() -> None:
    result = proto.run(seed=proto.SEED)
    assert result["protocol"] == "keep-testing-due_at-vs-lru-v2"
    assert len(result["sha"]) == 40
    int(result["sha"], 16)


def test_same_query_stream_both_arms() -> None:
    rng = proto.random.Random(proto.SEED)
    queries = proto.sample_queries(rng)
    result = proto.run(seed=proto.SEED)
    assert result["queries"] == queries
    assert len(queries) == proto.INTERFERENCE_HOURS
    assert set(queries) <= set(proto.atom_ids())


def test_schedule_review_stretch_on_success() -> None:
    atom = proto.make_pack()[40]
    first = inside_memory.schedule_review(atom, now=proto.ENCODE_AT)
    later = proto._shift_hours(proto.ENCODE_AT, 24)
    assert inside_memory.is_due(first, later)
    stretched = inside_memory.schedule_review(first, now=later, recalled=True)
    assert stretched["due_at"] > first["due_at"]
    assert int(stretched["review"]["reps"]) == 1
    assert stretched.get("valid_to") is None


def test_keep_testing_window_does_not_pin_query() -> None:
    pack = proto.encode_keep_testing(proto.make_pack())
    clock = proto._shift_hours(proto.ENCODE_AT, 24)
    tail_id = proto.tail_ids()[-1]
    window = proto.keep_testing_window(pack, now=clock, seeds=[tail_id], budget=proto.BUDGET)
    ids = [atom["id"] for atom in window]
    assert len(ids) == proto.BUDGET
    assert ids == proto.atom_ids()[: proto.BUDGET]
    assert tail_id not in ids


def test_lru_evicts_oldest() -> None:
    lru = proto.LruWindow(3)
    lru.access(["a000", "a001", "a002"])
    lru.access(["a003"])
    assert "a000" not in lru.ids()
    assert lru.ids() == ["a001", "a002", "a003"]


def test_same_budget_windows_and_real_stretch() -> None:
    result = proto.run(seed=proto.SEED)
    assert result["budget"] == proto.BUDGET
    assert len(result["lru_probe_window"]) == proto.BUDGET
    assert len(result["keep_testing_probe_window"]) <= proto.BUDGET
    assert len(result["keep_testing_last_window"]) <= proto.BUDGET
    assert result["stretch_n"] > 0
    assert result["keep_testing_store_n"] == proto.N_ATOMS
    assert result["lru_store_n"] == proto.BUDGET


def test_delta_is_not_due_over_budget_occupancy() -> None:
    result = proto.run(seed=proto.SEED)
    occupancy_8_8 = 1.0
    occupancy_16_8 = 0.5
    assert result["keep_testing_probe_tail_hit"] != occupancy_8_8
    assert result["keep_testing_probe_tail_hit"] != occupancy_16_8
    assert result["keep_testing_slot_tail"] != occupancy_16_8
    n_due = proto.N_ATOMS
    assert result["delta_slot_tail"] != n_due / proto.BUDGET


def test_keep_testing_beats_lru_on_long_tail() -> None:
    result = proto.run(seed=proto.SEED)
    assert result["delta_slot_tail"] > 0.0
    assert result["keep_testing_slot_tail"] > result["lru_slot_tail"]
    assert result["keep_testing_tail_coverage"] >= result["lru_tail_coverage"]


def test_deterministic() -> None:
    a = proto.run(seed=proto.SEED)
    b = proto.run(seed=proto.SEED)
    drop = {"measured_at"}
    assert {k: v for k, v in a.items() if k not in drop} == {
        k: v for k, v in b.items() if k not in drop
    }
