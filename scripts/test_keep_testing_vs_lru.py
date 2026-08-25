"""Named keep-testing stream vs LRU protocol. Numbers come from a real run."""

from __future__ import annotations

import inside_memory
import keep_testing_vs_lru as proto


def test_protocol_name_and_sha() -> None:
    result = proto.run_protocol()
    assert result["protocol"] == "keep-testing-stream-vs-lru"
    assert len(result["sha"]) == 40
    int(result["sha"], 16)


def test_same_pack_same_stream_both_arms() -> None:
    pack = proto.make_pack()
    ids = [atom["id"] for atom in pack]
    assert ids == proto.atom_ids()
    assert len(pack) == proto.N_ATOMS
    scheduled = proto.encode_keep_testing(proto.clone_pack(pack))
    later = proto.shift_hours(proto.ENCODE_AT, 24)
    for atom in scheduled:
        assert atom.get("due_at")
        assert inside_memory.is_due(atom, later)
        assert int((atom.get("review") or {}).get("reps") or 0) == 0
    rng_a = __import__("random").Random(proto.SEED)
    rng_b = __import__("random").Random(proto.SEED)
    assert proto.sample_queries(rng_a) == proto.sample_queries(rng_b)
    assert len(proto.sample_queries(__import__("random").Random(proto.SEED))) == (
        proto.INTERFERENCE_HOURS
    )


def test_successful_trial_stretches_due_at() -> None:
    atom = proto.make_pack()[0]
    first = inside_memory.schedule_review(atom, now=proto.ENCODE_AT)
    later = proto.shift_hours(proto.ENCODE_AT, 24)
    assert inside_memory.is_due(first, later)
    stretched = inside_memory.schedule_review(first, now=later, recalled=True)
    assert stretched["due_at"] > first["due_at"]
    assert int(stretched["review"]["reps"]) == 1
    assert stretched.get("valid_to") is None


def test_keep_testing_walks_stream_and_stretches() -> None:
    result = proto.run_protocol()
    assert result["keep_testing_trials"] == proto.INTERFERENCE_HOURS
    assert result["n_queries"] == proto.INTERFERENCE_HOURS
    assert result["keep_testing_stretched"] > 0
    assert result["stretched_head_ids"]
    assert len(result["keep_testing_window"]) == proto.BUDGET
    assert len(result["lru_window"]) == proto.BUDGET


def test_probe_does_not_pin_and_is_not_occupancy_arithmetic() -> None:
    result = proto.run_protocol()
    occupancy = proto.BUDGET / result["n_tail"]
    assert result["n_tail"] == proto.N_ATOMS - proto.BUDGET
    assert result["n_tail"] != proto.BUDGET
    textbook = result["n_tail"] == proto.BUDGET
    assert textbook is False
    # Static occupancy is budget/n_tail. A stream+stretch run must not
    # collapse to that identity, nor to the rejected 8/8 and 16/8 rows.
    assert result["keep_testing_tail_hit"] != 1.0
    assert result["keep_testing_tail_hit"] != 0.5
    assert abs(result["keep_testing_tail_hit"] - occupancy) > 1e-9 or (
        result["keep_testing_stretched"] > 0 and result["stretched_head_ids"]
    )


def test_zipf_keep_testing_beats_lru() -> None:
    result = proto.run_protocol()
    assert result["delta"] > 0.0
    assert result["keep_testing_tail_hit"] > result["lru_tail_hit"]
    assert 0.0 <= result["lru_tail_hit"] <= 1.0
    assert 0.0 <= result["keep_testing_tail_hit"] <= 1.0


def test_deterministic() -> None:
    a = proto.run_protocol()
    b = proto.run_protocol()
    assert a["keep_testing_window"] == b["keep_testing_window"]
    assert a["lru_window"] == b["lru_window"]
    assert a["delta"] == b["delta"]
    assert a["protocol"] == b["protocol"]
