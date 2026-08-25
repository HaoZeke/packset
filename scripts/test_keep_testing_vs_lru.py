"""Named keep-testing vs LRU protocol. Same pack, measured long-tail delta."""

from __future__ import annotations

import inside_memory
import keep_testing_vs_lru as proto


def test_protocol_name_and_sha() -> None:
    result = proto.run_protocol()
    assert result["protocol"] == "keep-testing-vs-lru-v1"
    assert len(result["sha"]) == 40
    int(result["sha"], 16)


def test_same_pack_both_arms() -> None:
    pack, head_ids, tail_ids = proto.build_pack(n_tail=8, schedule_at=proto.day_iso(0))
    ids = [atom["id"] for atom in pack]
    assert ids[:16] == head_ids
    assert ids[16:24] == tail_ids
    assert len(pack) == proto.N_TOTAL
    assert len(set(ids)) == proto.N_TOTAL
    for atom in pack:
        if atom["id"] in tail_ids:
            assert inside_memory.is_due(atom, proto.day_iso(1))
            assert atom.get("due_at")
        else:
            assert not inside_memory.is_due(atom, proto.day_iso(7))


def test_textbook_keep_testing_beats_lru() -> None:
    result = proto.run_split(proto.N_TAIL_TEXTBOOK)
    assert result["lru_tail_hit"] == 0.0
    assert result["keep_testing_tail_hit"] == 1.0
    assert result["delta"] == 1.0
    assert set(result["due_tail_at_probe"]) == set(result["tail_ids"])
    assert set(result["keep_testing_window"]) == set(result["tail_ids"])
    assert not set(result["lru_window"]) & set(result["tail_ids"])


def test_crowded_half_the_due_tail_fits() -> None:
    result = proto.run_split(proto.N_TAIL_CROWDED)
    assert result["lru_tail_hit"] == 0.0
    assert result["keep_testing_tail_hit"] == 0.5
    assert result["delta"] == 0.5
    assert len(result["due_tail_at_probe"]) == proto.N_TAIL_CROWDED
    assert len(result["keep_testing_window"]) == proto.BUDGET


def test_zipf_keep_testing_beats_lru() -> None:
    result = proto.run_zipf()
    assert result["delta"] > 0.0
    assert result["keep_testing_tail_hit"] > result["lru_tail_hit"]


def test_deterministic() -> None:
    a = proto.run_protocol()
    b = proto.run_protocol()
    assert a["textbook"] == b["textbook"]
    assert a["crowded"] == b["crowded"]
    assert a["zipf"] == b["zipf"]
    assert a["protocol"] == b["protocol"]
