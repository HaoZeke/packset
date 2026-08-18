#!/usr/bin/env python3
"""Tests for include-first recall. No daemon required."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inside_memory
import inside_recall

WS = "git:example.com/proj"


def _atom(ident, **fields):
    rec = {
        "id": ident,
        "workspace": WS,
        "kind": "habit",
        "text": f"Claim {ident}.",
        "trust": 1.0,
        "ts": "2026-08-11T12:00:00.000Z",
        "tombstone": False,
        "links": [],
        "valid_to": None,
    }
    rec.update(fields)
    return rec


def _crowd(n=70):
    return [_atom(f"x{i:03d}", text=f"Unrelated claim {i}.", trust=0.2) for i in range(n)]


class RecallTests(unittest.TestCase):
    def test_small_pack_returns_every_live_atom(self):
        atoms = [
            _atom("a", kind="voice", text="Speaks in short sentences."),
            _atom("b", kind="habit", text="Reviews open with a check."),
            _atom("c", kind="preference", text="Prefers Conventional Commits."),
        ]
        out = inside_recall.recall(WS, atoms=atoms)
        self.assertEqual({atom["id"] for atom in out}, {"a", "b", "c"})

    def test_small_pack_needs_no_seeds(self):
        atoms = [_atom("a"), _atom("b")]
        out = inside_recall.recall(WS, atoms=atoms, seeds=None, hints=None)
        self.assertEqual(len(out), 2)

    def test_large_pack_walks_one_hop(self):
        atoms = _crowd(70)
        seed = _atom(
            "seed",
            text="Action seed about JOSS.",
            links=["nbr", "x000"],
            trust=0.5,
        )
        neighbor = _atom("nbr", text="JOSS neighbour claim.", trust=0.4, links=["seed"])
        far = _atom("far", text="Far unlinked claim.", trust=0.99)
        atoms.extend([seed, neighbor, far])
        out = inside_recall.recall(WS, seeds=["seed"], atoms=atoms)
        ids = [atom["id"] for atom in out]
        self.assertIn("seed", ids)
        self.assertIn("nbr", ids)
        self.assertIn("x000", ids)
        self.assertNotIn("far", ids)
        self.assertLessEqual(len(out), 64)
        self.assertNotIn("x001", ids)

    def test_hints_select_seeds_by_text_and_entities(self):
        atoms = _crowd(70)
        seed = _atom(
            "seed",
            text="Reviews open on JOSS.",
            entities=["JOSS"],
            links=["nbr"],
        )
        neighbor = _atom("nbr", text="Linked review habit.", links=["seed"])
        other = _atom("other", text="Cache of github issues.", kind="cache-pointer")
        atoms.extend([seed, neighbor, other])
        by_text = inside_recall.recall(
            WS, hints={"user_text": "review this PR"}, atoms=atoms
        )
        self.assertEqual({atom["id"] for atom in by_text}, {"seed", "nbr"})
        by_ent = inside_recall.recall(
            WS, hints={"entities": ["JOSS"]}, atoms=atoms
        )
        self.assertEqual({atom["id"] for atom in by_ent}, {"seed", "nbr"})

    def test_tombstoned_and_expired_are_not_walked(self):
        atoms = _crowd(70)
        seed = _atom("seed", text="Live seed.", links=["dead", "stale", "nbr"])
        dead = _atom("dead", text="Tombstoned neighbour.", tombstone=True, links=["seed"])
        stale = _atom(
            "stale",
            text="Expired neighbour.",
            kind="cache-pointer",
            valid_to="2000-01-01T00:00:00.000Z",
            links=["seed"],
        )
        neighbor = _atom("nbr", text="Live neighbour.", links=["seed"])
        atoms.extend([seed, dead, stale, neighbor])
        out = inside_recall.recall(WS, seeds=["seed"], atoms=atoms)
        ids = {atom["id"] for atom in out}
        self.assertEqual(ids, {"seed", "nbr"})

    def test_tombstoned_seed_is_skipped(self):
        atoms = _crowd(70)
        dead = _atom("dead", text="Dead seed.", tombstone=True, links=["nbr"])
        neighbor = _atom("nbr", text="Would be a neighbour.", links=["dead"])
        atoms.extend([dead, neighbor])
        out = inside_recall.recall(WS, seeds=["dead"], atoms=atoms)
        self.assertEqual(out, [])

    def test_default_limit_is_64(self):
        atoms = _crowd(80)
        seed_links = [f"x{i:03d}" for i in range(80)]
        atoms.append(_atom("seed", text="Seed with many links.", links=seed_links))
        out = inside_recall.recall(WS, seeds=["seed"], atoms=atoms)
        self.assertLessEqual(len(out), 64)
        self.assertEqual(len(out), 64)
        huge = inside_recall.recall(WS, seeds=["seed"], atoms=atoms, limit=1000)
        self.assertEqual(len(huge), 64)

    def test_prefers_neighbours_then_trust_ts_id(self):
        atoms = _crowd(70)
        seed = _atom(
            "seed",
            text="Seed.",
            trust=0.1,
            ts="2026-08-01T00:00:00.000Z",
            links=["n1", "n2"],
        )
        n1 = _atom("n1", text="Neighbour one.", trust=0.5, ts="2026-08-10T00:00:00.000Z")
        n2 = _atom("n2", text="Neighbour two.", trust=0.9, ts="2026-08-09T00:00:00.000Z")
        atoms.extend([seed, n1, n2])
        out = inside_recall.recall(WS, seeds=["seed"], atoms=atoms)
        ids = [atom["id"] for atom in out]
        self.assertEqual(ids[:2], ["n2", "n1"])
        self.assertEqual(ids[2], "seed")

    def test_text_budget_caps_returned_text(self):
        long_a = _atom("a", text="A" * 20000, trust=1.0)
        long_b = _atom("b", text="B" * 20000, trust=0.9)
        long_c = _atom("c", text="C" * 20000, trust=0.8)
        out = inside_recall.recall(WS, atoms=[long_a, long_b, long_c])
        total = sum(len(atom.get("text") or "") for atom in out)
        self.assertLessEqual(total, inside_recall.TEXT_BUDGET)
        self.assertGreaterEqual(len(out), 1)
        self.assertLess(len(out), 3)

    def test_sort_is_trust_then_ts_then_id(self):
        atoms = [
            _atom("c", trust=0.5, ts="2026-08-11T00:00:00.000Z"),
            _atom("a", trust=0.9, ts="2026-08-01T00:00:00.000Z"),
            _atom("b", trust=0.9, ts="2026-08-10T00:00:00.000Z"),
            _atom("d", trust=0.9, ts="2026-08-10T00:00:00.000Z"),
        ]
        out = inside_recall.recall(WS, atoms=atoms)
        self.assertEqual([atom["id"] for atom in out], ["b", "d", "a", "c"])

    def test_home_load_skips_expired_and_tombstones(self):
        tmp = tempfile.TemporaryDirectory()
        home = Path(tmp.name)
        live = inside_memory.make_atom(
            workspace=WS,
            text="Live voice claim.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stale = inside_memory.make_atom(
            workspace=WS,
            text="Expired snapshot.",
            kind="cache-pointer",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to="2000-01-01T00:00:00.000Z",
        )
        stored = inside_memory.add_atom(live, home=home)
        inside_memory.add_atom(stale, home=home)
        dead = inside_memory.make_atom(
            workspace=WS,
            text="Habit that will be dropped.",
            kind="habit",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        dropped = inside_memory.add_atom(dead, home=home)
        inside_memory.delete_atom(WS, dropped["id"], home=home)
        out = inside_recall.recall(WS, home=home)
        ids = [atom["id"] for atom in out]
        self.assertEqual(ids, [stored["id"]])
        tmp.cleanup()

    def test_large_pack_without_seeds_returns_empty(self):
        out = inside_recall.recall(WS, atoms=_crowd(70))
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
