#!/usr/bin/env python3
"""Tests for shared identity and the memory pack."""
import subprocess
import tempfile
import unittest
from pathlib import Path

import inside_identity
import inside_memory


class IdentityTests(unittest.TestCase):
    def test_normalize_https_and_ssh(self):
        https = inside_identity.normalize_remote(
            "https://github.com/HaoZeke/grok-inside.git"
        )
        ssh = inside_identity.normalize_remote(
            "git@github.com:HaoZeke/grok-inside.git"
        )
        self.assertEqual(https, "git:github.com/HaoZeke/grok-inside")
        self.assertEqual(ssh, https)

    def test_same_workspace_for_two_clients(self):
        remote = "https://github.com/HaoZeke/joss-reviews.git"
        hermes = inside_identity.identity(
            harness="hermes", remote=remote, cwd=Path("/tmp/joss-reviews")
        )
        codex = inside_identity.identity(
            harness="codex", remote=remote, cwd=Path("/tmp/joss-reviews")
        )
        self.assertEqual(hermes["workspace"], codex["workspace"])
        self.assertEqual(hermes["workspace"], "git:github.com/HaoZeke/joss-reviews")
        self.assertEqual(hermes["agent_peer"], "hermes")
        self.assertEqual(codex["agent_peer"], "codex")

    def test_per_directory_and_global(self):
        root = Path("/tmp/some-tree").resolve()
        self.assertEqual(
            inside_identity.resolve_workspace(cwd=root, strategy="per-directory"),
            f"dir:{root}",
        )
        self.assertEqual(
            inside_identity.resolve_workspace(cwd=root, strategy="global"),
            "global",
        )

    def test_git_remote_from_repo(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@gitlab.com:me/proj.git"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        ident = inside_identity.identity(harness="pi", cwd=root)
        self.assertEqual(ident["workspace"], "git:gitlab.com/me/proj")
        tmp.cleanup()


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.ws = "git:github.com/HaoZeke/joss-reviews"

    def tearDown(self):
        self.tmp.cleanup()

    def test_user_overflow_is_an_error(self):
        with self.assertRaises(inside_memory.MemoryOverflow):
            inside_memory.set_user("x" * (inside_memory.USER_CAP + 1), home=self.home)

    def test_memory_overflow_is_an_error(self):
        with self.assertRaises(inside_memory.MemoryOverflow):
            inside_memory.set_memory(
                self.ws, "y" * (inside_memory.MEMORY_CAP + 1), home=self.home
            )

    def test_user_under_cap_round_trips(self):
        inside_memory.set_user("Prefers Conventional Commits.\n", home=self.home)
        self.assertIn(
            "Conventional Commits",
            inside_memory.read_text(inside_memory.user_path(self.home)),
        )

    def test_duplicate_add_is_noop(self):
        path = inside_memory.user_path(self.home)
        inside_memory.add_entry(path, "No thanks.", inside_memory.USER_CAP)
        inside_memory.add_entry(path, "No thanks.", inside_memory.USER_CAP)
        text = inside_memory.read_text(path)
        self.assertEqual(text.count("No thanks."), 1)

    def test_replace_needs_one_match(self):
        path = inside_memory.user_path(self.home)
        inside_memory.add_entry(path, "Alpha note", inside_memory.USER_CAP)
        inside_memory.add_entry(path, "Beta note", inside_memory.USER_CAP)
        with self.assertRaises(inside_memory.AtomError):
            inside_memory.replace_entry(path, "note", "Gamma", inside_memory.USER_CAP)
        inside_memory.replace_entry(path, "Alpha", "Gamma note", inside_memory.USER_CAP)
        self.assertIn("Gamma note", inside_memory.read_text(path))
        self.assertNotIn("Alpha note", inside_memory.read_text(path))

    def test_atom_add_update_delete(self):
        atom = inside_memory.make_atom(
            workspace=self.ws,
            text="Reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stored = inside_memory.add_atom(atom, home=self.home)
        again = inside_memory.add_atom(atom, home=self.home)
        self.assertEqual(stored["id"], again["id"])
        self.assertEqual(len(inside_memory.current_atoms(self.ws, self.home)), 1)

        updated = inside_memory.update_atom(
            self.ws,
            stored["id"],
            {"text": "Reviews open with scope and a reproducibility check."},
            home=self.home,
        )
        self.assertNotEqual(updated["text"], stored["text"])
        self.assertEqual(len(inside_memory.current_atoms(self.ws, self.home)), 1)

        inside_memory.delete_atom(self.ws, stored["id"], home=self.home)
        self.assertEqual(inside_memory.current_atoms(self.ws, self.home), [])
        log = inside_memory.load_atoms(self.ws, self.home)
        self.assertTrue(log[-1]["tombstone"])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(inside_memory.AtomError):
            inside_memory.make_atom(
                workspace=self.ws,
                text="nope",
                kind="vibes",
                about_peer="rgoswami",
                by_peer="pi",
            )

    def test_secret_text_rejected(self):
        with self.assertRaises(inside_memory.AtomError):
            inside_memory.set_user("api_key=sk-not-a-real-key", home=self.home)

    def test_english_secrets_is_not_a_credential(self):
        inside_memory.set_user(
            "Never commit secrets to git.\n", home=self.home
        )
        self.assertIn("secrets", inside_memory.read_text(inside_memory.user_path(self.home)))

    def test_expired_atom_absent_from_current(self):
        past = "2000-01-01T00:00:00.000Z"
        atom = inside_memory.make_atom(
            workspace=self.ws,
            text="Stale remote snapshot.",
            kind="cache-pointer",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to=past,
        )
        stored = inside_memory.add_atom(atom, home=self.home)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual(live, [])
        log = inside_memory.load_atoms(self.ws, self.home)
        self.assertEqual(log[-1]["id"], stored["id"])
        self.assertEqual(log[-1]["valid_to"], past)
        self.assertFalse(log[-1].get("tombstone"))

    def test_open_and_missing_valid_to_still_current(self):
        open_atom = inside_memory.make_atom(
            workspace=self.ws,
            text="Open validity claim.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to=None,
        )
        future = inside_memory.make_atom(
            workspace=self.ws,
            text="Still inside the validity window.",
            kind="habit",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to="2099-01-01T00:00:00.000Z",
        )
        bare = {
            "workspace": self.ws,
            "text": "No validity field at all.",
            "kind": "preference",
            "about_peer": "rgoswami",
            "by_peer": "pi",
        }
        inside_memory.add_atom(open_atom, home=self.home)
        inside_memory.add_atom(future, home=self.home)
        inside_memory.add_atom(bare, home=self.home)
        texts = {a["text"] for a in inside_memory.current_atoms(self.ws, self.home)}
        self.assertEqual(
            texts,
            {
                "Open validity claim.",
                "Still inside the validity window.",
                "No validity field at all.",
            },
        )

    def test_extract_entities_from_text(self):
        from_text = inside_memory.extract_entities(
            {"text": "See `AlphaRepo` and JOSS Reviews."}
        )
        self.assertIn("AlphaRepo", from_text)
        self.assertIn("JOSS", from_text)
        self.assertIn("Reviews", from_text)
        explicit = inside_memory.extract_entities(
            {"text": "ignored text", "entities": ["JOSS", "Reviews"]}
        )
        self.assertEqual(explicit, {"JOSS", "Reviews"})

    def test_entity_overlap_links_both_ways(self):
        first = inside_memory.make_atom(
            workspace=self.ws,
            text="JOSS reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        second = inside_memory.make_atom(
            workspace=self.ws,
            text="JOSS labels need a dated snapshot.",
            kind="habit",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        third = inside_memory.make_atom(
            workspace=self.ws,
            text="unrelated lowercase only.",
            kind="preference",
            about_peer="rgoswami",
            by_peer="pi",
            entities=["Unrelated"],
        )
        stored_a = inside_memory.add_atom(first, home=self.home)
        stored_b = inside_memory.add_atom(second, home=self.home)
        stored_c = inside_memory.add_atom(third, home=self.home)
        live = {a["id"]: a for a in inside_memory.current_atoms(self.ws, self.home)}
        self.assertIn(stored_b["id"], live[stored_a["id"]]["links"])
        self.assertIn(stored_a["id"], live[stored_b["id"]]["links"])
        self.assertEqual(live[stored_c["id"]]["links"], [])
        self.assertNotIn(stored_c["id"], live[stored_a["id"]]["links"])
        self.assertNotIn(stored_c["id"], live[stored_b["id"]]["links"])

    def test_new_claim_rewrites_old_live_links(self):
        first = inside_memory.make_atom(
            workspace=self.ws,
            text="AlphaRepo uses Conventional Commits.",
            kind="habit",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stored_a = inside_memory.add_atom(first, home=self.home)
        self.assertEqual(stored_a.get("links") or [], [])
        second = inside_memory.make_atom(
            workspace=self.ws,
            text="AlphaRepo reviews open with a check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stored_b = inside_memory.add_atom(second, home=self.home)
        live = {a["id"]: a for a in inside_memory.current_atoms(self.ws, self.home)}
        self.assertIn(stored_b["id"], live[stored_a["id"]]["links"])
        self.assertIn(stored_a["id"], live[stored_b["id"]]["links"])
        log = inside_memory.load_atoms(self.ws, self.home)
        a_versions = [rec for rec in log if rec["id"] == stored_a["id"]]
        self.assertGreaterEqual(len(a_versions), 2)
        self.assertIn(stored_b["id"], a_versions[-1]["links"])

    def test_tombstoned_atoms_leave_live_neighbourhoods(self):
        first = inside_memory.make_atom(
            workspace=self.ws,
            text="JOSS reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        second = inside_memory.make_atom(
            workspace=self.ws,
            text="JOSS labels need a dated snapshot.",
            kind="habit",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        stored_a = inside_memory.add_atom(first, home=self.home)
        stored_b = inside_memory.add_atom(second, home=self.home)
        inside_memory.delete_atom(self.ws, stored_a["id"], home=self.home)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["id"], stored_b["id"])
        self.assertNotIn(stored_a["id"], live[0].get("links") or [])

    def test_expired_atoms_are_not_in_live_graph(self):
        stale = inside_memory.make_atom(
            workspace=self.ws,
            text="Expired JOSS snapshot.",
            kind="cache-pointer",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to="2000-01-01T00:00:00.000Z",
            entities=["JOSS"],
        )
        live_atom = inside_memory.make_atom(
            workspace=self.ws,
            text="Live JOSS claim.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        inside_memory.add_atom(stale, home=self.home)
        stored_live = inside_memory.add_atom(live_atom, home=self.home)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual([a["id"] for a in live], [stored_live["id"]])
        self.assertEqual(live[0].get("links") or [], [])

    def test_cache_pointer_stores_timestamped_snapshot(self):
        stamp = "2026-08-11T12:00:00.000Z"
        ptr = inside_memory.make_cache_pointer(
            self.ws,
            f"open issues @ {stamp}: 3 labeled needs-review",
            valid_to="2099-01-01T00:00:00.000Z",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        self.assertEqual(ptr["kind"], "cache-pointer")
        self.assertIn(stamp, ptr["text"])
        stored = inside_memory.add_atom(ptr, home=self.home)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["id"], stored["id"])
        self.assertEqual(live[0]["kind"], "cache-pointer")

    def test_cache_pointer_live_only_while_valid_to_open(self):
        stale = inside_memory.make_cache_pointer(
            self.ws,
            "open issues @ 2000-01-01T00:00:00.000Z: 4 labeled needs-review",
            valid_to="2000-01-01T00:00:00.000Z",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        open_ptr = inside_memory.make_cache_pointer(
            self.ws,
            "open issues @ 2026-08-11T12:00:00.000Z: 2 labeled needs-review",
            valid_to="2099-01-01T00:00:00.000Z",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        inside_memory.add_atom(stale, home=self.home)
        stored = inside_memory.add_atom(open_ptr, home=self.home)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual([a["id"] for a in live], [stored["id"]])
        log = inside_memory.load_atoms(self.ws, self.home)
        self.assertEqual(len(log), 2)

    def test_cache_pointer_refresh_updates_text_and_validity(self):
        ptr = inside_memory.make_cache_pointer(
            self.ws,
            "open issues @ 2026-08-01T00:00:00.000Z: 4 labeled needs-review",
            valid_to="2099-01-01T00:00:00.000Z",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stored = inside_memory.add_atom(ptr, home=self.home)
        new_from = inside_memory.utcnow()
        new_to = "2099-06-01T00:00:00.000Z"
        new_text = "open issues @ 2026-08-11T12:00:00.000Z: 2 labeled needs-review"
        refreshed = inside_memory.update_atom(
            self.ws,
            stored["id"],
            {
                "text": new_text,
                "valid_from": new_from,
                "valid_to": new_to,
            },
            home=self.home,
        )
        self.assertEqual(refreshed["id"], stored["id"])
        self.assertEqual(refreshed["kind"], "cache-pointer")
        self.assertEqual(refreshed["text"], new_text)
        self.assertEqual(refreshed["valid_from"], new_from)
        self.assertEqual(refreshed["valid_to"], new_to)
        live = inside_memory.current_atoms(self.ws, self.home)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["text"], new_text)

    def test_tombstones_still_disappear_from_current(self):
        atom = inside_memory.make_atom(
            workspace=self.ws,
            text="Reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stored = inside_memory.add_atom(atom, home=self.home)
        inside_memory.delete_atom(self.ws, stored["id"], home=self.home)
        self.assertEqual(inside_memory.current_atoms(self.ws, self.home), [])
        log = inside_memory.load_atoms(self.ws, self.home)
        self.assertTrue(log[-1]["tombstone"])

    def test_hermes_home_view_is_write_through(self):
        inside_memory.set_user("No thanks.\n", home=self.home)
        inside_memory.set_memory(self.ws, "Read paper.pdf first.\n", home=self.home)
        isolated = self.home / "hermes-inside"
        views = inside_memory.install_home_view(
            isolated,
            layout="hermes",
            workspace=self.ws,
            pack_home=self.home,
        )
        self.assertTrue(views["user"].is_symlink())
        self.assertTrue(views["memory"].is_symlink())
        self.assertEqual(views["user"].read_text(encoding="utf-8"), "No thanks.\n")
        self.assertEqual(
            views["memory"].read_text(encoding="utf-8"), "Read paper.pdf first.\n"
        )
        views["memory"].write_text("Updated via Hermes.\n", encoding="utf-8")
        self.assertEqual(
            inside_memory.read_text(inside_memory.memory_path(self.ws, self.home)),
            "Updated via Hermes.\n",
        )
        sqlite = isolated / "memory.sqlite"
        lmdb_dir = isolated / "memory.lmdb"
        atoms = isolated / "memories" / "atoms.jsonl"
        self.assertFalse(sqlite.exists())
        self.assertFalse(lmdb_dir.exists())
        self.assertFalse(atoms.exists())

    def test_pi_home_view_writes_agents_snapshot(self):
        inside_memory.set_user("Be brief.\n", home=self.home)
        inside_memory.set_memory(self.ws, "Open with a check.\n", home=self.home)
        isolated = self.home / "pi-inside"
        views = inside_memory.install_home_view(
            isolated,
            layout="pi",
            workspace=self.ws,
            pack_home=self.home,
        )
        self.assertTrue(views["user"].is_symlink())
        self.assertTrue(views["memory"].is_symlink())
        agents = views["agents"].read_text(encoding="utf-8")
        self.assertIn("Seat memory (view)", agents)
        self.assertNotIn("Be brief.", agents)
        self.assertNotIn("Open with a check.", agents)
        self.assertIn("USER.md", agents)
        self.assertFalse((isolated / "memory.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
