#!/usr/bin/env python3
"""Ranked pack search. No Meilisearch process."""
import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import inside_memory
import inside_search


@contextlib.contextmanager
def _panel_env(**values: str):
    keys = (
        inside_search.ENV_FUSE,
        inside_search.ENV_DIVERSIFY,
        inside_search.ENV_DECAY,
    )
    prev = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key in keys:
            if prev[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev[key]


class SearchTests(unittest.TestCase):
    def test_empty_query(self):
        self.assertEqual(inside_search.search_pack({"user": "hi", "atoms": []}, ""), [])

    def test_user_and_memory_hit(self):
        pack = {
            "user": "No thanks. Be brief.\n",
            "memory": "Read paper.pdf first.\n",
            "atoms": [],
        }
        hits = inside_search.search_pack(pack, "brief")
        fields = [h["field"] for h in hits]
        self.assertIn("user", fields)
        hits = inside_search.search_pack(pack, "paper")
        self.assertEqual(hits[0]["field"], "memory")

    def test_live_atom_prefix_and_typo(self):
        atom = inside_memory.make_atom(
            workspace="git:ex/p",
            text="JOSS reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
            entities=["JOSS"],
        )
        pack = {"user": "", "memory": "", "atoms": [atom]}
        exact = inside_search.search_pack(pack, "joss")
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["id"], atom["id"])
        typo = inside_search.search_pack(pack, "reproducibility")
        self.assertEqual(typo[0]["id"], atom["id"])
        typo = inside_search.search_pack(pack, "reproducability")
        self.assertEqual(typo[0]["id"], atom["id"])

    def test_expired_atom_absent(self):
        stale = inside_memory.make_atom(
            workspace="git:ex/p",
            text="JOSS stale snapshot.",
            kind="cache-pointer",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to="2000-01-01T00:00:00.000Z",
        )
        pack = {"user": "", "memory": "", "atoms": [stale]}
        self.assertEqual(inside_search.search_pack(pack, "joss"), [])

    def test_pack_documents_skips_dead_atoms(self):
        live = inside_memory.make_atom(
            workspace="git:ex/p",
            text="JOSS reviews.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        stale = inside_memory.make_atom(
            workspace="git:ex/p",
            text="JOSS stale snapshot.",
            kind="cache-pointer",
            about_peer="rgoswami",
            by_peer="hermes",
            valid_to="2000-01-01T00:00:00.000Z",
        )
        docs = inside_search.pack_documents(
            {
                "workspace": "git:ex/p",
                "user": "Be brief.",
                "memory": "Read paper.pdf first.",
                "atoms": [live, stale],
            }
        )
        ids = {doc["id"] for doc in docs}
        self.assertIn("user", ids)
        self.assertIn(inside_search.document_id("memory", workspace="git:ex/p"), ids)
        self.assertTrue(
            inside_search.document_id("memory", workspace="git:ex/p").replace("_", "").isalnum()
        )
        self.assertIn(live["id"], ids)
        self.assertNotIn(stale["id"], ids)

    def test_linear_when_milli_absent(self):
        pack = {"workspace": "git:ex/p", "user": "Be brief.", "memory": "", "atoms": []}
        hits, engine = inside_search.search_pack_with_engine(pack, "brief")
        if inside_search.milli_bin() is None:
            self.assertEqual(engine, "linear")
        self.assertTrue(any(h["field"] == "user" for h in hits))

    def test_short_query_is_exact_not_prefix(self):
        pack = {
            "user": "Prefers Conventional Commits.\n",
            "memory": "",
            "atoms": [],
        }
        self.assertEqual(inside_search.search_pack(pack, "pr"), [])
        self.assertEqual(inside_search.search_pack(pack, "prefers")[0]["field"], "user")

    def test_stopwords_do_not_hit(self):
        pack = {
            "user": "The review queue is the bottleneck.\n",
            "memory": "",
            "atoms": [],
        }
        self.assertEqual(inside_search.search_pack(pack, "What color is the sky?"), [])
        self.assertEqual(inside_search.search_pack(pack, "review")[0]["field"], "user")

    def test_tombstone_absent(self):
        atom = inside_memory.make_atom(
            workspace="git:ex/p",
            text="JOSS reviews.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        atom["tombstone"] = True
        pack = {"user": "", "memory": "", "atoms": [atom]}
        self.assertEqual(inside_search.search_pack(pack, "joss"), [])

    def test_set_filter_on_full_atom_list(self):
        review = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Remember the zircon latch on every review.",
            kind="lesson",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="review",
        )
        debug = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Zircon latch belongs in the debug notebook.",
            kind="habit",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="debug",
        )
        pack = {
            "workspace": "git:ex/p",
            "user": "Open a review with the defect.\n",
            "memory": "",
            "atoms": [review, debug],
        }
        unscoped = inside_search.search_pack_linear(pack, "zircon")
        unscoped_ids = {h["id"] for h in unscoped if h.get("field") == "atom"}
        self.assertEqual(unscoped_ids, {review["id"], debug["id"]})

        review_hits = inside_search.search_pack(
            pack, "zircon", set_name="review"
        )
        review_ids = [h["id"] for h in review_hits if h.get("field") == "atom"]
        self.assertEqual(review_ids, [review["id"]])
        self.assertNotIn(debug["id"], review_ids)

        prose = inside_search.search_pack(pack, "defect", set_name="review")
        self.assertTrue(any(h["field"] == "user" for h in prose))
        self.assertTrue(
            any("defect" in (h.get("text") or "").lower() for h in prose)
        )

        doc = inside_search.atom_document(review)
        self.assertEqual(doc["set"], "review")
        self.assertEqual(inside_search.atom_document(debug)["set"], "debug")

    def test_reindex_atoms_skips_user_memory_docs(self):
        review = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Remember the zircon latch on every review.",
            kind="lesson",
            about_peer="user",
            by_peer="user",
            set_name="review",
        )
        pack = {
            "workspace": "git:ex/p",
            "set": "review",
            "user": "SETCARDONLYPHRASE for the pin.\n",
            "memory": "SETMEMORYPHRASE for the pin.\n",
            "atoms": [review],
        }
        docs = [
            inside_search.atom_document(a)
            for a in inside_search._live_atoms(pack)
            if a.get("id")
        ]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["field"], "atom")
        self.assertNotIn("user", {d["field"] for d in docs})
        # pack_documents would include set cards as workspace prose — never use that for set packs.
        full = inside_search.pack_documents(pack)
        self.assertIn("user", {d["field"] for d in full})

    def test_set_miss_does_not_poison_workspace_prose_in_index(self):
        if inside_search.milli_bin() is None:
            self.skipTest("inside-milli binary not available")
        review = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Remember the zircon latch on every review.",
            kind="lesson",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="review",
        )
        debug = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Zircon latch belongs in the debug notebook.",
            kind="habit",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="debug",
        )
        workspace_pack = {
            "workspace": "git:ex/p",
            "user": "Workspace prose mentions UNIQUEWORKSPACEPHRASE only.\n",
            "memory": "",
            "atoms": [review, debug],
        }
        set_pack = {
            "workspace": "git:ex/p",
            "set": "review",
            "user": "Set card mentions UNIQUESETPHRASE only.\n",
            "memory": "",
            "atoms": [review, debug],
        }
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "memory.milli"
            self.assertTrue(inside_search.replace_index(workspace_pack, index))
            # Pin miss / set prose hit: must not rewrite workspace user docs.
            set_hits = inside_search.search_pack(
                set_pack,
                "UNIQUESETPHRASE",
                index_dir=index,
                set_name="review",
            )
            self.assertTrue(
                any(
                    h.get("field") == "user"
                    and "UNIQUESETPHRASE" in (h.get("text") or "")
                    for h in set_hits
                )
            )
            # Empty atom path under pin (query with no atom match) also safe.
            inside_search.search_pack(
                set_pack,
                "nomatchtokenxyz",
                index_dir=index,
                set_name="review",
            )
            raw = inside_search._run_milli(
                [
                    "search",
                    "--index",
                    str(index),
                    "--q",
                    "UNIQUESETPHRASE",
                    "--workspace",
                    "git:ex/p",
                    "--limit",
                    "16",
                ]
            )
            self.assertIsNotNone(raw)
            for hit in raw.get("hits") or []:
                if hit.get("field") == "user":
                    self.assertNotIn(
                        "UNIQUESETPHRASE",
                        hit.get("text") or "",
                        "set card prose must not become the workspace user document",
                    )
            unscoped = inside_search.search_pack(
                workspace_pack,
                "UNIQUEWORKSPACEPHRASE",
                index_dir=index,
            )
            self.assertTrue(
                any(
                    h.get("field") == "user"
                    and "UNIQUEWORKSPACEPHRASE" in (h.get("text") or "")
                    for h in unscoped
                )
            )

    def test_stale_index_without_set_field_still_scopes(self):
        """Pack atoms own set membership even when projection docs omit set."""
        if inside_search.milli_bin() is None:
            self.skipTest("inside-milli binary not available")
        review = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Remember the zircon latch on every review.",
            kind="lesson",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="review",
        )
        debug = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Zircon latch belongs in the debug notebook.",
            kind="habit",
            about_peer="user",
            by_peer="user",
            entities=["zircon"],
            set_name="debug",
        )
        pack = {
            "workspace": "git:ex/p",
            "user": "",
            "memory": "",
            "atoms": [review, debug],
        }
        # Old projection: atom docs without set field.
        stale_docs = []
        for atom in (review, debug):
            doc = inside_search.atom_document(atom)
            doc.pop("set", None)
            stale_docs.append(doc)
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "memory.milli"
            payload = inside_search._run_milli(
                ["index", "--index", str(index), "--replace"],
                stdin="\n".join(json.dumps(d) for d in stale_docs) + "\n",
                timeout=60.0,
            )
            self.assertIsNotNone(payload)
            hits = inside_search.search_pack(
                pack, "zircon", index_dir=index, set_name="review"
            )
            atom_ids = [h["id"] for h in hits if h.get("field") == "atom"]
            self.assertIn(review["id"], atom_ids)
            self.assertNotIn(debug["id"], atom_ids)

    def test_borda_two_lists_of_three(self):
        left = [("f", "x"), ("f", "y"), ("f", "z")]
        right = [("f", "y"), ("f", "x"), ("f", "z")]
        self.assertEqual(inside_search.borda_merge([left, right], 3), left)
        left = [("f", "a"), ("f", "b"), ("f", "c")]
        right = [("f", "b"), ("f", "c"), ("f", "a")]
        self.assertEqual(
            inside_search.borda_merge([left, right], 3),
            [("f", "b"), ("f", "a"), ("f", "c")],
        )

    def test_rrf_two_lists_of_three(self):
        left = [("f", "x"), ("f", "y"), ("f", "z")]
        right = [("f", "y"), ("f", "x"), ("f", "z")]
        self.assertEqual(inside_search.rrf_merge([left, right], 60), left)

    def test_rrf_omitted_rank_lifts_z_above_borda_last(self):
        left = [("f", "x"), ("f", "y"), ("f", "z")]
        right = [("f", "y"), ("f", "x"), ("f", "z")]
        only_z = [("f", "z")]
        self.assertEqual(
            inside_search.borda_merge([left, right, only_z], 3),
            left,
        )
        self.assertEqual(
            inside_search.rrf_merge([left, right, only_z], 60)[0],
            ("f", "z"),
        )

    def test_dowdall_two_lists_of_three(self):
        left = [("f", "a"), ("f", "b"), ("f", "c")]
        right = [("f", "b"), ("f", "c"), ("f", "a")]
        self.assertEqual(
            inside_search.borda_merge([left, right], 3),
            [("f", "b"), ("f", "a"), ("f", "c")],
        )
        ranked, scores = inside_search.dowdall_scores([left, right], 3)
        self.assertEqual(ranked, [("f", "b"), ("f", "a"), ("f", "c")])
        self.assertAlmostEqual(scores[("f", "a")], 1.0 + 1.0 / 3.0)
        self.assertAlmostEqual(scores[("f", "b")], 0.5 + 1.0)
        self.assertAlmostEqual(scores[("f", "c")], 1.0 / 3.0 + 0.5)

    def test_dowdall_last_place_pile_keeps_first_place(self):
        first = [("f", "a"), ("f", "c"), ("f", "d"), ("f", "e"), ("f", "z")]
        second = [("f", "b"), ("f", "c"), ("f", "d"), ("f", "e"), ("f", "z")]
        third = [("f", "a"), ("f", "b"), ("f", "c"), ("f", "d"), ("f", "z")]
        self.assertEqual(
            inside_search.borda_merge([first, second, third], 5)[0],
            ("f", "c"),
        )
        self.assertEqual(
            inside_search.dowdall_merge([first, second, third], 5),
            [
                ("f", "a"),
                ("f", "b"),
                ("f", "c"),
                ("f", "d"),
                ("f", "z"),
                ("f", "e"),
            ],
        )

    def test_kemeny_two_lists_first_seen(self):
        left = [("f", "a"), ("f", "b"), ("f", "c")]
        right = [("f", "b"), ("f", "a"), ("f", "c")]
        self.assertEqual(inside_search.kemeny_merge([left, right], 3), left)
        left = [("f", "b"), ("f", "a"), ("f", "c")]
        right = [("f", "a"), ("f", "b"), ("f", "c")]
        self.assertEqual(inside_search.kemeny_merge([left, right], 3), left)

    def test_kemeny_cycle_of_three_pins_first_seen(self):
        first = [("f", "a"), ("f", "b"), ("f", "c")]
        second = [("f", "b"), ("f", "c"), ("f", "a")]
        third = [("f", "c"), ("f", "a"), ("f", "b")]
        self.assertEqual(
            inside_search.kemeny_merge([first, second, third], 3), first
        )

    def test_merge_is_borda_not_primary_dedupe(self):
        primary = [
            {"field": "atom", "id": "a", "text": "alpha one", "score": 1.0},
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.5},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.1},
        ]
        secondary = [
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.9},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.4},
            {"field": "atom", "id": "a", "text": "alpha one", "score": 0.2},
        ]
        with _panel_env():
            ids = [h["id"] for h in inside_search._merge_hits(primary, secondary, 3)]
        self.assertEqual(ids[0], "b")
        self.assertEqual(set(ids), {"a", "b", "c"})

    def test_mmr_after_borda_splits_near_duplicates(self):
        items = [
            (("f", "keep"), 1.0, {"review", "open", "repro"}),
            (("f", "dup"), 0.55, {"review", "open", "repro"}),
            (("f", "other"), 0.5, {"pin", "zircon", "index"}),
        ]
        order = inside_search.mmr_rerank(items, 0.7)
        self.assertEqual(order[0], ("f", "keep"))
        self.assertEqual(order[1], ("f", "other"))
        self.assertEqual(order[2], ("f", "dup"))

    def test_default_panel_matches_borda_then_mmr(self):
        primary = [
            {"field": "atom", "id": "a", "text": "alpha one", "score": 1.0},
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.5},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.1},
        ]
        secondary = [
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.9},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.4},
            {"field": "atom", "id": "a", "text": "alpha one", "score": 0.2},
        ]
        with _panel_env():
            self.assertEqual(inside_search.resolve_panel(), ("borda", "mmr", "off"))
            implicit = [
                h["id"] for h in inside_search._merge_hits(primary, secondary, 3)
            ]
            named = [
                h["id"]
                for h in inside_search._merge_hits(
                    primary, secondary, 3, fuse="borda", diversify="mmr"
                )
            ]
        self.assertEqual(implicit, named)
        self.assertEqual(implicit[0], "b")
        self.assertEqual(set(implicit), {"a", "b", "c"})

    def test_diversify_none_keeps_borda_order(self):
        primary = [
            {"field": "atom", "id": "a", "text": "alpha one", "score": 1.0},
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.5},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.1},
        ]
        secondary = [
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.9},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.4},
            {"field": "atom", "id": "a", "text": "alpha one", "score": 0.2},
        ]
        ids = [
            h["id"]
            for h in inside_search._merge_hits(
                primary, secondary, 3, diversify="none"
            )
        ]
        self.assertEqual(ids, ["b", "a", "c"])

    def test_parse_rrf_is_a_fuse(self):
        self.assertEqual(inside_search.parse_fuse("rrf"), "rrf")
        self.assertEqual(
            inside_search.resolve_panel("rrf", "mmr"), ("rrf", "mmr", "off")
        )
        left = [
            {"field": "atom", "id": "x", "text": "x", "score": 1.0},
            {"field": "atom", "id": "y", "text": "y", "score": 0.5},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        right = [
            {"field": "atom", "id": "y", "text": "y", "score": 1.0},
            {"field": "atom", "id": "x", "text": "x", "score": 0.5},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        ids = [
            h["id"]
            for h in inside_search._merge_hits(
                left, right, 3, fuse="rrf", diversify="none"
            )
        ]
        self.assertEqual(ids, ["x", "y", "z"])

    def test_parse_dowdall_is_a_fuse(self):
        self.assertEqual(inside_search.parse_fuse("dowdall"), "dowdall")
        self.assertEqual(
            inside_search.resolve_panel("dowdall", "none"),
            ("dowdall", "none", "off"),
        )
        first = [
            {"field": "atom", "id": "a", "text": "a", "score": 1.0},
            {"field": "atom", "id": "c", "text": "c", "score": 0.4},
            {"field": "atom", "id": "d", "text": "d", "score": 0.3},
            {"field": "atom", "id": "e", "text": "e", "score": 0.2},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        second = [
            {"field": "atom", "id": "b", "text": "b", "score": 1.0},
            {"field": "atom", "id": "c", "text": "c", "score": 0.4},
            {"field": "atom", "id": "d", "text": "d", "score": 0.3},
            {"field": "atom", "id": "e", "text": "e", "score": 0.2},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        ids = [
            h["id"]
            for h in inside_search._merge_hits(
                first, second, 5, fuse="dowdall", diversify="none"
            )
        ]
        self.assertEqual(ids[0], "a")

    def test_parse_kemeny_is_a_fuse(self):
        self.assertEqual(inside_search.parse_fuse("kemeny"), "kemeny")
        self.assertEqual(
            inside_search.resolve_panel("kemeny", "mmr"),
            ("kemeny", "mmr", "off"),
        )
        left = [
            {"field": "atom", "id": "a", "text": "a", "score": 1.0},
            {"field": "atom", "id": "b", "text": "b", "score": 0.5},
            {"field": "atom", "id": "c", "text": "c", "score": 0.1},
        ]
        right = [
            {"field": "atom", "id": "b", "text": "b", "score": 1.0},
            {"field": "atom", "id": "a", "text": "a", "score": 0.5},
            {"field": "atom", "id": "c", "text": "c", "score": 0.1},
        ]
        ids = [
            h["id"]
            for h in inside_search._merge_hits(
                left, right, 3, fuse="kemeny", diversify="none"
            )
        ]
        self.assertEqual(ids, ["a", "b", "c"])

    def test_unknown_voter_is_error(self):
        with self.assertRaises(inside_search.UnknownVoter):
            inside_search.parse_fuse("not-a-voter")
        with self.assertRaises(inside_search.UnknownVoter):
            inside_search.parse_diversify("not-a-voter")
        with self.assertRaises(inside_search.UnknownVoter):
            inside_search.parse_decay("maybe")
        with self.assertRaises(inside_search.UnknownVoter):
            inside_search._merge_hits([], [], 1, fuse="not-a-voter")

    def test_unimplemented_voter_is_error(self):
        for name in (
            "combmnz",
            "schulze",
            "copeland",
            "tideman",
        ):
            with self.assertRaises(inside_search.UnknownVoter) as ctx:
                inside_search.parse_fuse(name)
            self.assertIn("not implemented", str(ctx.exception))
        with self.assertRaises(inside_search.UnknownVoter) as ctx:
            inside_search.parse_diversify("dpp")
        self.assertIn("not implemented", str(ctx.exception))
        with self.assertRaises(inside_search.UnknownVoter):
            inside_search._merge_hits([], [], 1, fuse="schulze")

    def test_env_borda_mmr_matches_merge_fixtures(self):
        primary = [
            {"field": "atom", "id": "a", "text": "alpha one", "score": 1.0},
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.5},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.1},
        ]
        secondary = [
            {"field": "atom", "id": "b", "text": "beta two", "score": 0.9},
            {"field": "atom", "id": "c", "text": "gamma three", "score": 0.4},
            {"field": "atom", "id": "a", "text": "alpha one", "score": 0.2},
        ]
        with _panel_env(
            PACKSET_FUSE="borda", PACKSET_DIVERSIFY="mmr", PACKSET_DECAY="off"
        ):
            ids = [h["id"] for h in inside_search._merge_hits(primary, secondary, 3)]
        self.assertEqual(ids[0], "b")
        self.assertEqual(set(ids), {"a", "b", "c"})
        items = [
            (("f", "keep"), 1.0, {"review", "open", "repro"}),
            (("f", "dup"), 0.55, {"review", "open", "repro"}),
            (("f", "other"), 0.5, {"pin", "zircon", "index"}),
        ]
        order = inside_search.mmr_rerank(items, 0.7)
        self.assertEqual(order[0], ("f", "keep"))
        self.assertEqual(order[1], ("f", "other"))
        self.assertEqual(order[2], ("f", "dup"))

    def test_env_rrf_changes_omitted_rank_order(self):
        left = [
            {"field": "atom", "id": "x", "text": "x", "score": 1.0},
            {"field": "atom", "id": "y", "text": "y", "score": 0.5},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        right = [
            {"field": "atom", "id": "y", "text": "y", "score": 1.0},
            {"field": "atom", "id": "x", "text": "x", "score": 0.5},
            {"field": "atom", "id": "z", "text": "z", "score": 0.1},
        ]
        only_z = [{"field": "atom", "id": "z", "text": "z", "score": 1.0}]
        ballots = [left, right, only_z]
        with _panel_env(PACKSET_FUSE="borda", PACKSET_DIVERSIFY="none"):
            borda_ids = [
                h["id"] for h in inside_search._merge_ballots(ballots, 3)
            ]
        with _panel_env(PACKSET_FUSE="rrf", PACKSET_DIVERSIFY="none"):
            rrf_ids = [h["id"] for h in inside_search._merge_ballots(ballots, 3)]
        self.assertEqual(borda_ids, ["x", "y", "z"])
        self.assertEqual(rrf_ids[0], "z")
        self.assertNotEqual(borda_ids, rrf_ids)

    def test_env_unknown_fuse_is_error(self):
        with _panel_env(PACKSET_FUSE="not-a-voter"):
            with self.assertRaises(inside_search.UnknownVoter):
                inside_search.resolve_panel()
            with self.assertRaises(inside_search.UnknownVoter):
                inside_search._merge_hits([], [], 1)
        with _panel_env(PACKSET_FUSE=""):
            with self.assertRaises(inside_search.UnknownVoter):
                inside_search.resolve_panel()


if __name__ == "__main__":
    unittest.main()
