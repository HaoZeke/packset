#!/usr/bin/env python3
"""Active workflow set: pin, retrieve isolation, scoped Remember."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inside_extract
import inside_memory
import inside_policy
import inside_search
import inside_set


class PinFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.ws = "global"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_pin_by_default(self):
        self.assertEqual(inside_set.read_pin(self.ws, home=self.home), "")

    def test_write_and_clear_pin(self):
        self.assertEqual(
            inside_set.write_pin(self.ws, "review", home=self.home), "review"
        )
        self.assertEqual(inside_set.read_pin(self.ws, home=self.home), "review")
        self.assertEqual(
            inside_set.write_pin(self.ws, "debug", home=self.home), "debug"
        )
        self.assertEqual(inside_set.read_pin(self.ws, home=self.home), "debug")
        self.assertEqual(inside_set.write_pin(self.ws, "", home=self.home), "")
        self.assertEqual(inside_set.read_pin(self.ws, home=self.home), "")

    def test_bad_set_name(self):
        with self.assertRaises(inside_memory.AtomError):
            inside_set.check_set_name("Review This")
        with self.assertRaises(inside_memory.AtomError):
            inside_set.write_pin(self.ws, "../x", home=self.home)


class SelectPinnedTests(unittest.TestCase):
    def _review_pack(self, extra_atoms=None):
        return {
            "user": "Open a review with the defect and the call path.\n",
            "memory": "Reviews cite the commit.\nDebug traces stay in the debug set.\n",
            "atoms": extra_atoms or [],
        }

    def _debug_pack(self, extra_atoms=None):
        return {
            "user": "Name the failing test first.\n",
            "memory": "Debug traces stay in the debug set.\n",
            "atoms": extra_atoms or [],
        }

    def test_review_pack_prose_scores_without_debug_prose(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this pull request"}]}
        )
        selected = inside_policy.select(self._review_pack(), hints)
        block = inside_policy._selected_text(selected)
        self.assertIn("Open a review with the defect and the call path.", block)
        self.assertIn("Reviews cite the commit.", block)
        self.assertNotIn("Name the failing test first.", block)

    def test_debug_pin_hides_review_atom(self):
        review_atom = {
            "id": "r1",
            "kind": "lesson",
            "text": "Remember the zircon latch on every review.",
            "set": "review",
            "tombstone": False,
        }
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "debug the failing test"}]}
        )
        selected = inside_policy.select(self._debug_pack(), hints)
        block = inside_policy._selected_text(selected)
        self.assertIn("Name the failing test first.", block)
        self.assertNotIn("zircon latch", block)
        selected_with = inside_policy.select(
            self._debug_pack(extra_atoms=[review_atom]), hints
        )
        self.assertNotIn("zircon latch", inside_policy._selected_text(selected_with))

    def test_pinned_sky_splices_nothing(self):
        hints = inside_policy.inspect(
            {
                "messages": [
                    {"role": "user", "content": "review this PR"},
                    {"role": "user", "content": "What color is the sky?"},
                ]
            }
        )
        selected = inside_policy.select(self._review_pack(), hints)
        self.assertEqual(inside_policy._selected_text(selected), "")

    def test_retrieve_empty_pin_uses_pack_search(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "keep it brief"}]}
        )
        hits = [
            {
                "field": "atom",
                "id": "v1",
                "kind": "habit",
                "text": "Be brief.",
                "score": 4.0,
            }
        ]
        with (
            mock.patch.object(
                inside_policy,
                "fetch_pin_payload",
                return_value={"set": "", "instructions": ""},
            ),
            mock.patch.object(
                inside_policy, "fetch_search", return_value=hits
            ) as search,
        ):
            selected = inside_policy.retrieve("http://127.0.0.1:9", "global", hints)
        search.assert_called_once()
        self.assertIsNone(search.call_args.kwargs.get("set_name"))
        self.assertIn("Be brief.", inside_policy._selected_text(selected))

    def test_retrieve_uses_pinned_set(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this pull request"}]}
        )
        linear = inside_search.search_pack_linear(
            self._review_pack(), "review this pull request", limit=16
        )
        with (
            mock.patch.object(
                inside_policy,
                "fetch_pin_payload",
                return_value={"set": "review", "instructions": ""},
            ),
            mock.patch.object(
                inside_policy, "fetch_search", return_value=linear
            ) as search,
        ):
            selected = inside_policy.retrieve("http://127.0.0.1:9", "global", hints)
        self.assertEqual(search.call_args.kwargs.get("set_name"), "review")
        block = inside_policy._selected_text(selected)
        self.assertIn("Open a review with the defect and the call path.", block)
        self.assertNotIn("Name the failing test first.", block)

    def test_switch_then_clear_returns_empty(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this pull request"}]}
        )
        linear = inside_search.search_pack_linear(
            self._review_pack(), "review this pull request", limit=16
        )
        with (
            mock.patch.object(
                inside_policy,
                "fetch_pin_payload",
                return_value={"set": "review", "instructions": ""},
            ),
            mock.patch.object(inside_policy, "fetch_search", return_value=linear),
        ):
            self.assertIn(
                "Reviews cite the commit.",
                inside_policy._selected_text(
                    inside_policy.retrieve("http://127.0.0.1:9", "global", hints)
                ),
            )
        with (
            mock.patch.object(
                inside_policy,
                "fetch_pin_payload",
                return_value={"set": "", "instructions": ""},
            ),
            mock.patch.object(inside_policy, "fetch_search", return_value=[]),
        ):
            self.assertEqual(
                inside_policy._selected_text(
                    inside_policy.retrieve("http://127.0.0.1:9", "global", hints)
                ),
                "",
            )


class ExtractPinTests(unittest.TestCase):
    def test_remember_without_pin_writes_nothing(self):
        with mock.patch.object(inside_policy, "fetch_pin", return_value=""):
            with mock.patch.object(inside_extract, "post_atom") as post:
                got = inside_extract.extract_user_text(
                    "Remember: always pin the zircon index.",
                    url="http://127.0.0.1:9",
                    workspace="global",
                )
        self.assertIsNone(got)
        post.assert_not_called()

    def test_remember_pin_fetch_error_is_not_a_quiet_skip(self):
        with mock.patch.object(
            inside_policy,
            "fetch_pin",
            side_effect=TimeoutError("memd down"),
        ):
            with self.assertRaises(TimeoutError):
                inside_extract.extract_user_text(
                    "Remember: always pin the zircon index.",
                    url="http://127.0.0.1:9",
                    workspace="global",
                )

    def test_remember_write_error_is_not_a_quiet_skip(self):
        import urllib.error

        with mock.patch.object(inside_policy, "fetch_pin", return_value="review"):
            with self.assertRaises((urllib.error.URLError, TimeoutError, OSError)):
                inside_extract.extract_user_text(
                    "Remember: always pin the zircon index.",
                    url="http://127.0.0.1:1",
                    workspace="global",
                )

    def test_remember_with_pin_tags_the_set(self):
        posted = {}

        def _post(_url, atom):
            posted.update(atom)
            return atom

        with mock.patch.object(inside_policy, "fetch_pin", return_value="review"):
            with mock.patch.object(inside_extract, "post_atom", side_effect=_post):
                got = inside_extract.extract_user_text(
                    "Remember: always pin the zircon index.",
                    url="http://127.0.0.1:9",
                    workspace="global",
                )
        self.assertIsNotNone(got)
        self.assertEqual(posted.get("set"), "review")
        self.assertEqual(posted.get("kind"), "lesson")


if __name__ == "__main__":
    unittest.main()
