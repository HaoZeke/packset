#!/usr/bin/env python3
"""Explicit remember/prefer lines become seat atoms."""
import unittest

import inside_extract


class ExtractTests(unittest.TestCase):
    def test_remember_line(self):
        got = inside_extract.claim_from_user(
            "Remember: always pin the zircon index."
        )
        self.assertEqual(got, ("lesson", "always pin the zircon index"))

    def test_prefer_line(self):
        got = inside_extract.claim_from_user("Prefer conventional commits")
        self.assertEqual(got[0], "preference")
        self.assertIn("conventional", got[1])

    def test_quote_prompt_is_not_a_claim(self):
        self.assertIsNone(
            inside_extract.claim_from_user(
                "In one short sentence, quote the preference and the papers."
            )
        )

    def test_question_with_prefer_is_not_a_claim(self):
        self.assertIsNone(
            inside_extract.claim_from_user(
                "What latch do I prefer on reviews? One short sentence."
            )
        )

    def test_atom_kind_and_workspace(self):
        atom = inside_extract.atom_from_user(
            "Note that reviews cite the commit SHA.",
            workspace="global",
        )
        self.assertIsNotNone(atom)
        self.assertEqual(atom["kind"], "lesson")
        self.assertEqual(atom["workspace"], "global")
        self.assertIn("commit SHA", atom["text"])


if __name__ == "__main__":
    unittest.main()
