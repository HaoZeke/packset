#!/usr/bin/env python3
"""Hemingway-style gate. No editor binary."""
import unittest

import inside_memory
import inside_prose


class ProseTests(unittest.TestCase):
    def test_simple_claim_is_clean(self):
        report = inside_prose.refuse(
            "Reviews open with a reproducibility check.", role="atom"
        )
        self.assertLessEqual(report["sentences"], 2)
        self.assertEqual(report["very_hard_sentences"], 0)

    def test_adverb_slop_is_refused(self):
        text = (
            "We really actually basically just quite literally definitely "
            "probably certainly want this done."
        )
        with self.assertRaises(inside_prose.ProseError):
            inside_prose.refuse(text, role="file")

    def test_three_sentence_atom_is_refused(self):
        text = "One claim. Two claim. Three claim."
        with self.assertRaises(inside_prose.ProseError):
            inside_prose.refuse(text, role="atom")

    def test_validate_atom_stores_prose(self):
        atom = inside_memory.make_atom(
            workspace="git:ex/p",
            text="Reviews open with a reproducibility check.",
            kind="voice",
            about_peer="rgoswami",
            by_peer="hermes",
        )
        self.assertIn("prose", atom)
        self.assertGreaterEqual(atom["prose"]["words"], 4)

    def test_user_slop_is_an_error(self):
        import tempfile
        from pathlib import Path

        tmp = tempfile.TemporaryDirectory()
        with self.assertRaises((inside_prose.ProseError, inside_memory.AtomError)):
            inside_memory.set_user(
                "We really actually basically just quite literally "
                "definitely probably certainly want this done now.",
                home=Path(tmp.name),
            )
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
