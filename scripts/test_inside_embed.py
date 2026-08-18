#!/usr/bin/env python3
"""Local ONNX embedder. Works without the model present."""
import os
import unittest

import inside_embed


class EmbedTests(unittest.TestCase):
    def test_off_returns_none(self):
        prev = os.environ.get("INSIDE_EMBED")
        os.environ["INSIDE_EMBED"] = "off"
        try:
            self.assertFalse(inside_embed.enabled())
            self.assertIsNone(inside_embed.encode_one("Reviews open first."))
        finally:
            if prev is None:
                os.environ.pop("INSIDE_EMBED", None)
            else:
                os.environ["INSIDE_EMBED"] = prev

    def test_cosine_identical_and_orthogonal(self):
        self.assertAlmostEqual(inside_embed.cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(inside_embed.cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(inside_embed.cosine([], [1.0]), 0.0)

    def test_available_is_false_without_cache_or_download(self):
        prev_d = os.environ.get("INSIDE_EMBED_DOWNLOAD")
        prev_c = os.environ.get("INSIDE_EMBED_CACHE")
        os.environ["INSIDE_EMBED_DOWNLOAD"] = "0"
        os.environ["INSIDE_EMBED_CACHE"] = "/tmp/inside-embed-missing"
        try:
            # No onnx in that dir: encode stays off even if fastembed is installed.
            self.assertFalse(inside_embed.available())
        finally:
            if prev_d is None:
                os.environ.pop("INSIDE_EMBED_DOWNLOAD", None)
            else:
                os.environ["INSIDE_EMBED_DOWNLOAD"] = prev_d
            if prev_c is None:
                os.environ.pop("INSIDE_EMBED_CACHE", None)
            else:
                os.environ["INSIDE_EMBED_CACHE"] = prev_c


if __name__ == "__main__":
    unittest.main()
