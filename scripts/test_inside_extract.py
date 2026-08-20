import unittest

import inside_extract


class ExtractTests(unittest.TestCase):
    def test_listing_is_dump(self):
        listing = "\n".join(f"- file{i}.rs" for i in range(8))
        self.assertTrue(inside_extract.is_tool_dump(listing))
        self.assertFalse(inside_extract.is_tool_dump("Remember: keep the habit."))


if __name__ == "__main__":
    unittest.main()
