#!/usr/bin/env python3
"""Tests for inspect / select / splice. No daemon required."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inside_memory
import inside_policy


def _pack():
    return {
        "workspace": "git:example.com/proj",
        "user": "Prefers Conventional Commits.\n",
        "memory": "Read paper.pdf first.\n",
        "atoms": [
            {
                "id": "v1",
                "kind": "voice",
                "text": "Speaks in short sentences.",
                "tombstone": False,
            },
            {
                "id": "h1",
                "kind": "habit",
                "text": "Reviews open with a reproducibility check.",
                "tombstone": False,
            },
            {
                "id": "c1",
                "kind": "cache-pointer",
                "text": "Open issues snapshot 2026-08-01.",
                "tombstone": False,
            },
            {
                "id": "g1",
                "kind": "goal",
                "text": "Ship the memory splice.",
                "tombstone": False,
            },
            {
                "id": "b1",
                "kind": "belief",
                "text": "The review queue is the bottleneck.",
                "tombstone": False,
            },
        ],
    }


class InspectTests(unittest.TestCase):
    def test_review_turn_is_not_remote(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        self.assertEqual(hints["user_text"], "review this PR")
        self.assertEqual(hints["tool_names"], [])
        self.assertFalse(hints["wants_remote"])

    def test_github_or_host_sets_wants_remote(self):
        github = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "list github issues"}]}
        )
        self.assertTrue(github["wants_remote"])
        host = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "see docs.example.com/labels"}]}
        )
        self.assertTrue(host["wants_remote"])
        tools = inside_policy.inspect(
            {
                "messages": [{"role": "user", "content": "status"}],
                "tools": [{"type": "function", "function": {"name": "gh_issue_list"}}],
            }
        )
        self.assertEqual(tools["tool_names"], ["gh_issue_list"])
        self.assertTrue(tools["wants_remote"])

    def test_input_user_text(self):
        hints = inside_policy.inspect(
            {"input": [{"type": "message", "role": "user", "content": "hello"}]}
        )
        self.assertEqual(hints["user_text"], "hello")

    def test_type_user_without_role(self):
        # Stock Grok Build ACP / compaction turns.
        hints = inside_policy.inspect(
            {
                "messages": [
                    {
                        "type": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the seat fix token SEATOK1?",
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(hints["user_text"], "What is the seat fix token SEATOK1?")

    def test_user_query_wrapper_unwrapped(self):
        hints = inside_policy.inspect(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": (
                            "<user_query>\n"
                            "What is the seat fix token SEATOK1?\n"
                            "</user_query>"
                        ),
                    }
                ]
            }
        )
        self.assertEqual(hints["user_text"], "What is the seat fix token SEATOK1?")

    def test_inspect_uses_latest_user_turn(self):
        hints = inside_policy.inspect(
            {
                "messages": [
                    {"role": "user", "content": "review this PR"},
                    {"role": "assistant", "content": "Looking."},
                    {"role": "user", "content": "What color is the sky?"},
                ]
            }
        )
        self.assertEqual(hints["user_text"], "What color is the sky?")
        self.assertFalse(hints["wants_remote"])

    def test_inspect_skips_tool_result_user_messages(self):
        hints = inside_policy.inspect(
            {
                "messages": [
                    {"role": "user", "content": "review this PR"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "bash",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "ok",
                            }
                        ],
                    },
                ]
            }
        )
        self.assertEqual(hints["user_text"], "review this PR")


class SelectTests(unittest.TestCase):
    def test_review_turn_gets_voice_and_habit_not_cache(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        head_kinds = [atom["kind"] for atom in selected["head_atoms"]]
        tail_kinds = [atom["kind"] for atom in selected["tail_atoms"]]
        self.assertEqual(head_kinds, [])
        self.assertIn("habit", tail_kinds)
        self.assertNotIn("goal", tail_kinds)
        self.assertNotIn("cache-pointer", head_kinds + tail_kinds)
        self.assertNotIn("belief", tail_kinds)
        self.assertEqual(selected["user"], _pack()["user"])
        self.assertEqual(selected["memory"], _pack()["memory"])
        block = inside_policy._selected_text(selected)
        self.assertIn("Reviews open with a reproducibility check.", block)
        self.assertNotIn("Prefers Conventional Commits", block)
        self.assertNotIn("Speaks in short sentences", block)
        self.assertNotIn("Read paper.pdf first", block)

    def test_no_github_omits_cache_pointer(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "summarize the module"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        self.assertFalse(hints["wants_remote"])
        kinds = [atom["kind"] for atom in selected["tail_atoms"]]
        self.assertNotIn("cache-pointer", kinds)

    def test_remote_turn_keeps_cache_pointer(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "refresh github issues"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        self.assertTrue(hints["wants_remote"])
        self.assertIn(
            "cache-pointer", [atom["kind"] for atom in selected["tail_atoms"]]
        )

    def test_head_prefix_is_byte_stable(self):
        pack = _pack()
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        first = inside_policy.select(pack, hints)
        second = inside_policy.select(pack, hints)
        self.assertEqual(first["head_prefix"], second["head_prefix"])
        self.assertEqual(
            first["head_prefix"].encode("utf-8"),
            second["head_prefix"].encode("utf-8"),
        )
        self.assertIn("Reviews open with a reproducibility check.", first["tail_atoms"][0]["text"])
        self.assertNotIn("Speaks in short sentences.", first["head_prefix"])

    def test_unrelated_turn_selects_nothing(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "What color is the sky?"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        self.assertEqual(selected["tail_atoms"], [])
        self.assertEqual(inside_policy._selected_text(selected), "")

    def test_prior_review_then_sky_selects_nothing(self):
        hints = inside_policy.inspect(
            {
                "messages": [
                    {"role": "user", "content": "review this PR"},
                    {"role": "assistant", "content": "Looking."},
                    {"role": "user", "content": "What color is the sky?"},
                ]
            }
        )
        selected = inside_policy.select(_pack(), hints)
        self.assertEqual(selected["tail_atoms"], [])
        self.assertEqual(inside_policy._selected_text(selected), "")

    def test_short_token_does_not_prefix_match_prefers(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "pr"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        block = inside_policy._selected_text(selected)
        self.assertNotIn("Prefers Conventional Commits", block)
        self.assertEqual(selected["tail_atoms"], [])
        self.assertEqual(block, "")

    def test_duplicate_file_and_atom_text_is_kept_once(self):
        pack = _pack()
        pack["user"] = "Prefer zircon-latch-42 on every review.\n"
        pack["atoms"] = [
            {
                "id": "p1",
                "kind": "preference",
                "text": "Prefer zircon-latch-42 on every review.",
                "tombstone": False,
            }
        ]
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        selected = inside_policy.select(pack, hints)
        block = inside_policy._selected_text(selected)
        self.assertEqual(block.count("Prefer zircon-latch-42 on every review."), 1)
        self.assertEqual(selected["tail_atoms"], [])

    def test_judge_items_round_trip_drops_unkept(self):
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        selected = inside_policy.select(_pack(), hints)
        items = inside_policy.items_from_selected(selected)
        self.assertGreaterEqual(len(items), 1)
        kept = inside_policy.selected_from_items(selected, items[:1])
        self.assertEqual(len(inside_policy.items_from_selected(kept)), 1)
        self.assertIn(inside_policy.items_from_selected(kept)[0]["text"], items[0]["text"])

    def test_overflow_on_select_does_not_trim(self):
        pack = _pack()
        pack["user"] = "x" * (inside_memory.USER_CAP + 1)
        hints = inside_policy.inspect(
            {"messages": [{"role": "user", "content": "review this PR"}]}
        )
        with self.assertRaises(inside_memory.MemoryOverflow):
            inside_policy.select(pack, hints)


class SpliceTests(unittest.TestCase):
    def _selected(self, text="review this PR"):
        hints = inside_policy.inspect({"messages": [{"role": "user", "content": text}]})
        return inside_policy.select(_pack(), hints)

    def test_overflow_raises_and_does_not_trim(self):
        selected = {
            "user": "x" * (inside_memory.USER_CAP + 1),
            "memory": "ok",
            "head_prefix": "x",
            "head_atoms": [],
            "tail_atoms": [],
        }
        body = {"messages": [{"role": "user", "content": "hi"}]}
        with self.assertRaises(inside_memory.MemoryOverflow):
            inside_policy.splice(body, selected)
        selected = {
            "user": "ok",
            "memory": "y" * (inside_memory.MEMORY_CAP + 1),
            "head_prefix": "y",
            "head_atoms": [],
            "tail_atoms": [],
        }
        with self.assertRaises(inside_memory.MemoryOverflow):
            inside_policy.splice(body, selected)

    def test_chat_body_gets_system_then_user(self):
        selected = self._selected()
        raw = inside_policy.splice(
            {"messages": [{"role": "user", "content": "review this PR"}]},
            selected,
        )
        out = json.loads(raw)
        roles = [message["role"] for message in out["messages"]]
        self.assertIn(roles[0], ("system", "developer"))
        self.assertEqual(roles[1:], ["user"])
        self.assertTrue(out["messages"][0]["content"].startswith(selected["head_prefix"]))
        self.assertIn("Reviews open with a reproducibility check.", out["messages"][0]["content"])
        self.assertIn("End seat memory.", out["messages"][0]["content"])

    def test_anthropic_system_list_leaves_message_content_shapes(self):
        selected = self._selected()
        body = {
            "model": "v9",
            "system": [{"type": "text", "text": "You are Claude Code."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                    ],
                },
            ],
        }
        shapes = [type(m["content"]).__name__ for m in body["messages"]]
        out = json.loads(inside_policy.splice(body, selected))
        self.assertEqual(
            [type(m["content"]).__name__ for m in out["messages"]], shapes
        )
        self.assertEqual(out["messages"][2]["content"][0]["type"], "tool_result")
        self.assertTrue(out["system"][0]["text"].startswith("Seat memory:"))

    def test_splice_twice_is_idempotent(self):
        selected = self._selected()
        body = {"messages": [{"role": "user", "content": "review this PR"}]}
        once = inside_policy.splice(body, selected)
        twice = inside_policy.splice(once, selected)
        first = json.loads(once)
        second = json.loads(twice)
        self.assertEqual(first["messages"], second["messages"])
        heads = [
            message
            for message in second["messages"]
            if message.get("role") in ("system", "developer")
        ]
        self.assertEqual(len(heads), 1)

    def test_input_list_round_trips(self):
        selected = self._selected()
        raw = inside_policy.splice(
            {"input": [{"type": "message", "role": "user", "content": "review this PR"}]},
            selected,
        )
        out = json.loads(raw)
        self.assertEqual(out["input"][0]["role"], "system")
        self.assertEqual(out["input"][1]["role"], "user")
        again = json.loads(inside_policy.splice(raw, selected))
        self.assertEqual(out["input"], again["input"])
        self.assertEqual(sum(1 for item in again["input"] if item.get("role") == "system"), 1)


if __name__ == "__main__":
    unittest.main()
