"""Explicit remember/prefer lines become seat atoms."""

from __future__ import annotations

from pathlib import Path

import pytest

import inside_extract
import inside_memory
import inside_search


def test_remember_line() -> None:
    got = inside_extract.claim_from_user("Remember: always pin the zircon index.")
    assert got == ("lesson", "always pin the zircon index")


def test_prefer_line() -> None:
    kind, text = inside_extract.claim_from_user("Prefer: conventional commits")
    assert kind == "preference"
    assert "conventional" in text
    assert inside_extract.claim_from_user("Prefer conventional commits") is None


def test_quote_prompt_returns_none() -> None:
    assert (
        inside_extract.claim_from_user(
            "In one short sentence, quote the preference and the papers."
        )
        is None
    )


def test_question_with_prefer_returns_none() -> None:
    assert (
        inside_extract.claim_from_user("What latch do I prefer on reviews? One short sentence.")
        is None
    )


def test_atom_kind_and_workspace() -> None:
    atom = inside_extract.atom_from_user(
        "Remember: reviews cite the commit SHA.",
        workspace="global",
    )
    assert atom is not None
    assert atom["kind"] == "lesson"
    assert atom["workspace"] == "global"
    assert "commit SHA" in atom["text"]


def test_note_that_and_remember_that_are_not_claims() -> None:
    assert inside_extract.claim_from_user("Note that the test failed on line 12") is None
    assert inside_extract.claim_from_user("Remember that: pin the review set") is None
    assert inside_extract.claim_from_user("From now on, file a ticket first") is None


def test_listing_is_dump() -> None:
    listing = "\n".join(f"- file{i}.rs" for i in range(8))
    assert inside_extract.is_tool_dump(listing)
    assert not inside_extract.is_tool_dump("Remember: keep the habit.")


def test_extract_forbidden_off_compaction() -> None:
    assert not inside_extract.cheap_allowed("extract", "onDemand")
    assert inside_extract.cheap_allowed("extract", "compaction")
    with pytest.raises(inside_extract.CheapError):
        inside_extract.extract_propose(
            "Reviews close after the SHA is cited.",
            workspace="global",
            when="onDemand",
        )


def test_extract_propose_writes_inbox_not_atoms(tmp_path: Path) -> None:
    proposal = inside_extract.extract_propose(
        "Reviews close after the SHA is cited.",
        workspace="global",
        when="compaction",
        home=tmp_path,
    )
    assert proposal is not None
    assert proposal["schema"] == "inside.proposal/v1"
    assert proposal["status"] == "open"
    assert "id" in proposal
    inbox = inside_extract.list_proposals("global", home=tmp_path)
    assert [p["id"] for p in inbox] == [proposal["id"]]
    assert inside_memory.current_atoms("global", tmp_path) == []
    assert inside_extract.apply_pack("extractPropose", proposal) is None


def test_extract_accept_commits_atom(tmp_path: Path) -> None:
    proposal = inside_extract.extract_propose(
        "Reviews close after the SHA is cited.",
        workspace="global",
        when="compaction",
        home=tmp_path,
    )
    assert proposal is not None
    atom = inside_extract.accept_proposal(
        proposal["id"], workspace="global", home=tmp_path
    )
    assert atom["text"] == proposal["text"]
    assert atom["kind"] == "lesson"
    live = inside_memory.current_atoms("global", tmp_path)
    assert [a["id"] for a in live] == [atom["id"]]
    assert inside_extract.apply_pack("extractAccept", proposal) is not None


def test_compact_day_reads_archive_not_memory(tmp_path: Path) -> None:
    inside_memory.append_archive(
        "global",
        "The review latch is the zircon pin.",
        day="2026-08-24",
        home=tmp_path,
    )
    inside_memory.append_archive(
        "global",
        "The review latch is the zircon pin.",
        day="2026-08-24",
        home=tmp_path,
    )
    path = inside_memory.archive_path("global", day="2026-08-24", home=tmp_path)
    assert path.name == "2026-08-24.md"
    text = inside_memory.read_text(path)
    assert text.count("zircon pin") == 1
    proposed = inside_extract.compact_day(
        "global", day="2026-08-24", home=tmp_path
    )
    assert len(proposed) == 1
    assert "zircon" in proposed[0]["text"]
    pack = {
        "user": "",
        "memory": "",
        "atoms": [],
        "archive": text,
    }
    hits = inside_search.search_pack_linear(pack, "zircon", limit=8)
    assert hits == []


def test_fenced_splice_does_not_propose(tmp_path: Path) -> None:
    live = inside_memory.make_atom(
        workspace="global",
        text="always pin the zircon index",
        kind="lesson",
        about_peer="rgoswami",
        by_peer="hermes",
    )
    inside_memory.add_atom(live, home=tmp_path)
    got = inside_extract.extract_propose(
        "always pin the zircon index",
        workspace="global",
        when="compaction",
        home=tmp_path,
    )
    assert got is None
    assert inside_extract.list_proposals("global", home=tmp_path) == []


def test_new_prose_still_proposes_beside_fence(tmp_path: Path) -> None:
    live = inside_memory.make_atom(
        workspace="global",
        text="always pin the zircon index",
        kind="lesson",
        about_peer="rgoswami",
        by_peer="hermes",
    )
    inside_memory.add_atom(live, home=tmp_path)
    got = inside_extract.extract_propose(
        "Reviews close after the SHA is cited.",
        workspace="global",
        when="compaction",
        home=tmp_path,
    )
    assert got is not None
    assert "SHA" in got["text"]


def test_remember_still_commits_without_inbox(tmp_path: Path) -> None:
    atom = inside_extract.atom_from_user(
        "Remember: always pin the zircon index.",
        workspace="global",
    )
    assert atom is not None
    stored = inside_memory.add_atom(atom, home=tmp_path)
    assert inside_extract.list_proposals("global", home=tmp_path) == []
    assert stored["text"]
    assert inside_extract.apply_pack("remember", stored) == stored
