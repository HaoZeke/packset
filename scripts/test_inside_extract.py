"""Explicit remember/prefer lines become seat atoms."""

from __future__ import annotations

from pathlib import Path

import pytest

import inside_extract
import inside_memory


def test_remember_line() -> None:
    got = inside_extract.claim_from_user("Remember: always pin the zircon index.")
    assert got == ("lesson", "always pin the zircon index")


def test_prefer_line() -> None:
    kind, text = inside_extract.claim_from_user("Prefer conventional commits")
    assert kind == "preference"
    assert "conventional" in text


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
        "Note that reviews cite the commit SHA.",
        workspace="global",
    )
    assert atom is not None
    assert atom["kind"] == "lesson"
    assert atom["workspace"] == "global"
    assert "commit SHA" in atom["text"]


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
