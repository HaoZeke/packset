"""Ranked pack search. No Meilisearch process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import inside_memory
import inside_search

WS = "git:ex/p"


def voice(**fields: Any) -> dict[str, Any]:
    atom = inside_memory.make_atom(
        workspace=WS,
        text="JOSS reviews open with a reproducibility check.",
        kind="voice",
        about_peer="rgoswami",
        by_peer="hermes",
        entities=["JOSS"],
    )
    atom.update(fields)
    return atom


def test_empty_query() -> None:
    assert inside_search.search_pack({"user": "hi", "atoms": []}, "") == []


def test_user_and_memory_hit() -> None:
    pack = {
        "user": "No thanks. Be brief.\n",
        "memory": "Read paper.pdf first.\n",
        "atoms": [],
    }
    hits = inside_search.search_pack(pack, "brief")
    assert "user" in [h["field"] for h in hits]
    hits = inside_search.search_pack(pack, "paper")
    assert hits[0]["field"] == "memory"


def test_live_atom_prefix_and_typo() -> None:
    atom = voice()
    pack = {"user": "", "memory": "", "atoms": [atom]}
    exact = inside_search.search_pack(pack, "joss")
    assert len(exact) == 1
    assert exact[0]["id"] == atom["id"]
    typo = inside_search.search_pack(pack, "reproducibility")
    assert typo[0]["id"] == atom["id"]
    typo = inside_search.search_pack(pack, "reproducability")
    assert typo[0]["id"] == atom["id"]


def test_expired_atom_absent() -> None:
    stale = inside_memory.make_atom(
        workspace=WS,
        text="JOSS stale snapshot.",
        kind="cache-pointer",
        about_peer="rgoswami",
        by_peer="hermes",
        valid_to="2000-01-01T00:00:00.000Z",
    )
    assert inside_search.search_pack({"user": "", "memory": "", "atoms": [stale]}, "joss") == []


def test_pack_documents_skips_dead_atoms() -> None:
    live = inside_memory.make_atom(
        workspace=WS,
        text="JOSS reviews.",
        kind="voice",
        about_peer="rgoswami",
        by_peer="hermes",
    )
    stale = inside_memory.make_atom(
        workspace=WS,
        text="JOSS stale snapshot.",
        kind="cache-pointer",
        about_peer="rgoswami",
        by_peer="hermes",
        valid_to="2000-01-01T00:00:00.000Z",
    )
    docs = inside_search.pack_documents(
        {
            "workspace": WS,
            "user": "Be brief.",
            "memory": "Read paper.pdf first.",
            "atoms": [live, stale],
        }
    )
    ids = {doc["id"] for doc in docs}
    assert "user" in ids
    mem_id = inside_search.document_id("memory", workspace=WS)
    assert mem_id in ids
    assert mem_id.replace("_", "").isalnum()
    assert live["id"] in ids
    assert stale["id"] not in ids


def test_borda_two_lists_of_three() -> None:
    left = [("f", "x"), ("f", "y"), ("f", "z")]
    right = [("f", "y"), ("f", "x"), ("f", "z")]
    assert inside_search.borda_merge([left, right], 3) == left
    left = [("f", "a"), ("f", "b"), ("f", "c")]
    right = [("f", "b"), ("f", "c"), ("f", "a")]
    assert inside_search.borda_merge([left, right], 3) == [
        ("f", "b"),
        ("f", "a"),
        ("f", "c"),
    ]


def test_merge_is_borda_then_mmr() -> None:
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
    ids = [h["id"] for h in inside_search._merge_hits(primary, secondary, 3)]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_mmr_after_borda_splits_near_duplicates() -> None:
    items = [
        (("f", "keep"), 1.0, {"review", "open", "repro"}),
        (("f", "dup"), 0.55, {"review", "open", "repro"}),
        (("f", "other"), 0.5, {"pin", "zircon", "index"}),
    ]
    order = inside_search.mmr_rerank(items, 0.7)
    assert order[0] == ("f", "keep")
    assert order[1] == ("f", "other")
    assert order[2] == ("f", "dup")


def test_linear_when_milli_absent() -> None:
    if inside_search.milli_bin() is not None:
        pytest.skip("inside-milli binary is present")
    pack = {"workspace": WS, "user": "Be brief.", "memory": "", "atoms": []}
    hits, engine = inside_search.search_pack_with_engine(pack, "brief")
    assert engine == "linear"
    assert any(h["field"] == "user" for h in hits)


def test_short_query_is_exact() -> None:
    pack = {
        "user": "Prefers Conventional Commits.\n",
        "memory": "",
        "atoms": [],
    }
    assert inside_search.search_pack(pack, "pr") == []
    assert inside_search.search_pack(pack, "prefers")[0]["field"] == "user"


def test_stopwords_do_not_hit() -> None:
    pack = {
        "user": "The review queue is the bottleneck.\n",
        "memory": "",
        "atoms": [],
    }
    assert inside_search.search_pack(pack, "What color is the sky?") == []
    assert inside_search.search_pack(pack, "review")[0]["field"] == "user"


def test_tombstone_absent() -> None:
    atom = inside_memory.make_atom(
        workspace=WS,
        text="JOSS reviews.",
        kind="voice",
        about_peer="rgoswami",
        by_peer="hermes",
    )
    atom["tombstone"] = True
    assert inside_search.search_pack({"user": "", "memory": "", "atoms": [atom]}, "joss") == []


def test_set_filter_on_full_atom_list() -> None:
    review = inside_memory.make_atom(
        workspace=WS,
        text="Remember the zircon latch on every review.",
        kind="lesson",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="review",
    )
    debug = inside_memory.make_atom(
        workspace=WS,
        text="Zircon latch belongs in the debug notebook.",
        kind="habit",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="debug",
    )
    pack = {
        "workspace": WS,
        "user": "Open a review with the defect.\n",
        "memory": "",
        "atoms": [review, debug],
    }
    unscoped = inside_search.search_pack_linear(pack, "zircon")
    unscoped_ids = {h["id"] for h in unscoped if h.get("field") == "atom"}
    assert unscoped_ids == {review["id"], debug["id"]}

    review_hits = inside_search.search_pack(pack, "zircon", set_name="review")
    review_ids = [h["id"] for h in review_hits if h.get("field") == "atom"]
    assert review_ids == [review["id"]]
    assert debug["id"] not in review_ids

    prose = inside_search.search_pack(pack, "defect", set_name="review")
    assert any(h["field"] == "user" for h in prose)
    assert any("defect" in (h.get("text") or "").lower() for h in prose)

    assert inside_search.atom_document(review)["set"] == "review"
    assert inside_search.atom_document(debug)["set"] == "debug"


def test_reindex_atoms_skips_user_memory_docs() -> None:
    review = inside_memory.make_atom(
        workspace=WS,
        text="Remember the zircon latch on every review.",
        kind="lesson",
        about_peer="user",
        by_peer="user",
        set_name="review",
    )
    pack = {
        "workspace": WS,
        "set": "review",
        "user": "SETCARDONLYPHRASE for the pin.\n",
        "memory": "SETMEMORYPHRASE for the pin.\n",
        "atoms": [review],
    }
    live = [
        atom
        for atom in pack["atoms"]
        if isinstance(atom, dict) and inside_memory.is_live(atom) and atom.get("id")
    ]
    docs = [inside_search.atom_document(atom) for atom in live]
    assert len(docs) == 1
    assert docs[0]["field"] == "atom"
    assert "user" not in {d["field"] for d in docs}
    full = inside_search.pack_documents(pack)
    assert "user" in {d["field"] for d in full}


def test_set_miss_does_not_poison_workspace_prose_in_index(tmp_path: Path) -> None:
    if inside_search.milli_bin() is None:
        pytest.skip("inside-milli binary not available")
    review = inside_memory.make_atom(
        workspace=WS,
        text="Remember the zircon latch on every review.",
        kind="lesson",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="review",
    )
    debug = inside_memory.make_atom(
        workspace=WS,
        text="Zircon latch belongs in the debug notebook.",
        kind="habit",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="debug",
    )
    workspace_pack = {
        "workspace": WS,
        "user": "Workspace prose mentions UNIQUEWORKSPACEPHRASE only.\n",
        "memory": "",
        "atoms": [review, debug],
    }
    set_pack = {
        "workspace": WS,
        "set": "review",
        "user": "Set card mentions UNIQUESETPHRASE only.\n",
        "memory": "",
        "atoms": [review, debug],
    }
    index = tmp_path / "memory.milli"
    assert inside_search.replace_index(workspace_pack, index)
    set_hits = inside_search.search_pack(
        set_pack,
        "UNIQUESETPHRASE",
        index_dir=index,
        set_name="review",
    )
    assert any(
        h.get("field") == "user" and "UNIQUESETPHRASE" in (h.get("text") or "") for h in set_hits
    )
    inside_search.search_pack(
        set_pack,
        "nomatchtokenxyz",
        index_dir=index,
        set_name="review",
    )
    unscoped_set = inside_search.search_pack(workspace_pack, "UNIQUESETPHRASE", index_dir=index)
    for hit in unscoped_set:
        if hit.get("field") == "user":
            assert "UNIQUESETPHRASE" not in (hit.get("text") or "")
    unscoped = inside_search.search_pack(
        workspace_pack,
        "UNIQUEWORKSPACEPHRASE",
        index_dir=index,
    )
    assert any(
        h.get("field") == "user" and "UNIQUEWORKSPACEPHRASE" in (h.get("text") or "")
        for h in unscoped
    )


def test_stale_index_without_set_field_still_scopes(tmp_path: Path) -> None:
    if inside_search.milli_bin() is None:
        pytest.skip("inside-milli binary not available")
    review = inside_memory.make_atom(
        workspace=WS,
        text="Remember the zircon latch on every review.",
        kind="lesson",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="review",
    )
    debug = inside_memory.make_atom(
        workspace=WS,
        text="Zircon latch belongs in the debug notebook.",
        kind="habit",
        about_peer="user",
        by_peer="user",
        entities=["zircon"],
        set_name="debug",
    )
    pack = {
        "workspace": WS,
        "user": "",
        "memory": "",
        "atoms": [review, debug],
    }
    stale_docs = []
    for atom in (review, debug):
        doc = inside_search.atom_document(atom)
        doc.pop("set", None)
        stale_docs.append(doc)
    index = tmp_path / "memory.milli"
    payload = inside_search._run_milli(
        ["index", "--index", str(index), "--replace"],
        stdin="\n".join(json.dumps(d) for d in stale_docs) + "\n",
        timeout=60.0,
    )
    assert payload is not None
    hits = inside_search.search_pack(pack, "zircon", index_dir=index, set_name="review")
    atom_ids = [h["id"] for h in hits if h.get("field") == "atom"]
    assert review["id"] in atom_ids
    assert debug["id"] not in atom_ids
