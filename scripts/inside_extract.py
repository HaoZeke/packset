#!/usr/bin/env python3
"""Seated Remember plus compaction extract.

Remember (claim_from_user) writes an atom when the operator
says to keep a line. Cheap extract writes a Proposal inbox
row and commits nothing until accept.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import inside_memory

_PATTERNS = (
    (re.compile(r"(?i)^\s*remember(?:\s+that)?[:\s]+(.+)$"), "lesson"),
    (re.compile(r"(?i)^\s*note that[:\s]+(.+)$"), "lesson"),
    (re.compile(r"(?i)^\s*from now on[,:]?\s+(.+)$"), "habit"),
    (re.compile(r"(?i)^\s*prefer[:\s]+(.+)$"), "preference"),
)
_FIRST_SENTENCE = re.compile(r"\s*[.!?]+\s*")
_QUESTION = re.compile(
    r"(?i)^\s*(what|which|who|whom|whose|where|when|why|how|"
    r"do|does|did|is|are|can|could|would|should)\b"
)


def claim_from_user(text: str) -> tuple[str, str] | None:
    """Return (kind, claim) when a line is an explicit keep directive."""
    blob = (text or "").strip()
    if not blob:
        return None
    # Stock Grok Build wraps the turn; keep directives are inside it.
    try:
        from inside_policy import _unwrap_user_query

        blob = _unwrap_user_query(blob)
    except Exception:
        pass
    last = blob.splitlines()[-1].strip()
    if last.endswith("?") or _QUESTION.match(last):
        return None
    for pat, kind in _PATTERNS:
        match = pat.match(last)
        if not match:
            continue
        claim = match.group(1).strip()
        claim = _FIRST_SENTENCE.split(claim, 1)[0].strip().rstrip(".,;:")
        if len(claim) < 8:
            return None
        return kind, claim
    return None


def atom_from_user(
    text: str,
    *,
    workspace: str,
    about_peer: str = "user",
    by_peer: str = "user",
    set_name: str | None = None,
) -> dict[str, Any] | None:
    parsed = claim_from_user(text)
    if parsed is None:
        return None
    kind, claim = parsed
    return inside_memory.make_atom(
        workspace=workspace,
        text=claim,
        kind=kind,
        about_peer=about_peer,
        by_peer=by_peer,
        level="explicit",
        set_name=set_name,
    )


def post_atom(url: str, atom: dict[str, Any]) -> dict[str, Any]:
    target = (url or "").rstrip("/") + "/v1/atoms"
    payload = json.dumps(atom).encode("utf-8")
    req = urllib.request.Request(
        target,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("atom response is not an object")
    return body


def extract_user_text(
    text: str,
    *,
    url: str,
    workspace: str,
    about_peer: str = "user",
    by_peer: str = "user",
) -> dict[str, Any] | None:
    import inside_policy

    pin = inside_policy.fetch_pin(url, workspace)
    if not pin:
        return None
    atom = atom_from_user(
        text,
        workspace=workspace,
        about_peer=about_peer,
        by_peer=by_peer,
        set_name=pin,
    )
    if atom is None:
        return None
    return post_atom(url, atom)


CHEAP_ALLOWED = frozenset(
    {
        ("extract", "compaction"),
        ("linkRewrite", "compaction"),
        ("fidelity", "onDemand"),
        ("dueSuggest", "onDemand"),
    }
)
PROPOSAL_SCHEMA = "inside.proposal/v1"


class CheapError(ValueError):
    """Cheap-model job is not allowed at this when."""


def cheap_allowed(job: str, when: str) -> bool:
    return (job, when) in CHEAP_ALLOWED


def proposals_path(workspace: str, home: Path | None = None) -> Path:
    return inside_memory.workspace_dir(workspace, home) / "proposals.jsonl"


def list_proposals(workspace: str, home: Path | None = None) -> list[dict[str, Any]]:
    path = proposals_path(workspace, home)
    if not path.exists():
        return []
    seen: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and rec.get("id"):
            seen[rec["id"]] = rec
    return [p for p in seen.values() if p.get("status") == "open"]


def _append_proposal(workspace: str, rec: dict[str, Any], home: Path | None = None) -> None:
    path = proposals_path(workspace, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _norm_claim(text: str) -> str:
    return " ".join((text or "").lower().split()).rstrip(".,;:")


def fence_texts(workspace: str, home: Path | None = None) -> set[str]:
    """Cards and live atoms already in the splice. Compaction must not re-capture them."""
    out: set[str] = set()
    for path in (
        inside_memory.user_path(home),
        inside_memory.memory_path(workspace, home),
    ):
        blob = inside_memory.read_text(path)
        for part in inside_memory._entries(blob):
            n = _norm_claim(part)
            if n:
                out.add(n)
    for atom in inside_memory.current_atoms(workspace, home):
        n = _norm_claim(str(atom.get("text") or ""))
        if n:
            out.add(n)
    return out


def is_fenced(claim: str, fence: set[str]) -> bool:
    n = _norm_claim(claim)
    if not n:
        return False
    for item in fence:
        if n == item or n in item or item in n:
            return True
    return False


def extract_propose(
    text: str,
    *,
    workspace: str,
    when: str,
    job: str = "extract",
    home: Path | None = None,
    fence: set[str] | None = None,
) -> dict[str, Any] | None:
    """Cheap extract. Inbox only. Forbidden off compaction."""
    if not cheap_allowed(job, when):
        raise CheapError(f"{job} is not allowed on {when}")
    blob = (text or "").strip()
    if not blob or is_tool_dump(blob):
        return None
    claim = _FIRST_SENTENCE.split(blob, 1)[0].strip().rstrip(".,;:")
    if len(claim) < 8:
        return None
    wall = fence if fence is not None else fence_texts(workspace, home)
    if is_fenced(claim, wall):
        return None
    rec = {
        "schema": PROPOSAL_SCHEMA,
        "id": inside_memory.new_id(),
        "workspace": workspace,
        "text": claim,
        "job": job,
        "when": when,
        "status": "open",
        "ts": inside_memory.utcnow(),
    }
    _append_proposal(workspace, rec, home)
    return rec


def accept_proposal(
    proposal_id: str, *, workspace: str, home: Path | None = None
) -> dict[str, Any]:
    open_ones = {p["id"]: p for p in list_proposals(workspace, home)}
    if proposal_id not in open_ones:
        raise CheapError(f"no open proposal {proposal_id}")
    rec = dict(open_ones[proposal_id])
    atom = inside_memory.make_atom(
        workspace=workspace,
        text=rec["text"],
        kind="lesson",
        about_peer="user",
        by_peer="extract",
        level="derived",
    )
    stored = inside_memory.add_atom(atom, home=home)
    rec["status"] = "accepted"
    rec["atom_id"] = stored["id"]
    rec["ts"] = inside_memory.utcnow()
    _append_proposal(workspace, rec, home)
    return stored


def apply_pack(kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Mirror Lean applyPack: extractPropose commits nothing."""
    if kind == "remember":
        return payload
    if kind == "extractPropose":
        return None
    if kind == "extractAccept":
        return {"text": payload["text"]}
    raise CheapError(f"unknown pack write {kind}")


def is_tool_dump(text: str) -> bool:
    """Tool stdout, listings, and fetched bodies are attach, not atoms."""
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    if "```" in lower and ("stdout" in lower or "stderr" in lower):
        return True
    if lower.startswith("<!doctype") or lower.startswith("<html"):
        return True
    lines = t.splitlines()
    if len(lines) >= 8 and sum(
        1
        for line in lines
        if line.startswith("-") or line.startswith("drwx") or line.startswith("total ")
    ) >= 6:
        return True
    return False

