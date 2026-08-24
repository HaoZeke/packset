#!/usr/bin/env python3
"""Ask the model which retrieved claims belong on this turn.

Lexical search proposes a neighbourhood. This pass keeps only the claims
that would change the answer. The caller supplies complete(); this module
does not open a socket. The prompt carries packset ids, not atom bodies
(A-MemGuard: raw snippets are an injection surface). Judge-off and
unreadable replies keep nothing, so a failed audit cannot append the
atom as executable context.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_NONE = re.compile(r"(?i)^\s*(none|no|nil|\[\])\b")
_NUMBER = re.compile(r"\d+")


def build_prompt(turn: str, items: list[dict[str, Any]]) -> str:
    """Numbered claims plus the turn. Reply is NONE or a number list."""
    lines = [
        "Which seat claims belong in context for this turn?",
        "Keep a claim only if using it would change the answer.",
        "Turn:",
        (turn or "").strip() or "(empty)",
        "",
        "Claims:",
    ]
    if not items:
        lines.append("(none)")
    else:
        for index, item in enumerate(items, start=1):
            kind = item.get("kind") or item.get("field") or "claim"
            aid = item.get("id") or item.get("field") or "claim"
            lines.append(f"{index}. `packset:{kind}:{aid}`")
    lines.append("")
    lines.append(
        "Reply with NONE or a comma-separated list of claim numbers. No other text."
    )
    return "\n".join(lines)


def parse_reply(text: str, n_items: int) -> list[int] | None:
    """0-based indices, or None when the reply cannot be read."""
    if n_items <= 0:
        return []
    blob = (text or "").strip()
    if not blob:
        return None
    first = blob.splitlines()[0].strip()
    if _NONE.match(first):
        return []
    found = [int(tok) for tok in _NUMBER.findall(first)]
    if not found:
        found = [int(tok) for tok in _NUMBER.findall(blob)]
    if not found:
        return None
    out: list[int] = []
    seen: set[int] = set()
    for number in found:
        index = number - 1
        if index < 0 or index >= n_items or index in seen:
            continue
        seen.add(index)
        out.append(index)
    if not out:
        return None
    return out


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{4,}")
# Common English glue; not distinctive enough to anchor a keep.
_STOP = frozenset(
    {
        "about",
        "after",
        "again",
        "being",
        "claim",
        "claims",
        "could",
        "from",
        "have",
        "memory",
        "only",
        "quote",
        "seat",
        "should",
        "stored",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "token",
        "under",
        "using",
        "what",
        "which",
        "would",
        "write",
    }
)


def _tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _TOKEN.finditer(text or "")
        if m.group(0).lower() not in _STOP
    }


def _item_pointer(item: dict[str, Any]) -> str:
    kind = item.get("kind") or item.get("field") or "claim"
    aid = item.get("id") or item.get("field") or "claim"
    return f"packset {kind} {aid}"


def anchored_indices(turn: str, items: list[dict[str, Any]]) -> list[int]:
    """Indices whose packset id the turn names. Never reads atom bodies."""
    turn_toks = _tokens(turn)
    if not turn_toks:
        return []
    out: list[int] = []
    for index, item in enumerate(items):
        if turn_toks & _tokens(_item_pointer(item)):
            out.append(index)
    return out


def judge(
    turn: str,
    items: list[dict[str, Any]],
    complete: Callable[[str], str],
    *,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Subset of items the model kept. Off or complete() failure keeps nothing."""
    if not enabled or not items:
        return []
    prompt = build_prompt(turn, items)
    try:
        raw = complete(prompt)
    except Exception:
        return []
    picked = parse_reply(raw if isinstance(raw, str) else str(raw or ""), len(items))
    if picked is None:
        return []
    if picked:
        return [items[index] for index in picked]
    # Model said NONE. Keep claims the turn already names by id.
    anchored = anchored_indices(turn, items)
    if anchored:
        return [items[index] for index in anchored]
    return []
