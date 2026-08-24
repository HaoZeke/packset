#!/usr/bin/env python3
"""Inspect the in-flight request and splice the matching pack bits.

A pass on the model wire: look at this request, rewrite the body so the
model sees a ranked neighbourhood of the pack. Nothing rides every turn.
The query is the current user turn, not the joined transcript.
USER.md and MEMORY.md contribute scored paragraphs only. The shim
retrieves through GET /v1/search (milli BM25, linear fallback) and
fronts GET /v1/recall through due_atoms so the review clock reaches
the Facts tail and close_live cannot drop a still-due atom from
retrieve. Due atom text is withheld to a packset id. Atoms
are at most MAX_ATOMS, never a kind dump. cache-pointer also needs a
remote-shaped turn. A pin scopes retrieve to that set; no pin uses
the pack search. Empty neighbourhood is a no-op splice. A later
judge pass may drop claims the ranker proposed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import inside_memory
import inside_search

# Files are paragraph-scored. Atoms are ranked. Nothing is a default
# head. USER/MEMORY caps bound the files; the atom cap bounds the rest.
_CARD_KINDS = frozenset({"voice", "preference", "habit"})
_FACT_KINDS = frozenset({"lesson", "goal", "conclusion", "belief"})
_ATOM_KINDS = _CARD_KINDS | _FACT_KINDS
_CACHE_KIND = "cache-pointer"
_MAX_ATOMS = 4
_SEARCH_LIMIT = 16
_MIN_SCORE = 2.0
_SYSTEM_ROLES = frozenset({"system", "developer"})
_FILE_EXT = frozenset(
    {
        "c",
        "cc",
        "cpp",
        "css",
        "el",
        "go",
        "h",
        "hpp",
        "html",
        "js",
        "json",
        "md",
        "org",
        "pdf",
        "png",
        "py",
        "rs",
        "sh",
        "svg",
        "toml",
        "ts",
        "txt",
        "xml",
        "yaml",
        "yml",
    }
)
_REMOTE_WORD = re.compile(
    r"(?i)(?:^|[^a-z0-9])(github|issues|labels|gh)(?:[^a-z0-9]|$)"
)
_URL = re.compile(r"(?i)https?://")
_DOTTED = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.I)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for part in (_as_text(item) for item in value) if part)
    if isinstance(value, dict):
        if "text" in value:
            return value.get("text") or ""
        return _as_text(value.get("content"))
    return ""


_TOOL_ONLY = frozenset({"tool_result", "tool_use"})


def _user_turn_text(content: Any) -> str:
    """Visible user text. Tool-result blocks are not a turn."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in _TOOL_ONLY:
                continue
            text = (
                _user_turn_text(item)
                if isinstance(item, (list, dict))
                else _as_text(item)
            )
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        if content.get("type") in _TOOL_ONLY:
            return ""
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content:
            return _user_turn_text(content.get("content"))
        return _as_text(content).strip()
    return _as_text(content).strip()


_USER_QUERY = re.compile(
    r"(?is)<\s*user_query\s*>\s*(.*?)\s*<\s*/\s*user_query\s*>"
)


def _unwrap_user_query(text: str) -> str:
    """Stock Grok Build wraps the turn in <user_query>; search the inner text."""
    blob = (text or "").strip()
    if not blob:
        return ""
    match = _USER_QUERY.search(blob)
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner
    return blob


def _is_user_turn(item: dict[str, Any]) -> bool:
    """True for chat and Responses user turns, including type=user without role."""
    role = item.get("role")
    kind = item.get("type")
    if role in {"user", "human"}:
        return True
    # Grok Build ACP / compaction: {"type": "user", "content": [...]} (no role).
    if kind in {"user", "human"}:
        return True
    if kind == "message" and role in {None, "user", "human"}:
        return True
    return False


def _last_user_text(items: list[Any]) -> str:
    last = ""
    for item in items:
        if not isinstance(item, dict):
            continue
        if not _is_user_turn(item):
            continue
        text = _user_turn_text(item.get("content") if "content" in item else item)
        text = _unwrap_user_query(text)
        if text:
            last = text
    return last


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    inner = tool.get("function")
    if isinstance(inner, dict):
        nested = inner.get("name")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return ""


def _looks_like_host(token: str) -> bool:
    last = token.rsplit(".", 1)[-1].lower()
    if last in _FILE_EXT:
        return False
    return "." in token and len(last) >= 2


def _mentions_remote(text: str) -> bool:
    if not text:
        return False
    if _REMOTE_WORD.search(text):
        return True
    if _URL.search(text):
        return True
    return any(_looks_like_host(match.group(1)) for match in _DOTTED.finditer(text))


def inspect(body: dict) -> dict[str, Any]:
    """Read the current user turn and tool names. Does not rewrite the body."""
    if not isinstance(body, dict):
        body = {}
    user_text = ""
    messages = body.get("messages")
    if isinstance(messages, list):
        user_text = _last_user_text(messages)
    if not user_text:
        raw_input = body.get("input")
        if isinstance(raw_input, str) and raw_input.strip():
            user_text = raw_input.strip()
        elif isinstance(raw_input, list):
            user_text = _last_user_text(raw_input)
    tool_names: list[str] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            name = _tool_name(tool)
            if name:
                tool_names.append(name)
    remote_hay = user_text + "\n" + "\n".join(tool_names)
    return {
        "user_text": user_text,
        "tool_names": tool_names,
        "wants_remote": _mentions_remote(remote_hay),
    }


def _atom_sort_key(atom: dict[str, Any]) -> tuple[str, str]:
    return (str(atom.get("id") or ""), atom.get("text") or "")


def _fact_sort_key(atom: dict[str, Any]) -> tuple[int, str, str]:
    due = 0 if atom.get("withheld") or inside_memory.is_due(atom) else 1
    ident, text = _atom_sort_key(atom)
    return (due, ident, text)


def _due_pointer(atom: dict[str, Any]) -> dict[str, Any]:
    rec = dict(atom)
    aid = str(rec.get("id") or "")
    kind = rec.get("kind") or "atom"
    rec["text"] = f"`packset:{kind}:{aid}`"
    rec["withheld"] = True
    return rec


def _front_due(
    due: list[dict[str, Any]], tail: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for atom in due:
        if not isinstance(atom, dict):
            continue
        aid = str(atom.get("id") or "")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        out.append(_due_pointer(atom))
    for atom in tail:
        if not isinstance(atom, dict):
            continue
        aid = str(atom.get("id") or "")
        if aid and aid in seen:
            continue
        if aid:
            seen.add(aid)
        out.append(atom)
        if len(out) >= _MAX_ATOMS:
            break
    return out[:_MAX_ATOMS]


def _query(hints: dict[str, Any]) -> str:
    parts = [str(hints.get("user_text") or "")]
    for name in hints.get("tool_names") or []:
        parts.append(str(name))
    return " ".join(part for part in parts if part.strip())


def _scored_paragraphs(text: str, query: str) -> str:
    if not query.strip():
        return ""
    hits = [
        para
        for para in inside_search.paragraphs(text)
        if inside_search.score_text(query, para) >= _MIN_SCORE
    ]
    return "\n".join(hits)


_BLOCK_OPEN = "Seat memory:"
_BLOCK_CLOSE = "End seat memory."
_CARD_MARK = "Cards:"
_FACT_MARK = "Facts:"
_ATOM_BUDGET = 800


def _head_prefix(user: str, memory: str, voice_atoms: list[dict[str, Any]]) -> str:
    """Cards only. voice_atoms is ignored; facts stay on the tail."""
    del voice_atoms
    parts: list[str] = [_BLOCK_OPEN]
    cards = [p for p in ((user or "").strip(), (memory or "").strip()) if p]
    if not cards:
        return ""
    parts.append(_CARD_MARK)
    parts.extend(cards)
    return "\n".join(parts)


def _selected_text(selected: dict[str, Any]) -> str:
    cards: list[str] = []
    instructions = (selected.get("instructions") or "").strip()
    if instructions:
        cards.append(instructions)
    user_bits = (selected.get("user_bits") or "").strip()
    memory_bits = (selected.get("memory_bits") or "").strip()
    if user_bits:
        cards.append(user_bits)
    if memory_bits:
        cards.append(memory_bits)
    for atom in selected.get("head_atoms") or []:
        text = (atom.get("text") or "").strip()
        if text:
            cards.append(text)
    if not cards:
        head = selected.get("head_prefix") or ""
        if head.startswith(_BLOCK_OPEN):
            rest = head[len(_BLOCK_OPEN) :].strip("\n")
            if rest:
                cards.append(rest)
        elif head.strip():
            cards.append(head.strip())
    facts: list[str] = []
    used = 0
    for atom in sorted(selected.get("tail_atoms") or [], key=_fact_sort_key):
        if not isinstance(atom, dict):
            continue
        rec = atom if atom.get("withheld") else _due_pointer(atom)
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        if used and used + len(text) + 1 > _ATOM_BUDGET:
            break
        facts.append(text)
        used += len(text) + 1
    attach = (selected.get("attach") or "").strip()
    if attach:
        label = (selected.get("attach_label") or "").strip()
        extra = f"{label}: {attach}" if label else attach
        if used + len(extra) + 1 <= _ATOM_BUDGET:
            facts.append(extra)
    inner: list[str] = []
    if cards:
        inner.append(_CARD_MARK)
        inner.extend(cards)
    if facts:
        inner.append(_FACT_MARK)
        inner.extend(facts)
    if not inner:
        return ""
    return _BLOCK_OPEN + "\n" + "\n".join(inner) + "\n" + _BLOCK_CLOSE


def _check_caps(user: str, memory: str) -> None:
    if len(user) > inside_memory.USER_CAP:
        raise inside_memory.MemoryOverflow(
            f"USER.md is {len(user)} characters; cap is {inside_memory.USER_CAP}"
        )
    if len(memory) > inside_memory.MEMORY_CAP:
        raise inside_memory.MemoryOverflow(
            f"MEMORY.md is {len(memory)} characters; cap is {inside_memory.MEMORY_CAP}"
        )


def select_from_hits(
    hits: list[dict[str, Any]],
    hints: dict,
    *,
    atoms: dict[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Turn /v1/search hits into a splice neighbourhood."""
    hints = hints if isinstance(hints, dict) else {}
    query = _query(hints)
    wants_remote = bool(hints.get("wants_remote"))
    live = atoms if isinstance(atoms, dict) else {}
    user_parts: list[str] = []
    memory_parts: list[str] = []
    picked: list[dict[str, Any]] = []
    card_atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_text: set[str] = set()
    ranked = (
        [hit for hit in hits if isinstance(hit, dict)] if isinstance(hits, list) else []
    )

    def _take_file(field: str, parts: list[str]) -> None:
        for hit in ranked:
            if hit.get("field") != field:
                continue
            bits = _scored_paragraphs(str(hit.get("text") or ""), query)
            for line in bits.splitlines():
                norm = " ".join(line.lower().split())
                if not norm or norm in seen_text:
                    continue
                seen_text.add(norm)
                parts.append(line)

    _take_file("user", user_parts)
    _take_file("memory", memory_parts)
    for hit in ranked:
        if hit.get("field") != "atom":
            continue
        if float(hit.get("score") or 0.0) < _MIN_SCORE:
            continue
        atom = live.get(hit.get("id"))
        if atom is None:
            kind = hit.get("kind")
            text = str(hit.get("text") or "").strip()
            if not text or not hit.get("id"):
                continue
            atom = {
                "id": hit.get("id"),
                "kind": kind,
                "text": text,
                "due_at": hit.get("due_at"),
            }
        kind = atom.get("kind")
        if kind == _CACHE_KIND:
            if not wants_remote:
                continue
        elif kind in _CARD_KINDS:
            pass
        elif kind not in _FACT_KINDS:
            continue
        if atom.get("due_at") and not inside_memory.is_due(atom):
            continue
        aid = str(atom.get("id"))
        if aid in seen:
            continue
        norm = " ".join((atom.get("text") or "").lower().split())
        if not norm or norm in seen_text:
            continue
        seen.add(aid)
        seen_text.add(norm)
        if kind in _CARD_KINDS:
            card_atoms.append(atom)
            continue
        picked.append(atom)
        if len(picked) >= _MAX_ATOMS:
            break
    user_bits = "\n".join(user_parts)
    memory_bits = "\n".join(memory_parts)
    return {
        "user": "",
        "memory": "",
        "user_bits": user_bits,
        "memory_bits": memory_bits,
        "head_atoms": card_atoms,
        "tail_atoms": picked,
        "head_prefix": _head_prefix(user_bits, memory_bits, []),
    }


def select(pack: dict, hints: dict) -> dict[str, Any]:
    """Rank a local pack. Tests only; the shim calls /v1/search."""
    pack = pack if isinstance(pack, dict) else {}
    hints = hints if isinstance(hints, dict) else {}
    user = pack.get("user") or ""
    memory = pack.get("memory") or ""
    if not isinstance(user, str):
        user = str(user)
    if not isinstance(memory, str):
        memory = str(memory)
    _check_caps(user, memory)
    query = _query(hints)
    atoms = pack.get("atoms") if isinstance(pack.get("atoms"), list) else []
    live = {
        atom.get("id"): atom
        for atom in atoms
        if isinstance(atom, dict)
        and atom.get("id")
        and (inside_memory.is_live(atom) or inside_memory.is_due(atom))
    }
    hits = inside_search.search_pack_linear(pack, query, limit=_SEARCH_LIMIT)
    selected = select_from_hits(hits, hints, atoms=live)
    due = inside_memory.due_atoms(
        str(pack.get("workspace") or ""),
        atoms=list(live.values()),
    )
    selected["tail_atoms"] = _front_due(due, selected.get("tail_atoms") or [])
    selected["user"] = user
    selected["memory"] = memory
    return selected


def items_from_selected(selected: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten scored file lines and tail atoms for the relevance pass."""
    selected = selected if isinstance(selected, dict) else {}
    items: list[dict[str, Any]] = []
    for line in (selected.get("user_bits") or "").splitlines():
        text = line.strip()
        if text:
            items.append({"field": "user", "kind": "user", "text": text})
    for line in (selected.get("memory_bits") or "").splitlines():
        text = line.strip()
        if text:
            items.append({"field": "memory", "kind": "memory", "text": text})
    for atom in selected.get("tail_atoms") or []:
        if not isinstance(atom, dict):
            continue
        text = (atom.get("text") or "").strip()
        if not text:
            continue
        items.append(
            {
                "field": "atom",
                "kind": atom.get("kind") or "atom",
                "text": text,
                "atom": atom,
            }
        )
    return items


def selected_from_items(
    selected: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Rebuild the splice payload from the claims the judge kept."""
    selected = dict(selected) if isinstance(selected, dict) else {}
    user_bits_parts: list[str] = []
    memory_bits_parts: list[str] = []
    tail: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        if field == "user":
            text = (item.get("text") or "").strip()
            if text:
                user_bits_parts.append(text)
            continue
        if field == "memory":
            text = (item.get("text") or "").strip()
            if text:
                memory_bits_parts.append(text)
            continue
        atom = item.get("atom")
        if isinstance(atom, dict) and len(tail) < _MAX_ATOMS:
            tail.append(atom)
    user_bits = "\n".join(user_bits_parts)
    memory_bits = "\n".join(memory_bits_parts)
    selected["user_bits"] = user_bits
    selected["memory_bits"] = memory_bits
    selected["tail_atoms"] = tail
    selected["head_prefix"] = _head_prefix(user_bits, memory_bits, [])
    return selected


def _message_text(message: dict[str, Any]) -> str:
    return _as_text(message.get("content"))


def _set_message_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        block = dict(content[0])
        if "text" in block:
            block["text"] = text
            message["content"] = [block, *list(content[1:])]
            return message
    message["content"] = text
    return message


def _splice_message_list(messages: list[Any], text: str, prefix: str) -> list[Any]:
    out = list(messages)
    if (
        out
        and isinstance(out[0], dict)
        and out[0].get("role") in _SYSTEM_ROLES
        and _message_text(out[0]).startswith(prefix)
    ):
        first = dict(out[0])
        _set_message_text(first, text)
        out[0] = first
        return out
    out.insert(0, {"role": "system", "content": text})
    return out


def _splice_input_list(items: list[Any], text: str, prefix: str) -> list[Any]:
    out = list(items)
    if out and isinstance(out[0], dict):
        first = out[0]
        role = first.get("role")
        body = _as_text(first.get("content") if "content" in first else first)
        if role in _SYSTEM_ROLES and body.startswith(prefix):
            replaced = dict(first)
            replaced["role"] = role
            if "type" not in replaced:
                replaced["type"] = "message"
            _set_message_text(replaced, text)
            out[0] = replaced
            return out
    out.insert(0, {"type": "message", "role": "system", "content": text})
    return out


def _splice_system_field(payload: dict[str, Any], text: str, prefix: str) -> None:
    system = payload.get("system")
    if isinstance(system, str):
        if not system or system.startswith(prefix):
            payload["system"] = text
        else:
            payload["system"] = text + "\n" + system
        return
    if isinstance(system, list):
        for index, block in enumerate(system):
            if isinstance(block, dict) and _as_text(block).startswith(prefix):
                updated = dict(block)
                updated["text"] = text
                system[index] = updated
                payload["system"] = system
                return
            if isinstance(block, str) and block.startswith(prefix):
                system[index] = text
                payload["system"] = system
                return
        payload["system"] = [{"type": "text", "text": text}, *system]
        return
    payload["system"] = text


def splice(body: bytes | dict, selected: dict) -> bytes:
    """Insert the ranked neighbourhood. Idempotent. Does not trim."""
    selected = selected if isinstance(selected, dict) else {}
    user = selected.get("user") or ""
    memory = selected.get("memory") or ""
    if not isinstance(user, str):
        user = str(user)
    if not isinstance(memory, str):
        memory = str(memory)
    _check_caps(user, memory)
    raw_in: bytes | None
    if isinstance(body, (bytes, bytearray)):
        raw_in = bytes(body)
        if not raw_in:
            return raw_in
        try:
            payload = json.loads(raw_in)
        except (ValueError, UnicodeDecodeError):
            return raw_in
    elif isinstance(body, dict):
        raw_in = None
        payload = body
    else:
        return b""
    if not isinstance(payload, dict):
        return raw_in if raw_in is not None else json.dumps(payload).encode("utf-8")
    text = _selected_text(selected)
    prefix = selected.get("head_prefix") or ""
    if not text:
        if raw_in is not None:
            return raw_in
        return json.dumps(payload).encode("utf-8")
    payload = dict(payload)
    messages = payload.get("messages")
    inputs = payload.get("input")
    if isinstance(messages, list):
        if isinstance(payload.get("system"), (str, list)):
            _splice_system_field(payload, text, prefix)
        else:
            payload["messages"] = _splice_message_list(messages, text, prefix)
    elif isinstance(inputs, list):
        payload["input"] = _splice_input_list(inputs, text, prefix)
    elif isinstance(inputs, str):
        payload["input"] = [
            {"type": "message", "role": "system", "content": text},
            {"type": "message", "role": "user", "content": inputs},
        ]
    elif "system" in payload:
        _splice_system_field(payload, text, prefix)
    else:
        payload["messages"] = [{"role": "system", "content": text}]
    # Responses also carries durable system text on `instructions`. Stock
    # Grok Build always sets it; splicing only into input can leave the
    # model reading a system that never saw the seat pack.
    if "instructions" in payload or (
        isinstance(inputs, list) and not isinstance(messages, list)
    ):
        existing = payload.get("instructions")
        if isinstance(existing, str):
            if prefix and existing.startswith(prefix):
                payload["instructions"] = text
            elif text not in existing:
                payload["instructions"] = (
                    (existing.rstrip() + "\n\n" + text) if existing.strip() else text
                )
        elif existing is None or existing == "":
            payload["instructions"] = text
    return json.dumps(payload).encode("utf-8")


def fetch_pin_payload(url: str, workspace: str) -> dict[str, Any]:
    """GET {url}/v1/pin?workspace=... Object with set and instructions."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    query = urllib.parse.urlencode({"workspace": workspace})
    target = f"{base}/v1/pin?{query}"
    with urllib.request.urlopen(target, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pin response is not an object")
    return payload


def fetch_pin(url: str, workspace: str) -> str:
    """GET {url}/v1/pin?workspace=... Empty string when nothing is pinned."""
    return str(fetch_pin_payload(url, workspace).get("set") or "").strip()


def fetch_attach(url: str, workspace: str, *, peek: bool = False) -> dict[str, Any]:
    """GET {url}/v1/attach?workspace=... Consumes unless peek=True."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    fields = {"workspace": workspace}
    if peek:
        fields["peek"] = "1"
    target = f"{base}/v1/attach?{urllib.parse.urlencode(fields)}"
    with urllib.request.urlopen(target, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("attach response is not an object")
    return payload


def fetch_pack(url: str, workspace: str) -> dict[str, Any]:
    """GET {url}/v1/pack?workspace=... Not the splice retrieve."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    query = urllib.parse.urlencode({"workspace": workspace})
    target = f"{base}/v1/pack?{query}"
    with urllib.request.urlopen(target, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pack response is not an object")
    return payload


def fetch_search(
    url: str,
    workspace: str,
    query: str,
    *,
    limit: int = _SEARCH_LIMIT,
    set_name: str | None = None,
) -> list[dict[str, Any]]:
    """GET {url}/v1/search?workspace=&q=&limit=&set=."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    fields = {
        "workspace": workspace,
        "q": query,
        "limit": max(0, int(limit)),
    }
    if set_name:
        fields["set"] = set_name
    params = urllib.parse.urlencode(fields)
    target = f"{base}/v1/search?{params}"
    with urllib.request.urlopen(target, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("search response is not an object")
    hits = payload.get("hits")
    if not isinstance(hits, list):
        raise ValueError("search response hits are not a list")
    return [hit for hit in hits if isinstance(hit, dict)]


def fetch_recall(
    url: str,
    workspace: str,
    *,
    limit: int = 64,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """GET {url}/v1/recall?workspace=&limit=&q=."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    fields: dict[str, Any] = {
        "workspace": workspace,
        "limit": max(0, int(limit)),
    }
    if query:
        fields["q"] = query
    params = urllib.parse.urlencode(fields)
    target = f"{base}/v1/recall?{params}"
    with urllib.request.urlopen(target, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recall response is not an object")
    atoms = payload.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError("recall response atoms are not a list")
    return [atom for atom in atoms if isinstance(atom, dict)]


def post_grade(
    url: str, workspace: str, atom_id: str, *, recalled: bool = True
) -> dict[str, Any]:
    """POST {url}/v1/grade. Complementary review-clock write."""
    base = (url or "").rstrip("/")
    if not base:
        raise ValueError("memory url is empty")
    payload = json.dumps(
        {"workspace": workspace, "id": atom_id, "recalled": bool(recalled)}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/grade",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("grade response is not an object")
    return body


def grade_due(url: str, workspace: str, due: list[dict[str, Any]]) -> None:
    """Stretch due_at after a retrieve. Fail-open on a dead grade."""
    for atom in due:
        aid = str(atom.get("id") or "")
        if not aid:
            continue
        try:
            post_grade(url, workspace, aid)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue


def retrieve(url: str, workspace: str, hints: dict) -> dict[str, Any]:
    """Search the pack, then front the due queue from /v1/recall."""
    pin_info = fetch_pin_payload(url, workspace)
    pin = str(pin_info.get("set") or "").strip()
    query = _query(hints)
    selected = select_from_hits(
        fetch_search(url, workspace, query, set_name=pin or None),
        hints,
    )
    try:
        recalled = fetch_recall(url, workspace, query=query or None)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        recalled = []
    due = inside_memory.due_atoms(workspace, atoms=recalled)
    if pin:
        due = [atom for atom in due if str(atom.get("set") or "") == pin]
    selected["tail_atoms"] = _front_due(due, selected.get("tail_atoms") or [])
    grade_due(url, workspace, due)
    instructions = str(pin_info.get("instructions") or "").strip()
    if instructions:
        selected["instructions"] = instructions
    return selected
