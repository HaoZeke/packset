#!/usr/bin/env python3
"""Write the corpus the Rust port is checked against.

The Python modules are the reference while the port runs alongside them, so
parity is a generated file rather than a second reading of the same rules. Run
`just goldens` after changing either side and read the diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import inside_extract
import inside_identity
import inside_memory
import inside_prose
import inside_set

OUT = Path(__file__).resolve().parent.parent / "crates/packset-core/tests/python_goldens.json"

TEXTS = [
    "",
    "Reviews open with a reproducibility check.",
    "One thing. Another thing. A third thing.",
    "The header was parsed by the reader.",
    "only family apply early daily",
    "quickly and very carefully the reviewer really basically just moved on",
    "word " * 20 + ".",
    "word " * 30 + ".",
    "Prefer ripgrep over grep for repository search because it respects gitignore.",
    "Always open review links after pushing, so the CI result is visible before the next change lands.",
    "The seat pack keeps one claim per atom and the cards stay files on disk.",
    "Tool dumps and fetched bodies are not atoms.",
    "packsetd listens on 127.0.0.1 only and never on localhost.",
    "AAA",
    "Make the change, then run the tests.",
]

REMOTES = [
    "https://github.com/HaoZeke/grok-inside.git",
    "git@github.com:HaoZeke/grok-inside.git",
    "ssh://git@github.com/HaoZeke/vissue.git",
    "https://example.com:8443/team/repo.git",
    "http://gitlab.com/group/sub/project.git",
    "git@bitbucket.org:team/repo",
]

WORKSPACES = [
    "git:github.com/HaoZeke/vissue",
    "global",
    "dir:/home/x/y",
    "///",
    "",
    "a.b_c-d",
]

ATOMS = [
    # kind / level / text / workspace, then the fields validate normalizes.
    {"kind": "voice", "text": "Reviews open with a reproducibility check.", "workspace": "w"},
    {"kind": "vibes", "text": "nope", "workspace": "w"},
    {"kind": "voice", "text": "nope", "workspace": "w", "level": "guessed"},
    {"kind": "voice", "text": "   ", "workspace": "w"},
    {"kind": "voice", "text": "no workspace", "workspace": ""},
    {"kind": "voice", "text": "x" * 501, "workspace": "w"},
    {"kind": "voice", "text": "api_key=sk-not-a-real-key-here", "workspace": "w"},
    {"kind": "voice", "text": "One. Two. Three.", "workspace": "w"},
    {"kind": "voice", "text": "Set scoped.", "workspace": "w", "set": "Review"},
    {"kind": "voice", "text": "Bad set.", "workspace": "w", "set": "../etc"},
    {"kind": "voice", "text": "Cited.", "workspace": "w", "entities": ["deed-patch-overlay", "JOSS"]},
    {"kind": "voice", "text": "Bare.", "workspace": "w", "entities": ["deed-"]},
    {"kind": "voice", "text": "Split.", "workspace": "w", "entities": ["deed-a,b"]},
]

TOOL_DUMPS = [
    "",
    "   ",
    "Reviews open with a check.",
    "Here is the run:\n```\nstdout: everything fine\n```",
    "```\nnothing named\n```",
    "<!DOCTYPE html><html><body>hi</body></html>",
    "<html>no doctype</html>",
    "total 48\n" + "\n".join(f"-rw-r--r-- 1 x x 0 Jan 1 00:00 file{i}" for i in range(7)),
    "total 48\n" + "\n".join(f"-rw-r--r-- 1 x x 0 Jan 1 00:00 file{i}" for i in range(4)),
    "\n".join(f"drwxr-xr-x 2 x x 4096 Jan 1 00:00 dir{i}" for i in range(8)),
]

CLAIMS = [
    "Remember: always pin the review set",
    "Prefer: ripgrep over grep",
    "Remember that: pin the review set",
    "Note that the test failed on line 12",
    "Prefer conventional commits",
    "remember: lowercase still counts",
    "",
]

ENTITY_TEXTS = [
    "The Parser reads the Header block.",
    "Use `ripgrep` not `grep` here.",
    "no capitals at all here",
    "A single A and an AB pair",
    "Mixed `back tick` and Capitalized runs",
    "",
]

LINK_CLOCK = "2026-01-01T00:00:00.000Z"
LINK_LIVE = [
    {"id": "one", "text": "one", "entities": ["Parser", "Header"], "links": []},
    {"id": "two", "text": "two", "entities": ["Parser", "Header", "Record"], "links": []},
    {"id": "three", "text": "three", "entities": ["Unrelated"], "links": ["subject"]},
    {"id": "four", "text": "four", "entities": ["Parser"], "valid_to": "2020-01-01T00:00:00.000Z"},
]
LINK_SUBJECT = {"id": "subject", "text": "subject", "entities": ["Parser", "Header"]}

REVIEW_CLOCK = "2026-01-01T00:00:00.000Z"
REVIEW_CASES = [
    ({}, "initial", None),
    ({}, "initial", 3600),
    ({}, "recalled", None),
    ({}, "lapsed", None),
    ({"reps": 3, "stability": 4.0, "difficulty": 6.0, "last": "2025-12-20T00:00:00.000Z"}, "recalled", None),
    ({"reps": 3, "stability": 4.0, "difficulty": 6.0, "last": "2025-12-20T00:00:00.000Z"}, "lapsed", None),
    ({"reps": 1, "stability": 0.2, "difficulty": 9.9, "last": "2026-01-01T00:00:00.000Z"}, "recalled", None),
]

SET_NAMES = [
    "Review",
    "  joss-reviews  ",
    "a",
    "1review",
    "-review",
    "",
    "../etc",
    "a/b",
    "a b",
    "a_b",
    "a" * 32,
    "a" * 33,
]


def main() -> int:
    prose = [{"text": t, **inside_prose.assess(t)} for t in TEXTS]

    remote = []
    for url in REMOTES:
        try:
            remote.append({"remote": url, "workspace": inside_identity.normalize_remote(url)})
        except ValueError as exc:
            remote.append({"remote": url, "error": str(exc)})

    slug = [
        {"workspace": w, "slug": inside_identity.workspace_slug(w)} for w in WORKSPACES
    ]

    sets = []
    for name in SET_NAMES:
        try:
            sets.append({"name": name, "set": inside_set.check_set_name(name)})
        except inside_memory.AtomError:
            sets.append({"name": name, "error": True})

    atoms = []
    for raw in ATOMS:
        try:
            atoms.append({"input": raw, "ok": inside_memory.validate_atom(dict(raw))})
        except (inside_memory.AtomError, inside_prose.ProseError) as exc:
            atoms.append({"input": raw, "error": str(exc)})

    dumps = [
        {"text": t, "is_dump": inside_extract.is_tool_dump(t)} for t in TOOL_DUMPS
    ]

    claims = []
    for line in CLAIMS:
        got = inside_extract.claim_from_user(line)
        claims.append(
            {"line": line, "kind": got[0], "claim": got[1]} if got else {"line": line}
        )

    entities = [
        {"text": t, "entities": sorted(inside_memory.extract_entities({"text": t}))}
        for t in ENTITY_TEXTS
    ]

    subject = dict(LINK_SUBJECT)
    live = [dict(a) for a in LINK_LIVE]
    rewritten = inside_memory.apply_links(subject, live, now=LINK_CLOCK)
    links = {
        "clock": LINK_CLOCK,
        "live": LINK_LIVE,
        "subject_in": LINK_SUBJECT,
        "subject_links": subject["links"],
        "rewritten": sorted(
            ({"id": p["id"], "links": p["links"]} for p in rewritten),
            key=lambda p: p["id"],
        ),
    }

    filtered = [
        {"id": "a", "links": ["b", "gone"]},
        {"id": "b", "links": ["a"]},
    ]
    inside_memory.filter_live_links(filtered)
    links["filtered"] = filtered

    review = []
    for block, grade, interval in REVIEW_CASES:
        atom = {"id": "a", "review": dict(block)} if block else {"id": "a"}
        out = inside_memory.schedule_review(
            atom,
            now=REVIEW_CLOCK,
            interval_s=interval,
            recalled=(grade == "recalled"),
            lapse=(grade == "lapsed"),
        )
        review.append(
            {
                "review_in": block,
                "grade": grade,
                "interval_s": interval,
                "due_at": out["due_at"],
                "review": out["review"],
            }
        )

    OUT.write_text(
        json.dumps(
            {
                "prose": prose,
                "remote": remote,
                "slug": slug,
                "set": sets,
                "atom": atoms,
                "entities": entities,
                "tool_dump": dumps,
                "claim": claims,
                "links": links,
                "review_clock": REVIEW_CLOCK,
                "review": review,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
