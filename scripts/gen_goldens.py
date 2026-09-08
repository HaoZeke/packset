#!/usr/bin/env python3
"""Write the corpus the Rust port is checked against.

The Python modules are the reference while the port runs alongside them, so
parity is a generated file rather than a second reading of the same rules. Run
`just goldens` after changing either side and read the diff.
"""
from __future__ import annotations

import json
from pathlib import Path

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

    OUT.write_text(
        json.dumps({"prose": prose, "remote": remote, "slug": slug, "set": sets}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
