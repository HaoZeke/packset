#!/usr/bin/env python3
"""Prove the two writers share one store, in both directions.

The port is a swap and not a migration, which is a claim about the file on
disk rather than about the code. So: Python writes, Rust reads it back; Rust
writes, Python reads it back. Any drift in the key layout, the JSON, or the
live-set rules shows up here rather than on somebody's seat.

    python3 scripts/interop_check.py <path-to-cargo-target-dir>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import packsetd  # noqa: E402


def run_rust(binary: Path, *args: str, stdin: str = "") -> str:
    proc = subprocess.run(
        [str(binary), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{binary.name} failed: {proc.stderr.strip()}")
    return proc.stdout


def sorted_records(lines: str) -> list[dict]:
    out = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return sorted(out, key=lambda r: (r.get("workspace", ""), r.get("id", "")))


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/examples")
    dump = target / "dump_store"
    write = target / "write_store"
    for binary in (dump, write):
        if not binary.is_file():
            raise SystemExit(f"missing {binary}; cargo build -p packset-daemon --examples")

    root = Path(tempfile.mkdtemp(prefix="packset-interop-"))
    try:
        # 1. Python writes.
        store = packsetd.Store(root)
        written = []
        for i, (text, kind, entities) in enumerate(
            [
                ("Reviews open with a check.", "voice", ["JOSS"]),
                ("Prefer ripgrep for search.", "preference", ["ripgrep"]),
                ("The overlay landed.", "conclusion", ["deed-patch-overlay"]),
            ]
        ):
            atom = store.add(
                {
                    "workspace": "git:github.com/HaoZeke/vissue",
                    "text": text,
                    "kind": kind,
                    "about_peer": "rgoswami",
                    "by_peer": "hermes",
                    "entities": entities,
                }
            )
            written.append(atom)
        # A second workspace whose name is a prefix of nothing, plus one that
        # is: the key separator is what keeps these apart.
        store.add(
            {
                "workspace": "git",
                "text": "A short workspace name.",
                "kind": "voice",
                "about_peer": "rgoswami",
                "by_peer": "hermes",
            }
        )
        store.close()

        py_all = sorted_records(
            "\n".join(json.dumps(r) for r in _scan_with_python(root, None))
        )
        rust_all = sorted_records(run_rust(dump, str(root)))
        _compare("python wrote, rust read", py_all, rust_all)

        py_scoped = sorted_records(
            "\n".join(
                json.dumps(r)
                for r in _scan_with_python(root, "git:github.com/HaoZeke/vissue")
            )
        )
        rust_scoped = sorted_records(
            run_rust(dump, str(root), "git:github.com/HaoZeke/vissue")
        )
        _compare("the workspace scan agrees", py_scoped, rust_scoped)
        if len(rust_scoped) != 3:
            raise SystemExit(
                f"the prefix scan leaked: expected 3, got {len(rust_scoped)}"
            )

        # 2. Rust writes.
        from_rust = {
            "id": "written-by-rust",
            "workspace": "git:github.com/HaoZeke/vissue",
            "kind": "lesson",
            "level": "explicit",
            "text": "The store is shared by both writers.",
            "about_peer": "rgoswami",
            "by_peer": "packsetd",
            "ts": "2026-01-01T00:00:00.000Z",
            "valid_from": "2026-01-01T00:00:00.000Z",
            "valid_to": None,
            "due_at": None,
            "links": [],
            "embedding": None,
            "trust": 1.0,
            "tombstone": False,
            "entities": ["packsetd"],
            "unmodelled": {"kept": [1, 2, 3]},
        }
        run_rust(write, str(root), stdin=json.dumps(from_rust))

        back = _scan_with_python(root, "git:github.com/HaoZeke/vissue")
        mine = [r for r in back if r.get("id") == "written-by-rust"]
        if not mine:
            raise SystemExit("rust wrote a record python cannot see")
        got = mine[0]
        for key, want in from_rust.items():
            if got.get(key) != want:
                raise SystemExit(
                    f"field {key} changed crossing the store: {got.get(key)!r} != {want!r}"
                )
        print("rust wrote, python read: ok (all fields, unmodelled included)")

        # 3. Python's own live-set view still holds with the Rust record in it.
        store = packsetd.Store(root)
        live = store.current("git:github.com/HaoZeke/vissue")
        store.close()
        if not any(r["id"] == "written-by-rust" for r in live):
            raise SystemExit("the rust record is not in python's live set")
        print(f"python's live set carries it: {len(live)} atoms")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _scan_with_python(root: Path, workspace: str | None) -> list[dict]:
    store = packsetd.Store(root)
    try:
        return store._scan(workspace)
    finally:
        store.close()


def _compare(label: str, left: list[dict], right: list[dict]) -> None:
    if left != right:
        for a, b in zip(left, right):
            if a != b:
                raise SystemExit(f"{label}: first difference\n  python {a}\n  rust   {b}")
        raise SystemExit(f"{label}: {len(left)} vs {len(right)} records")
    print(f"{label}: ok ({len(left)} records)")


if __name__ == "__main__":
    raise SystemExit(main())
