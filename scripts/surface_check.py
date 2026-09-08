#!/usr/bin/env python3
"""Run one request sequence against both writers and diff the answers.

A port is finished when a client cannot tell which one it is talking to, so
this asks both the same questions in the same order and compares every status
code and body. Volatile fields (ids, timestamps, the home path, the port) are
normalised, because they differ between two runs of the same writer too.

    python3 scripts/surface_check.py <path-to-packsetd-binary>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

WS = "git:github.com/HaoZeke/vissue"

VOLATILE = {
    "id",
    "ts",
    "valid_from",
    "due_at",
    "last",
    "last_write_ts",
    "home",
    "index_dir",
    # Seat facts, not writer facts: the same on one machine, different on two.
    "user_peer",
    "cwd",
}


def request(port: int, method: str, path: str, body: dict | None = None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
            return resp.status, _parse(raw)
    except urllib.error.HTTPError as exc:
        return exc.code, _parse(exc.read().decode())


def _parse(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def collect_ids(value, ids: dict[str, str]) -> None:
    """Name every atom id after its own text.

    Order of appearance will not do: the ids are random per run, so the two
    writers hand back the same atoms in different orders and an
    order-of-appearance name would encode the order rather than the atom.
    """
    if isinstance(value, dict):
        atom_id = value.get("id")
        text = value.get("text")
        if isinstance(atom_id, str) and isinstance(text, str):
            ids.setdefault(atom_id, f"<id:{text}>")
        for item in value.values():
            collect_ids(item, ids)
    elif isinstance(value, list):
        for item in value:
            collect_ids(item, ids)


def normalise(value, ids: dict[str, str]):
    """Replace volatile values and name ids after their atom."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "id" and isinstance(item, str):
                out[key] = ids.get(item, "<id:unknown>")
            elif key == "links" and isinstance(item, list):
                out[key] = sorted(
                    ids.get(x, "<id:unknown>") if isinstance(x, str) else x for x in item
                )
            elif key in VOLATILE:
                out[key] = f"<{key}>" if item is not None else None
            else:
                out[key] = normalise(item, ids)
        return out
    if isinstance(value, list):
        return [normalise(v, ids) for v in value]
    return value


def wait_for(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.15)
    raise SystemExit(f"nothing came up on {port}")


def sequence(port: int):
    """The same calls, in the same order, against whichever writer is listening."""
    steps: list[tuple[str, object]] = []

    def call(label, method, path, body=None):
        steps.append((label, request(port, method, path, body)))

    call("workspaces empty", "GET", "/v1/workspaces")
    call("status empty", "GET", f"/v1/status?workspace={WS}")
    call("atoms empty", "GET", f"/v1/atoms?workspace={WS}")
    call("pack empty", "GET", f"/v1/pack?workspace={WS}")
    call("pin empty", "GET", f"/v1/pin?workspace={WS}")

    call("atoms needs a workspace", "GET", "/v1/atoms")
    call("unknown route", "GET", "/v1/nope")
    call("old health is gone", "GET", "/__inside_memd/health")

    base = {
        "workspace": WS,
        "about_peer": "rgoswami",
        "by_peer": "hermes",
    }
    call("add one", "POST", "/v1/atoms",
         {**base, "text": "Reviews open with a check.", "kind": "voice",
          "entities": ["Parser", "Header"]})
    call("add it again", "POST", "/v1/atoms",
         {**base, "text": "Reviews open with a check.", "kind": "voice",
          "entities": ["Parser", "Header"]})
    call("add a peer", "POST", "/v1/atoms",
         {**base, "text": "The Header sits before the Parser body.", "kind": "conclusion",
          "entities": ["Parser", "Header"]})
    call("add an unrelated one", "POST", "/v1/atoms",
         {**base, "text": "Prefer ripgrep for search.", "kind": "preference",
          "entities": ["ripgrep"]})

    call("bad kind", "POST", "/v1/atoms", {**base, "text": "nope", "kind": "vibes"})
    call("no text", "POST", "/v1/atoms", {**base, "text": "   ", "kind": "voice"})
    call("no workspace", "POST", "/v1/atoms",
         {"text": "nope", "kind": "voice", "workspace": "", "about_peer": "a", "by_peer": "b"})
    call("bad set", "POST", "/v1/atoms",
         {**base, "text": "nope", "kind": "voice", "set": "../etc"})
    call("bare accession", "POST", "/v1/atoms",
         {**base, "text": "nope", "kind": "voice", "entities": ["deed-"]})
    call("split accession", "POST", "/v1/atoms",
         {**base, "text": "nope", "kind": "voice", "entities": ["deed-a,b"]})
    call("three sentences", "POST", "/v1/atoms",
         {**base, "text": "One. Two. Three.", "kind": "voice"})
    call("a credential", "POST", "/v1/atoms",
         {**base, "text": "api_key=abcd1234efgh", "kind": "voice"})
    listing = "total 48\n" + "\n".join(
        f"-rw-r--r-- 1 x x 0 Jan 1 00:00 file{i}" for i in range(7)
    )
    call("a tool dump", "POST", "/v1/atoms", {**base, "text": listing, "kind": "voice"})

    call("atoms after writes", "GET", f"/v1/atoms?workspace={WS}")
    call("workspaces after writes", "GET", "/v1/workspaces")
    call("status after writes", "GET", f"/v1/status?workspace={WS}")

    call("cards", "PUT", "/v1/user", {"text": "Open review links after pushing.\n"})
    call("memory", "PUT", "/v1/memory", {"workspace": WS, "text": "The port is a swap.\n"})
    call("card overflow", "PUT", "/v1/user", {"text": "One small claim. " * 200})
    call("memory needs a workspace", "PUT", "/v1/memory", {"text": "x"})
    call("pack with cards", "GET", f"/v1/pack?workspace={WS}")

    call("pin a set", "PUT", "/v1/pin", {"workspace": WS, "set": "Review"})
    call("read the pin", "GET", f"/v1/pin?workspace={WS}")
    call("bad pin", "PUT", "/v1/pin", {"workspace": WS, "set": "../etc"})
    call("set pack", "GET", f"/v1/set?workspace={WS}&name=review")
    call("set pack needs a name", "GET", f"/v1/set?workspace={WS}")
    call("clear the pin", "PUT", "/v1/pin", {"workspace": WS, "set": ""})

    call("recall", "GET", f"/v1/recall?workspace={WS}")
    call("recall limited", "GET", f"/v1/recall?workspace={WS}&limit=2")
    call("recall zero", "GET", f"/v1/recall?workspace={WS}&limit=0")
    call("recall negative", "GET", f"/v1/recall?workspace={WS}&limit=-3")
    call("recall over the cap", "GET", f"/v1/recall?workspace={WS}&limit=9999")
    call("recall bad limit", "GET", f"/v1/recall?workspace={WS}&limit=abc")
    call("recall by hint", "GET", f"/v1/recall?workspace={WS}&q=ripgrep")
    call("recall needs a workspace", "GET", "/v1/recall")

    call("identity", "GET", "/v1/identity?cwd=/tmp&harness=hermes")

    here = str(Path(__file__).resolve().parent.parent)
    call("rules", "GET", f"/v1/rules?cwd={here}")
    call("rules with bodies", "GET", f"/v1/rules?cwd={here}&body=1")
    call("rules outside a tree", "GET", "/v1/rules?cwd=/tmp")
    call("skills", "GET", f"/v1/skills?cwd={here}")
    call("skills by name", "GET", f"/v1/skills?cwd={here}&name=nothing-here")
    call("map", "GET", f"/v1/map?cwd={here}")
    call("map outside a tree", "GET", "/v1/map?cwd=/")

    call("attach", "POST", "/v1/attach", {"workspace": WS, "text": "a log body", "label": "build"})
    call("peek", "GET", f"/v1/attach?workspace={WS}&peek=1")
    call("take", "GET", f"/v1/attach?workspace={WS}")
    call("take again", "GET", f"/v1/attach?workspace={WS}")

    return steps


def _sort_atoms(body):
    """Atom order follows the random ids, so it is not part of the contract.

    Both writers scan the store in key order, and the key carries the id, so
    two runs of the same writer come out in different orders too.
    """
    if isinstance(body, dict):
        out = {}
        for key, value in body.items():
            if key == "atoms" and isinstance(value, list):
                out[key] = sorted(value, key=lambda a: json.dumps(a, sort_keys=True))
            else:
                out[key] = _sort_atoms(value)
        return out
    if isinstance(body, list):
        return [_sort_atoms(v) for v in body]
    return body


def mask_root(value, root: str):
    """Replace this writer's own pack home wherever it appears.

    Each daemon runs on its own temp home, so a path into it is a fact about
    the run rather than about the writer. It shows up inside strings, not only
    as a whole field, which is why this is a substring replace.
    """
    if isinstance(value, str):
        return value.replace(root, "<home>")
    if isinstance(value, dict):
        return {key: mask_root(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_root(item, root) for item in value]
    return value


def with_ids(steps, root: str):
    ids: dict[str, str] = {}
    for _, (_, body) in steps:
        collect_ids(body, ids)
    # Named first, then sorted, so the sort key does not carry a random id.
    return [
        (label, code, _sort_atoms(normalise(mask_root(body, root), ids)))
        for label, (code, body) in steps
    ]


def main() -> int:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/packsetd")
    if not binary.is_file():
        raise SystemExit(f"missing {binary}; cargo build -p packset-daemon")
    here = Path(__file__).resolve().parent

    results = {}
    for label, argv, port in (
        ("python", [sys.executable, str(here / "packsetd.py")], 8771),
        ("rust", [str(binary)], 8772),
    ):
        root = Path(tempfile.mkdtemp(prefix=f"packset-{label}-"))
        proc = subprocess.Popen(
            [*argv, "--port", str(port), "--home", str(root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=str(here),
        )
        try:
            wait_for(port)
            results[label] = with_ids(sequence(port), str(root))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            shutil.rmtree(root, ignore_errors=True)

    differences = 0
    for (label, py_code, py_body), (_, rs_code, rs_body) in zip(
        results["python"], results["rust"], strict=True
    ):
        if py_code != rs_code or py_body != rs_body:
            differences += 1
            print(f"--- {label}")
            print(f"    python {py_code}: {json.dumps(py_body, sort_keys=True)[:400]}")
            print(f"    rust   {rs_code}: {json.dumps(rs_body, sort_keys=True)[:400]}")
    total = len(results["python"])
    if differences:
        print(f"\n{differences} of {total} calls differ")
        return 1
    print(f"all {total} calls agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
