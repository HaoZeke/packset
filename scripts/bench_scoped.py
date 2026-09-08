#!/usr/bin/env python3
"""A set-scoped search against an unscoped one, with the projection present."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

WS = "w"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8810


def call(path, body=None, method="GET"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def timed(fn, reps=30):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2] * 1000


root = Path(tempfile.mkdtemp())
proc = subprocess.Popen([sys.argv[1], "--port", str(PORT), "--home", str(root)],
                        stderr=subprocess.DEVNULL)
try:
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    for i in range(400):
        atom = {"workspace": WS, "kind": "conclusion", "about_peer": "a", "by_peer": "b",
                "text": f"The parser number {i} settled the question.", "entities": [f"n{i}"]}
        if i % 2 == 0:
            atom["set"] = "review"
        call("/v1/atoms", atom, "POST")
    plain = timed(lambda: call(f"/v1/search?workspace={WS}&q=parser&limit=16"))
    scoped = timed(lambda: call(f"/v1/search?workspace={WS}&q=parser&limit=16&set=review"))
    engine = call(f"/v1/search?workspace={WS}&q=parser&limit=16")["engine"]
    print(f"engine {engine}: unscoped {plain:.1f} ms, set-scoped {scoped:.1f} ms"
          f"  ({scoped / max(plain, 1e-9):.1f}x)")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(root, ignore_errors=True)
