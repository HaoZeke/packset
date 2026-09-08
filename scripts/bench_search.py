#!/usr/bin/env python3
"""Where search time goes, as the pack grows and as callers pile up."""
import json, subprocess, sys, tempfile, time, shutil, os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BIN = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8801
WS = "git:github.com/HaoZeke/vissue"

def req(path, body=None, method="GET"):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if data: r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())

def wait():
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); return
        except OSError: time.sleep(0.1)
    raise SystemExit("no daemon")

WORDS = ["parser","header","overlay","ripgrep","review","manifest","record","token",
         "commit","branch","index","atom","workspace","daemon","search","pack"]

def seed(n, start=0):
    for i in range(start, start + n):
        w = WORDS[i % len(WORDS)]
        req("/v1/atoms", {
            "workspace": WS, "kind": "conclusion", "about_peer": "a", "by_peer": "b",
            "text": f"The {w} number {i} settled the question.",
            "entities": [w, f"n{i}"],
        }, "POST")

def timed(fn, reps=20):
    best = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); best.append(time.perf_counter() - t)
    best.sort()
    return best[len(best)//2] * 1000, best[-1] * 1000  # median, max in ms

root = tempfile.mkdtemp(prefix="packset-bench-")
env = dict(os.environ)
env.pop("PACKSET_MILLI", None) if "--no-milli" in sys.argv else None
proc = subprocess.Popen([BIN, "--port", str(PORT), "--home", root],
                        stderr=subprocess.DEVNULL, env=env)
try:
    wait()
    print(f"{'atoms':>7} {'search ms':>12} {'search max':>12} {'recall ms':>11} {'atoms ms':>10} {'write ms':>10}")
    total = 0
    for target in (50, 200, 1000, 4000):
        seed(target - total, total); total = target
        s_med, s_max = timed(lambda: req(f"/v1/search?workspace={WS}&q=parser&limit=16"))
        r_med, _ = timed(lambda: req(f"/v1/recall?workspace={WS}&limit=64"))
        a_med, _ = timed(lambda: req(f"/v1/atoms?workspace={WS}"), reps=10)
        counter = [total]
        def one_write():
            counter[0] += 1
            req("/v1/atoms", {"workspace": WS, "kind": "voice", "about_peer":"a","by_peer":"b",
                              "text": f"A distinct claim {counter[0]} here."}, "POST")
        w_med, _ = timed(one_write, reps=10)
        total = counter[0]
        print(f"{total:>7} {s_med:>12.1f} {s_max:>12.1f} {r_med:>11.1f} {a_med:>10.1f} {w_med:>10.1f}")

    print()
    for workers in (1, 4, 16, 64):
        def hit(_): return req(f"/v1/search?workspace={WS}&q=parser&limit=16")
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(hit, range(workers * 8)))
        dt = time.perf_counter() - t
        n = workers * 8
        print(f"{workers:>3} concurrent: {n} searches in {dt*1000:7.0f} ms  "
              f"= {n/dt:8.1f} req/s")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()
    shutil.rmtree(root, ignore_errors=True)
