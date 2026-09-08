#!/usr/bin/env python3
"""How many links a pack accumulates as it grows."""
import json, subprocess, sys, tempfile, time, shutil, urllib.request

BIN, PORT = sys.argv[1], int(sys.argv[2])
WS = "w"
def req(path, body=None, method="GET"):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                               data=json.dumps(body).encode() if body else None, method=method)
    if body: r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=120) as resp: return json.loads(resp.read().decode())

WORDS = ["parser","header","overlay","ripgrep","review","manifest","record","token",
         "commit","branch","index","atom","workspace","daemon","search","pack"]
root = tempfile.mkdtemp()
p = subprocess.Popen([BIN,"--port",str(PORT),"--home",root], stderr=subprocess.DEVNULL)
try:
    for _ in range(200):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
        except OSError: time.sleep(0.1)
    total = 0
    print(f"{'atoms':>7} {'links':>9} {'links/atom':>11} {'bytes':>10}")
    for target in (50, 200, 1000, 4000):
        for i in range(total, target):
            w = WORDS[i % len(WORDS)]
            req("/v1/atoms", {"workspace": WS, "kind":"conclusion","about_peer":"a","by_peer":"b",
                              "text": f"The {w} number {i} settled the question.",
                              "entities": [w, f"n{i}"]}, "POST")
        total = target
        atoms = req(f"/v1/atoms?workspace={WS}")["atoms"]
        links = sum(len(a.get("links") or []) for a in atoms)
        size = len(json.dumps(atoms))
        print(f"{total:>7} {links:>9} {links/max(len(atoms),1):>11.1f} {size:>10}")
finally:
    p.terminate()
    try: p.wait(timeout=5)
    except subprocess.TimeoutExpired: p.kill()
    shutil.rmtree(root, ignore_errors=True)
