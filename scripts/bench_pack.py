#!/usr/bin/env python3
"""What a pack costs as it grows, and what the graph inside it looks like.

Two questions, because the answer to the second explains the first. A pack
whose atoms all link to each other is quadratic in its own size, and every read
pays for that whether or not the caller wanted the links.

    just bench                       # against the built writer
    python3 scripts/bench_pack.py <packsetd> [port] [--sizes 50,200,1000]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKSPACE = "git:github.com/HaoZeke/vissue"
DEFAULT_SIZES = (50, 200, 1000, 4000)
CONCURRENCY = (1, 4, 16, 64)

# Sixteen subjects, so atoms repeat an entity often enough for the link rule to
# fire. A corpus where nothing overlaps would measure nothing.
SUBJECTS = (
    "parser", "header", "overlay", "ripgrep", "review", "manifest", "record",
    "token", "commit", "branch", "index", "atom", "workspace", "daemon",
    "search", "pack",
)


class Client:
    """The bits of the surface this measures."""

    def __init__(self, port: int) -> None:
        self.port = port

    def call(self, path: str, body: dict | None = None, method: str = "GET"):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())

    def wait(self, seconds: float = 20.0) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1
                )
                return
            except OSError:
                time.sleep(0.1)
        raise SystemExit(f"nothing came up on {self.port}")

    def write(self, index: int) -> None:
        subject = SUBJECTS[index % len(SUBJECTS)]
        self.call(
            "/v1/atoms",
            {
                "workspace": WORKSPACE,
                "kind": "conclusion",
                "about_peer": "bench",
                "by_peer": "bench",
                "text": f"The {subject} number {index} settled the question.",
                "entities": [subject, f"n{index}"],
            },
            "POST",
        )


def median_and_worst(call, reps: int = 20) -> tuple[float, float]:
    """Milliseconds. The worst matters: it is the first read after a write."""
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2] * 1000, times[-1] * 1000


def graph_shape(atoms: list[dict]) -> tuple[int, float, int]:
    links = sum(len(atom.get("links") or []) for atom in atoms)
    per_atom = links / len(atoms) if atoms else 0.0
    return links, per_atom, len(json.dumps(atoms))


def run(binary: Path, port: int, sizes: tuple[int, ...]) -> int:
    root = Path(tempfile.mkdtemp(prefix="packset-bench-"))
    daemon = subprocess.Popen(
        [str(binary), "--port", str(port), "--home", str(root)],
        stderr=subprocess.DEVNULL,
    )
    client = Client(port)
    try:
        client.wait()
        header = (
            f"{'atoms':>7} {'search':>9} {'worst':>8} {'recall':>8} "
            f"{'list':>8} {'write':>8} {'links':>9} {'per atom':>9} {'bytes':>10}"
        )
        print(header)
        print("-" * len(header))

        written = 0
        for target in sizes:
            for index in range(written, target):
                client.write(index)
            written = target

            search, worst = median_and_worst(
                lambda: client.call(f"/v1/search?workspace={WORKSPACE}&q=parser&limit=16")
            )
            recall, _ = median_and_worst(
                lambda: client.call(f"/v1/recall?workspace={WORKSPACE}&limit=64")
            )
            listed, _ = median_and_worst(
                lambda: client.call(f"/v1/atoms?workspace={WORKSPACE}"), reps=10
            )

            counter = [written]

            def one_write() -> None:
                counter[0] += 1
                client.write(counter[0])

            write, _ = median_and_worst(one_write, reps=10)
            written = counter[0]

            atoms = client.call(f"/v1/atoms?workspace={WORKSPACE}")["atoms"]
            links, per_atom, size = graph_shape(atoms)
            print(
                f"{written:>7} {search:>8.1f}m {worst:>7.1f}m {recall:>7.1f}m "
                f"{listed:>7.1f}m {write:>7.1f}m {links:>9} {per_atom:>9.1f} {size:>10}"
            )

        print()
        for workers in CONCURRENCY:
            requests = workers * 8

            def one_search(_ignored: int):
                return client.call(
                    f"/v1/search?workspace={WORKSPACE}&q=parser&limit=16"
                )

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one_search, range(requests)))
            elapsed = time.perf_counter() - start
            print(
                f"{workers:>3} callers: {requests:>4} searches in {elapsed * 1000:>7.0f} ms"
                f"  = {requests / elapsed:>8.1f} req/s"
            )
        return 0
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    binary = Path(argv[0])
    if not binary.is_file():
        raise SystemExit(f"missing {binary}; cargo build --release -p packset-daemon")
    port = 8801
    sizes = DEFAULT_SIZES
    rest = argv[1:]
    if rest and rest[0].isdigit():
        port = int(rest[0])
        rest = rest[1:]
    if rest and rest[0] == "--sizes" and len(rest) > 1:
        sizes = tuple(int(part) for part in rest[1].split(",") if part)
    return run(binary, port, sizes)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
