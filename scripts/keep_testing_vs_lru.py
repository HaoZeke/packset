#!/usr/bin/env python3
"""Named protocol: keep-testing due_at vs LRU eviction (Hu 2025 p.59).

Same pack. Two arms. Long-tail hit rate at a later horizon.
GrokOS-42bh / GrokOS-96iq. Not a sort-order unit test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import inside_memory
import inside_recall

PROTOCOL = "keep-testing-vs-lru-v1"
WS = "git:example.com/keep-testing-vs-lru"
DAY0 = datetime(2026, 8, 1, tzinfo=UTC)
INTERVAL_S = 86400
N_HEAD = 16
N_TAIL_TEXTBOOK = 8
N_TAIL_CROWDED = 16
N_TOTAL = 80
BUDGET = 8
WARMUP_DAYS = 7
PROBE_DAY = 7
ZIPF_N = 64
ZIPF_DRAWS = 200
ZIPF_TAIL = 16
ZIPF_SEED = 42

ROOT = Path(__file__).resolve().parents[1]


def day_iso(n: int) -> str:
    stamp = DAY0 + timedelta(days=n)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def packset_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _atom(ident: str, text: str, *, ts: str) -> dict[str, Any]:
    atom = inside_memory.make_atom(
        workspace=WS,
        text=text,
        kind="lesson",
        about_peer="rgoswami",
        by_peer="hermes",
        atom_id=ident,
    )
    atom["ts"] = ts
    atom["valid_from"] = ts
    return atom


def build_pack(*, n_tail: int, schedule_at: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Head + tail + filler. Only tail atoms get schedule_review."""
    n_filler = N_TOTAL - N_HEAD - n_tail
    head_ids = [f"head-{i:03d}" for i in range(N_HEAD)]
    tail_ids = [f"tail-{i:03d}" for i in range(n_tail)]
    filler_ids = [f"fill-{i:03d}" for i in range(n_filler)]
    pack: list[dict[str, Any]] = []
    for ident in head_ids:
        pack.append(_atom(ident, f"Hot claim {ident} used every day.", ts=schedule_at))
    for ident in tail_ids:
        raw = _atom(
            ident,
            f"Long-tail claim {ident} is seldom accessed but essential.",
            ts=schedule_at,
        )
        pack.append(inside_memory.schedule_review(raw, now=schedule_at, interval_s=INTERVAL_S))
    for ident in filler_ids:
        pack.append(_atom(ident, f"Filler claim {ident} is never queried.", ts=schedule_at))
    return pack, head_ids, tail_ids


class LruWindow:
    """Hu p.59 baseline: last-budget accessed ids. Evicts the long tail."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.order: OrderedDict[str, None] = OrderedDict()

    def access(self, ids: list[str]) -> None:
        for ident in ids:
            if ident in self.order:
                self.order.move_to_end(ident)
            else:
                self.order[ident] = None
                while len(self.order) > self.budget:
                    self.order.popitem(last=False)

    def ids(self) -> list[str]:
        return list(self.order)


def keep_testing_window(
    pack: list[dict[str, Any]],
    *,
    seeds: list[str],
    now: str,
    budget: int,
) -> list[dict[str, Any]]:
    return inside_recall.recall(WS, seeds=seeds, atoms=pack, limit=budget, now=now)


def tail_hit(window_ids: set[str], tail_ids: list[str]) -> float:
    if not tail_ids:
        return 0.0
    return sum(1 for ident in tail_ids if ident in window_ids) / len(tail_ids)


def run_split(n_tail: int) -> dict[str, Any]:
    pack, head_ids, tail_ids = build_pack(n_tail=n_tail, schedule_at=day_iso(0))
    lru = LruWindow(BUDGET)
    for _day in range(1, WARMUP_DAYS + 1):
        lru.access(head_ids)
    probe = day_iso(PROBE_DAY)
    keep_ids = {
        atom["id"]
        for atom in keep_testing_window(pack, seeds=head_ids, now=probe, budget=BUDGET)
    }
    lru_ids = set(lru.ids())
    keep_rate = tail_hit(keep_ids, tail_ids)
    lru_rate = tail_hit(lru_ids, tail_ids)
    return {
        "n_tail": n_tail,
        "n_head": N_HEAD,
        "n_total": N_TOTAL,
        "budget": BUDGET,
        "warmup_days": WARMUP_DAYS,
        "probe_day": PROBE_DAY,
        "keep_testing_tail_hit": keep_rate,
        "lru_tail_hit": lru_rate,
        "delta": keep_rate - lru_rate,
        "keep_testing_window": sorted(keep_ids),
        "lru_window": sorted(lru_ids),
        "tail_ids": tail_ids,
        "due_tail_at_probe": [
            ident
            for ident, atom in ((a["id"], a) for a in pack)
            if ident in tail_ids and inside_memory.is_due(atom, probe)
        ],
    }


def _zipf_stream(n: int, draws: int, seed: int) -> list[int]:
    # Deterministic LCG. Rank 0 is hottest.
    state = seed & 0xFFFFFFFF
    out: list[int] = []
    # Harmonic weights 1/(i+1)
    weights = [1.0 / (i + 1) for i in range(n)]
    total = sum(weights)
    cdf: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    for _ in range(draws):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u = (state + 1) / 4294967296.0
        idx = 0
        while idx < n - 1 and u > cdf[idx]:
            idx += 1
        out.append(idx)
    return out


def run_zipf() -> dict[str, Any]:
    ids = [f"z-{i:03d}" for i in range(ZIPF_N)]
    tail_idx = list(range(ZIPF_N - ZIPF_TAIL, ZIPF_N))
    tail_ids = [ids[i] for i in tail_idx]
    pack: list[dict[str, Any]] = []
    t0 = day_iso(0)
    for i, ident in enumerate(ids):
        raw = _atom(ident, f"Zipf claim {ident} rank {i}.", ts=t0)
        if i in tail_idx:
            pack.append(inside_memory.schedule_review(raw, now=t0, interval_s=INTERVAL_S))
        else:
            pack.append(raw)
    stream = _zipf_stream(ZIPF_N, ZIPF_DRAWS, ZIPF_SEED)
    lru = LruWindow(BUDGET)
    for rank in stream:
        lru.access([ids[rank]])
    probe = day_iso(max(1, ZIPF_DRAWS // ZIPF_N))
    keep_ids = {
        atom["id"]
        for atom in keep_testing_window(pack, seeds=[ids[0]], now=probe, budget=BUDGET)
    }
    lru_ids = set(lru.ids())
    keep_rate = tail_hit(keep_ids, tail_ids)
    lru_rate = tail_hit(lru_ids, tail_ids)
    return {
        "n": ZIPF_N,
        "draws": ZIPF_DRAWS,
        "tail": ZIPF_TAIL,
        "budget": BUDGET,
        "seed": ZIPF_SEED,
        "keep_testing_tail_hit": keep_rate,
        "lru_tail_hit": lru_rate,
        "delta": keep_rate - lru_rate,
        "keep_testing_window": sorted(keep_ids),
        "lru_window": sorted(lru_ids),
    }


def run_protocol() -> dict[str, Any]:
    textbook = run_split(N_TAIL_TEXTBOOK)
    crowded = run_split(N_TAIL_CROWDED)
    zipf = run_zipf()
    return {
        "protocol": PROTOCOL,
        "sha": packset_sha(),
        "citation": "Hu 2025 p.59 LRU may eliminate long-tail knowledge",
        "textbook": textbook,
        "crowded": crowded,
        "zipf": zipf,
    }


def as_org(result: dict[str, Any]) -> str:
    tb = result["textbook"]
    cr = result["crowded"]
    zf = result["zipf"]
    lines = [
        "#+title: keep-testing vs LRU",
        "#+options: toc:nil num:nil",
        "",
        f"Protocol: ={result['protocol']}=",
        f"Packset SHA: ={result['sha']}=",
        f"Citation: {result['citation']}",
        "",
        "Same pack. Keep-testing arm is =schedule_review= + =recall= due-first.",
        "LRU arm is last-budget accessed ids. Probe is later-horizon tail hit.",
        "",
        "| Split | keep-testing tail hit | LRU tail hit | delta |",
        f"| textbook {tb['n_tail']} tail / budget {tb['budget']} | {tb['keep_testing_tail_hit']:.3f} | {tb['lru_tail_hit']:.3f} | {tb['delta']:+.3f} |",
        f"| crowded {cr['n_tail']} tail / budget {cr['budget']} | {cr['keep_testing_tail_hit']:.3f} | {cr['lru_tail_hit']:.3f} | {cr['delta']:+.3f} |",
        f"| zipf n={zf['n']} draws={zf['draws']} tail={zf['tail']} | {zf['keep_testing_tail_hit']:.3f} | {zf['lru_tail_hit']:.3f} | {zf['delta']:+.3f} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of org")
    parser.add_argument(
        "--write",
        type=Path,
        default=ROOT / "docs/orgmode/keep-testing-vs-lru.org",
        help="write org results to this path",
    )
    args = parser.parse_args()
    result = run_protocol()
    org = as_org(result)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(org + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(org, end="")


if __name__ == "__main__":
    main()
