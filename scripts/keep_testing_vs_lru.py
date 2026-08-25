#!/usr/bin/env python3
"""Named protocol: keep-testing stream vs LRU eviction (Hu 2025 p.59).

Same pack. Same Zipf query stream. Same budget. Keep-testing arm is
due-first recall plus schedule_review(recalled=True) on a successful
trial. LRU arm is last-budget accessed ids. Probe is one later-horizon
budget window; the queried id is not pinned.

SCORECARD Measured stays empty until this script is executed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import inside_memory
import inside_recall

PROTOCOL = "keep-testing-stream-vs-lru"
WS = "git:example.com/keep-testing-stream-vs-lru"
ENCODE_AT = "2026-08-01T00:00:00.000Z"
N_ATOMS = 64
BUDGET = 8
ZIPF_S = 1.2
SEED = 42
INTERFERENCE_HOURS = 7 * 24
TAIL_START = BUDGET

ROOT = Path(__file__).resolve().parents[1]


def packset_sha(repo: Path | None = None) -> str:
    root = repo or ROOT
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def shift_hours(now: str, hours: int) -> str:
    dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def probe_at() -> str:
    return shift_hours(ENCODE_AT, INTERFERENCE_HOURS)


def atom_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(N_ATOMS)]


def tail_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(TAIL_START, N_ATOMS)]


def head_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(TAIL_START)]


def clone_pack(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(atom, review=dict(atom.get("review") or {})) for atom in atoms]


def make_pack() -> list[dict[str, Any]]:
    pack: list[dict[str, Any]] = []
    for i, ident in enumerate(atom_ids()):
        atom = inside_memory.make_atom(
            workspace=WS,
            text=f"Long-tail claim {ident} rank {i}.",
            kind="lesson",
            about_peer="rgoswami",
            by_peer="protocol",
            atom_id=ident,
        )
        atom["ts"] = ENCODE_AT
        atom["valid_from"] = ENCODE_AT
        atom["trust"] = 1.0
        pack.append(atom)
    return pack


def encode_keep_testing(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [inside_memory.schedule_review(atom, now=ENCODE_AT) for atom in atoms]


def zipf_weights(n: int, s: float = ZIPF_S) -> list[float]:
    return [1.0 / ((i + 1) ** s) for i in range(n)]


def sample_queries(rng: random.Random) -> list[str]:
    ids = atom_ids()
    return rng.choices(ids, weights=zipf_weights(len(ids)), k=INTERFERENCE_HOURS)


class LruWindow:
    """Hu p.59 baseline: last-budget accessed ids. Evicts the long tail."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.order: OrderedDict[str, None] = OrderedDict()

    def access(self, ident: str) -> None:
        if ident in self.order:
            self.order.move_to_end(ident)
        else:
            self.order[ident] = None
            while len(self.order) > self.budget:
                self.order.popitem(last=False)

    def ids(self) -> list[str]:
        return list(self.order)


def replace(pack: list[dict[str, Any]], updated: dict[str, Any]) -> list[dict[str, Any]]:
    aid = updated.get("id")
    return [updated if atom.get("id") == aid else atom for atom in pack]


def keep_testing_window(
    pack: list[dict[str, Any]],
    *,
    now: str,
    seeds: list[str] | None,
    budget: int,
) -> list[dict[str, Any]]:
    """Due-first recall. The queried id is a seed, never a pin."""
    return inside_recall.recall(
        WS,
        seeds=seeds,
        atoms=pack,
        limit=budget,
        now=now,
    )


def run_lru(queries: list[str]) -> LruWindow:
    lru = LruWindow(BUDGET)
    for qid in queries:
        lru.access(qid)
    return lru


def run_keep_testing(
    pack: list[dict[str, Any]], queries: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stretched = 0
    trials = 0
    for hour, qid in enumerate(queries):
        clock = shift_hours(ENCODE_AT, hour + 1)
        window = keep_testing_window(pack, now=clock, seeds=[qid], budget=BUDGET)
        window_ids = {str(atom.get("id")) for atom in window}
        trials += 1
        target = next(atom for atom in pack if atom.get("id") == qid)
        if qid in window_ids and inside_memory.is_due(target, clock):
            pack = replace(
                pack,
                inside_memory.schedule_review(target, now=clock, recalled=True),
            )
            stretched += 1
    return pack, {"trials": trials, "stretched": stretched}


def tail_hit(window_ids: set[str], tails: list[str]) -> float:
    if not tails:
        return 0.0
    return sum(1 for ident in tails if ident in window_ids) / len(tails)


def run_protocol(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    queries = sample_queries(rng)
    tails = tail_ids()
    lru = run_lru(queries)
    kt_pack, kt_stats = run_keep_testing(
        encode_keep_testing(clone_pack(make_pack())), queries
    )
    clock = probe_at()
    keep_window = keep_testing_window(
        kt_pack, now=clock, seeds=None, budget=BUDGET
    )
    keep_ids = {str(atom.get("id")) for atom in keep_window}
    lru_ids = set(lru.ids())
    keep_rate = tail_hit(keep_ids, tails)
    lru_rate = tail_hit(lru_ids, tails)
    due_at_probe = [
        str(atom.get("id"))
        for atom in kt_pack
        if inside_memory.is_due(atom, clock)
    ]
    stretched_heads = [
        str(atom.get("id"))
        for atom in kt_pack
        if atom.get("id") in set(head_ids())
        and int((atom.get("review") or {}).get("reps") or 0) > 0
    ]
    return {
        "protocol": PROTOCOL,
        "citation": "Hu 2025 p.59 LRU may eliminate long-tail knowledge",
        "sha": packset_sha(),
        "n_atoms": N_ATOMS,
        "budget": BUDGET,
        "n_tail": len(tails),
        "zipf_s": ZIPF_S,
        "seed": seed,
        "interference_hours": INTERFERENCE_HOURS,
        "encode_at": ENCODE_AT,
        "probe_at": clock,
        "n_queries": len(queries),
        "keep_testing_trials": kt_stats["trials"],
        "keep_testing_stretched": kt_stats["stretched"],
        "stretched_head_ids": stretched_heads,
        "due_at_probe": due_at_probe,
        "n_due_at_probe": len(due_at_probe),
        "keep_testing_window": sorted(keep_ids),
        "lru_window": sorted(lru_ids),
        "keep_testing_tail_hits": sum(1 for ident in tails if ident in keep_ids),
        "lru_tail_hits": sum(1 for ident in tails if ident in lru_ids),
        "keep_testing_tail_hit": keep_rate,
        "lru_tail_hit": lru_rate,
        "delta": keep_rate - lru_rate,
        "occupancy_budget_over_tail": BUDGET / len(tails) if tails else 0.0,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def as_org(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#+title: keep-testing vs LRU",
            "#+options: toc:nil num:nil",
            "",
            f"Protocol: ={result['protocol']}=",
            f"Packset SHA: ={result['sha']}=",
            f"Citation: {result['citation']}",
            "",
            "Same pack. Same Zipf stream. Keep-testing arm is =recall=",
            "due-first plus =schedule_review(recalled=True)= on a successful",
            "trial. LRU arm is last-budget accessed ids. Probe is one",
            "later-horizon budget window. The queried id is not pinned.",
            "",
            f"- n_atoms={result['n_atoms']} budget={result['budget']} n_tail={result['n_tail']}",
            f"- zipf_s={result['zipf_s']} hours={result['interference_hours']} seed={result['seed']}",
            f"- keep-testing trials={result['keep_testing_trials']} stretched={result['keep_testing_stretched']}",
            f"- stretched heads: {len(result['stretched_head_ids'])}",
            f"- due at probe: {result['n_due_at_probe']}",
            f"- LRU tail hit: {result['lru_tail_hits']}/{result['n_tail']}"
            f" ({result['lru_tail_hit']:.4f})",
            f"- keep-testing tail hit: {result['keep_testing_tail_hits']}/{result['n_tail']}"
            f" ({result['keep_testing_tail_hit']:.4f})",
            f"- delta (keep-testing - LRU): {result['delta']:+.4f}",
            f"- budget/n_tail occupancy: {result['occupancy_budget_over_tail']:.4f}",
            "",
            "| Split | keep-testing tail hit | LRU tail hit | delta |",
            f"| zipf n={result['n_atoms']} hours={result['interference_hours']} tail={result['n_tail']} | {result['keep_testing_tail_hit']:.3f} | {result['lru_tail_hit']:.3f} | {result['delta']:+.3f} |",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of org")
    parser.add_argument(
        "--write",
        type=Path,
        default=ROOT / "docs/orgmode/keep-testing-vs-lru.org",
        help="write org results to this path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "research/keep-testing-stream-vs-lru.json",
        help="write JSON results to this path",
    )
    args = parser.parse_args()
    result = run_protocol()
    org = as_org(result)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(org + "\n", encoding="utf-8")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(org, end="")


if __name__ == "__main__":
    main()
