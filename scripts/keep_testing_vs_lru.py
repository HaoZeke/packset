#!/usr/bin/env python3
"""Named protocol: keep-testing due_at vs LRU eviction on one pack.

Hu 2025 p.59: LRU drops long-tail knowledge. Karpicke 2008: keep-testing
is the review queue. This run is the measurement; it is not a sort-order
unit test. SCORECARD Measured stays empty until this script is executed.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import inside_memory

PROTOCOL = "keep-testing-due_at-vs-lru"
N_ATOMS = 64
BUDGET = 8
SEED = 42
ZIPF_S = 1.2
ENCODE_AT = "2026-08-01T00:00:00.000Z"
INTERFERENCE_HOURS = 7 * 24
HOUR_S = 3600
WS = "git:github.com/HaoZeke/packset"


def packset_sha(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[1]
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


def _shift_hours(now: str, hours: int) -> str:
    dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def test_at() -> str:
    return _shift_hours(ENCODE_AT, INTERFERENCE_HOURS)


def make_pack() -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for i in range(N_ATOMS):
        atom = inside_memory.make_atom(
            workspace=WS,
            text=f"Long-tail claim {i:03d} about topic {i:03d}.",
            kind="lesson",
            about_peer="rgoswami",
            by_peer="protocol",
            atom_id=f"a{i:03d}",
        )
        atom["ts"] = ENCODE_AT
        atom["valid_from"] = ENCODE_AT
        atom["trust"] = 1.0
        atoms.append(atom)
    return atoms


def clone_pack(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(atom, review=dict(atom.get("review") or {})) for atom in atoms]


def zipf_weights(n: int, s: float = ZIPF_S) -> list[float]:
    return [1.0 / ((i + 1) ** s) for i in range(n)]


def tail_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(BUDGET, N_ATOMS)]


def lru_last(atom: dict[str, Any]) -> str:
    return str(atom.get("lru_last") or atom.get("ts") or "")


def lru_window(atoms: list[dict[str, Any]], budget: int = BUDGET) -> list[dict[str, Any]]:
    """Recency cache. Oldest last-access leaves first."""
    ranked = sorted(atoms, key=lambda atom: (lru_last(atom), str(atom.get("id") or "")), reverse=True)
    return ranked[:budget]


def evict_lru(held: list[dict[str, Any]], incoming: dict[str, Any], budget: int = BUDGET) -> list[dict[str, Any]]:
    by_id = {str(atom.get("id")): atom for atom in held if atom.get("id")}
    by_id[str(incoming["id"])] = incoming
    return lru_window(list(by_id.values()), budget)


def keep_testing_window(
    atoms: list[dict[str, Any]],
    now: str,
    query_id: str | None = None,
    budget: int = BUDGET,
) -> list[dict[str, Any]]:
    """Due queue first. The queried due atom is the keep-testing trial."""
    due = [atom for atom in atoms if inside_memory.is_due(atom, now)]
    rest = [atom for atom in atoms if not inside_memory.is_due(atom, now)]
    pinned = [atom for atom in due if query_id and atom.get("id") == query_id]
    other_due = [atom for atom in due if atom.get("id") != query_id]
    other_due.sort(key=lambda atom: (lru_last(atom), str(atom.get("id") or "")), reverse=True)
    rest.sort(key=lambda atom: (lru_last(atom), str(atom.get("id") or "")), reverse=True)
    ordered = pinned + other_due + rest
    return ordered[:budget]


def _by_id(atoms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(atom["id"]): atom for atom in atoms if atom.get("id")}


def encode_keep_testing(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for atom in atoms:
        scheduled = inside_memory.schedule_review(atom, now=ENCODE_AT)
        scheduled["lru_last"] = ENCODE_AT
        out.append(scheduled)
    return out


def encode_lru(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for atom in atoms:
        rec = dict(atom)
        rec["lru_last"] = ENCODE_AT
        out.append(rec)
    return out


def run_interference_lru(
    atoms: list[dict[str, Any]], queries: list[str]
) -> list[dict[str, Any]]:
    held: list[dict[str, Any]] = lru_window(atoms, BUDGET)
    index = _by_id(atoms)
    for hour, qid in enumerate(queries):
        clock = _shift_hours(ENCODE_AT, hour + 1)
        atom = dict(index[qid])
        atom["lru_last"] = clock
        index[qid] = atom
        held = evict_lru(held, atom, BUDGET)
    held_ids = {str(atom.get("id")) for atom in held}
    return [atom for atom in index.values() if atom.get("id") in held_ids]


def run_interference_keep_testing(
    atoms: list[dict[str, Any]], queries: list[str]
) -> list[dict[str, Any]]:
    index = _by_id(atoms)
    for hour, qid in enumerate(queries):
        clock = _shift_hours(ENCODE_AT, hour + 1)
        pack = list(index.values())
        window = keep_testing_window(pack, clock, query_id=qid, budget=BUDGET)
        window_ids = {str(atom.get("id")) for atom in window}
        target = index[qid]
        if qid in window_ids and inside_memory.is_due(target, clock):
            stretched = inside_memory.schedule_review(target, now=clock, recalled=True)
            stretched["lru_last"] = clock
            index[qid] = stretched
        else:
            rec = dict(target)
            rec["lru_last"] = clock
            index[qid] = rec
    return list(index.values())


def sample_queries(rng: random.Random) -> list[str]:
    weights = zipf_weights(N_ATOMS)
    ids = [f"a{i:03d}" for i in range(N_ATOMS)]
    return rng.choices(ids, weights=weights, k=INTERFERENCE_HOURS)


def delayed_hits(
    atoms: list[dict[str, Any]],
    *,
    arm: str,
    clock: str,
) -> tuple[int, int]:
    tail = tail_ids()
    index = _by_id(atoms)
    hits = 0
    for qid in tail:
        if qid not in index:
            continue
        if arm == "lru":
            window = lru_window(atoms, BUDGET)
        else:
            window = keep_testing_window(atoms, clock, query_id=qid, budget=BUDGET)
        if any(atom.get("id") == qid for atom in window):
            hits += 1
    return hits, len(tail)


def queue_retention(atoms: list[dict[str, Any]]) -> tuple[int, int]:
    tail = tail_ids()
    index = _by_id(atoms)
    kept = sum(1 for qid in tail if qid in index and index[qid].get("due_at"))
    return kept, len(tail)


def run(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    base = make_pack()
    queries = sample_queries(rng)
    clock = test_at()
    lru_pack = run_interference_lru(encode_lru(clone_pack(base)), queries)
    kt_pack = run_interference_keep_testing(encode_keep_testing(clone_pack(base)), queries)
    lru_hits, n_tail = delayed_hits(lru_pack, arm="lru", clock=clock)
    kt_hits, _ = delayed_hits(kt_pack, arm="keep_testing", clock=clock)
    kt_retained, _ = queue_retention(kt_pack)
    lru_rate = lru_hits / n_tail if n_tail else 0.0
    kt_rate = kt_hits / n_tail if n_tail else 0.0
    return {
        "protocol": PROTOCOL,
        "n_atoms": N_ATOMS,
        "budget": BUDGET,
        "n_tail": n_tail,
        "interference_hours": INTERFERENCE_HOURS,
        "encode_at": ENCODE_AT,
        "test_at": clock,
        "seed": seed,
        "lru_tail_hits": lru_hits,
        "keep_testing_tail_hits": kt_hits,
        "lru_tail_hit_rate": lru_rate,
        "keep_testing_tail_hit_rate": kt_rate,
        "keep_testing_queue_retention": kt_retained / n_tail if n_tail else 0.0,
        "delta_tail_hit_rate": kt_rate - lru_rate,
        "sha": packset_sha(),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def render_org(result: dict[str, Any]) -> str:
    sha = result.get("sha") or ""
    return "\n".join(
        [
            f"- protocol: {result['protocol']}",
            f"- packset SHA: {sha}",
            f"- n_atoms={result['n_atoms']} budget={result['budget']} n_tail={result['n_tail']}",
            f"- LRU tail hit: {result['lru_tail_hits']}/{result['n_tail']}"
            f" ({result['lru_tail_hit_rate']:.4f})",
            f"- keep-testing tail hit: {result['keep_testing_tail_hits']}/{result['n_tail']}"
            f" ({result['keep_testing_tail_hit_rate']:.4f})",
            f"- delta (keep-testing - LRU): {result['delta_tail_hit_rate']:.4f}",
            f"- keep-testing queue retention: {result['keep_testing_queue_retention']:.4f}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    result = run()
    if "--json" in args:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_org(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
