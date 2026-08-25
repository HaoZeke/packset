#!/usr/bin/env python3
"""Named protocol: keep-testing due_at vs LRU eviction (Hu 2025 p.59).

Same pack. Same Zipf query stream. Same splice budget on both arms.
Keep-testing arm is recall() due-first plus schedule_review(recalled=True)
on a spliced due atom (at most once per day). LRU arm is last-budget
accessed ids. The store on the keep-testing arm stays the full pack;
eviction under comparison is the size-8 splice, not deletion.

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

PROTOCOL = "keep-testing-due_at-vs-lru-v2"
WS = "git:example.com/keep-testing-vs-lru"
N_ATOMS = 64
BUDGET = 8
SEED = 42
ZIPF_S = 1.2
ENCODE_AT = "2026-08-01T00:00:00.000Z"
INTERFERENCE_HOURS = 7 * 24
GRADE_COOLDOWN_H = 24
HOUR_S = 3600
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


def _shift_hours(now: str, hours: int) -> str:
    dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def test_at() -> str:
    return _shift_hours(ENCODE_AT, INTERFERENCE_HOURS)


def atom_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(N_ATOMS)]


def tail_ids() -> list[str]:
    return [f"a{i:03d}" for i in range(BUDGET, N_ATOMS)]


def make_pack() -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for i, ident in enumerate(atom_ids()):
        atom = inside_memory.make_atom(
            workspace=WS,
            text=f"Long-tail claim {i:03d} about topic {i:03d}.",
            kind="lesson",
            about_peer="rgoswami",
            by_peer="protocol",
            atom_id=ident,
        )
        atom["ts"] = ENCODE_AT
        atom["valid_from"] = ENCODE_AT
        atom["trust"] = 1.0
        atoms.append(atom)
    return atoms


def clone_pack(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(atom, review=dict(atom.get("review") or {})) for atom in atoms]


def encode_keep_testing(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [inside_memory.schedule_review(atom, now=ENCODE_AT) for atom in atoms]


def zipf_weights(n: int, s: float = ZIPF_S) -> list[float]:
    return [1.0 / ((i + 1) ** s) for i in range(n)]


def sample_queries(rng: random.Random) -> list[str]:
    weights = zipf_weights(N_ATOMS)
    ids = atom_ids()
    return rng.choices(ids, weights=weights, k=INTERFERENCE_HOURS)


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


def _replace(pack: list[dict[str, Any]], updated: dict[str, Any]) -> list[dict[str, Any]]:
    aid = updated.get("id")
    return [updated if atom.get("id") == aid else atom for atom in pack]


def keep_testing_window(
    pack: list[dict[str, Any]],
    *,
    now: str,
    seeds: list[str] | None,
    budget: int = BUDGET,
) -> list[dict[str, Any]]:
    """Real recall() due-first splice. No extra pin of the query id."""
    return inside_recall.recall(WS, seeds=seeds, atoms=pack, limit=budget, now=now)


def _hours_apart(earlier: str, later: str) -> float:
    a = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
    b = datetime.fromisoformat(later.replace("Z", "+00:00"))
    return (b - a).total_seconds() / HOUR_S


def grade_due_in_window(
    pack: list[dict[str, Any]],
    window: list[dict[str, Any]],
    *,
    now: str,
    last_grade: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Test spliced due atoms. Stretch on success. At most one grade per day."""
    stretches = 0
    out = pack
    by_id = {str(atom.get("id")): atom for atom in out if atom.get("id")}
    for row in window:
        aid = str(row.get("id") or "")
        current = by_id.get(aid)
        if current is None or not inside_memory.is_due(current, now):
            continue
        prev = last_grade.get(aid)
        if prev is not None and _hours_apart(prev, now) < GRADE_COOLDOWN_H:
            continue
        stretched = inside_memory.schedule_review(current, now=now, recalled=True)
        last_grade[aid] = now
        out = _replace(out, stretched)
        by_id[aid] = stretched
        stretches += 1
    return out, stretches


def tail_fraction(window_ids: list[str], tail: set[str]) -> float:
    if not tail:
        return 0.0
    return sum(1 for ident in window_ids if ident in tail) / len(tail)


def slot_rate(windows: list[list[str]], tail: set[str]) -> float:
    slots = sum(len(window) for window in windows)
    if slots == 0:
        return 0.0
    hits = sum(1 for window in windows for ident in window if ident in tail)
    return hits / slots


def coverage(windows: list[list[str]], tail: list[str]) -> float:
    seen = {ident for window in windows for ident in window}
    if not tail:
        return 0.0
    return sum(1 for ident in tail if ident in seen) / len(tail)


def run(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    queries = sample_queries(rng)
    tail = tail_ids()
    tail_set = set(tail)
    clock_end = test_at()

    lru = LruWindow(BUDGET)
    lru_windows: list[list[str]] = []
    for qid in queries:
        lru.access([qid])
        lru_windows.append(list(lru.ids()))
    lru_probe = list(lru.ids())

    pack = encode_keep_testing(clone_pack(make_pack()))
    last_grade: dict[str, str] = {}
    kt_windows: list[list[str]] = []
    stretch_n = 0
    for hour, qid in enumerate(queries, start=1):
        now = _shift_hours(ENCODE_AT, hour)
        window = keep_testing_window(pack, now=now, seeds=[qid], budget=BUDGET)
        pack, n = grade_due_in_window(pack, window, now=now, last_grade=last_grade)
        stretch_n += n
        kt_windows.append([str(atom.get("id")) for atom in window if atom.get("id")])
    kt_probe_atoms = keep_testing_window(pack, now=clock_end, seeds=[], budget=BUDGET)
    kt_probe = [str(atom.get("id")) for atom in kt_probe_atoms if atom.get("id")]

    lru_probe_rate = tail_fraction(lru_probe, tail_set)
    kt_probe_rate = tail_fraction(kt_probe, tail_set)
    lru_slot = slot_rate(lru_windows, tail_set)
    kt_slot = slot_rate(kt_windows, tail_set)
    kt_last = kt_windows[-1] if kt_windows else []
    lru_last = lru_windows[-1] if lru_windows else []
    final_day = 24
    kt_final_slot = slot_rate(kt_windows[-final_day:], tail_set)
    lru_final_slot = slot_rate(lru_windows[-final_day:], tail_set)
    return {
        "protocol": PROTOCOL,
        "citation": "Hu 2025 p.59 LRU may eliminate long-tail knowledge",
        "n_atoms": N_ATOMS,
        "budget": BUDGET,
        "n_tail": len(tail),
        "interference_hours": INTERFERENCE_HOURS,
        "encode_at": ENCODE_AT,
        "test_at": clock_end,
        "seed": seed,
        "zipf_s": ZIPF_S,
        "queries": queries,
        "stretch_n": stretch_n,
        "keep_testing_store_n": len(pack),
        "lru_store_n": len(lru_probe),
        "keep_testing_probe_window": sorted(kt_probe),
        "lru_probe_window": sorted(lru_probe),
        "keep_testing_last_window": sorted(kt_last),
        "lru_last_window": sorted(lru_last),
        "keep_testing_probe_tail_hit": kt_probe_rate,
        "lru_probe_tail_hit": lru_probe_rate,
        "delta_probe_tail_hit": kt_probe_rate - lru_probe_rate,
        "keep_testing_last_tail_hit": tail_fraction(kt_last, tail_set),
        "lru_last_tail_hit": tail_fraction(lru_last, tail_set),
        "delta_last_tail_hit": tail_fraction(kt_last, tail_set) - tail_fraction(lru_last, tail_set),
        "keep_testing_slot_tail": kt_slot,
        "lru_slot_tail": lru_slot,
        "delta_slot_tail": kt_slot - lru_slot,
        "keep_testing_final_day_slot": kt_final_slot,
        "lru_final_day_slot": lru_final_slot,
        "delta_final_day_slot": kt_final_slot - lru_final_slot,
        "keep_testing_tail_coverage": coverage(kt_windows, tail),
        "lru_tail_coverage": coverage(lru_windows, tail),
        "sha": packset_sha(),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def as_org(result: dict[str, Any]) -> str:
    sha = result.get("sha") or ""
    return "\n".join(
        [
            "#+title: keep-testing vs LRU",
            "#+options: toc:nil num:nil",
            "",
            f"Protocol: ={result['protocol']}=",
            f"Packset SHA: ={sha}=",
            f"Citation: {result['citation']}",
            "",
            "Same pack. Same Zipf stream. Keep-testing arm is =recall= due-first",
            "plus =schedule_review(recalled=True)= on a spliced due atom.",
            "LRU arm is last-budget accessed ids. Primary metric is interference",
            "tail slot rate (fraction of splice slots that are tail ranks 8-63).",
            "Rest-seat probe is seedless =recall= at hour 168; an empty window",
            "means the due queue is caught up. No query-id pin.",
            "",
            f"n_atoms={result['n_atoms']} budget={result['budget']} "
            f"n_tail={result['n_tail']} hours={result['interference_hours']} "
            f"seed={result['seed']} stretch_n={result['stretch_n']}",
            "",
            "| Metric | keep-testing | LRU | delta |",
            f"| interference tail slot | {result['keep_testing_slot_tail']:.4f} "
            f"| {result['lru_slot_tail']:.4f} "
            f"| {result['delta_slot_tail']:+.4f} |",
            f"| final-day tail slot | {result['keep_testing_final_day_slot']:.4f} "
            f"| {result['lru_final_day_slot']:.4f} "
            f"| {result['delta_final_day_slot']:+.4f} |",
            f"| last-stream tail hit | {result['keep_testing_last_tail_hit']:.4f} "
            f"| {result['lru_last_tail_hit']:.4f} "
            f"| {result['delta_last_tail_hit']:+.4f} |",
            f"| rest-seat tail hit | {result['keep_testing_probe_tail_hit']:.4f} "
            f"| {result['lru_probe_tail_hit']:.4f} "
            f"| {result['delta_probe_tail_hit']:+.4f} |",
            f"| tail coverage | {result['keep_testing_tail_coverage']:.4f} "
            f"| {result['lru_tail_coverage']:.4f} | |",
            "",
        ]
    )


def measured_line(result: dict[str, Any]) -> str:
    return (
        f"slot-tail {result['keep_testing_slot_tail']:.3f}/"
        f"{result['lru_slot_tail']:.3f} (keep/LRU); "
        f"last-stream {result['keep_testing_last_tail_hit']:.3f}/"
        f"{result['lru_last_tail_hit']:.3f}; "
        f"delta-slot {result['delta_slot_tail']:+.3f}"
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
        default=ROOT / "research/keep-testing-vs-lru.json",
        help="write JSON results to this path",
    )
    args = parser.parse_args()
    result = run()
    org = as_org(result)
    printable = {k: v for k, v in result.items() if k != "queries"}
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(org + "\n", encoding="utf-8")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(printable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(printable, indent=2, sort_keys=True))
    else:
        print(org, end="")
        print(measured_line(result))


if __name__ == "__main__":
    main()
