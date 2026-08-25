#!/usr/bin/env python3
"""Named protocol: FSRS due_at re-inject vs always-include vs never-review.

Karpicke 2008 three-cell map on the packset splice:

- fsrs: keep-testing. recall() prepends due_at; a spliced due atom is a
  test trial and schedule_review(recalled=True) stretches the interval.
- always_include: extra study, no review clock. Live atoms sort by trust
  only. Over-budget seats restudy the high-trust crowd.
- never_review: drop-from-testing. After the shared first test, due_at
  is cleared. Unprompted seats do not re-inject.

Shared first study+test at t0 (all 40 targets). Days 1-7 are over-budget
coding seats whose hints do not mention the targets. Week-later metrics
are mean FSRS-lite retrievability at t0+7d and the mean target splice
hit rate on those seven seats.

This module prints measured numbers. It does not invent them.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import inside_memory
import inside_recall

PROTOCOL = "review-clock-3cell"
WS = "git:example.com/review-clock-3cell"
T0 = "2026-08-01T00:00:00.000Z"
N_TARGETS = 40
N_CROWD = 80
SEAT_LIMIT = 8
SEAT_DAYS = 7
POLICIES = ("fsrs", "always_include", "never_review")


def _shift_days(now: str, days: int) -> str:
    return inside_memory._shift_iso(now, days * 86400)


def _atom(ident: str, **fields: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": ident,
        "workspace": WS,
        "kind": "lesson",
        "text": f"Claim {ident}.",
        "trust": 1.0,
        "ts": T0,
        "tombstone": False,
        "links": [],
        "valid_to": None,
    }
    rec.update(fields)
    return rec


def _pack() -> tuple[list[dict[str, Any]], set[str]]:
    targets = [
        _atom(
            f"t{i:02d}",
            text=f"Lesson {i}: packset due_at is a keep-testing queue.",
            trust=0.05,
            kind="lesson",
        )
        for i in range(N_TARGETS)
    ]
    crowd = [
        _atom(
            f"c{i:03d}",
            text=f"Crowd filler {i} about compiler flags and linker scripts.",
            trust=9.0,
            kind="habit",
        )
        for i in range(N_CROWD)
    ]
    return targets + crowd, {row["id"] for row in targets}


def _sort_ignore_due(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(atoms)
    out.sort(key=lambda atom: str(atom.get("id") or ""))
    out.sort(key=lambda atom: str(atom.get("ts") or ""), reverse=True)
    out.sort(key=lambda atom: inside_recall._trust(atom), reverse=True)
    return out


def _always_include(
    atoms: list[dict[str, Any]], *, now: str, limit: int
) -> list[dict[str, Any]]:
    live = [
        atom
        for atom in atoms
        if inside_memory.is_live(atom, now) or inside_memory.is_due(atom, now)
    ]
    return inside_recall._apply_budget(_sort_ignore_due(live)[:limit])


def _first_test(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for atom in targets:
        scheduled = inside_memory.schedule_review(atom, now=T0)
        out.append(inside_memory.schedule_review(scheduled, now=T0, recalled=True))
    return out


def _index_by_id(atoms: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(atom["id"]): atom for atom in atoms if atom.get("id")}


def _splice(
    policy: str,
    atoms: list[dict[str, Any]],
    *,
    now: str,
    dropped: set[str],
) -> list[dict[str, Any]]:
    if policy == "always_include":
        return _always_include(atoms, now=now, limit=SEAT_LIMIT)
    patched = []
    for atom in atoms:
        rec = dict(atom)
        if policy == "never_review" and rec.get("id") in dropped:
            rec.pop("due_at", None)
        patched.append(rec)
    return inside_recall.recall(
        WS,
        atoms=patched,
        now=now,
        limit=SEAT_LIMIT,
        hints=None,
    )


def _apply_policy(
    policy: str,
    spliced: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    targets: set[str],
    *,
    now: str,
    dropped: set[str],
) -> int:
    trials = 0
    for row in spliced:
        aid = row.get("id")
        if not aid or aid not in targets:
            continue
        current = by_id[str(aid)]
        if policy == "fsrs" and inside_memory.is_due(current, now):
            by_id[str(aid)] = inside_memory.schedule_review(
                current, now=now, recalled=True
            )
            trials += 1
        elif policy == "never_review":
            dropped.add(str(aid))
    return trials


def _mean_retrievability(
    by_id: dict[str, dict[str, Any]], targets: set[str], now: str
) -> float:
    scores = [inside_memory.retrievability(by_id[tid], now) for tid in sorted(targets)]
    return sum(scores) / len(scores)


def packset_sha(root: Path | None = None) -> str:
    cwd = root or Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def measured_line(row: dict[str, Any]) -> str:
    cells = row["cells"]
    return (
        "R "
        f"{cells['fsrs']['week_later_mean_retrievability']:.6f}/"
        f"{cells['always_include']['week_later_mean_retrievability']:.6f}/"
        f"{cells['never_review']['week_later_mean_retrievability']:.6f} "
        "(FSRS/always/never); hit "
        f"{cells['fsrs']['seat_later_target_splice_hit_rate']:.6f}/"
        f"{cells['always_include']['seat_later_target_splice_hit_rate']:.6f}/"
        f"{cells['never_review']['seat_later_target_splice_hit_rate']:.6f}"
    )


def default_report_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research" / "review-clock-3cell.json"


def write_report(
    path: Path | None = None, row: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = row if row is not None else run()
    dest = path or default_report_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run(sha: str | None = None) -> dict[str, Any]:
    """Run the three cells. Returns the measured scorecard row."""
    probe = _shift_days(T0, SEAT_DAYS)
    cells: dict[str, Any] = {}
    for policy in POLICIES:
        base, targets = _pack()
        first = _first_test([row for row in base if row["id"] in targets])
        by_id = _index_by_id(base)
        by_id.update(_index_by_id(first))
        dropped: set[str] = set(targets) if policy == "never_review" else set()
        hits = 0
        trials = 0
        per_seat: list[float] = []
        for day in range(1, SEAT_DAYS + 1):
            clock = _shift_days(T0, day)
            pack = list(by_id.values())
            spliced = _splice(policy, pack, now=clock, dropped=dropped)
            shown = {str(row["id"]) for row in spliced if row.get("id")}
            n_hit = len(shown & targets)
            hits += n_hit
            per_seat.append(n_hit / len(targets))
            trials += _apply_policy(
                policy, spliced, by_id, targets, now=clock, dropped=dropped
            )
        cells[policy] = {
            "week_later_mean_retrievability": round(
                _mean_retrievability(by_id, targets, probe), 6
            ),
            "seat_later_target_splice_hit_rate": round(hits / (SEAT_DAYS * len(targets)), 6),
            "target_test_trials": trials,
            "per_seat_hit_rate": [round(x, 6) for x in per_seat],
        }
    fsrs = cells["fsrs"]
    always = cells["always_include"]
    never = cells["never_review"]
    retr_delta_always = round(
        fsrs["week_later_mean_retrievability"] - always["week_later_mean_retrievability"],
        6,
    )
    retr_delta_never = round(
        fsrs["week_later_mean_retrievability"] - never["week_later_mean_retrievability"],
        6,
    )
    hit_delta_always = round(
        fsrs["seat_later_target_splice_hit_rate"]
        - always["seat_later_target_splice_hit_rate"],
        6,
    )
    hit_delta_never = round(
        fsrs["seat_later_target_splice_hit_rate"]
        - never["seat_later_target_splice_hit_rate"],
        6,
    )
    return {
        "protocol": PROTOCOL,
        "packset_sha": sha if sha is not None else packset_sha(),
        "n_targets": N_TARGETS,
        "n_crowd": N_CROWD,
        "seat_limit": SEAT_LIMIT,
        "seat_days": SEAT_DAYS,
        "t0": T0,
        "probe": probe,
        "cells": cells,
        "delta": {
            "retrievability_fsrs_minus_always_include": retr_delta_always,
            "retrievability_fsrs_minus_never_review": retr_delta_never,
            "hit_rate_fsrs_minus_always_include": hit_delta_always,
            "hit_rate_fsrs_minus_never_review": hit_delta_never,
        },
        "fsrs_beats_always_include": retr_delta_always > 0 and hit_delta_always > 0,
        "fsrs_beats_never_review": retr_delta_never > 0 and hit_delta_never > 0,
    }


def main() -> None:
    row = write_report()
    print(json.dumps(row, indent=2, sort_keys=True))
    print(measured_line(row))


if __name__ == "__main__":
    main()
