#!/usr/bin/env python3
"""Seat-later review-clock three-cell protocol.

Compares FSRS due_at re-inject, always-include, and never-review on
one over-budget pack. Day 0 is a shared first test of the lesson
set. Days 1-7 measure keep-testing splice hits under limit=8.

- fsrs: recall() due-first, schedule_review(recalled=True) on a hit
- always-include: trust-only live dump; due_at is ignored
- never-review: clear due_at after the first test (drop-from-testing)

Primary metric is keep-test rate: lesson appearances in day-1..7
splices, divided by (n_lessons * 7). Week-later splice is the
day-7 lesson fraction. Week-later R is mean FSRS retrievability
of the lesson set at day 7 (the schedule's own R(t)).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import inside_memory
import inside_recall

PROTOCOL = "review-clock-three-cell"
WS = "git:example.com/review-clock"
N_LESSONS = 8
N_DISTRACTORS = 72
LIMIT = 8
KEEP_DAYS = 7
DAY0 = "2026-08-17T00:00:00.000Z"
CELLS = ("fsrs", "always_include", "never_review")


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


def day_iso(day: int) -> str:
    return inside_memory._shift_iso(DAY0, day * 86400)


def retrievability(atom: dict[str, Any], now: str) -> float:
    review = atom.get("review") or {}
    try:
        stability = float(review.get("stability") or inside_memory.DEFAULT_STABILITY)
    except (TypeError, ValueError):
        stability = inside_memory.DEFAULT_STABILITY
    elapsed = inside_memory._review_elapsed_days(review.get("last"), now)
    if stability <= 0:
        return 0.0
    retr = 0.9 ** (elapsed / stability)
    return min(0.99, max(0.01, retr))


def _trust(atom: dict[str, Any]) -> float:
    try:
        return float(atom.get("trust") if atom.get("trust") is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def trust_sort(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(atoms)
    out.sort(key=lambda atom: str(atom.get("id") or ""))
    out.sort(key=lambda atom: str(atom.get("ts") or ""), reverse=True)
    out.sort(key=lambda atom: _trust(atom), reverse=True)
    return out


def lesson_ids() -> list[str]:
    return [f"lesson-{i:02d}" for i in range(N_LESSONS)]


def fixture(now: str = DAY0) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for i, ident in enumerate(lesson_ids()):
        atoms.append(
            {
                "id": ident,
                "workspace": WS,
                "kind": "lesson",
                "text": f"Review lesson {i} on the due clock.",
                "trust": 0.01,
                "ts": now,
                "tombstone": False,
                "links": [],
                "valid_to": None,
                "due_at": now,
            }
        )
    for i in range(N_DISTRACTORS):
        atoms.append(
            {
                "id": f"hot-{i:03d}",
                "workspace": WS,
                "kind": "habit",
                "text": f"High trust not due distractor {i}.",
                "trust": 9.0,
                "ts": now,
                "tombstone": False,
                "links": [],
                "valid_to": None,
            }
        )
    return atoms


def replace(pack: list[dict[str, Any]], updated: dict[str, Any]) -> list[dict[str, Any]]:
    aid = updated.get("id")
    return [updated if atom.get("id") == aid else atom for atom in pack]


def first_test(pack: list[dict[str, Any]], now: str) -> list[dict[str, Any]]:
    out = pack
    for aid in lesson_ids():
        atom = next(row for row in out if row["id"] == aid)
        out = replace(out, inside_memory.schedule_review(atom, now=now, recalled=True))
    return out


def drop_from_testing(pack: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dropped: list[dict[str, Any]] = []
    lessons = set(lesson_ids())
    for atom in pack:
        if atom.get("id") in lessons:
            row = dict(atom)
            row["due_at"] = None
            dropped.append(row)
        else:
            dropped.append(atom)
    return dropped


def splice_fsrs(pack: list[dict[str, Any]], now: str) -> list[dict[str, Any]]:
    return inside_recall.recall(WS, atoms=pack, limit=LIMIT, now=now)


def splice_trust(pack: list[dict[str, Any]], now: str) -> list[dict[str, Any]]:
    live = [atom for atom in pack if inside_memory.is_live(atom, now)]
    return inside_recall._apply_budget(trust_sort(live)[:LIMIT])


def grade_fsrs(
    pack: list[dict[str, Any]], spliced: list[dict[str, Any]], now: str
) -> list[dict[str, Any]]:
    lessons = set(lesson_ids())
    out = pack
    by_id = {atom["id"]: atom for atom in pack}
    for row in spliced:
        aid = row.get("id")
        if aid not in lessons:
            continue
        current = by_id.get(aid)
        if current is None or not inside_memory.is_due(current, now):
            continue
        updated = inside_memory.schedule_review(current, now=now, recalled=True)
        by_id[aid] = updated
        out = replace(out, updated)
    return out


def run_cell(policy: str) -> dict[str, Any]:
    if policy not in CELLS:
        raise ValueError(f"unknown policy {policy}")
    pack = first_test(fixture(DAY0), DAY0)
    if policy == "never_review":
        pack = drop_from_testing(pack)
    keep_hits = 0
    daily: list[dict[str, Any]] = []
    week_ids: set[str] = set()
    for day in range(1, KEEP_DAYS + 1):
        now = day_iso(day)
        if policy == "fsrs":
            spliced = splice_fsrs(pack, now)
            pack = grade_fsrs(pack, spliced, now)
        else:
            spliced = splice_trust(pack, now)
        ids = {str(atom.get("id")) for atom in spliced}
        hits = ids & set(lesson_ids())
        keep_hits += len(hits)
        if day == KEEP_DAYS:
            week_ids = hits
        daily.append({"day": day, "now": now, "hits": sorted(hits), "n": len(hits)})
    opportunities = N_LESSONS * KEEP_DAYS
    week_now = day_iso(KEEP_DAYS)
    lessons = [atom for atom in pack if atom.get("id") in set(lesson_ids())]
    week_r = [retrievability(atom, week_now) for atom in lessons]
    return {
        "policy": policy,
        "keep_test_hits": keep_hits,
        "keep_test_opportunities": opportunities,
        "keep_test_rate": keep_hits / opportunities,
        "week_later_splice": len(week_ids) / N_LESSONS,
        "week_later_R": sum(week_r) / len(week_r) if week_r else 0.0,
        "days": daily,
    }


def run_protocol(sha: str | None = None) -> dict[str, Any]:
    cells = {name: run_cell(name) for name in CELLS}
    rates = {name: float(cells[name]["keep_test_rate"]) for name in CELLS}
    ranking = sorted(CELLS, key=lambda name: (-rates[name], name))
    return {
        "protocol": PROTOCOL,
        "packset_sha": sha if sha is not None else packset_sha(),
        "fixture": {
            "lessons": N_LESSONS,
            "distractors": N_DISTRACTORS,
            "limit": LIMIT,
            "keep_days": KEEP_DAYS,
            "day0": DAY0,
            "pack_size": N_LESSONS + N_DISTRACTORS,
        },
        "cells": {
            name: {
                "keep_test_hits": cells[name]["keep_test_hits"],
                "keep_test_opportunities": cells[name]["keep_test_opportunities"],
                "keep_test_rate": cells[name]["keep_test_rate"],
                "week_later_splice": cells[name]["week_later_splice"],
                "week_later_R": cells[name]["week_later_R"],
            }
            for name in CELLS
        },
        "daily": {name: cells[name]["days"] for name in CELLS},
        "ranking": ranking,
        "fsrs_beats_always_include": rates["fsrs"] > rates["always_include"],
        "fsrs_beats_never_review": rates["fsrs"] > rates["never_review"],
    }


def measured_line(report: dict[str, Any]) -> str:
    cells = report["cells"]
    return (
        "keep-test "
        f"{cells['fsrs']['keep_test_rate']:.3f}/"
        f"{cells['always_include']['keep_test_rate']:.3f}/"
        f"{cells['never_review']['keep_test_rate']:.3f} "
        "(FSRS/always/never); week-later R "
        f"{cells['fsrs']['week_later_R']:.3f}/"
        f"{cells['always_include']['week_later_R']:.3f}/"
        f"{cells['never_review']['week_later_R']:.3f}"
    )


def write_report(path: Path, report: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = report if report is not None else run_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def default_report_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research" / "review-clock-three-cell.json"


def main() -> None:
    report = write_report(default_report_path())
    print(json.dumps(report, indent=2, sort_keys=True))
    print(measured_line(report))


if __name__ == "__main__":
    main()
