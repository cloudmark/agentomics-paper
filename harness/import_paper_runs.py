"""Seed runs/ from the paper's data/report-data.json so a reader can compare paper results to harness reruns.

For every run (complete and partial):
  - creates runs/<key>/<arm>-<effort>/attempt-1/events.json  (status: done, with plausible timestamps)
  - creates runs/<key>/<arm>-<effort>/attempt-1/log/usage.jsonl  (empty)
  - copies data/predictions/<session>/<n>-test.csv → attempt-1/scores/<n>-test.csv

For complete runs only:
  - also writes result.json so the harness treats the run as done and skips it

Partial runs get the attempt-1 folder but no result.json, so the harness runs them fresh and
produces a new result alongside the original paper data for comparison.

Runs that already have a result.json are skipped unless --force is passed.

Usage:  python -m harness.import_paper_runs [--force]
  --force  overwrite existing result.json files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Base epoch for 2026-11-18 00:00:00 UTC (spread original runs across Nov 18-19)
_EPOCH_BASE = 1794960000


def _start_time(session: str) -> float:
    """Derive a plausible start epoch from the session id (spreads across two days)."""
    h = int(hashlib.md5(session.encode()).hexdigest()[:8], 16)
    return _EPOCH_BASE + (h % (2 * 86400))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing result.json")
    args = parser.parse_args()

    data = json.loads((REPO / "data" / "report-data.json").read_text())
    cells = {(c["model"], c["arm"], c["effort"]): c for c in data["cells"]}
    iters = {(c["model"], c["arm"], c["effort"]): c for c in data["iters"]}
    preds_root = REPO / "data" / "predictions"
    runs_root = REPO / "runs"

    seeded = skipped_partial = skipped_existing = 0

    for key_tuple, cell in cells.items():
        is_partial = bool(cell.get("partial"))
        rdir = runs_root / cell["key"] / f"{cell['arm']}-{cell['effort']}"
        result_path = rdir / "result.json"

        if result_path.exists() and not args.force:
            print(f"  skip (exists):    {cell['key']} {cell['arm']}-{cell['effort']}")
            skipped_existing += 1
            continue

        rdir.mkdir(parents=True, exist_ok=True)

        # Build plausible events.json from per-iteration timing
        session = cell.get("session", "")
        start = _start_time(session)
        it = iters.get(key_tuple, {})
        per = it.get("per", [])
        iteration_ends = {p["it"]: start + p["cum_time"] * 60 for p in per}
        last_end = max(iteration_ends.values()) if iteration_ends else start + cell.get("time", 0) * 60
        events = {
            "attempt": "attempt-1",
            "status": "done",
            "start": start,
            "iteration_ends": iteration_ends,
            "last_iteration_end": last_end,
            "end": last_end + 300,
        }

        # Create attempt-1 with harness-shaped folders
        attempt = rdir / "attempt-1"
        attempt.mkdir(exist_ok=True)
        (attempt / "events.json").write_text(json.dumps(events, indent=1))
        log = attempt / "log"
        log.mkdir(exist_ok=True)
        (log / "usage.jsonl").write_text("")

        # Copy prediction CSVs into scores/
        scores = attempt / "scores"
        scores.mkdir(exist_ok=True)
        src_dir = preds_root / session
        if src_dir.exists():
            for f in src_dir.glob("*-test.csv"):
                shutil.copy2(f, scores / f.name)
        else:
            print(f"  WARNING: no predictions at {src_dir}")

        # Build the result record
        paper_record = {**cell, "per": per, "after": it.get("after", []),
                        "overall": it.get("overall", {}), "extra": []}

        if is_partial:
            # No result.json — harness will run this fresh
            print(f"  paper data only:  {cell['key']} {cell['arm']}-{cell['effort']}  "
                  f"(partial, harness will rerun)  session={session}")
            skipped_partial += 1
        else:
            # Write result.json so harness skips this run
            result_path.write_text(json.dumps(paper_record, indent=1))
            print(f"  seeded:           {cell['key']} {cell['arm']}-{cell['effort']}  "
                  f"test={cell['test_auprc']}  session={session}")
            seeded += 1

    print(f"\nDone: {seeded} seeded as done, {skipped_partial} with paper data only (harness will rerun), "
          f"{skipped_existing} skipped (result.json already exists)")
    partial_names = [f"{c['key']} {c['arm']}-{c['effort']}" for c in data['cells'] if c.get('partial')]
    print(f"Runs the harness will run: {partial_names}")


if __name__ == "__main__":
    main()
