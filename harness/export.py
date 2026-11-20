"""Collects every finished run's result.json into a report-data.json shaped like data/report-data.json, and copies each
run's per-iteration test predictions where bootstrap.py and bootstrap_first.py read them.

Usage: python -m harness.export [--runs-dir runs] [--out data/harness]
Writes <out>/report-data.json and <out>/predictions/<session>/<iteration>-test.csv; data/report-data.json is not touched.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from harness.results import attempts, read_json, usage_path, write_json
from harness.usage import read_usage

REPO = Path(__file__).resolve().parent.parent
CELL_FIELDS = ["model", "key", "arm", "effort", "test_auprc", "test_acc", "test_auroc", "train_acc", "val", "recovered",
               "partial", "note", "cost", "calls", "time", "model_time", "tools_time", "session", "input_tokens",
               "cached_tokens", "output_tokens", "reasoning_tokens"]


def _counted(rdir: Path) -> bool:
    """Runs under a hidden folder (runs/.superseded, runs/.datasets) are not part of the data."""
    return not rdir.parent.name.startswith(".")


def collect(runs_dir: Path) -> list[tuple[Path, dict]]:
    found = [(p.parent, read_json(p)) for p in sorted(runs_dir.glob("*/*/result.json")) if _counted(p.parent)]
    return [(rdir, r) for rdir, r in found if r]


def export(runs_dir: Path, out: Path) -> dict:
    runs = collect(runs_dir)
    cells, iters, extra = [], [], []
    predictions = out / "predictions"
    for rdir, r in runs:
        cells.append({k: r.get(k) for k in CELL_FIELDS})
        iters.append({k: r[k] for k in ("model", "key", "arm", "effort", "per", "after", "overall")})
        extra.extend(e for e in r.get("extra", []) if e["calls"])
        live = attempts(rdir)
        scores_dir = live[-1] / "scores" if live else None
        scores = scores_dir
        target = predictions / r["session"]
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for f in scores.glob("*-test.csv"):
            shutil.copy2(f, target / f.name)
    # runs that never finished cost money too
    for rdir in sorted(p for p in runs_dir.glob("*/*")
                       if p.is_dir() and _counted(p) and not (p / "result.json").exists()):
        for a in attempts(rdir):
            calls = read_usage(usage_path(a))
            if calls:
                extra.append({"label": f"[attempt] {rdir.name}/{rdir.parent.name} {a.name} (unfinished)",
                              "cost": round(sum(c["cost"] for c in calls), 2), "calls": len(calls)})
    data = {
        "cells": cells, "iters": iters, "extra": extra,
        "total_cost": round(sum(c["cost"] or 0 for c in cells) + sum(e["cost"] for e in extra), 2),
        "total_time": round(sum(c["time"] or 0 for c in cells), 1),
        "total_time_extra_note": "training time only: model and command minutes up to the end of the last iteration",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report-data.json", data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "harness")
    args = parser.parse_args()
    data = export(args.runs_dir, args.out)
    print(f"{len(data['cells'])} runs, {len(data['extra'])} extra attempts, ${data['total_cost']} -> {args.out}")


if __name__ == "__main__":
    main()
