"""Builds a run's result.json, in the shape of one cell and one iteration record of data/report-data.json.

A run lives in runs/<model-key>/<method>-<reasoning>/ with one folder per attempt (attempt-1, attempt-2, ...), each
holding workspace/, events.json and log/usage.jsonl. The run is made of its last attempt and, if that attempt continued an
earlier one, the part of the earlier attempt up to the iteration it was continued from. Everything else an attempt
spent is reported as extra cost, as the paper's `extra` entries are.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from harness.dataset_prep import TEST_LABELS
from harness.scoring import score
from harness.usage import read_usage

PLANNED_ITERATIONS = 3


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def usage_path(attempt: Path) -> Path:
    return attempt / "log" / "usage.jsonl"


def attempts(run_dir: Path) -> list[Path]:
    """Return numbered attempt folders (attempt-1, attempt-2, …) in order. Non-numeric folders like
    attempt-paper are excluded — they hold reference data, not live runs."""
    def _num(p: Path):
        try:
            return int(p.name.split("-")[1])
        except (IndexError, ValueError):
            return None
    return sorted((p for p in run_dir.glob("attempt-*") if p.is_dir() and _num(p) is not None), key=_num)


# ── what each method recorded about its iterations ────────────────────────────

def skill_iterations(workspace: Path, events: dict) -> list[dict]:
    """Per iteration: id, recorded validation AU-PRC (None if its check never passed), end time."""
    record = read_json(workspace / "results" / "metrics.json") or {}
    passed = {str(i["iteration"]): (i.get("validation") or {}).get("AUPRC")
              for i in record.get("iterations", []) if i.get("status") == "ok"}
    folders = sorted((p.name for p in (workspace / "iterations").glob("*") if p.is_dir() and p.name.isdigit()), key=int)
    ends = events.get("iteration_ends", {})
    return [{"it": k, "val_recorded": passed.get(k), "end": ends.get(k, events.get("last_iteration_end"))} for k in folders]


def skill_pick(workspace: Path) -> tuple[str | None, float | None]:
    record = read_json(workspace / "results" / "metrics.json") or {}
    best = record.get("best")
    if not best:
        return None, None
    item = next((i for i in record.get("iterations", []) if i["iteration"] == best["iteration"]), {})
    return str(best["iteration"]), (item.get("train") or {}).get("ACC")


def sm_iterations(workspace: Path) -> list[dict]:
    out = []
    for folder in sorted((workspace / "run").glob("iteration_*"), key=lambda p: int(p.name.split("_")[1])):
        state = read_json(folder / "runtime_info" / "iteration_state.json") or {}
        validation = read_json(folder / "validation_evaluation" / "output.json") or {}
        metrics = (validation.get("payload") or {}).get("metrics") or {}
        out.append({"it": folder.name.split("_")[1], "val_recorded": metrics.get("validation/AUPRC"),
                    "train_acc": metrics.get("train/ACC"), "end": state.get("ended_at")})
    return out


def sm_pick(workspace: Path, iterations: list[dict]) -> tuple[str | None, float | None]:
    meta = read_json(workspace / "best_iteration_snapshot" / "runtime_info" / "iteration_metadata.json")
    if not meta:
        return None, None
    pick = str(meta["iteration"])
    return pick, next((i["train_acc"] for i in iterations if i["it"] == pick), None)


# ── assembling the record ─────────────────────────────────────────────────────

def _r(x, n=4):
    return None if x is None else round(x, n)


def build(run_dir: Path, meta: dict) -> dict:
    """meta: model (display name), key, arm ("skill" | "sm"), effort ("none" | "high"), model_id."""
    chain = attempts(run_dir)
    last = chain[-1]
    events = read_json(last / "events.json") or {}
    workspace = last / "workspace"

    # the training segments the run is made of: the last attempt up to its last iteration's end, preceded by any attempt
    # it continued from, up to the point it continued from
    segments = []
    attempt, attempt_events = last, events
    until = events.get("last_iteration_end") or events["end"]
    while True:
        segments.insert(0, (attempt, attempt_events["start"], until))
        if not attempt_events.get("fork_from"):
            break
        until = attempt_events["fork_point"]
        attempt = run_dir / attempt_events["fork_from"]
        attempt_events = read_json(attempt / "events.json") or {}
    run_end = segments[-1][2]

    def elapsed(t: float) -> float:
        """Training minutes from the run's start to time t, across its segments."""
        total = 0.0
        for _, start, end in segments:
            if t <= start:
                break
            total += min(t, end) - start
        return total / 60

    usage = {a: read_usage(usage_path(a)) for a in chain}
    training = [c for a, start, end in segments for c in usage[a] if start <= c["start"] < end]
    test_step = [c for c in usage[last] if c["start"] >= run_end]
    in_run = {id(c) for c in training + test_step}

    if meta["arm"] == "skill":
        iterations = skill_iterations(workspace, events)
        pick, train_acc = skill_pick(workspace)
        final_predictions = workspace / "final" / "test_predictions.csv"
    else:
        iterations = sm_iterations(workspace)
        pick, train_acc = sm_pick(workspace, iterations)
        final_predictions = workspace / "best_iteration_snapshot" / "eval_predictions_test.csv"

    scores = last / "scores"
    per = []
    for item in iterations:
        ok = item["val_recorded"] is not None
        test = score(scores / f"{item['it']}-test.csv", TEST_LABELS) if ok else None
        val = score(scores / f"{item['it']}-val.csv", scores / "val-labels.csv") if ok else None
        end = min(item["end"] or run_end, run_end)
        per.append({
            "it": item["it"],
            "test_auprc": test and test["auprc"], "test_acc": test and test["acc"],
            "val_auprc": val and val["auprc"], "val_recorded": _r(item["val_recorded"]),
            "cum_time": round(elapsed(end), 1),
            "cum_cost": round(sum(c["cost"] for c in training if c["start"] < end), 2),
            "wall": False,
        })

    after = []
    for k in range(PLANNED_ITERATIONS):
        seen = [p for p in per[:k + 1] if p["val_recorded"] is not None]
        if k >= len(per) or (per[k]["val_recorded"] is None and seen):
            after.append(None)
        elif not seen:
            after.append({"none": True, "cum_time": per[k]["cum_time"], "cum_cost": per[k]["cum_cost"]})
        else:
            best = max(seen, key=lambda p: p["val_recorded"])
            after.append({"pick": best["it"], "test_auprc": best["test_auprc"], "test_acc": best["test_acc"],
                          "cum_time": per[k]["cum_time"], "cum_cost": per[k]["cum_cost"]})

    final = score(final_predictions, TEST_LABELS) if final_predictions.exists() else None
    time_min = elapsed(run_end)
    model_min = sum(min(c["end"], end) - c["start"] for a, start, end in segments for c in usage[a]
                    if start <= c["start"] < end) / 60
    cost = sum(c["cost"] for c in training + test_step)
    passed = sum(1 for p in per if p["val_recorded"] is not None)

    return {
        "model": meta["model"], "key": meta["key"], "arm": meta["arm"], "effort": meta["effort"],
        "model_id": meta["model_id"], "session": f"{meta['key']}-{meta['arm']}-{meta['effort']}",
        "pick": pick,
        "test_auprc": final and final["auprc"], "test_acc": final and final["acc"], "test_auroc": final and final["auroc"],
        "train_acc": _r(train_acc),
        "val": [p["val_recorded"] for p in per],
        "recovered": len(segments) > 1,
        "partial": passed < PLANNED_ITERATIONS,
        "note": (f"continued from {segments[0][0].name} after an interruption, from the end of its "
                 f"{events.get('fork_step')} step of iteration {events.get('fork_iteration')}") if len(segments) > 1 else None,
        "cost": round(cost, 2),
        "calls": len(training) + len(test_step),
        "time": round(time_min, 1), "model_time": round(model_min, 1), "tools_time": round(time_min - model_min, 1),
        "input_tokens": sum(c["input_tokens"] for c in training + test_step),
        "cached_tokens": sum(c["cached_tokens"] for c in training + test_step),
        "output_tokens": sum(c["output_tokens"] for c in training + test_step),
        "reasoning_tokens": sum(c["reasoning_tokens"] for c in training + test_step),
        "per": per, "after": after,
        "overall": {"test_auprc": final and final["auprc"], "test_acc": final and final["acc"],
                    "time": round(time_min, 1), "cost": round(cost, 2)},
        "extra": [{"label": f"[attempt] {meta['arm']}/{meta['key']}/{meta['effort']} {a.name}",
                   "cost": round(sum(c["cost"] for c in usage[a] if id(c) not in in_run), 2),
                   "calls": sum(1 for c in usage[a] if id(c) not in in_run)}
                  for a in chain if any(id(c) not in in_run for c in usage[a])],
    }
