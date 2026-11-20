"""Scores a finished run outside the agent: each iteration's own model is run on the test inputs and on the validation
split, from a copy of the workspace, and the predictions are kept as <iteration>-test.csv / <iteration>-val.csv."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sklearn import metrics as sk

from harness.dataset_prep import AGO2_TEST_SRC, DATASET_NAME, REPO, sm_dataset
from harness.sandbox import run_offline

TIMEOUT = 1800


def read_labels(path: Path) -> dict[str, int]:
    rows = list(csv.reader(open(path)))
    return {r[0]: int(float(r[1])) for r in rows[1:]}


def score(predictions: Path, labels: Path) -> dict | None:
    """AU-PRC, accuracy and AUROC of a predictions file (id, prediction, probability_1) against a labels file."""
    try:
        p = {r["id"]: r for r in csv.DictReader(open(predictions))}
        truth = read_labels(labels)
        ids = [i for i in truth if i in p]
        if len(ids) < len(truth):
            return None
        y = [truth[i] for i in ids]
        prob = [float(p[i]["probability_1"]) for i in ids]
        hard = [int(float(p[i]["prediction"])) for i in ids]
        return {"auprc": round(sk.average_precision_score(y, prob), 4), "acc": round(sk.accuracy_score(y, hard), 4),
                "auroc": round(sk.roc_auc_score(y, prob), 4)}
    except (OSError, KeyError, ValueError):
        return None


def _inputs_only(source: Path, destination: Path) -> Path:
    """A split folder holding only input/, so no labels sit beside the inputs."""
    shutil.copytree(source / "input", destination / "input")
    return destination


def score_skill_iterations(workspace: Path, out: Path) -> None:
    """Runs iterations/<k>/inference.py of a skill run in the image it ran in, without network, from a copy of its
    workspace and with its own virtualenv."""
    workspace, out = workspace.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # the venv's python links to the image's own interpreter, so on this machine only the link itself exists
    if not os.path.lexists(workspace / ".venv" / "bin" / "python"):
        raise RuntimeError(f"{workspace} has no .venv; its iterations cannot be run with their own libraries")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        copy = tmp / "ws"
        shutil.copytree(workspace, copy, symlinks=True,
                        ignore=shutil.ignore_patterns(".venv", "uploads", "__pycache__"))
        _inputs_only(AGO2_TEST_SRC, tmp / "test")
        _inputs_only(copy / "data" / "validation", tmp / "val")
        mounts = {copy: "/workspace", workspace / ".venv": "/venv", tmp / "test": "/inputs/test",
                  tmp / "val": "/inputs/val", out: "/out"}
        for folder in sorted((copy / "iterations").glob("*"), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not (folder.name.isdigit() and (folder / "inference.py").exists()):
                continue
            k = folder.name
            artifacts = f"iterations/{k}/training_artifacts" if (folder / "training_artifacts").exists() else f"iterations/{k}"
            for split in ("test", "val"):
                if (out / f"{k}-{split}.csv").exists():
                    continue
                done = run_offline(mounts, ["/venv/bin/python", f"iterations/{k}/inference.py",
                                            "--input", f"/inputs/{split}/input", "--output", f"/out/{k}-{split}.csv",
                                            "--artifacts-dir", artifacts], "/workspace", TIMEOUT)
                (out / f"{k}-{split}.log").write_text(done.stdout + done.stderr)
    shutil.copy2(workspace / "data" / "validation" / "labels.csv", out / "val-labels.csv")


def score_sm_iterations(workspace: Path, out: Path) -> None:
    """Runs every archived iteration of an Agentomics-ML run through its own inference container."""
    workspace, out = workspace.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # the validation split the run trained against: the provided one, or one its agent made (earlier runs)
    made = workspace / "run" / "shared" / "splits" / "split_0" / "validation"
    split = made if (made / "labels.csv").is_file() and not (made / "labels.csv").is_symlink() \
        else sm_dataset() / DATASET_NAME / "validation"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs = {"test": _inputs_only(AGO2_TEST_SRC, tmp / "test"), "val": _inputs_only(split, tmp / "val")}
        for name, split_dir in inputs.items():
            if all((out / f"{k}-{name}.csv").exists() for k in _sm_iterations(workspace)):
                continue
            produced = tmp / f"{name}-out"
            produced.mkdir()
            done = subprocess.run(
                [sys.executable, "-m", "agentomics.cli.inference", "--agent-dir", str(workspace),
                 "--input", str(split_dir), "--output", str(produced / "predictions.csv"),
                 "--all-iterations", "--cpu-only"],
                cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT * 3)
            (out / f"{name}.log").write_text(done.stdout + done.stderr)
            for k in _sm_iterations(workspace):
                made = produced / f"iteration_{k}_predictions.csv"
                if made.exists():
                    shutil.copy2(made, out / f"{k}-{name}.csv")
    shutil.copy2(split / "labels.csv", out / "val-labels.csv")


def _sm_iterations(workspace: Path) -> list[int]:
    return sorted(int(p.name.removeprefix("iteration_")) for p in (workspace / "run").glob("iteration_*")
                  if p.name.removeprefix("iteration_").isdigit())
