"""Runs Agentomics-ML on AGO2 CLASH through its own Docker launcher."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from harness.dataset_prep import DATASET_NAME, REPO, sm_dataset

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ITERATION_START = re.compile(r"=== ITERATION (\d+) / \d+ ===")
TEST_STEP = "Evaluating the best iteration"


def run_agentomics(
    model_id: str,
    workspace: Path,
    api_key: str,
    reasoning_effort: str,
    usage_dir: Path,
    iterations: int = 3,
    fork_from: Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> dict:
    """Run Agentomics-ML in its container; once its iterations end it scores its best model on the test split.

    The launcher mounts only the training and validation splits into the agent's container (see sm_dataset). With
    `fork_from` (an earlier attempt's workspace), the run continues from that workspace's newest checkpoint, the end of
    a step or of an iteration, and keeps that run's total of iterations.
    Returns the run's timing events and the test predictions' path.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "harness.sm_launch", reasoning_effort, str(usage_dir),
        "--model", model_id,
        "--provider", "openrouter",
        "--dataset", DATASET_NAME,
        "--task-type", "classification",
        "--val-metric", "AUPRC",
        "--cpu-only",
        "--datasets-dir", str(sm_dataset()),
        "--workspace-dir", str(workspace),
        "--verbosity", "summary",
    ]
    command += ["--fork-from-run", str(fork_from)] if fork_from is not None else ["--iterations", str(iterations)]
    events: dict = {"start": time.time(), "iteration_starts": {}}
    process = subprocess.Popen(
        command, cwd=REPO, env={**os.environ, "OPENROUTER_API_KEY": api_key},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    with open(usage_dir / "agentomics.log", "a") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            text = ANSI.sub("", line).rstrip()
            started = ITERATION_START.search(text)
            if started:
                events["iteration_starts"].setdefault(started.group(1), time.time())
            if TEST_STEP in text:
                events.setdefault("last_iteration_end", time.time())
            if text and on_event:
                on_event(text)
    code = process.wait()
    events["end"] = time.time()
    events.setdefault("last_iteration_end", events["end"])
    if code != 0:
        raise RuntimeError(f"Agentomics-ML exited with code {code}; see {usage_dir / 'agentomics.log'}")

    predictions = workspace / "best_iteration_snapshot" / "eval_predictions_test.csv"
    return {"events": events, "predictions_path": predictions if predictions.exists() else None}
