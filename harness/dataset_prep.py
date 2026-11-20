"""Prepare the AGO2 CLASH dataset in the layout each condition expects."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET_NAME = "AGO2_CLASH_Hejret2023"
UPLOAD_NAME = "ago2-clash"   # the folder name the paper's skill runs were given
AGO2_SRC = REPO / "data" / "dataset" / DATASET_NAME
AGO2_TEST_SRC = AGO2_SRC / "test"
TEST_LABELS = AGO2_TEST_SRC / "labels.csv"
SKILL_SRC = REPO / "skill"


def prep_skill_workspace(workspace: Path) -> None:
    """Set up a workspace for the skill agent.

    uploads/ago2-clash/
      train/input/data.csv
      train/labels.csv
      metadata.json
      dataset_description.md
    .skills/ml-research/   (the skill scripts)
    """
    workspace.mkdir(parents=True, exist_ok=True)

    upload = workspace / "uploads" / UPLOAD_NAME
    upload.mkdir(parents=True, exist_ok=True)

    train_input = upload / "train" / "input"
    train_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGO2_SRC / "train" / "input" / "data.csv", train_input / "data.csv")
    shutil.copy2(AGO2_SRC / "train" / "labels.csv", upload / "train" / "labels.csv")
    shutil.copy2(AGO2_SRC / "metadata.json", upload / "metadata.json")
    if (AGO2_SRC / "dataset_description.md").exists():
        shutil.copy2(AGO2_SRC / "dataset_description.md", upload / "dataset_description.md")

    skill_dst = workspace / ".skills" / "ml-research"
    skill_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SKILL_SRC, skill_dst, dirs_exist_ok=True)


def deliver_test_inputs(workspace: Path) -> None:
    """Hand over the test inputs, without labels, once the agent's iterations are done."""
    clean_input = workspace / "uploads" / "input"
    clean_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGO2_TEST_SRC / "input" / "data.csv", clean_input / "data.csv")


SM_DATASETS = REPO / "runs" / ".datasets"


def sm_dataset() -> Path:
    """The dataset as Agentomics-ML gets it: the training pairs already split into train/ and validation/ by the skill's
    own seeded, stratified setup script, so both conditions use the same split and, as in the paper's Setup state, the
    split is made by code rather than by the model. Built once; returns the datasets folder holding it."""
    target = SM_DATASETS / DATASET_NAME
    if (target / "validation" / "labels.csv").exists():
        return SM_DATASETS
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        upload = work / "uploads" / UPLOAD_NAME
        shutil.copytree(AGO2_SRC, upload, ignore=shutil.ignore_patterns("test"))
        done = subprocess.run([sys.executable, str(SKILL_SRC / "scripts" / "setup.py"), "--dataset", f"uploads/{UPLOAD_NAME}",
                               "--metric", "AUPRC", "--iterations", "3"], cwd=work, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"the skill's setup could not split the data: {done.stdout}{done.stderr}")
        staging = SM_DATASETS / f".{DATASET_NAME}.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for split in ("train", "validation"):
            shutil.copytree(work / "data" / split, staging / split)
        shutil.copytree(AGO2_TEST_SRC, staging / "test")
        for name in ("metadata.json", "dataset_description.md"):
            shutil.copy2(AGO2_SRC / name, staging / name)
        staging.rename(target)
    return SM_DATASETS
