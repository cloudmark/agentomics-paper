---
name: ml-research
description: Trains the best model you can on a labelled dataset the person uploaded, over a few research iterations (plan, explore, represent, design, train, infer, check, analyse), keeps the best one and writes a report. Use it when the person asks you to train, fit or build a model on a dataset in uploads/, to predict a label or a number from their data, or to do ML research on it. Its scripts split the data, check each iteration's scripts and compute every score.
license: Apache-2.0
compatibility: Its commands run in the thread's workspace with Python 3, in a virtualenv it makes there (numpy, pandas, scikit-learn, scipy); making it needs the workspace to have the internet.
metadata:
  icon: icon.svg
---

# ML research

The person will judge your model on a test set you never see. Three rules always hold:

- **Every score you report comes from `results/metrics.json`**, which only the scripts here write. A number you worked out yourself, or remember, is not one: the person compares runs, and a number nobody can reproduce is worse than none.
- **There is no test set for you.** Never look for one. If the upload holds a `test/` folder or test labels, stop and tell the person.
- **Train on `data/train/` only.** `data/validation/` is for scoring and early stopping. Fitting anything else on its labels (a model, a threshold, a feature selection) makes the validation score a lie and the test score a disappointment.

Run every command with `execute` in the workspace, from the workspace's top, and **never pass `skill` to `execute`**: with it the command runs in the skill's own computer, which has no packages, no internet and not your virtualenv, and the scripts refuse to run there. `SK` below stands for `.skills/ml-research`, and `PY` for `.venv/bin/python`.

## The Python you work with, once

The workspace's own Python has no scientific libraries. Before anything else, make a virtualenv and use it for every Python command, the scripts here included:

    python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn scipy

Install what else a plan needs the same way (`.venv/bin/pip install lightgbm`); a large package can take minutes, so run it in the background and say in the plan why it is worth it. If the install fails for want of internet, stop and tell the person.

## What you are given

A dataset folder under `uploads/`:

```
uploads/<name>/
├── train/input/data.csv      a row per sample, with an id column
├── train/labels.csv          id,label
├── metadata.json             {"task_type": "classification" | "regression"}
└── dataset_description.md    what the data is, when there is one
```

The person's message may say how many iterations to run (3 when it doesn't) and the metric to be judged on (AUROC for classification, MAE for regression when it doesn't). Others: ACC, AUPRC, F1, LOG_LOSS, MCC; MSE, RMSE, R2, PEARSON, SPEARMAN.

## "Train a model on this dataset"

### 1. Set up, once

    PY SK/scripts/setup.py --dataset uploads/<name> --metric AUPRC --iterations 3

It checks the upload and writes `data/train/` (80%), `data/validation/` (20%, stratified, seeded), `data/mini_train/` (up to 100 rows per class, for quick runs), `data/description.md` and the run's record, `results/metrics.json`. Classification labels become class ids `0…k-1`; `data/description.md` says which is which. Read it before anything else.

Start `lessons.md` in the workspace's top, empty. After any step that crashed, gave a wrong result or wasted real time, add one short line: what to do or avoid, and the condition that triggered it (a data property, a library, an input size). Read it before each plan.

### 2. Iterate

For iteration `n` from 0 (the first is `iterations/0/`), work in `iterations/<n>/` and do the eight stages in order. [stages.md](stages.md) says what each one writes and what makes it good; read it once before iteration 0.

| Stage | You write | In short |
|---|---|---|
| Plan | `plan.md` | Iteration 0: a baseline that finishes fast and exposes problems. Later: one coherent change the evidence points at. |
| Explore | `exploration.md` | Measured facts about the data, from a script you ran. Later iterations reuse it unless the plan needs a new fact. |
| Represent | in `plan.md` | How each input becomes features; what is fitted, on train only. |
| Design | in `plan.md` | Model, loss, hyperparameters, early stopping; an estimate that it fits the time. |
| Train | `train.py` | Run it on `data/train`. |
| Infer | `inference.py` | Predictions from inputs alone. |
| Check | nothing | `PY SK/scripts/check_iteration.py --iteration <n>` runs both scripts, scores, records. |
| Analyse | `analysis.md` | Where the model is wrong, from the saved predictions. |

The two scripts' interfaces:

    PY iterations/<n>/train.py --train-data data/train --validation-data data/validation --artifacts-dir iterations/<n>/training_artifacts
    PY iterations/<n>/inference.py --input <split>/input --output <predictions.csv> --artifacts-dir <dir>

**The check is not optional and not yours to do by hand.** It runs `train.py` on `data/mini_train` into a scratch folder, runs `inference.py` on copies of the validation and train inputs with your real artifacts, checks the predictions files, scores them, records the scores, and copies the iteration to `best/` when its validation score strictly beats the best so far. When it fails it says what is wrong: fix that and run it again, up to 3 times. Still failing: record it with `--give-up "<why>"`, add a lesson, and plan the next iteration around it.

A command has 10 minutes. Training that needs longer runs in the background (`nohup .venv/bin/python … > iterations/<n>/train.log 2>&1 &`) and you poll its log. Prefer a model that trains in minutes: three finished iterations beat one still running.

Stop after the planned number of iterations. Do not add one because the last looked promising; say so in the report.

### 3. Report

    PY SK/scripts/report.py

It writes `results/report.md` from the record. Add a short paragraph under "Notes": what the best model is, and what you would try next. Then tell the person in a few lines: the best iteration, its validation scores as recorded, that the model is in `best/`, and that the test score is theirs to compute.

## "Here is the test set"

The person gives test inputs after the last iteration: a folder with `input/data.csv` and no labels.

    PY SK/scripts/predict.py --input uploads/<test name>/input

It runs `best/inference.py` once on the test inputs and once on the train split, and writes `final/test_predictions.csv` and `final/train_predictions.csv`. Change nothing in `best/` after seeing test inputs, and do not train again.

## What makes a run fail its purpose

- A reported score that `results/metrics.json` does not hold.
- `inference.py` reading `labels.csv`, or anything outside its `--input` and `--artifacts-dir`.
- Picking the best iteration by anything but the recorded validation score.
- Editing the scripts in `SK/scripts/`, `results/metrics.json` or the `data/` splits by hand.

[reference.md](reference.md) has the file formats, the metrics' definitions and exactly what the check runs, for when its message is not enough.
