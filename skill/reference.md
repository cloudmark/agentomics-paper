# ML research: formats and metrics

## Files

| File | Columns |
|---|---|
| `<split>/input/data.csv` | `id`, then the dataset's own columns, as uploaded |
| `<split>/labels.csv` | `id`, `label` (a class id `0…k-1` for classification, a number for regression) |
| predictions | `id`, `prediction`, and for classification `probability_0 … probability_<k-1>` |

A predictions file has one row for every sample of the split it was made for, each id once, no NaN or infinite
value. `prediction` is a class id (classification) or a number (regression). Probabilities need not sum to exactly 1.

## The record

`results/metrics.json`, written only by the scripts:

```json
{
  "dataset": "…", "task": "classification", "metric": "AUPRC", "classes": {"0": 0, "1": 1},
  "iterations_planned": 3,
  "iterations": [{"iteration": 0, "status": "ok", "attempts": 1, "validation": {"ACC": 0.0, "AUROC": 0.0}, "train": {"ACC": 0.0, "AUROC": 0.0}}],
  "best": {"iteration": 0, "validation": {}}
}
```

An iteration's `status` is `ok`, `failed` (with `why`) or `unchecked` (a check started and did not pass yet).

## Metrics

Computed with scikit-learn on the validation split.

Classification: `ACC`; `AUROC` (the positive class's probability for two classes, one-vs-rest otherwise); `AUPRC`
(average precision; macro over classes for more than two); `F1` (macro); `LOG_LOSS` (probabilities clipped to
[1e-15, 1 - 1e-15]); `MCC`. Lower is better only for `LOG_LOSS`.

Regression: `MSE`, `RMSE`, `MAE` (lower is better); `R2`, `PEARSON`, `SPEARMAN` (higher is better).

The best iteration is the one with the best recorded validation score on the run's metric; a tie keeps the earlier.

## What the check runs

1. `train.py --train-data data/mini_train --validation-data data/validation --artifacts-dir <scratch>`: must exit 0
   and write at least one file, within 10 minutes.
2. `inference.py --input <a copy of data/validation/input> --output <scratch>/predictions.csv --artifacts-dir <a copy
   of the iteration's training_artifacts>`: both folders are copies somewhere else, so a script that reads a fixed
   path instead of its arguments fails here rather than on the test set.
3. The same on `data/train/input`, for the train scores recorded beside the validation ones.

`inference.py` is also refused, before it runs, when its text names `labels.csv`, a `data/` split or an `iterations/`
folder.
