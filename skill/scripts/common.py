"""What the ml-research scripts share: the run's record, the metrics, and reading a predictions file."""
import csv
import json
import math
import os
import sys

RECORD = "results/metrics.json"
CLASSIFICATION = "classification"
REGRESSION = "regression"
METRICS = {
    CLASSIFICATION: ["ACC", "AUROC", "AUPRC", "F1", "LOG_LOSS", "MCC"],
    REGRESSION: ["MSE", "RMSE", "MAE", "R2", "PEARSON", "SPEARMAN"],
}
LOWER_IS_BETTER = {"LOG_LOSS", "MSE", "RMSE", "MAE"}
DEFAULT_METRIC = {CLASSIFICATION: "AUROC", REGRESSION: "MAE"}


def fail(message):
    """Ends the script with what is wrong, said for whoever has to fix it."""
    print("FAILED: " + message)
    sys.exit(1)


def workspace_python():
    """Ends the script, before it records anything, when this Python lacks what the scripts and the models need."""
    try:
        import numpy, pandas, scipy, sklearn  # noqa: F401
    except ImportError as missing:
        fail("this Python (%s) has no %s. Nothing was recorded and no check was used up. Run this script with the "
             "workspace's virtualenv, .venv/bin/python, through execute with its skill argument left unset: the skill's "
             "own computer has no packages and no internet. If .venv is missing, make it as SKILL.md says."
             % (sys.executable, missing.name))


def read_csv(path):
    """A CSV's header and rows (each a dict)."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_record():
    if not os.path.exists(RECORD):
        fail(RECORD + " is missing: run setup.py first.")
    with open(RECORD) as handle:
        return json.load(handle)


def save_record(record):
    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    with open(RECORD, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def beats(metric, candidate, best):
    """Whether candidate beats best on metric; a tie doesn't, so the earlier iteration stays."""
    return candidate < best if metric in LOWER_IS_BETTER else candidate > best


def read_predictions(path, ids, task, classes):
    """
    The predictions at path for the samples ids, checked against the contract: an id and a prediction column, for
    classification a probability_<k> per class id, and a row for every id, once. Returns (predictions, probabilities)
    in ids' order, probabilities a list of rows (empty for regression).
    """
    if not os.path.exists(path):
        fail("inference.py wrote no predictions file at " + path)
    header, rows = read_csv(path)
    wanted = ["id", "prediction"] + (["probability_%d" % k for k in range(classes)] if task == CLASSIFICATION else [])
    missing = [column for column in wanted if column not in header]
    if missing:
        fail("the predictions file lacks the column(s) %s; it has %s" % (", ".join(missing), ", ".join(header)))
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            fail("the predictions file has sample %s twice" % row["id"])
        by_id[row["id"]] = row
    absent = [sample for sample in ids if sample not in by_id]
    if absent:
        fail("the predictions file has no row for %d sample(s), such as %s" % (len(absent), ", ".join(absent[:5])))
    if len(by_id) != len(ids):
        fail("the predictions file has %d row(s) for samples that are not in the split" % (len(by_id) - len(ids)))
    predictions, probabilities = [], []
    for sample in ids:
        row = by_id[sample]
        try:
            if task == CLASSIFICATION:
                predictions.append(int(float(row["prediction"])))
                probabilities.append([float(row["probability_%d" % k]) for k in range(classes)])
            else:
                predictions.append(float(row["prediction"]))
        except ValueError:
            fail("sample %s has a value that is not a number: %s" % (sample, row))
        values = probabilities[-1] if task == CLASSIFICATION else [predictions[-1]]
        if any(math.isnan(value) or math.isinf(value) for value in values):
            fail("sample %s has a NaN or infinite value" % sample)
    if task == CLASSIFICATION and any(p < 0 or p >= classes for p in predictions):
        fail("a prediction is not a class id between 0 and %d" % (classes - 1))
    return predictions, probabilities


def score(task, classes, truth, predictions, probabilities):
    """Every metric of the task, by scikit-learn, as a dict of name to value."""
    import numpy as np
    from sklearn import metrics as sk

    if task == REGRESSION:
        from scipy import stats

        y, p = np.array(truth, dtype=float), np.array(predictions, dtype=float)
        mse = float(sk.mean_squared_error(y, p))
        return {
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "MAE": float(sk.mean_absolute_error(y, p)),
            "R2": float(sk.r2_score(y, p)),
            "PEARSON": float(stats.pearsonr(y, p)[0]),
            "SPEARMAN": float(stats.spearmanr(y, p)[0]),
        }
    y, p, proba = np.array(truth, dtype=int), np.array(predictions, dtype=int), np.array(probabilities, dtype=float)
    labels = list(range(classes))
    if classes == 2:
        auroc = sk.roc_auc_score(y, proba[:, 1])
        auprc = sk.average_precision_score(y, proba[:, 1])
    else:
        onehot = np.eye(classes)[y]
        auroc = sk.roc_auc_score(y, proba, multi_class="ovr", labels=labels)
        auprc = sk.average_precision_score(onehot, proba, average="macro")
    return {
        "ACC": float(sk.accuracy_score(y, p)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(sk.f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        "LOG_LOSS": float(sk.log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=labels)),
        "MCC": float(sk.matthews_corrcoef(y, p)),
    }
