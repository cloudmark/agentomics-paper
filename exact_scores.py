"""Test AU-PRC and accuracy of every iteration's saved predictions, unrounded, so tables round each score once.

Usage: python3 exact_scores.py <report-data.json> <predictions-dir> <test-labels.csv> <exact-scores.json>
Writes {session: {iteration: {"auprc": ..., "acc": ...}}} for the sessions the report data names.
"""
import csv, json, os, sys
from sklearn.metrics import average_precision_score

data, out, labels_csv, dest = sys.argv[1:5]
D = json.load(open(data))
labels = {r["id"]: int(r["label"]) for r in csv.DictReader(open(labels_csv))}
ids = sorted(labels)
y = [labels[i] for i in ids]
exact = {}
for cell in D["cells"]:
    folder = os.path.join(out, cell["session"])
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith("-test.csv"):
            continue
        rows = {r["id"]: r for r in csv.DictReader(open(os.path.join(folder, name)))}
        if set(rows) != set(ids) or "probability_1" not in rows[ids[0]]:
            continue
        prob = [float(rows[i]["probability_1"]) for i in ids]
        hard = [int(float(rows[i]["prediction"])) for i in ids] if "prediction" in rows[ids[0]] else None
        exact.setdefault(cell["session"], {})[name[:-len("-test.csv")]] = {
            "auprc": float(average_precision_score(y, prob)),
            "acc": None if hard is None else sum(int(a == b) for a, b in zip(hard, y)) / len(y)}
json.dump(exact, open(dest, "w"), indent=1)
print(sum(len(v) for v in exact.values()), "iterations of", len(exact), "sessions ->", dest)
