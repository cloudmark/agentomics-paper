"""Paired bootstrap for the single-shot comparison: each run's model after its first iteration (the `after[0]` pick of the
report data), state machine minus skill per model and reasoning setting, on the same resamples as bootstrap.py.
Usage: python3 bootstrap_first.py <report-data.json> <out-dir> <test-labels.csv> <bootstrap-first.json>"""
import csv, json, sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

data, out, labels_csv, dest = sys.argv[1:5]
D = json.load(open(data))
labels = {r["id"]: int(r["label"]) for r in csv.DictReader(open(labels_csv))}
ids = sorted(labels); y = np.array([labels[i] for i in ids])
B = 2000; rng = np.random.default_rng(20261119); draws = rng.integers(0, len(ids), size=(B, len(ids)))
ap = lambda s, idx=None: average_precision_score(y if idx is None else y[idx], s if idx is None else s[idx])
runs = {}
for cell, it in zip(D["cells"], D["iters"]):
    assert (cell["model"], cell["arm"], cell["effort"]) == (it["model"], it["arm"], it["effort"])
    first = it["after"][0] if it["after"] else None
    if not first or first.get("none") or first.get("test_auprc") is None: continue
    p = {r["id"]: float(r["probability_1"]) for r in csv.DictReader(open("%s/%s/%s-test.csv" % (out, cell["session"], first["pick"])))}
    s = np.array([p[i] for i in ids]); point = ap(s)
    assert abs(point - first["test_auprc"]) < 5e-4, (cell["model"], cell["arm"], cell["effort"], point, first["test_auprc"])
    runs[(cell["model"], cell["arm"], cell["effort"])] = (point, np.array([ap(s, d) for d in draws]))
pairs, diffs = [], []
for m in dict.fromkeys(c["model"] for c in D["cells"]):
    for e in ("none", "high"):
        a, b = runs.get((m, "sm", e)), runs.get((m, "skill", e))
        if not a or not b: continue
        d = a[1] - b[1]
        pairs.append({"model": m, "effort": e, "sm": a[0], "skill": b[0], "diff": a[0] - b[0],
                      "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))})
        diffs.append(a[0] - b[0])
stat = wilcoxon(diffs)
S = {"pairs": pairs, "wilcoxon": {"n": len(diffs), "p": float(stat.pvalue), "median_diff": float(np.median(diffs))},
     "pairs_excluding_zero": {"sm": sum(p["lo"] > 0 for p in pairs), "skill": sum(p["hi"] < 0 for p in pairs)}}
json.dump(S, open(dest, "w"), indent=1); print(json.dumps({k: S[k] for k in ("wilcoxon", "pairs_excluding_zero")}))
