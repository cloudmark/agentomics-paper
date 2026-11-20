"""Bootstrap 95% CIs on each run's test AU-PRC and paired skill-vs-state-machine tests, from the saved predictions.

Reads the report data (which iteration each run submitted) and out/<session>/<it>-test.csv; writes bootstrap.json.
Usage: python3 bootstrap.py <report-data.json> <out-dir> <test-labels.csv> <bootstrap.json>
"""
import csv, json, sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

data, out, labels_csv, dest = sys.argv[1:5]
D = json.load(open(data))
labels = {r["id"]: int(r["label"]) for r in csv.DictReader(open(labels_csv))}
ids = sorted(labels)
y = np.array([labels[i] for i in ids])
B = 2000
rng = np.random.default_rng(20261119)
draws = rng.integers(0, len(ids), size=(B, len(ids)))  # the same resamples for every run, so differences are paired


def scores(path):
    p = {r["id"]: float(r["probability_1"]) for r in csv.DictReader(open(path))}
    return np.array([p[i] for i in ids])


def ap(s, idx=None):
    return average_precision_score(y if idx is None else y[idx], s if idx is None else s[idx])


runs = {}
for cell, it in zip(D["cells"], D["iters"]):
    assert (cell["model"], cell["arm"], cell["effort"]) == (it["model"], it["arm"], it["effort"])
    final = it["overall"]["test_auprc"]
    # The submitted iteration is the one whose own test score is the run's final score.
    pick = next((p["it"] for p in it["per"] if p["test_auprc"] is not None and abs(p["test_auprc"] - final) < 5e-5), None)
    path = "%s/%s/%s-test.csv" % (out, cell["session"], pick)
    try:
        s = scores(path)
    except (FileNotFoundError, TypeError):
        print("missing", cell["model"], cell["arm"], cell["effort"], path)
        continue
    point = ap(s)
    boot = np.array([ap(s, d) for d in draws])
    key = "%s|%s|%s" % (cell["model"], cell["arm"], cell["effort"])
    runs[key] = {"ap": point, "reported": cell["test_auprc"], "lo": float(np.percentile(boot, 2.5)),
                 "hi": float(np.percentile(boot, 97.5)), "boot": boot}
    print("%-45s %.4f (reported %.4f) [%.3f, %.3f]" % (key, point, cell["test_auprc"], runs[key]["lo"], runs[key]["hi"]))

pairs, diffs = [], []
for m in dict.fromkeys(c["model"] for c in D["cells"]):
    for e in ("none", "high"):
        a, b = runs.get("%s|sm|%s" % (m, e)), runs.get("%s|skill|%s" % (m, e))
        if not a or not b:
            continue
        d = a["boot"] - b["boot"]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        pairs.append({"model": m, "effort": e, "diff": a["ap"] - b["ap"], "lo": lo, "hi": hi})
        diffs.append(a["ap"] - b["ap"])
stat = wilcoxon(diffs)
half = [(r["hi"] - r["lo"]) / 2 for r in runs.values()]
summary = {
    "runs": {k: {x: v[x] for x in ("ap", "reported", "lo", "hi")} for k, v in runs.items()},
    "pairs": pairs,
    "wilcoxon": {"n": len(diffs), "statistic": float(stat.statistic), "p": float(stat.pvalue),
                 "median_diff": float(np.median(diffs)), "sm_wins": int(sum(d > 0 for d in diffs))},
    "ci_halfwidth": {"min": float(min(half)), "median": float(np.median(half)), "max": float(max(half))},
    "pairs_excluding_zero": {"sm": sum(p["lo"] > 0 for p in pairs), "skill": sum(p["hi"] < 0 for p in pairs)},
}
json.dump(summary, open(dest, "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("wilcoxon", "ci_halfwidth", "pairs_excluding_zero")}, indent=1))
