"""Checks an uploaded dataset and writes the splits a research run works from. See SKILL.md, step 1."""
import argparse
import csv
import json
import os
import random

from common import CLASSIFICATION, DEFAULT_METRIC, METRICS, RECORD, REGRESSION, fail, read_csv, save_record, workspace_python

SEED = 13
VALIDATION_SHARE = 0.2
MINI_PER_CLASS = 100


def write_split(folder, header, rows, label_of):
    os.makedirs(os.path.join(folder, "input"), exist_ok=True)
    with open(os.path.join(folder, "input", "data.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row, _ in rows)
    with open(os.path.join(folder, "labels.csv"), "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "label"])
        writer.writerows([row["id"], label_of(label)] for row, label in rows)


def main():
    workspace_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="the uploaded dataset's folder, uploads/<name>")
    parser.add_argument("--metric", help="the metric the run is judged on; the task's default when left out")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if os.path.exists(RECORD):
        fail(RECORD + " exists already: this workspace has a run set up. Carry on with it.")
    dataset = args.dataset.rstrip("/")
    if os.path.isdir(os.path.join(dataset, "test")):
        fail(dataset + " carries a test/ folder. A test set the agent can read scores nothing honest: stop and tell the person.")
    for needed in ("metadata.json", "train/labels.csv", "train/input/data.csv"):
        if not os.path.exists(os.path.join(dataset, needed)):
            fail("%s has no %s" % (dataset, needed))
    with open(os.path.join(dataset, "metadata.json")) as handle:
        task = str(json.load(handle).get("task_type", "")).lower()
    if task not in (CLASSIFICATION, REGRESSION):
        fail("metadata.json's task_type is '%s'; it must be classification or regression" % task)
    metric = (args.metric or DEFAULT_METRIC[task]).upper()
    if metric not in METRICS[task]:
        fail("%s is not a %s metric; choose one of %s" % (metric, task, ", ".join(METRICS[task])))

    header, rows = read_csv(os.path.join(dataset, "train/input/data.csv"))
    if "id" not in header:
        fail("train/input/data.csv has no id column")
    _, label_rows = read_csv(os.path.join(dataset, "train/labels.csv"))
    labels = {}
    for row in label_rows:
        values = list(row.values())
        labels[values[0]] = values[1]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        fail("train/input/data.csv has an id more than once")
    unlabelled = [sample for sample in ids if sample not in labels]
    if unlabelled:
        fail("%d sample(s) have no label, such as %s" % (len(unlabelled), ", ".join(unlabelled[:5])))
    if len(labels) != len(ids):
        fail("train/labels.csv has %d label(s) for samples that are not in data.csv" % (len(labels) - len(ids)))

    classes = sorted(set(labels.values())) if task == CLASSIFICATION else []
    class_id = {label: index for index, label in enumerate(classes)}
    label_of = (lambda label: class_id[label]) if task == CLASSIFICATION else (lambda label: label)
    if task == REGRESSION:
        for sample, label in labels.items():
            try:
                float(label)
            except ValueError:
                fail("sample %s's label '%s' is not a number" % (sample, label))

    # stratified by class (regression: one stratum), seeded, so every run of a dataset cuts the same split
    rng = random.Random(SEED)
    strata = {}
    for row in rows:
        strata.setdefault(labels[row["id"]] if task == CLASSIFICATION else "", []).append(row)
    train, validation, mini = [], [], []
    for stratum in sorted(strata):
        members = strata[stratum][:]
        rng.shuffle(members)
        cut = max(1, round(len(members) * VALIDATION_SHARE)) if len(members) > 1 else 0
        validation += members[:cut]
        train += members[cut:]
        mini += members[cut:cut + MINI_PER_CLASS]
    order = {sample: index for index, sample in enumerate(ids)}
    for part in (train, validation, mini):
        part.sort(key=lambda row: order[row["id"]])
    pair = lambda part: [(row, labels[row["id"]]) for row in part]
    write_split("data/train", header, pair(train), label_of)
    write_split("data/validation", header, pair(validation), label_of)
    write_split("data/mini_train", header, pair(mini), label_of)

    description = ""
    if os.path.exists(os.path.join(dataset, "dataset_description.md")):
        with open(os.path.join(dataset, "dataset_description.md")) as handle:
            description = handle.read().strip()
    mapping = ", ".join("%s → %d" % (label, index) for label, index in class_id.items())
    with open("data/description.md", "w") as handle:
        handle.write(description + "\n\n" if description else "")
        handle.write("## This run\n\n- Task: %s\n- Judged on: %s\n" % (task, metric))
        if classes:
            handle.write("- Class ids (the `label` column of every labels.csv here): %s\n" % mapping)
        handle.write("- Splits: data/train (%d samples), data/validation (%d), data/mini_train (%d, a part of train)\n"
                     % (len(train), len(validation), len(mini)))

    save_record({
        "dataset": os.path.basename(dataset), "task": task, "metric": metric, "classes": class_id,
        "iterations_planned": args.iterations, "iterations": [], "best": None,
    })
    print("OK: %s, %s, judged on %s, %d iteration(s) planned" % (os.path.basename(dataset), task, metric, args.iterations))
    print("train %d, validation %d, mini_train %d%s" % (len(train), len(validation), len(mini),
                                                        "; class ids: " + mapping if classes else ""))
    print("Read data/description.md next.")


if __name__ == "__main__":
    main()
