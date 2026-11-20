"""Checks one iteration's scripts the way the final test will run them, and records its scores. See SKILL.md, step 2.5."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from common import beats, fail, load_record, read_csv, read_predictions, save_record, score, workspace_python

LIMIT_SECONDS = 600
MAX_ATTEMPTS = 3


def run(command, what):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=LIMIT_SECONDS)
    except subprocess.TimeoutExpired:
        fail("%s took over %d seconds" % (what, LIMIT_SECONDS))
    if done.returncode != 0:
        fail("%s exited %d:\n%s" % (what, done.returncode, (done.stderr or done.stdout)[-4000:]))


def entry(record, iteration):
    for item in record["iterations"]:
        if item["iteration"] == iteration:
            return item
    item = {"iteration": iteration, "status": "unchecked", "attempts": 0}
    record["iterations"].append(item)
    record["iterations"].sort(key=lambda each: each["iteration"])
    return item


def main():
    workspace_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--give-up", help="records the iteration as failed, with why")
    args = parser.parse_args()

    record = load_record()
    folder = "iterations/%d" % args.iteration
    item = entry(record, args.iteration)
    if item["status"] == "ok":
        fail("iteration %d is recorded already (%s %.4f). A changed idea is a new iteration."
             % (args.iteration, record["metric"], item["validation"][record["metric"]]))
    if args.give_up:
        item.update(status="failed", why=args.give_up)
        save_record(record)
        print("Recorded iteration %d as failed. Plan the next one around it." % args.iteration)
        return
    if item["attempts"] >= MAX_ATTEMPTS:
        fail("iteration %d has used its %d checks. Record it with --give-up \"<why>\" and move on." % (args.iteration, MAX_ATTEMPTS))
    item["attempts"] += 1
    save_record(record)

    train, inference, artifacts = folder + "/train.py", folder + "/inference.py", folder + "/training_artifacts"
    for path in (train, inference):
        if not os.path.exists(path):
            fail(path + " is missing")
    if not os.path.isdir(artifacts) or not os.listdir(artifacts):
        fail(artifacts + " is missing or empty: run train.py on data/train first")
    with open(inference) as handle:
        source = handle.read()
    if re.search(r"labels\.csv", source):
        fail("inference.py mentions labels.csv. Inference gets only input/; a model that needs labels cannot be tested.")
    if re.search(r"iterations/\d|data/(train|validation|mini_train)", source):
        fail("inference.py reads a fixed path (another iteration's folder or a data split). It may read only --input and --artifacts-dir.")

    # train.py must run from scratch: on mini_train, into a folder of its own
    with tempfile.TemporaryDirectory() as scratch:
        run([sys.executable, train, "--train-data", "data/mini_train", "--validation-data", "data/validation",
             "--artifacts-dir", scratch], "train.py on data/mini_train")
        if not os.listdir(scratch):
            fail("train.py on data/mini_train wrote nothing into --artifacts-dir")

    # inference.py on validation's inputs alone, with the iteration's real artifacts copied elsewhere
    task, classes = record["task"], len(record["classes"])
    def scored(split):
        with tempfile.TemporaryDirectory() as scratch:
            moved = os.path.join(scratch, "artifacts")
            shutil.copytree(artifacts, moved)
            inputs = os.path.join(scratch, "input")
            shutil.copytree("data/%s/input" % split, inputs)
            predictions_path = os.path.join(scratch, "predictions.csv")
            run([sys.executable, inference, "--input", inputs, "--output", predictions_path, "--artifacts-dir", moved],
                "inference.py on data/%s/input" % split)
            _, label_rows = read_csv("data/%s/labels.csv" % split)
            predictions, probabilities = read_predictions(predictions_path, [row["id"] for row in label_rows], task, classes)
            shutil.copy(predictions_path, "%s/predictions_%s.csv" % (folder, split))
        truth = [float(row["label"]) if task == "regression" else int(row["label"]) for row in label_rows]
        return score(task, classes, truth, predictions, probabilities)

    scores = scored("validation")
    # the same model on what it was trained on: the gap to validation is what analysis reads overfitting from
    train_scores = scored("train")

    metric = record["metric"]
    item.update(status="ok", validation=scores, train=train_scores)
    item.pop("why", None)
    best = record["best"]
    improved = best is None or beats(metric, scores[metric], best["validation"][metric])
    if improved:
        record["best"] = {"iteration": args.iteration, "validation": scores}
        shutil.rmtree("best", ignore_errors=True)
        os.makedirs("best")
        shutil.copy(train, "best/train.py")
        shutil.copy(inference, "best/inference.py")
        shutil.copytree(artifacts, "best/training_artifacts")
    save_record(record)

    print("OK: iteration %d checked. Scores (recorded in results/metrics.json):" % args.iteration)
    print("  %-9s %10s %10s" % ("", "validation", "train"))
    for name, value in scores.items():
        print("  %-9s %10.4f %10.4f%s" % (name, value, train_scores[name], "   <- judged on (validation)" if name == metric else ""))
    print("Best so far: iteration %d (%s %.4f)%s" % (record["best"]["iteration"], metric, record["best"]["validation"][metric],
                                                    ", this one, copied to best/" if improved else ""))
    left = record["iterations_planned"] - len([each for each in record["iterations"] if each["status"] != "unchecked"])
    print("Write %s/analysis.md next. %s" % (folder, "%d iteration(s) left." % left if left > 0 else "That was the last planned iteration: run report.py."))


if __name__ == "__main__":
    main()
