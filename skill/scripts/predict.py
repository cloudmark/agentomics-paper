"""Runs the best iteration's inference once on test inputs, writing final/test_predictions.csv. See SKILL.md, step 3."""
import argparse
import os
import subprocess
import sys

from common import fail, load_record, read_csv, read_predictions, workspace_python


def main():
    workspace_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="the test set's input/ folder; no labels")
    args = parser.parse_args()

    record = load_record()
    if record["best"] is None:
        fail("there is no best model to predict with.")
    data = os.path.join(args.input, "data.csv")
    if not os.path.exists(data):
        fail(args.input + " has no data.csv")
    if os.path.exists(os.path.join(os.path.dirname(args.input.rstrip("/")), "labels.csv")):
        fail("the test set's folder holds labels.csv. Labels the agent can read score nothing honest: stop and tell the person.")
    os.makedirs("final", exist_ok=True)
    # the same model on the train split too, so the person can put train beside test
    for inputs, output in ((args.input, "final/test_predictions.csv"), ("data/train/input", "final/train_predictions.csv")):
        done = subprocess.run([sys.executable, "best/inference.py", "--input", inputs, "--output", output,
                               "--artifacts-dir", "best/training_artifacts"], capture_output=True, text=True)
        if done.returncode != 0:
            fail("best/inference.py on %s exited %d:\n%s" % (inputs, done.returncode, (done.stderr or done.stdout)[-4000:]))
        _, rows = read_csv(os.path.join(inputs, "data.csv"))
        read_predictions(output, [row["id"] for row in rows], record["task"], len(record["classes"]))
        print("OK: wrote %s for %d sample(s), with iteration %d's model." % (output, len(rows), record["best"]["iteration"]))
    print("The person scores them.")


if __name__ == "__main__":
    main()
