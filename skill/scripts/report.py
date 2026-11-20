"""Writes results/report.md from the run's record. See SKILL.md, step 3."""
import os

from common import METRICS, fail, load_record, workspace_python


def first_lines(path, count=6):
    if not os.path.exists(path):
        return "(none written)"
    with open(path) as handle:
        lines = [line.rstrip() for line in handle if line.strip()]
    return "\n".join(lines[:count])


def main():
    workspace_python()
    record = load_record()
    if record["best"] is None:
        fail("no iteration has passed its check, so there is no model to report.")
    names = METRICS[record["task"]]
    out = ["# %s: research report" % record["dataset"], ""]
    out += ["- Task: %s" % record["task"], "- Judged on: %s (validation)" % record["metric"],
            "- Iterations: %d planned, %d recorded" % (record["iterations_planned"], len(record["iterations"])),
            "- Best: iteration %d, in `best/`" % record["best"]["iteration"], ""]
    metric = record["metric"]
    out += ["## Validation scores", "", "| Iteration | Status | " + " | ".join(names) + " | Train " + metric + " |",
            "|---|---|" + "---|" * (len(names) + 1)]
    for item in record["iterations"]:
        values = [("%.4f" % item["validation"][name]) if item.get("validation") else "" for name in names]
        values.append(("%.4f" % item["train"][metric]) if item.get("train") else "")
        status = item["status"] + (" (best)" if item["iteration"] == record["best"]["iteration"] else "")
        out.append("| %d | %s | %s |" % (item["iteration"], status, " | ".join(values)))
    out += ["", "These are scores on the validation split, which the agent could see. The test score is computed by the "
            "person, from `final/test_predictions.csv`.", ""]
    for item in record["iterations"]:
        out += ["## Iteration %d" % item["iteration"], ""]
        if item.get("why"):
            out += ["Failed: " + item["why"], ""]
        out += ["Plan:", "", first_lines("iterations/%d/plan.md" % item["iteration"]), "",
                "Analysis:", "", first_lines("iterations/%d/analysis.md" % item["iteration"]), ""]
    out += ["## Notes", "", ""]
    with open("results/report.md", "w") as handle:
        handle.write("\n".join(out))
    print("OK: wrote results/report.md. Add your paragraph under Notes, then tell the person.")


if __name__ == "__main__":
    main()
