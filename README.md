# Skill or State Machine?

Companion repository for the paper *"Skill or State Machine? Guidance versus Control for LLM Research Agents on miRNA–Target
Prediction"*. The compiled paper is [report.pdf](report.pdf).

We compare two ways of giving a language model an ML research procedure on the AGO2 CLASH miRNA-target classification task:

| Condition | Description |
|-----------|-------------|
| **Skill** | A general-purpose agent reads a plain-text skill document (`ml-research`) and follows it on its own, with no enforced steps. |
| **State machine** | The Agentomics-ML procedure run as a strict state machine: fixed steps, deterministic checks, and the model works only inside a step. |

Nine models × two methods × two reasoning settings = 36 runs, each planned for three iterations, scored on a held-out test set
by AU-PRC.

## What is here

```
report.pdf        The compiled paper
report.tex        Main LaTeX document
sections/         LaTeX source, one file per section
figures/          Diagrams and charts (TikZ / pgfplots)
references.bib    Bibliography
generate.py       Writes generated.tex (every number, table and chart series) from the files in data/
generated.tex     Output of generate.py
Makefile          Runs generate.py, then latexmk

data/             Everything the paper's numbers come from (see "Data" below)
bootstrap.py      95% bootstrap intervals and paired tests on the final scores
bootstrap_first.py  The same for the first iteration only
exact_scores.py   Unrounded score of every iteration, so tables round each score once

harness/          The run harness: both conditions on any OpenRouter model
skill/            The ml-research skill, as given to the agent
```

`runs/`, where the harness keeps each run's workspace and logs, is not in the repository.

## How the runs were made

All 36 runs reported in the paper were made with the harness in this repository (`harness/`): a state machine for one
condition and a general-purpose agent with the skill for the other.

**27 runs (18–20 November 2026)** were run on a separate machine using this harness. Their results are stored here: each
run's scores, cost, time and model calls are in `data/report-data.json`, and the test predictions of every iteration are
in `data/predictions/`. `python -m harness.import_paper_runs` loads them into `runs/` so they appear alongside any new
runs you make.

**9 runs (20–21 November 2026)** repeat, and replace, the same nine configurations using the same harness run directly
from this repository: all four GPT-4.1 runs, both GPT-6.2 Astra runs with reasoning high, and the state-machine runs of
Claude Opus 5.5, Gemini 4 Hydrogen and Claude Sonnet 5.5 with reasoning none.

The 9 repeated runs differ from the original 27 in their parameter settings: round limits (40 per step against 100),
the number of model steps per iteration (five against eight), how the validation split is made, prompts and how the
reasoning setting is sent. The paper's *Repeated runs* subsection gives the reasons for the repeats and lists these
differences.

The repeats were launched like this:

```bash
python harness/run.py --model openai/gpt-4.1 --reasoning both --force
```

```bash
python harness/run.py --model openai/gpt-6.2-astra --reasoning high --force
```

```bash
python harness/run.py --model anthropic/claude-opus-5.5 google/gemini-4-hydrogen anthropic/claude-sonnet-5.5 --condition sm --reasoning none --force
```

The harness sends the reasoning setting to OpenRouter as given, and OpenRouter refused `none` for Opus, Gemini, Sonnet and
Astra, whose reasoning it lists as mandatory. The last three state-machine repeats were therefore run with the same harness
code and the effort set by hand to the lowest value the provider accepted: `minimal` for Opus and Gemini, `low` for Sonnet.
That value is not a command-line option; each run's note in `data/report-data.json` records what was sent.

Attempts that did not finish, runs that were replaced and nine pilot runs of the harness on DeepSeek are not among the 36. They
are listed in the paper's appendix and in `data/report-data.json` under `extra`.

## Running the harness

### Prerequisites

Docker must be running: both conditions execute the model's commands in the `biogemt/agentomics:1.2.0` image.

```bash
pip install git+https://github.com/BioGeMT/agentomics-ml.git
```

```bash
pip install langgraph langchain-openai langchain-community rich numpy scipy scikit-learn
```

```bash
export OPENROUTER_API_KEY=sk-or-...
```

The key is also read from a `.env` file in this folder, which git ignores.

### Commands

```bash
# Both conditions on one model
python harness/run.py --model openai/gpt-4o-mini
```

```bash
# The paper's whole matrix: nine models, both reasoning settings, both conditions
python harness/run.py --model paper --reasoning both
```

```bash
# Two models, skill only, reasoning high
python harness/run.py --model anthropic/claude-haiku-4.5 deepseek/deepseek-v4.1-flash --reasoning high --condition skill
```

```
--model       one or more OpenRouter model ids, or `paper` for the paper's nine  (default: openai/gpt-4o-mini)
--reasoning   none | high | both   (default: none)
--condition   skill | sm | both    (default: both)
--api-key     overrides OPENROUTER_API_KEY
--runs-dir    where runs are kept  (default: ./runs/)
--no-resume   start a stopped state-machine run from scratch instead of continuing from its last step
--force       set aside the result of each matching run (as result.before-<time>.json) so that it runs again
```

Runs go one at a time, cheapest model first (by OpenRouter's listed price): for each model, the skill with reasoning off and on,
then the state machine with reasoning off and on. The display shows the plan as a matrix, a results table for every run in the
plan, and the running run's log.

### Where the harness is

Both conditions are in `harness/`, started by the same command (`harness/run.py`), which picks one with `--condition`.

| | 1. State machine (`--condition sm`) | 2. Skill (`--condition skill`) |
|---|---|---|
| **Entry point** | `harness/agentomics_runner.py` | `harness/skill_agent.py` |
| **What it runs** | Agentomics-ML 1.2.0, unmodified, from its container image `biogemt/agentomics:1.2.0` | A LangGraph agent with file and shell tools, defined in `skill_agent.py` |
| **Supporting files** | `harness/sm_launch.py` (starts Agentomics-ML's own launcher), `harness/container/sitecustomize.py` (reasoning effort, retry count, disabled tool, call log) | `harness/sandbox.py` (the container its commands run in), `skill/` (the ml-research skill it loads) |
| **Upstream source** | https://github.com/BioGeMT/agentomics-ml | https://github.com/palexiou/agentomics-distill |

Shared by both: `harness/protocol.py` (round limits and iterations), `harness/dataset_prep.py` (the dataset and its split),
`harness/scoring.py` (scores each iteration on test and validation), `harness/usage.py` (cost and tokens per call),
`harness/runs.py` and `harness/results.py` (attempts, resuming, the run's record) and `harness/export.py`.

### What the harness does

- **State machine.** It runs Agentomics-ML's published container image through Agentomics-ML's own launcher and changes three
  things from outside (`harness/container/sitecustomize.py`): the reasoning effort is set on every model request, a failing
  script is returned to its step three times, not the default five, and its tool for fetching papers is switched off. Each
  model step has 100 rounds.
- **Skill.** A LangGraph agent with file and shell tools works in a container of the same image, with the skill in `skill/`
  available to load. It has 1,500 rounds for its one turn, then receives the test inputs in the same conversation.
- **Scoring.** After a run's iterations, each iteration's model is run on the test inputs and the validation split from a copy
  of the workspace, without network.

A run that OpenRouter refuses for its reasoning setting fails on its first model call and costs nothing. Agentomics-ML's
launcher mounts `~/.codex`, if it exists, read-only into the model's container.

### Runs, resuming and what is recorded

Each run has its own folder, `runs/<model>/<method>-<reasoning>/`, and every try at it is an `attempt-N/` inside:

```
attempt-N/workspace/     the agent's workspace
attempt-N/log/           usage.jsonl (one line per model call: time, iteration, cost, tokens) and Agentomics-ML's output
attempt-N/events.json    start, iteration ends, end of the last iteration, status
attempt-N/scores/        each iteration's own model run on the test inputs and the validation split
result.json              the run's record, written once it is scored
```

Running the same command again picks up where it left off, so no work is done twice:

- **Done.** A run with a `result.json` is not run again; the plan shows its score.
- **Finished but not scored.** Only the scoring runs again.
- **State machine, stopped or crashed.** Agentomics-ML commits a git checkpoint in the workspace after every step. A new
  attempt continues from the newest one; the run is marked `recovered`, and its note says which attempt and step it continued
  from. `--no-resume` starts it from scratch instead.
- **Skill, stopped.** It starts again in a new attempt; the skill agent keeps no checkpoint.
- Every attempt is kept; what an unfinished attempt spent is reported as extra cost.
- An iteration that fails inside a run is not retried. The run is recorded as `partial` with that iteration's scores missing.

To see the paper's runs in the harness's display alongside your own, seed `runs/` from the data first:

```bash
python -m harness.import_paper_runs
```

## Generating the paper

To rebuild `report.pdf` from the data in this repository:

```bash
make
```

`make` runs `generate.py`, which writes `generated.tex` from the files in `data/`, and then `latexmk`. It needs Python 3 and a
LaTeX distribution with `latexmk`, `pgfplots`, `booktabs`, `tikz` and `subcaption`. `python3 generate.py` writes `generated.tex`
only, and `make clean` removes the build files.

No number is typed into the LaTeX source: every score, count, cost, table and chart series is a macro in `generated.tex`.

### From runs to the paper

After new or repeated runs, the paper is brought up to date in five steps.

1. Collect the finished runs. This writes `data/harness/report-data.json` and `data/harness/predictions/`, which git ignores:

```bash
python -m harness.export
```

2. Carry each new run into `data/report-data.json` by hand (its `cells` and `iters` records), add the run it replaces to
   `replaced` and `extra`, and copy its predictions folder into `data/predictions/`.

3. Recompute the statistics from the saved predictions:

```bash
python3 bootstrap.py data/report-data.json data/predictions data/dataset/AGO2_CLASH_Hejret2023/test/labels.csv data/bootstrap.json
```

```bash
python3 bootstrap_first.py data/report-data.json data/predictions data/dataset/AGO2_CLASH_Hejret2023/test/labels.csv data/bootstrap-first.json
```

```bash
python3 exact_scores.py data/report-data.json data/predictions data/dataset/AGO2_CLASH_Hejret2023/test/labels.csv data/exact-scores.json
```

4. The paper also quotes the tallies with the nine original runs in place of their repeats. Those come from the same scripts
   over `data/report-data-originals.json`:

```bash
python3 bootstrap.py data/report-data-originals.json data/predictions data/dataset/AGO2_CLASH_Hejret2023/test/labels.csv data/bootstrap-originals.json
```

```bash
python3 bootstrap_first.py data/report-data-originals.json data/predictions data/dataset/AGO2_CLASH_Hejret2023/test/labels.csv data/bootstrap-first-originals.json
```

5. Build:

```bash
make
```

Steps 3 and 4 are deterministic: run on the data in this repository, they reproduce the committed files exactly.

## Data

| File | Holds |
|------|-------|
| `data/report-data.json` | The 36 reported runs: score, cost, time and calls (`cells`), each iteration (`iters`), the attempts, replaced runs and pilots that are not among the 36 (`extra`), and the nine original runs the repeats replaced (`replaced`). |
| `data/report-data-originals.json` | The same records with the nine original runs in place of their repeats. |
| `data/predictions/<session>/<n>-test.csv` | The test predictions of iteration `n` (from 0) of the run with that `session` id: the 36 reported runs and the nine replaced ones. |
| `data/bootstrap.json`, `data/bootstrap-first.json` | 95% paired bootstrap intervals and Wilcoxon tests for the 18 final pairs and the 17 first-iteration pairs. |
| `data/bootstrap-originals.json`, `data/bootstrap-first-originals.json` | The same with the nine original runs. |
| `data/exact-scores.json` | Unrounded test AU-PRC and accuracy of every iteration's saved predictions. |
| `data/dataset/AGO2_CLASH_Hejret2023/` | The dataset every run was given: 8,140 training pairs and 1,018 test pairs. The harness reads it from there. |

## Citation

> Vlastimil Martinek, Andrea Gariboldi, Dimosthenis Tzimotoudis, Mark Galea, Elissavet Zacharopoulou, Panagiotis Alexiou.
> *Skill or State Machine? Guidance versus Control for LLM Research Agents on miRNA–Target Prediction.*
> 2026.
