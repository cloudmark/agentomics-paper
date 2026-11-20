# The stages of an iteration

Each stage leaves a short record in `iterations/<n>/`, so the next stage, and the next iteration, work from what was decided and measured, not from memory. Keep observed facts apart from guesses in all of them.

## Plan (`plan.md`)

Read `data/description.md`, `results/metrics.json`, `lessons.md` and every earlier `analysis.md` first.

- **Iteration 0** is a baseline: a representation and a model that are simple, finish in a few minutes and show whether the data and the scripts work. A baseline that runs beats an ambitious model that doesn't.
- **Later iterations** make one coherent change, and say which evidence points at it: a validation score, a pattern in the errors, a gap between train and validation. First decide whether the last result was a model problem (too simple, overfitting) or a data or measurement problem (a leak, a wrong encoding, a bug); they need different fixes. Repeating the last idea with another seed is not an iteration, and neither is a strategy that already failed, unless something new justifies it.
- Say how the time is spent: how long training may take within the 10 minutes a command has, leaving room for one fix.
- Never write the score you expect to get.

The plan also holds the next two stages' decisions, under their own headings.

## Explore (`exploration.md`)

Write a small script, run it, and record what it printed. Do this in iteration 0, and again only when a plan depends on a fact you have not measured.

- What one sample is, what the label means, and how an id maps to the model's inputs.
- Counts per class, missing values, lengths or shapes of each input, duplicate ids and duplicate samples.
- Which inputs or simple statistics separate the classes, with numbers.
- Structure that could leak: the same entity (a subject, a sequence, a group, a date) on both sides of the split, or appearing with both labels. Ids being different does not mean samples are unrelated.
- For paired inputs (two sequences, a query and a document), what relates the pair, and whether one side alone predicts the label.

Do not split, relabel, fit or train here.

## Represent (in `plan.md`, under "Representation")

How each input becomes model features: encodings, tokens or k-mers, padding and truncation, how a pair's two sides are combined, scaling, any feature selection. For each step say whether it is fitted, and if so that it is fitted on `data/train` only and saved into the training artifacts so inference can apply it. A pretrained model or a download of data from outside is out: the model is trained on this dataset alone. You may run a short script to confirm a size or a memory estimate; you may not use validation labels to choose features.

## Design (in `plan.md`, under "Model")

The model family and structure, the loss, how class imbalance or the target's scale is handled, the hyperparameters with their values and seeds, any small search and its budget, and what early stopping watches. Estimate training time and memory from the data's size and the representation. From iteration 1, say how the design answers the last analysis.

## Train (`train.py`)

Implements the plan exactly. Reads `<split>/input/data.csv` and `<split>/labels.csv` from `--train-data`, may use `--validation-data` for early stopping only, and writes everything inference needs into `--artifacts-dir`: the model, fitted encoders, the feature list, the class order. Sets its seeds. Run it yourself on `data/train` into `iterations/<n>/training_artifacts` before the check.

## Infer (`inference.py`)

Reads only `--input` (a folder with `data.csv`) and `--artifacts-dir`, and writes `--output`: `id`, `prediction`, and for classification a `probability_<k>` column per class id. One row per sample, every id once. It must work from any folder, for rows it has never seen, without labels.

## Check

    .venv/bin/python .skills/ml-research/scripts/check_iteration.py --iteration <n>

Read its output; it is the iteration's result. It prints validation and train scores side by side: a train score far above validation means overfitting, and both low means the model or the representation is too weak.

## Analyse (`analysis.md`)

Work from the saved `iterations/<n>/predictions_validation.csv` and `data/validation`; do not run inference or train again. Write a script, run it, and report:

- the confusion matrix and errors per class (or residuals by target range, for regression);
- how the probabilities are spread per class, and the most confident errors;
- errors by the input properties exploration found (length, composition, an entity seen in training or not);
- the train and validation scores from the check, and what their gap says.

End with what the evidence suggests about the data and the model for the next iteration, in two or three sentences. Then add to `lessons.md` anything another iteration could trip over.
