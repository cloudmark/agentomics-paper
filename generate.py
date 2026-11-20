"""Writes generated.tex for report.tex from data/report-data.json: number macros, the tables and the chart coordinates.

Run `make` in this folder (or `python3 generate.py && latexmk`). Every number in the report comes from the data file.
Tables follow the plain booktabs layout of the Agentomics-ML paper: bold header, one row per entry, best value in bold.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "report-data.json")))
# Bootstrap CIs and paired tests, from bootstrap.py over the runs' saved test predictions.
BOOT = json.load(open(os.path.join(HERE, "data", "bootstrap.json")))
CELLS, ITERS = D["cells"], D["iters"]
# Scores are stored to four decimals; printing those to three rounds twice. Where the unrounded score of the same
# prediction file is at hand (exact_scores.py) and agrees with the stored one, it is used, so each score is rounded once.
EXACT_PATH = os.path.join(HERE, "data", "exact-scores.json")
EXACT = json.load(open(EXACT_PATH)) if os.path.exists(EXACT_PATH) else {}


def _exact(stored, value):
    return value if stored is not None and value is not None and abs(stored - value) < 0.00006 else stored


for _c, _r in zip(CELLS, ITERS):
    assert (_c["model"], _c["arm"], _c["effort"]) == (_r["model"], _r["arm"], _r["effort"])
    _x = EXACT.get(_c["session"], {})
    for _p in _r["per"]:
        _e = _x.get(str(_p["it"]))
        if _e:
            _p["test_auprc"], _p["test_acc"] = _exact(_p.get("test_auprc"), _e["auprc"]), _exact(_p.get("test_acc"), _e["acc"])
    _by = {str(_p["it"]): _p for _p in _r["per"]}
    for _a in _r["after"]:
        if _a and _a.get("pick") is not None and str(_a["pick"]) in _by:
            _a["test_auprc"], _a["test_acc"] = _exact(_a.get("test_auprc"), _by[str(_a["pick"])]["test_auprc"]), _exact(_a.get("test_acc"), _by[str(_a["pick"])]["test_acc"])
    for _e in _x.values():
        _c["test_auprc"], _c["test_acc"] = _exact(_c.get("test_auprc"), _e["auprc"]), _exact(_c.get("test_acc"), _e["acc"])
    _r["overall"]["test_auprc"], _r["overall"]["test_acc"] = _c["test_auprc"], _exact(_r["overall"].get("test_acc"), _c.get("test_acc"))
ORDER = [("skill", "none"), ("skill", "high"), ("sm", "none"), ("sm", "high")]
ARM = {"skill": "skill", "sm": "state machine"}
ARM_SHORT = {"skill": "skill", "sm": "SM"}
METHOD = {"skill": "Skill", "sm": "State machine"}
# Model tiers by each provider's own line-up (top-of-line vs. mid-range), sorted within tier by relative capability.
FLAGSHIP = ["Claude Fable 5.2", "Claude Opus 5.5", "GPT-6.2 Astra", "Gemini 4 Hydrogen"]
SMALLER = ["Claude Sonnet 5.5", "Qwen 3.5 122B-A10B", "Claude Haiku 4.5", "DeepSeek V4.1 Flash", "GPT-4.1"]
MODELS = FLAGSHIP + SMALLER
ALL_MODELS = MODELS
# A run that did not finish has no test score: shown as a dash, and left out of every mean, best and comparison.
DAGGER = "$^{\\dagger}$"
DDAGGER = "$^{\\ddagger}$"
REPEATS = {(r["model"], r["arm"], r["effort"]) for r in D.get("replaced", [])}
# Runs whose test predictions we produced ourselves from the run's selected model.
HAND = {(c["model"], c["arm"], c["effort"]) for c in CELLS if c.get("recovered") and "continued from" not in (c.get("note") or "")}


def marks(m, a, e):
    return (DAGGER if (m, a, e) in REPEATS else "") + (DDAGGER if (m, a, e) in HAND else "")


def signed(v):
    t = "%+.3f" % v
    return "0.000" if t in ("-0.000", "+0.000") else t


def interval(lo, hi):
    return "$[%s,\\;%s]$" % (signed(lo), signed(hi))


def delta(v, strong=False):
    """A signed difference in math mode, without a negative zero."""
    t = signed(v)
    return "$\\mathbf{%s}$" % t if strong else "$%s$" % t


def cell(model, arm, effort):
    return next(c for c in CELLS if c["model"] == model and c["arm"] == arm and c["effort"] == effort)


def it_of(model, arm, effort):
    return next(r for r in ITERS if r["model"] == model and r["arm"] == arm and r["effort"] == effort)


def f3(v):
    return "--" if v is None else "%.3f" % v


def money(v):
    return "\\$%.2f" % (v or 0)


def mins(v):
    if v is None:
        return "--"
    if v < 0.5:
        return "$<$1\\,min"
    return "%.1f\\,h" % (v / 60) if v >= 60 else "%d\\,min" % round(v)


def tex(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def scored(cells):
    return [c for c in cells if c["test_auprc"] is not None]


def bold(s):
    return "\\textbf{%s}" % s


out = []
w = out.append

# ── Number macros used by the prose ──
skill_base = [c for c in CELLS if c["arm"] == "skill"]
sm_base    = [c for c in CELLS if c["arm"] == "sm"]
best = max(scored(CELLS), key=lambda c: c["test_auprc"])
# Reasoning none against high, where both runs of a model and method have a score.
effort_pairs = [(cell(c["model"], c["arm"], "none"), c) for c in scored(CELLS) if c["effort"] == "high"
                and cell(c["model"], c["arm"], "none")["test_auprc"] is not None]


def avg_cost(models, arm):
    return money(mean([c["cost"] for c in scored(CELLS) if c["model"] in models and c["arm"] == arm]))

macros = {
    "SkillCost": money(sum(c["cost"] for c in skill_base if c["cost"] is not None)),
    "SkillTime": mins(sum(c["time"] for c in skill_base if c["time"] is not None)),
    "SmCost":    money(sum(c["cost"] for c in sm_base    if c["cost"] is not None)),
    "SmTime":    mins(sum(c["time"] for c in sm_base     if c["time"] is not None)),
    "BestScore": f3(best["test_auprc"]), "BestCell": tex("%s, %s, reasoning " % (best["model"], ARM[best["arm"]])) + "\\texttt{%s}" % best["effort"],
    "SkillCalls": "{:,}".format(sum(c["calls"] for c in skill_base if c["calls"] is not None)).replace(",", "{,}"),
    "SmCalls":    "{:,}".format(sum(c["calls"] for c in sm_base    if c["calls"] is not None)).replace(",", "{,}"),
    "HighWins": str(sum(1 for none, high in effort_pairs if high["test_auprc"] > none["test_auprc"])),
    "HighPairs": str(len(effort_pairs)),
    "ScoredRuns": str(len(scored(CELLS))), 
    "FlagSkillAvgCost": avg_cost(FLAGSHIP, "skill"), "FlagSmAvgCost": avg_cost(FLAGSHIP, "sm"),
    "SmallSkillAvgCost": avg_cost(SMALLER, "skill"), "SmallSmAvgCost": avg_cost(SMALLER, "sm"),
    "CostRatio": "%.0f\\%%" % (100 * sum(c["cost"] for c in skill_base) / sum(c["cost"] for c in sm_base)),
    "TimeRatio": "%.0f\\%%" % (100 * sum(c["time"] for c in skill_base) / sum(c["time"] for c in sm_base)),
    # The same two ratios over the models that finished under both methods.
}
for name, value in macros.items():
    w("\\newcommand{\\%s}{%s}" % (name, value))

# Per way of running, over its scored runs: mean, median and best test AU-PRC (the comparison with Agentomics-ML),
# and how many runs beat the published references: Agentomics-ML's best (0.778) and the human state of the art (0.860).
for name, xs in (("Skill", skill_base), ("Sm", sm_base)):
    scores = sorted(c["test_auprc"] for c in scored(xs))
    w("\\newcommand{\\%sMean}{%s}\\newcommand{\\%sMedian}{%s}\\newcommand{\\%sBest}{%s}" % (
        name, f3(mean(scores)), name, f3(median(scores)), name, f3(scores[-1])))
    w("\\newcommand{\\%sAboveAgentomics}{%d}\\newcommand{\\%sAboveSota}{%d}" % (
        name, sum(s > 0.778 for s in scores), name, sum(s > 0.860 for s in scores)))


def best_of(model, arm):
    return max((c["test_auprc"] for c in scored(cell(model, arm, e) for e in ("none", "high"))), default=None)


# ── GPT-4.1's own runs, beside the figures Agentomics-ML published for it (tab:agentomics) ──
gpt_sm = [cell("GPT-4.1", "sm", e)["test_auprc"] for e in ("none", "high")]
w("\\newcommand{\\GptSmNone}{%s}\\newcommand{\\GptSmHigh}{%s}\\newcommand{\\GptSmMean}{%s}\\newcommand{\\GptSmBest}{%s}" % (
    f3(gpt_sm[0]), f3(gpt_sm[1]), f3(mean(gpt_sm)), f3(max(gpt_sm))))
gpt_skill = [cell("GPT-4.1", "skill", e)["test_auprc"] for e in ("none", "high")]
w("\\newcommand{\\GptSkillNone}{%s}\\newcommand{\\GptSkillHigh}{%s}\\newcommand{\\GptSkillMean}{%s}\\newcommand{\\GptSkillBest}{%s}" % (
    f3(gpt_skill[0]), f3(gpt_skill[1]), f3(mean(gpt_skill)), f3(max(gpt_skill))))


# ── Paired comparison: state machine minus skill, per model and reasoning setting, with bootstrap 95% CIs ──
pairs = {(p["model"], p["effort"]): p for p in BOOT["pairs"]}
wil = BOOT["wilcoxon"]
half = BOOT["ci_halfwidth"]
sig_sm = [k for k, p in pairs.items() if p["lo"] > 0]
sig_skill = [k for k, p in pairs.items() if p["hi"] < 0]
small = SMALLER
for name, value in {
    "WilcoxonP": "%.2f" % wil["p"], "WilcoxonN": str(wil["n"]), "MedianDiff": "%.3f" % wil["median_diff"],
    "CIHalfMed": "%.3f" % half["median"], 
    "PairsSm": str(len(sig_sm)), "PairsSkill": str(len(sig_skill)), "PairsTie": str(len(pairs) - len(sig_sm) - len(sig_skill)),
    "FlagHighTies": str(sum(1 for m in FLAGSHIP if pairs[(m, "high")]["lo"] <= 0 <= pairs[(m, "high")]["hi"])),
    "FlagHighMax": "%.3f" % max(abs(pairs[(m, "high")]["diff"]) for m in FLAGSHIP),
    "SmallSm": str(sum(1 for k in sig_sm if k[0] in small)), "SmallSkill": str(sum(1 for k in sig_skill if k[0] in small)),
    "FlagScoreMin": f3(min(cell(m, a, e)["test_auprc"] for m in FLAGSHIP for a in ("skill", "sm") for e in ("none", "high"))),
    "FlagScoreMax": f3(max(cell(m, a, e)["test_auprc"] for m in FLAGSHIP for a in ("skill", "sm") for e in ("none", "high"))),
    "SmallScoreMin": f3(min(c["test_auprc"] for c in scored(CELLS) if c["model"] in small)),
    "SmallScoreMax": f3(max(c["test_auprc"] for c in scored(CELLS) if c["model"] in small)),
    "SmallPairs": str(sum(1 for k in pairs if k[0] in small)),
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, value))

PAIR_HEAD = ("\\textbf{Model} & \\textbf{Reas.} & \\textbf{Skill} & \\textbf{SM} & "
             "\\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{Winner}\\\\\\midrule")
w("\\newcommand{\\PairTable}{%")
w("\\begin{tabular}{@{}l l r r r r l@{}}\\toprule")
w(PAIR_HEAD)
for title, group in (("Top-of-line models", FLAGSHIP), ("Mid-range models", small)):
    w("\\multicolumn{7}{@{}l}{\\emph{%s}}\\\\" % title)
    for mi, m in enumerate(group):
        if mi > 0:
            w("\\addlinespace[3pt]")
        for e in ("none", "high"):
            skill_score = f3(cell(m, "skill", e)["test_auprc"])
            sm_score    = f3(cell(m, "sm",    e)["test_auprc"])
            skill_mark, sm_mark = marks(m, "skill", e), marks(m, "sm", e)
            p = pairs.get((m, e))
            if p is None:
                # No pair without both scores: nothing to compare.
                w("%s & %s & %s & %s & -- & -- & --\\\\" % (tex(m), e, skill_score + DAGGER, sm_score))
                continue
            d = delta(p["diff"], p["lo"] > 0 or p["hi"] < 0)
            if p["lo"] > 0:
                winner = bold("SM")
                sm_score = bold(sm_score)
            elif p["hi"] < 0:
                winner = bold("Skill")
                skill_score = bold(skill_score)
            else:
                winner = "tie"
            ci = interval(p["lo"], p["hi"])
            w("%s & %s & %s & %s & %s & %s & %s\\\\" % (tex(m), e, skill_score + skill_mark, sm_score + sm_mark, d, ci, winner))
    w("\\midrule" if group is FLAGSHIP else "")
w("\\bottomrule\\end{tabular}}")
w("\\newcommand{\\PairTableNote}{\\footnotesize SM: state machine wins; Skill: skill wins; tie: 95\\% CI includes zero."
  + (" " + DAGGER + ": the skill run did not finish, so there is no pair to compare." if len(scored(CELLS)) < len(CELLS) else "") + "}")

for i, m in enumerate(MODELS):
    tag = "abcdefghi"[i]
    w("\\newcommand{\\BestSkill%s}{%s}\\newcommand{\\BestSm%s}{%s}" % (tag, f3(best_of(m, "skill")), tag, f3(best_of(m, "sm"))))

# ── Model x condition tables: accuracy, cost, time. Bold: the model's best; bold and underlined: the best overall. ──
HEADER = ("& \\multicolumn{2}{c}{\\textbf{Skill}} & \\multicolumn{2}{c}{\\textbf{SM}}\\\\"
          "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}"
          "\\textbf{Model} & \\textbf{none} & \\textbf{high} & \\textbf{none} & \\textbf{high}\\\\\\midrule")


def matrix(name, value, fmt, better, dnf="dash", mark_repeats=False):
    """Model x condition table; best per model in bold, overall best bold+underlined, over the runs that finished.
    A run that did not finish shows a dash, or (dnf="value") what it used, marked with a dagger; neither is ranked."""
    pick = max if better == "high" else min
    done = lambda m, a, e: value(cell(m, a, e)) if cell(m, a, e)["test_auprc"] is not None else None
    overall = pick(v for m in MODELS for a, e in ORDER if (v := done(m, a, e)) is not None)
    w("\\newcommand{\\%s}{%%" % name)
    w("\\begin{tabular}{@{}l rr rr@{}}\\toprule")
    w(HEADER)
    for m in MODELS:
        row = [done(m, a, e) for a, e in ORDER]
        top = pick(v for v in row if v is not None)
        shown = []
        for (a, e), v in zip(ORDER, row):
            if v is None:
                shown.append((fmt(value(cell(m, a, e))) if dnf == "value" else "--") + DAGGER)
                continue
            s = fmt(v)
            if fmt(v) == fmt(overall):
                s = "\\underline{\\textbf{%s}}" % s
            elif fmt(v) == fmt(top):
                s = bold(s)
            shown.append(s + (marks(m, a, e) if mark_repeats else ""))
        w("%s & %s\\\\" % (tex(m), " & ".join(shown)))
    w("\\midrule")
    columns = [[v for m in MODELS if (v := done(m, a, e)) is not None] for a, e in ORDER]
    w("Mean & %s\\\\" % " & ".join(fmt(mean(c)) for c in columns))
    w("Median & %s\\\\" % " & ".join(fmt(median(c)) for c in columns))
    w("\\bottomrule\\end{tabular}}")


matrix("MatrixTable", lambda c: c["test_auprc"], f3, "high", mark_repeats=True)
matrix("CostTable", lambda c: round(c["cost"], 2), lambda v: "%.2f" % v, "low", dnf="value")
matrix("TimeTable", lambda c: round(c["time"]), lambda v: "%.0f" % v, "low", dnf="value")

# Per condition (way of running x reasoning): mean test AU-PRC, cost, time and calls over the eight models.

# ── Table 4's rows: each method's runs by model tier, so top-of-line and mid-range models are not blended ──
for tier, group in (("Flag", FLAGSHIP), ("Mid", SMALLER)):
    for arm, tag in (("sm", "Sm"), ("skill", "Skill")):
        xs = [cell(m, arm, e)["test_auprc"] for m in group for e in ("none", "high")]
        w("\\newcommand{\\%s%sTierMean}{%s}\\newcommand{\\%s%sTierMedian}{%s}\\newcommand{\\%s%sTierBest}{%s}\\newcommand{\\%s%sTierRuns}{%d}" % (
            tier, tag, f3(mean(xs)), tier, tag, f3(median(xs)), tier, tag, f3(max(xs)), tier, tag, len(xs)))

# ── Single shot: the model a run has after its first iteration, against what it finally submitted ──
def first_of(m, a, e):
    x = it_of(m, a, e)["after"][0] if it_of(m, a, e)["after"] else None
    return x if x and not x.get("none") and x.get("test_auprc") is not None else None


first_pairs = [(first_of(m, "skill", e), first_of(m, "sm", e)) for m in MODELS for e in ("none", "high")]
first_pairs = [(sk["test_auprc"], sm["test_auprc"]) for sk, sm in first_pairs if sk and sm]
first_all = [(first_of(m, a, e), cell(m, a, e)) for m in MODELS for a, e in ORDER if first_of(m, a, e)]
for name, val in {
    "FirstRuns": str(len(first_all)),
    "FirstAboveSota": str(sum(1 for f, _ in first_all if f["test_auprc"] > 0.860)),
    "FirstFlagGain": f3(mean([c["test_auprc"] - f["test_auprc"] for f, c in first_all if c["model"] in FLAGSHIP])),
    "FirstSmallGain": f3(mean([c["test_auprc"] - f["test_auprc"] for f, c in first_all if c["model"] in SMALLER])),
    "FirstSkillMean": f3(mean([f["test_auprc"] for f, c in first_all if c["arm"] == "skill"])),
    "FirstSmMean": f3(mean([f["test_auprc"] for f, c in first_all if c["arm"] == "sm"])),
    "FirstGap": f3(mean([f["test_auprc"] for f, c in first_all if c["arm"] == "sm"]) - mean([f["test_auprc"] for f, c in first_all if c["arm"] == "skill"])),
    "FinalGap": f3(mean([c["test_auprc"] for c in CELLS if c["arm"] == "sm"]) - mean([c["test_auprc"] for c in CELLS if c["arm"] == "skill"])),
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, val))

# ── Single shot, pair by pair: state machine minus skill after the first iteration, with bootstrap 95% CIs ──
FIRST = json.load(open(os.path.join(HERE, "data", "bootstrap-first.json"))) if os.path.exists(os.path.join(HERE, "data", "bootstrap-first.json")) else {"pairs": [], "wilcoxon": {"n": 0, "p": 1.0, "median_diff": 0.0}}
fpairs = {(p["model"], p["effort"]): p for p in FIRST["pairs"]}
w("\\newcommand{\\FirstPairTable}{%")
w("\\begin{tabular}{@{}l l r r r r l@{}}\\toprule")
w(PAIR_HEAD)
for title, group in (("Top-of-line models", FLAGSHIP), ("Mid-range models", SMALLER)):
    w("\\multicolumn{7}{@{}l}{\\emph{%s}}\\\\" % title)
    for mi, m in enumerate(group):
        if mi > 0:
            w("\\addlinespace[3pt]")
        for e in ("none", "high"):
            p = fpairs.get((m, e))
            if p is None:
                sk, sm = first_of(m, "skill", e), first_of(m, "sm", e)
                w("%s & %s & %s & %s & -- & -- & --\\\\" % (tex(m), e, (f3(sk["test_auprc"]) if sk else "N/A") + marks(m, "skill", e), (f3(sm["test_auprc"]) if sm else "N/A") + marks(m, "sm", e)))
                continue
            sk, sm = f3(first_of(m, "skill", e)["test_auprc"]), f3(first_of(m, "sm", e)["test_auprc"])
            d = delta(p["diff"], p["lo"] > 0 or p["hi"] < 0)
            if p["lo"] > 0: winner, sm = bold("SM"), bold(sm)
            elif p["hi"] < 0: winner, sk = bold("Skill"), bold(sk)
            else: winner = "tie"
            w("%s & %s & %s & %s & %s & %s & %s\\\\" % (tex(m), e, sk + marks(m, "skill", e), sm + marks(m, "sm", e), d, interval(p["lo"], p["hi"]), winner))
    w("\\midrule" if group is FLAGSHIP else "")
w("\\bottomrule\\end{tabular}}")
fs = lambda group, test: sum(1 for k, p in fpairs.items() if k[0] in group and test(p))
for name, val in {
    "FirstCiSm": str(fs(MODELS, lambda p: p["lo"] > 0)), "FirstCiSkill": str(fs(MODELS, lambda p: p["hi"] < 0)),
    "FirstCiTie": str(fs(MODELS, lambda p: p["lo"] <= 0 <= p["hi"])),
    "FirstFlagCiSm": str(fs(FLAGSHIP, lambda p: p["lo"] > 0)), "FirstFlagCiPairs": str(fs(FLAGSHIP, lambda p: True)),
    "FirstMidCiSm": str(fs(SMALLER, lambda p: p["lo"] > 0)), "FirstMidCiPairs": str(fs(SMALLER, lambda p: True)),
    "FirstWilcoxonP": ("=%.3f" % FIRST["wilcoxon"]["p"]) if FIRST["wilcoxon"]["p"] >= 0.001 else "<0.001",
    "FirstMedianDiff": "%.3f" % FIRST["wilcoxon"]["median_diff"],
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, val))

# ── Reasoning effort by method, and the iteration budget: what one, two and three iterations give and cost ──
def arm_mean(a, e):
    return mean([cell(m, a, e)["test_auprc"] for m in MODELS])


for name, val in {
    "FableSkillHighCost": money(cell("Claude Fable 5.2", "skill", "high")["cost"]), "FableSmHighCost": money(cell("Claude Fable 5.2", "sm", "high")["cost"]),
    "RunsCost": money(sum(c["cost"] for c in CELLS)),
    "SkillNoneMean": f3(arm_mean("skill", "none")), "SkillHighMean": f3(arm_mean("skill", "high")),
    "SmNoneMean": f3(arm_mean("sm", "none")), "SmHighMean": f3(arm_mean("sm", "high")),
    "SkillHighWins": str(sum(1 for m in MODELS if cell(m, "skill", "high")["test_auprc"] > cell(m, "skill", "none")["test_auprc"])),
    "SmHighWins": str(sum(1 for m in MODELS if cell(m, "sm", "high")["test_auprc"] > cell(m, "sm", "none")["test_auprc"])),
    "SkillReasonGain": f3(arm_mean("skill", "high") - arm_mean("skill", "none")),
    "SmReasonGain": f3(arm_mean("sm", "high") - arm_mean("sm", "none")),
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, val))


def at_budget(m, a, e, k):
    """(test score or None, cumulative cost, cumulative time) of the run after its first k+1 iterations; a run with
    fewer iterations keeps what it ended with."""
    c, after = cell(m, a, e), it_of(m, a, e)["after"]
    x = after[k] if k < len(after) and after[k] else None
    if x is None:
        return c["test_auprc"], c["cost"], c["time"]
    return (None if x.get("none") else x.get("test_auprc")), x["cum_cost"], x["cum_time"]


w("\\newcommand{\\BudgetTable}{\\begin{tabular}{@{}l c r r r r@{}}\\toprule")
w("\\textbf{Method} & \\textbf{Iterations} & \\textbf{Runs with a model} & \\textbf{Mean test AU-PRC} & \\textbf{Cost} & \\textbf{Time}\\\\\\midrule")
budget = {}
for a in ("skill", "sm"):
    runs_ = [(m, e) for m in MODELS for e in ("none", "high")]
    tot_cost = sum(cell(m, a, e)["cost"] for m, e in runs_); tot_time = sum(cell(m, a, e)["time"] for m, e in runs_)
    for k in range(3):
        got = [at_budget(m, a, e, k) for m, e in runs_]
        if k == 2:
            got = [(cell(m, a, e)["test_auprc"], cell(m, a, e)["cost"], cell(m, a, e)["time"]) for m, e in runs_]
        scores = [s_ for s_, _, _ in got if s_ is not None]
        budget[(a, k)] = (mean(scores), sum(c_ for _, c_, _ in got) / tot_cost, sum(t_ for _, _, t_ in got) / tot_time, len(scores))
        w("%s & %d & %d & %s & %d\\%% & %d\\%%\\\\" % (("Skill" if a == "skill" else "State machine") if k == 0 else "", k + 1, len(scores),
          f3(mean(scores)), round(100 * budget[(a, k)][1]), round(100 * budget[(a, k)][2])))
    w("\\midrule" if a == "skill" else "")
w("\\bottomrule\\end{tabular}}")
all_runs = [(m, a, e) for m in MODELS for a, e in ORDER]
two = [(at_budget(m, a, e, 1)[0], cell(m, a, e)["test_auprc"]) for m, a, e in all_runs]
two = [(t, f) for t, f in two if t is not None]
for name, val in {
    "TwoRuns": str(len(two)), "TwoWithin": str(sum(1 for t, f in two if abs(f - t) <= 0.005)),
    "TwoLoss": f3(mean([f for _, f in two]) - mean([t for t, _ in two])),
    "TwoCostShare": str(round(100 * sum(at_budget(m, a, e, 1)[1] for m, a, e in all_runs) / sum(cell(m, a, e)["cost"] for m, a, e in all_runs))),
    "TwoTimeShare": str(round(100 * sum(at_budget(m, a, e, 1)[2] for m, a, e in all_runs) / sum(cell(m, a, e)["time"] for m, a, e in all_runs))),
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, val))

# ── Dot plot coordinates: one row per model ──
# pgfplots expands neither macros in `coordinates` nor in tick lists: whole \\addplot lines and a style instead.
w("\\pgfplotsset{dotrows/.style={ytick={%s}, yticklabels={%s}}}" % (",".join(str(len(ALL_MODELS) - i) for i in range(len(ALL_MODELS))), ",".join("{%s}" % tex(m) for m in ALL_MODELS)))
w("\\newcommand{\\DotPlots}{%")
# Jitter overlapping points with fractional y-offsets (not yshift, which leaks into legend markers)
_yoff = {("skill","none"): -0.19, ("skill","high"): -0.06, ("sm","none"): +0.06, ("sm","high"): +0.19}
for a, e in ORDER:
    # each point with its 95% bootstrap interval as a thin horizontal bar
    def _pt(i, m):
        x = cell(m, a, e)["test_auprc"]; ci = BOOT["runs"].get("%s|%s|%s" % (m, a, e))
        err = " += (%.4f,0) -= (%.4f,0)" % (max(0.0, ci["hi"] - x), max(0.0, x - ci["lo"])) if ci else ""
        return "(%.4f,%.2f)%s" % (x, len(ALL_MODELS) - i + _yoff[(a, e)], err)
    pts = " ".join(_pt(i, m) for i, m in enumerate(ALL_MODELS) if cell(m, a, e)["test_auprc"] is not None)
    # skill: circle marks; sm: square marks — readable in greyscale
    mark = ("square*" if e == "high" else "square") if a == "sm" else ("*" if e == "high" else "o")
    w("\\addplot[%s, mark=%s, error bars/.cd, x dir=both, x explicit, error mark=none, error bar style={line width=0.55pt, opacity=0.55}] coordinates {%s};" % (a, mark, pts))
w("}")

# ── Iterations: small multiples (after k) and the table ──
w("\\newcommand{\\IterPlots}{%")
# Series are nudged sideways a little so that runs with the same score stay apart.
_dx = {("skill", "none"): -0.07, ("skill", "high"): -0.025, ("sm", "none"): 0.025, ("sm", "high"): 0.07}
_names = {("skill", "none"): "skill (\\texttt{none})", ("skill", "high"): "skill (\\texttt{high})",
          ("sm", "none"): "SM (\\texttt{none})", ("sm", "high"): "SM (\\texttt{high})"}
for i, m in enumerate(ALL_MODELS):
    opts = ["title={%s}" % tex(m)]
    if i % 3 == 0: opts.append("ylabel={test AU-PRC}")
    if i >= len(ALL_MODELS) - 3: opts.append("xlabel={iterations}")
    if i == 0: opts.append("legend to name=iterlegend, legend columns=4, legend style={font=\\footnotesize, draw=black!20, /tikz/every even column/.append style={column sep=0.6em}}")
    w("\\nextgroupplot[%s]" % ", ".join(opts))
    for a, e in ORDER:
        r = it_of(m, a, e)
        pts = " ".join("(%.3f,%.4f)" % (k + 1 + _dx[(a, e)], x["test_auprc"]) for k, x in enumerate(r["after"]) if x and not x.get("none"))
        # skill: circle marks; sm: square marks — readable in greyscale
        mark = ("square*" if e == "high" else "square") if a == "sm" else ("*" if e == "high" else "o")
        style = "%s,%s,mark=%s" % (a, "solid" if e == "high" else "densely dashed", mark)
        if pts:
            w("\\addplot[%s] coordinates {%s};" % (style, pts))
            if i == 0: w("\\addlegendentry{%s}" % _names[(a, e)])
    # the human state of the art, where the panel's range reaches it
    w("\\draw[black!55, line width=0.5pt, densely dotted] (axis cs:0.7,0.860) -- (axis cs:3.3,0.860);")
w("}")

# ── Iterations, summarised: mean test AU-PRC after 1, 2 and 3 iterations by tier, method and reasoning ──
w("\\newcommand{\\IterMeanPlots}{%")
for gi, (title, group) in enumerate((("Top-of-line models", FLAGSHIP), ("Mid-range models", SMALLER))):
    opts = ["title={%s}" % title, "xlabel={iterations}"]
    if gi == 0: opts += ["ylabel={mean test AU-PRC}", "legend to name=itermeanlegend, legend columns=4, legend style={font=\\footnotesize, draw=black!20, /tikz/every even column/.append style={column sep=0.6em}}"]
    w("\\nextgroupplot[%s]" % ", ".join(opts))
    for a_, e in ORDER:
        pts = []
        for k in range(3):
            xs = [at_budget(m, a_, e, k)[0] for m in group]
            xs = [x for x in xs if x is not None]
            if xs: pts.append("(%d,%.4f)" % (k + 1, mean(xs)))
        mark = ("square*" if e == "high" else "square") if a_ == "sm" else ("*" if e == "high" else "o")
        w("\\addplot[%s,%s,mark=%s] coordinates {%s};" % (a_, "solid" if e == "high" else "densely dashed", mark, " ".join(pts)))
        if gi == 0: w("\\addlegendentry{%s}" % _names[(a_, e)])
    w("\\draw[black!55, line width=0.5pt, densely dotted] (axis cs:0.7,0.860) -- (axis cs:3.3,0.860);")
    if gi == 1: w("\\node[font=\\scriptsize, text=black!60, anchor=south east] at (axis cs:3.28,0.860) {human state of the art};")
w("}")


def after_cell(x):
    """Test AU-PRC of the model submitted after k iterations; N/A when none had run on new inputs, -- when stopped."""
    if x is None:
        return "--"
    return "N/A" if x.get("none") else f3(x["test_auprc"])


def number(r, it):
    """Iteration folder name to its 1-based number: most runs number folders from 0, a few from 1."""
    base = min(int(p["it"]) for p in r["per"])
    return int(it) - base + 1


def picked(r):
    """The iteration the run handed in: the best on validation among all it made (its only working one, if just one)."""
    scored = [p for p in r["per"] if p.get("val_auprc") is not None]
    last = next((x for x in reversed(r["after"]) if x and not x.get("none") and x.get("pick") is not None), None)
    if last is not None and len(scored) <= len(r["after"]):
        return number(r, last["pick"])
    return number(r, max(scored, key=lambda p: p["val_auprc"])["it"]) if scored else None


def best_on_test(r):
    """The iteration whose own model scored highest on test, and that score."""
    scored = [p for p in r["per"] if p.get("test_auprc") is not None]
    if not scored:
        return None, None
    top = max(scored, key=lambda p: p["test_auprc"])
    return number(r, top["it"]), top["test_auprc"]


ITER_HEAD = ("\\textbf{Model} & \\textbf{Method} & \\textbf{Reas.} & \\textbf{After 1} & \\textbf{After 2} & \\textbf{After 3} "
             "& \\textbf{Final} & \\textbf{Picked} & \\textbf{Best on test}\\\\\\midrule")
w("\\newcommand{\\IterTable}[1]{%")
w("\\begin{longtable}{@{}l l l r r r r c r@{}}\\caption{#1}\\\\\\toprule")
w(ITER_HEAD + "\\endfirsthead")
w("\\toprule " + ITER_HEAD + "\\endhead")
same = 0
for m in MODELS:
    for a, e in ORDER:
        r = it_of(m, a, e)
        pick, (top, top_score) = picked(r), best_on_test(r)
        same += pick is not None and pick == top
        best_cell = "--" if top is None else "%d (%s)" % (top, f3(top_score))
        if top is not None and top != pick:
            best_cell = bold(best_cell)
        cells3 = ([after_cell(x) for x in r["after"]] + ["--"] * 3)[:3]
        after_cols = " & ".join(cells3)
        w("%s & %s & %s & %s & %s & %s & %s\\\\" % (tex(m) + marks(m, a, e), METHOD[a], e, after_cols,
                                                  f3(r["overall"]["test_auprc"]), "--" if pick is None else pick, best_cell))
    if m != MODELS[-1]:
        w("\\midrule")
w("\\bottomrule\\end{longtable}}")
w("\\newcommand{\\PickedBest}{%d}" % same)

# ── Time: a stacked bar per run, its iterations then the final step ──
def model_short(m):
    return (m.replace("Claude ", "")
             .replace(" V4.1 Flash", "")
             .replace(" 122B-A10B", "")
             .replace("Gemini 4 Hydrogen", "Gemini"))

runs = []
for m in ALL_MODELS:
    for a, e in ORDER:
        r = it_of(m, a, e)
        prev, segs_raw = 0.0, []
        for n, p in enumerate(r["per"]):
            if p["cum_time"] is None:
                if n != len(r["per"]) - 1 or p.get("test_auprc") is None:
                    continue
                p = dict(p, cum_time=cell(m, a, e)["time"])
            segs_raw.append(max(0.0, p["cum_time"] - prev))
            prev = max(prev, p["cum_time"])
        # More than three iterations: the extra ones are drawn with the third.
        segs = (segs_raw + [0.0] * 3)[:3] if len(segs_raw) <= 3 else segs_raw[:2] + [sum(segs_raw[2:])]
        runs.append(("%s, %s, %s" % (model_short(m), ARM_SHORT[a], e), segs, cell(m, a, e)["time"], a, e))
rows = runs[::-1]  # the first run at the top
w("\\pgfplotsset{timerows/.style={ytick={%s}, yticklabels={%s}}}" % (",".join(str(i) for i in range(len(rows))), ",".join("{%s}" % tex(l) for l, *_ in rows)))
# Left panel: each run's total training time, on a log axis.
w("\\newcommand{\\TimeTotalPlots}{%")
for i, (_, _, total, a, e) in enumerate(rows):
    w("\\addplot[forget plot, black!22, line width=0.5pt, mark=none] coordinates {(0.5,%d) (%.2f,%d)};" % (i, total, i))
for a, e in ORDER:
    mark = ("square*" if e == "high" else "square") if a == "sm" else ("*" if e == "high" else "o")
    w("\\addplot[%s, only marks, mark=%s] coordinates {%s};" % (a, mark, " ".join("(%.2f,%d)" % (t, i) for i, (_, _, t, a2, e2) in enumerate(rows) if (a2, e2) == (a, e))))
for i, (_, _, total, a, e) in enumerate(rows):
    w("\\node[font=\\tiny, anchor=west, inner sep=3.5pt] at (axis cs:%.2f,%d) {%s};" % (total, i, ("%.0f" % total) if total >= 10 else ("%.1f" % total)))
w("}")
# Cost panel: each run's cost in USD, on a log axis, same rows.
w("\\newcommand{\\CostTotalPlots}{%")
costs = [cell(m, a, e)["cost"] for m in ALL_MODELS for a, e in ORDER][::-1]
for i, c_ in enumerate(costs):
    w("\\addplot[forget plot, black!22, line width=0.5pt, mark=none] coordinates {(0.02,%d) (%.3f,%d)};" % (i, c_, i))
for a, e in ORDER:
    mark = ("square*" if e == "high" else "square") if a == "sm" else ("*" if e == "high" else "o")
    w("\\addplot[%s, only marks, mark=%s] coordinates {%s};" % (a, mark, " ".join("(%.3f,%d)" % (costs[i], i) for i, (_, _, _, a2, e2) in enumerate(rows) if (a2, e2) == (a, e))))
for i, c_ in enumerate(costs):
    w("\\node[font=\\tiny, anchor=west, inner sep=3.5pt] at (axis cs:%.3f,%d) {%s};" % (c_, i, "%.2f" % c_))
w("}")
# Right panel: how each run's training time splits across its iterations, in per cent.
w("\\newcommand{\\TimeSharePlots}{%")
for k, col in enumerate(["seg1", "seg2", "seg3"]):
    w("\\addplot[fill=%s, draw=white, line width=0.3pt] coordinates {%s};" % (col, " ".join("(%.2f,%d)" % (100 * segs[k] / sum(segs), i) for i, (_, segs, *_r) in enumerate(rows))))
w("}")


# ── Every run ──
def whole(v):
    """Minutes as a whole number, so one column has one unit."""
    return "--" if v is None else ("$<$1" if v < 0.5 else "%d" % round(v))


RUNS_HEAD = ("\\textbf{Model} & \\textbf{Method} & \\textbf{Reas.} & \\textbf{It.} & \\textbf{Test AU-PRC} & \\textbf{Test acc.} & \\textbf{Train acc.} "
             "& \\textbf{Time} & \\textbf{Model time} & \\textbf{Code time} & \\textbf{Calls} & \\textbf{Reas. tok.} & \\textbf{Cost}\\\\\\midrule")
w("\\newcommand{\\RunsTable}[1]{%")
w("\\begin{longtable}{@{}l l l r r r r r r r r r r@{}}\\caption{#1}\\\\\\toprule")
w(RUNS_HEAD + "\\endfirsthead")
w("\\toprule " + RUNS_HEAD + "\\endhead")
for m in MODELS:
    for a, e in ORDER:
        c = cell(m, a, e)
        made = sum(1 for p in it_of(m, a, e)["per"] if p.get("test_auprc") is not None)
        cost_str = money(c["cost"]) if c.get("cost") is not None else "--"
        tokens = "--" if c.get("reasoning_tokens") is None else "{:,}".format(c["reasoning_tokens"]).replace(",", "{,}")
        w("%s%s & %s & %s & %d & %s & %s & %s & %s & %s & %s & %s & %s & %s\\\\" % (
            tex(m), marks(m, a, e), {"skill": "Skill", "sm": "SM"}[a], e, made, f3(c["test_auprc"]), f3(c["test_acc"]), f3(c.get("train_acc")),
            whole(c["time"]), whole(c.get("model_time")), whole(c.get("tools_time")), c.get("calls") or "--", tokens, cost_str))
w("\\bottomrule\\end{longtable}}")


# ── Repeats: the nine replaced original runs beside the runs that replaced them ──
import math
import re
REPLACED = D.get("replaced", [])
ORIG = json.load(open(os.path.join(HERE, "data", "bootstrap-originals.json")))
ORIG_FIRST = json.load(open(os.path.join(HERE, "data", "bootstrap-first-originals.json")))
by_key = {(c["key"], c["arm"], c["effort"]): c for c in CELLS}
repeat_rows = sorted(REPLACED, key=lambda r: (MODELS.index(r["model"]), ORDER.index((r["arm"], r["effort"]))))
repeat_diffs = [by_key[(r["key"], r["arm"], r["effort"])]["test_auprc"] - r["test_auprc"] for r in repeat_rows]
w("\\newcommand{\\RepeatTable}{\\begin{tabular}{@{}l l l r r r r r@{}}\\toprule")
w("\\textbf{Model} & \\textbf{Method} & \\textbf{Reas.} & \\textbf{Original} & \\textbf{It.} & \\textbf{Repeat} & \\textbf{It.} & $\\Delta$\\\\\\midrule")
for r, d in zip(repeat_rows, repeat_diffs):
    now = by_key[(r["key"], r["arm"], r["effort"])]
    made = sum(1 for q in it_of(r["model"], r["arm"], r["effort"])["per"] if q.get("test_auprc") is not None)
    w("%s & %s & %s & %s & %d & %s & %d & $%+.3f$\\\\" % (tex(r["model"]), METHOD[r["arm"]], r["effort"], f3(r["test_auprc"]),
                                                       r["iterations"], f3(now["test_auprc"]), made, d))
w("\\bottomrule\\end{tabular}}")
abs_diffs = sorted(abs(d) for d in repeat_diffs)
o_sm = sum(1 for q in ORIG["pairs"] if q["lo"] > 0); o_skill = sum(1 for q in ORIG["pairs"] if q["hi"] < 0)
of_sm = sum(1 for q in ORIG_FIRST["pairs"] if q["lo"] > 0); of_skill = sum(1 for q in ORIG_FIRST["pairs"] if q["hi"] < 0)
high_wins = sum(1 for none, high in effort_pairs if high["test_auprc"] > none["test_auprc"])
sign_p = min(1.0, 2 * sum(math.comb(len(effort_pairs), i) for i in range(max(high_wins, len(effort_pairs) - high_wins), len(effort_pairs) + 1)) / 2 ** len(effort_pairs))
hand = [c for c in CELLS if (c["model"], c["arm"], c["effort"]) in HAND]
extra_cost = sum(e["cost"] for e in D["extra"])
cls = lambda q: "sm" if q["lo"] > 0 else "skill" if q["hi"] < 0 else "tie"
orig_cls = {(q["model"], q["effort"]): cls(q) for q in ORIG["pairs"]}
changed = [k for k, q in pairs.items() if cls(q) != orig_cls[k]]
for name, val in {
    "ChangedPairs": str(len(changed)),
    "HandSmAhead": str(sum(1 for (m, e) in sig_sm if (m, "sm", e) in HAND)),
    "RepeatMedianDiff": "%.3f" % median(abs_diffs), "RepeatMaxDiff": "%.3f" % abs_diffs[-1],
    "OrigPairsSm": str(o_sm), "OrigPairsSkill": str(o_skill), "OrigPairsTie": str(len(ORIG["pairs"]) - o_sm - o_skill),
    "OrigWilcoxonP": "%.2f" % ORIG["wilcoxon"]["p"], 
    "HandCount": str(len(hand)), "HandSm": str(sum(1 for c in hand if c["arm"] == "sm")),
    "ExtraCost": money(extra_cost), "ExtraCount": str(len(D["extra"])),
}.items():
    w("\\newcommand{\\%s}{%s}" % (name, val))

# ── Attempts and replaced runs that are not among the 36 ──
KIND = {"pilot": "Pilot of the second implementation", "attempt": "Unfinished attempt", "attempt, stopped at setup": "Attempt stopped at the skill's setup",
        "replaced by rerun": "Run replaced by a later run", "skillv1": "Run with an earlier version of the skill"}
name_of = {c["key"]: c["model"] for c in CELLS}
w("\\newcommand{\\ExtraTable}{\\begin{tabular}{@{}l l l l r r@{}}\\toprule")
w("\\textbf{Model} & \\textbf{Method} & \\textbf{Reas.} & \\textbf{What} & \\textbf{Calls} & \\textbf{Cost}\\\\\\midrule")
extra_rows = []
for e in D["extra"]:
    m = re.match(r"^(?:\[(.+?)\] )?(skillv1|skill|sm)/([^/]+)/(\w+)", e["label"])
    kind, arm, key, effort = m.group(1), m.group(2), m.group(3), m.group(4)
    if arm == "skillv1":
        kind, arm = "skillv1", "skill"
    extra_rows.append((MODELS.index(name_of[key]), ORDER.index((arm, effort)), name_of[key], arm, effort, KIND[kind], e["calls"], e["cost"]))
for _, _, model, arm, effort, kind, calls, cost in sorted(extra_rows, key=lambda r: r[:2]):
    w("%s & %s & %s & %s & %d & %s\\\\" % (tex(model), METHOD[arm], effort, kind, calls, money(cost)))
w("\\midrule\\textbf{Total} & & & & & %s\\\\" % money(extra_cost))
w("\\bottomrule\\end{tabular}}")

open(os.path.join(HERE, "generated.tex"), "w").write("\n".join(l for l in out if l) + "\n")
print("generated.tex", len(out), "lines")
