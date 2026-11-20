#!/usr/bin/env python3
"""AGO2 CLASH agent-matrix harness with Rich live display.

Usage:
    python harness/run.py --model openai/gpt-4o-mini --condition both
    python harness/run.py --model paper --reasoning both
    python harness/run.py --model deepseek/deepseek-v4.1-flash anthropic/claude-haiku-4.5 --reasoning high --condition skill

Runs go one at a time, cheapest model first. Each run keeps its own folder, runs/<model>/<method>-<reasoning>/; a run
with a result.json is done and is not run again, and an Agentomics-ML run that was stopped continues from its last
finished step (--no-resume starts it again instead). Every attempt is kept.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path as _Path

# load .env from repo root if present
_env = _Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# the nine models of the paper, by OpenRouter id
PAPER_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.5-122b-a10b",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-5.5",
    "google/gemini-4-hydrogen",
    "anthropic/claude-opus-5.5",
    "openai/gpt-6.2-astra",
    "anthropic/claude-fable-5.2",
]
EFFORTS = {"none": ["none"], "high": ["high"], "both": ["none", "high"]}


def fetch_prices() -> dict[str, float]:
    """OpenRouter's listed price per model (USD per million tokens, input + output); empty if it can't be read."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=10) as r:
            models = json.load(r)["data"]
        return {m["id"]: (float(m["pricing"]["prompt"]) + float(m["pricing"]["completion"])) * 1e6 for m in models}
    except Exception:
        return {}


class PlanStep:
    def __init__(self, n: int, model_id: str, effort: str, price: float | None, states: list["RunState"],
                 cells: list["RunState"]):
        self.cells = cells       # every method's run for the plan matrix, whether or not this run includes it
        self.n = n
        self.model_id = model_id
        self.effort = effort
        self.price = price
        self.states = states

    @property
    def status(self) -> str:
        statuses = {s.status for s in self.states}
        if statuses == {"skipped"}:
            return "skipped"
        if "running" in statuses:
            return "running"
        if statuses == {"waiting"}:
            return "waiting"
        return "error" if "error" in statuses else "done"


# ── state shared between threads and the display ─────────────────────────────

class RunState:
    def __init__(self, label: str, model_id: str, effort: str, short: str):
        self.label = label
        self.model_id = model_id
        self.effort = effort
        self.short = short       # column name in the plan box
        self.algo = ""           # last detected ML algorithm name
        self.status = "waiting"
        self.step = ""
        self.iteration = 0
        self.log: deque[str] = deque(maxlen=60)
        self.scores: list[float | None] = []
        self.iter_costs: list[float] = []
        self.final_score: float | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.cost: float | None = None
        self.minutes: float | None = None    # training time of a finished run, from its result
        self.note = ""                       # what waiting means for this run: new, resumable, ...
        self.error: str | None = None

    def elapsed(self) -> str:
        if self.minutes is not None and self.start_time is None:
            t = self.minutes * 60
        elif self.start_time is None:
            return "--:--"
        else:
            t = (self.end_time or time.time()) - self.start_time
        return f"{int(t//60):02d}:{int(t%60):02d}"

    def add_log(self, text: str) -> None:
        for line in text.strip().splitlines():
            if line.strip():
                self.log.append(line.strip()[:400])


# ── rich rendering ────────────────────────────────────────────────────────────

STATUS_STYLE = {
    "waiting":  "dim",
    "running":  "bold cyan",
    "done":     "bold green",
    "error":    "bold red",
    "skipped":  "dim",
}

# how each method is badged everywhere in the display: name, badge style, one-line meaning
METHOD = {
    "Skill": ("SKILL", "bold black on green", "no enforced steps: the agent follows the ml-research skill on its own"),
    "State machine": ("STATE MACHINE", "bold white on blue", "Agentomics-ML enforces every step and checks each one"),
}

STEP_ICON = {
    "waiting":  "○",
    "running":  "⠿",
    "done":     "✓",
    "error":    "✗",
    "skipped":  "·",
}


def _fmt_score(v):
    return f"[green]{v:.4f}[/green]" if v is not None else "[dim]—[/dim]"


def render_run_panel(state: RunState, log_lines: int = 12) -> Panel:
    icon = STEP_ICON.get(state.status, "·")
    style = STATUS_STYLE.get(state.status, "")
    border_style = {"waiting": "dim", "running": "cyan", "done": "green", "error": "red"}.get(state.status, "white")

    t = Text(no_wrap=True, overflow="ellipsis")

    name, badge_style, meaning = METHOD[state.short]
    t.append(f" {name} ", style=badge_style)
    t.append(f"  {meaning}\n", style="dim")

    # model + algo subtitle
    if state.model_id:
        t.append(state.model_id, style="bold white")
        t.append("  ·  reasoning ", style="dim")
        t.append(state.effort, style="bold magenta" if state.effort == "high" else "bold white")
        if state.algo:
            t.append(f"  ·  {state.algo}", style="yellow")
        t.append("\n")

    # status line
    t.append(f"{icon} {state.status.upper()}", style=style)
    if state.step:
        t.append(f"  ·  {state.step}", style="dim")
    if state.iteration:
        t.append(f"  ·  iter {state.iteration}/3", style="cyan")
    t.append(f"  ·  {state.elapsed()}\n", style="dim")

    # scores grid
    iter_scores = (state.scores + [None, None, None])[:3]
    score_parts = "   ".join(_fmt_score(s) for s in iter_scores)
    t.append_text(Text.from_markup(f"Iter  {score_parts}\n"))

    if state.final_score is not None:
        t.append_text(Text.from_markup(f"[bold]AU-PRC  [green]{state.final_score:.4f}[/green][/bold]\n"))

    # per-iteration cost row
    if state.iter_costs or state.cost:
        ic = (state.iter_costs + [None, None, None])[:3]
        cost_parts = "   ".join(f"[yellow]${c:.4f}[/yellow]" if c is not None else "[dim]—[/dim]" for c in ic)
        total_str = f"[yellow]${state.cost:.4f}[/yellow]" if state.cost else "[dim]—[/dim]"
        t.append_text(Text.from_markup(f"Cost  {cost_parts}   total {total_str}\n"))

    if state.error:
        t.append_text(Text.from_markup(f"[red]{state.error[:160]}[/red]\n"))

    t.append("\n")

    # log — filter noise, keep meaningful lines
    skip_tokens = ("[exit", "[stderr]", "Traceback", "File \"/", "  File", "subprocess")
    entries = [e for e in state.log if not any(tok in e for tok in skip_tokens)]
    for entry in entries[-log_lines:]:
        t.append(f"  {entry}\n", style="dim")

    return Panel(t, title=f"[bold]{state.label}[/bold]", border_style=border_style, expand=True)


def _mean_std(vals: list) -> str:
    nums = [v for v in vals if v is not None]
    if not nums:
        return "[dim]—[/dim]"
    mean = np.mean(nums)
    if len(nums) > 1:
        std = np.std(nums, ddof=1)
        return f"[cyan]{mean:.3f}[/cyan][dim]±{std:.3f}[/dim]"
    return f"[cyan]{mean:.4f}[/cyan]"


def render_results_table(states: list[RunState],
                         paper_rows: list[dict] | None = None) -> Table:
    t = Table(box=None, expand=True, show_header=True, header_style="dim", padding=(0, 1))
    t.add_column("Model", style="bold", no_wrap=True, overflow="ellipsis",
                 max_width=13 if console.size.width < 170 else None)
    t.add_column("Method", no_wrap=True, min_width=15)
    t.add_column("Reason", no_wrap=True, min_width=6)
    t.add_column("Iter 1", justify="right", no_wrap=True, min_width=6)
    t.add_column("Iter 2", justify="right", no_wrap=True, min_width=6)
    t.add_column("Iter 3", justify="right", no_wrap=True, min_width=6)
    t.add_column("Test", justify="right", style="bold", no_wrap=True, min_width=6)
    t.add_column("Cost", justify="right", no_wrap=True, min_width=7)
    t.add_column("Time", justify="right", no_wrap=True, min_width=5)

    for s in states:
        def fmt(v):
            return f"[green]{v:.4f}[/green]" if v is not None else "[dim]—[/dim]"

        sc = s.scores + [None, None, None]
        final = f"[bold green]{s.final_score:.4f}[/bold green]" if s.final_score else "[dim]—[/dim]"
        total_cost = s.cost or sum(s.iter_costs)
        cost = f"${total_cost:.4f}" if total_cost else "[dim]—[/dim]"
        name, badge_style, _ = METHOD[s.short]
        reasoning = Text("on", style="bold magenta") if s.effort == "high" else Text("off")
        row_style = "dim" if s.status == "waiting" else ""
        t.add_row(Text(s.model_id.split("/")[-1], style=row_style or "bold"), Text(f" {name} ", style=badge_style),
                  reasoning, fmt(sc[0]), fmt(sc[1]), fmt(sc[2]),
                  final, cost, s.elapsed(), style=row_style)

    return t


PLAN_SHORT = {"State machine": "SM", "Skill": "Skill"}
PLAN_WIDTH = 62


def _plan_cell(state: RunState) -> Text:
    if state.final_score is not None:
        return Text(f"{state.final_score:.3f}", style="bold green")
    if state.status == "error":
        return Text("failed", style="red")
    if state.status == "running":
        return Text(f"▶ it {state.iteration}" if state.iteration else "▶ start", style="bold black on cyan")
    if state.status == "done":
        return Text("none", style="yellow")
    if state.status == "skipped":
        return Text("·", style="dim")
    if "continue" in state.note:
        return Text("resume", style="yellow")
    return Text("queued", style="white")


def render_plan(plan: list[PlanStep], priced: bool) -> Panel:
    """One row per model; one column per method and reasoning setting."""
    efforts = {"none": "off", "high": "on"}
    shorts = [s.short for s in plan[0].cells]
    t = Table(box=None, expand=True, show_header=True, pad_edge=False, padding=(0, 1))
    t.add_column("Model", no_wrap=True, min_width=19)
    for short in shorts:
        colour = METHOD[short][1].split(" on ")[-1]
        for label in efforts.values():
            t.add_column(f"[bold {colour}]{PLAN_SHORT[short]}[/] {label}", justify="center", no_wrap=True, width=len(PLAN_SHORT[short]) + 4)

    steps = {(p.model_id, p.effort): p for p in plan}
    for model_id in dict.fromkeys(p.model_id for p in plan):
        row = [steps[(model_id, effort)] for effort in efforts]
        in_run = any(p.n for p in row)
        cells = [_plan_cell(step.cells[i]) for i in range(len(shorts)) for step in row]
        t.add_row(Text(model_id.split("/")[-1], style="bold" if in_run else "dim"), *cells)

    runs = [s for p in plan if p.n for s in p.states]
    done = sum(1 for s in runs if s.status in ("done", "error"))
    order = "cheapest first" if priced else "order as given"
    return Panel(t, title=f"[bold]Plan[/bold]  ·  {done}/{len(runs)} runs done",
                 subtitle=f"[dim]{order}  ·  off/on = reasoning[/dim]",
                 border_style="white")


def render_layout(header: str, states: list[RunState],
                  paper_rows: list[dict] | None = None,
                  plan: list[PlanStep] | None = None, priced: bool = True,
                  runs: list[RunState] | None = None) -> Layout:
    """Header; the plan matrix beside the results table; then one full-width row per running method."""
    models = len(dict.fromkeys(p.model_id for p in plan))
    # the results table lists every run of this plan; a long plan shows the runs around the current one
    runs = runs or [s for p in plan if p.n for s in p.states]
    top = max(models, min(len(runs), 10)) + 3
    room = top - 3
    first = max(0, min(runs.index(states[0]) - 1, len(runs) - room))
    shown = runs[first:first + room]

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="top", size=top),
        *[Layout(name=f"run{i}") for i in range(len(states))],
    )
    layout["header"].update(Panel(
        Text(header, justify="center", style="bold white"),
        style="on grey7",
        box=box.DOUBLE_EDGE,
    ))
    layout["top"].split_row(Layout(name="plan", size=PLAN_WIDTH), Layout(name="results"))
    layout["top"]["plan"].update(render_plan(plan, priced))
    layout["top"]["results"].update(Panel(
        render_results_table(shown, paper_rows=paper_rows),
        title=f"Results  ·  runs {first + 1}–{first + len(shown)} of {len(runs)}", border_style="white",
    ))
    log_lines = max(3, (console.size.height - 4 - top) // len(states) - 7)
    for i, s in enumerate(states):
        layout[f"run{i}"].update(render_run_panel(s, log_lines))
    return layout


# ── running one run (in a thread) ─────────────────────────────────────────────

def _live_scores(arm: str, workspace: Path) -> list[float | None]:
    """Validation AU-PRC per iteration so far, as the run itself recorded it."""
    from harness import results
    if arm == "skill":
        from harness.skill_agent import _read_iter_scores
        return _read_iter_scores(workspace)
    return [i["val_recorded"] and round(i["val_recorded"], 4) for i in results.sm_iterations(workspace)]


def load_result(state: RunState, result: dict) -> None:
    state.status = "done"
    state.final_score = result.get("test_auprc")
    state.scores = list(result.get("val") or [])
    state.cost = result.get("cost")
    state.minutes = result.get("time")
    state.note = "done" + (" (continued after an interruption)" if result.get("recovered") else "")


def run_thread(state: RunState, rdir: Path, arm: str, api_key: str, resume: bool) -> None:
    import re
    from harness import results, runs
    from harness.usage import read_usage

    state.status = "running"
    state.start_time = time.time()
    step_line = re.compile(r"ITERATION (\d+) \| (.+?) STEP")
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(3):
            tries = results.attempts(rdir)
            if not tries:
                continue
            attempt = tries[-1]
            try:
                state.scores = _live_scores(arm, attempt / "workspace")
                state.cost = sum(c["cost"] for c in read_usage(results.usage_path(attempt)))
            except Exception:
                pass

    def on_event(kind: str, text: str) -> None:
        m = step_line.search(text)
        if m:
            state.iteration = int(m.group(1)) + 1
            state.step = m.group(2).lower()
        elif kind:
            state.step = kind
        state.add_log(text)

    def on_step(text: str) -> None:
        state.step = text
        state.add_log(f"── {text}")

    threading.Thread(target=poll, daemon=True).start()
    try:
        result = runs.execute(rdir, arm, state.model_id, state.effort, api_key, resume, on_event, on_step)
        load_result(state, result)
        state.add_log(f"Result written: test AU-PRC {result.get('test_auprc')}, ${result.get('cost')}")
    except Exception as e:
        state.error = str(e)
        state.status = "error"
        state.add_log(f"ERROR: {e}")
    finally:
        stop.set()
        state.end_time = time.time()


# ── main ─────────────────────────────────────────────────────────────────────

def _stop_containers() -> None:
    """Stops the containers this harness started: Agentomics-ML's (workspace folder "workspace") and the skill's."""
    import shutil
    import subprocess
    docker = shutil.which("docker") or "/usr/local/bin/docker"
    ids = [i for label in ("agentomics.workspace=workspace", "ago2-harness=skill")
           for i in subprocess.run([docker, "ps", "-q", "--filter", f"label={label}"],
                                   capture_output=True, text=True).stdout.split()]
    if ids:
        subprocess.run([docker, "stop", *ids], capture_output=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="AGO2 CLASH agent-matrix harness")
    parser.add_argument("--model", nargs="+", default=["openai/gpt-4o-mini"],
                        help="OpenRouter model ids, or 'paper' for the paper's nine (default: openai/gpt-4o-mini)")
    parser.add_argument("--reasoning", choices=list(EFFORTS), default="none",
                        help="Reasoning effort per model: none, high, or both (default: none)")
    parser.add_argument("--condition", choices=["skill", "sm", "both"], default="both",
                        help="Which condition to run (default: both)")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="Directory for runs (default: ./runs)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start stopped runs again instead of continuing them")
    parser.add_argument("--force", action="store_true",
                        help="Delete result.json for matching runs so they run again")
    args = parser.parse_args()

    if not args.api_key:
        console.print("[red]Set OPENROUTER_API_KEY or pass --api-key[/red]")
        sys.exit(1)

    from harness import runs

    models = list(dict.fromkeys(m for given in args.model for m in (PAPER_MODELS if given == "paper" else [given])))
    prices = fetch_prices()
    priced = all(m in prices for m in models)
    if priced:
        models.sort(key=lambda m: prices[m])

    names = {"skill": ("Skill  (ml-research)", "Skill"), "sm": ("State Machine  (Agentomics-ML)", "State machine")}
    chosen = ["skill", "sm"] if args.condition == "both" else [args.condition]

    if args.force:
        for m in models:
            for c in chosen:
                for effort in EFFORTS[args.reasoning]:
                    r = runs.run_dir(args.runs_dir, m, c, effort) / "result.json"
                    if r.exists():
                        # kept beside the run, so a rerun that fails loses nothing; export reads result.json only
                        r.rename(r.with_name(f"result.before-{time.strftime('%Y%m%d-%H%M%S')}.json"))
                        console.print(f"[yellow]set aside result.json: {r.parent.parent.name}/{r.parent.name}[/yellow]")

    # the plan box lists the paper's whole matrix; runs outside this plan show what they hold on disk, if anything
    listed = list(dict.fromkeys(models + PAPER_MODELS))
    if all(m in prices for m in listed):
        listed.sort(key=lambda m: prices[m])
    else:
        listed = models + [m for m in PAPER_MODELS if m not in models]
    board: list[PlanStep] = []
    plan: list[PlanStep] = []
    for model_id in listed:
        for effort in EFFORTS["both"]:
            in_run = model_id in models and effort in EFFORTS[args.reasoning]
            cells = {c: RunState(names[c][0], model_id, effort, names[c][1]) for c in ("sm", "skill")}
            for c, cell in cells.items():
                found, note = runs.state_of(runs.run_dir(args.runs_dir, model_id, c, effort), c)
                cell.note = note
                if found == "complete":
                    load_result(cell, runs.read_json(runs.run_dir(args.runs_dir, model_id, c, effort) / "result.json"))
                elif not (in_run and c in chosen):
                    cell.status = "skipped"
                elif found == "resumable" and args.no_resume:
                    cell.note = note.split(";")[0] + "; will start again"
            step = PlanStep(len(plan) + 1 if in_run else 0, model_id, effort, prices.get(model_id),
                            [cells[c] for c in chosen], list(cells.values()))
            if in_run:
                plan.append(step)
            board.append(step)

    # one run at a time: per model, the skill off then on, then the state machine off then on; finished runs are skipped
    queue = [(step, c, state)
             for model_id in dict.fromkeys(p.model_id for p in plan)
             for c in chosen
             for step in plan if step.model_id == model_id
             for cc, state in zip(chosen, step.states) if cc == c]
    order = [state for _, _, state in queue]
    todo = [(step, c, state) for step, c, state in queue if state.status != "done"]

    console.print(f"[bold]{len(queue)} runs in this plan[/bold]: {len(queue) - len(todo)} already done, {len(todo)} to run")
    for state in order:
        console.print(f"  {state.model_id.split('/')[-1]:<22} {METHOD[state.short][0]:<14} reasoning {state.effort:<5} {state.note}")
    if not todo:
        return

    def header_for(n: int, state: RunState) -> str:
        return (f"AGO2 CLASH Agent Matrix  ·  run {n}/{len(todo)}  ·  method: {METHOD[state.short][0]}\n"
                f"{state.model_id}  ·  reasoning {state.effort}  ·  metric: AU-PRC")

    def draw(n: int, state: RunState) -> Layout:
        return render_layout(header_for(n, state), [state], plan=board, priced=priced, runs=order)

    try:
        with Live(draw(1, todo[0][2]), refresh_per_second=4, console=console) as live:
            for n, (step, c, state) in enumerate(todo, 1):
                rdir = runs.run_dir(args.runs_dir, step.model_id, c, step.effort)
                thread = threading.Thread(target=run_thread, args=(state, rdir, c, args.api_key, not args.no_resume),
                                          daemon=True)
                thread.start()
                while thread.is_alive():
                    live.update(draw(n, state))
                    time.sleep(0.25)
                live.update(draw(n, state))
    except KeyboardInterrupt:
        _stop_containers()
        console.print("[yellow]Stopped. Finished runs are kept; run the same command again to carry on.[/yellow]")
        return

    # print final summary
    console.print()
    console.rule("[bold white]Final Results")
    console.print(render_plan(board, priced))
    console.print(render_results_table(order))
    console.print(f"\nRuns are in: {args.runs_dir}")


if __name__ == "__main__":
    # allow `python harness/run.py` from the repo root
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
