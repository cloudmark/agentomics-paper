"""One run per model, method and reasoning setting, kept in runs/<model-key>/<method>-<reasoning>/.

Each try is an attempt folder (attempt-1, attempt-2, ...) holding workspace/, events.json, log/ (the per-call usage
log, and Agentomics-ML's output) and, once the attempt has finished, scores/. A run is complete when result.json exists; it is written only after scoring.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Callable

from harness import results
from harness.results import read_json, write_json
from harness.usage import UsageLog

ITERATIONS = 3

# the paper's nine models: OpenRouter id -> (name in the paper, key in its data)
PAPER = {
    "anthropic/claude-fable-5.2": ("Claude Fable 5.2", "fable-5-2"),
    "anthropic/claude-opus-5.5": ("Claude Opus 5.5", "opus-5-5"),
    "openai/gpt-6.2-astra": ("GPT-6.2 Astra", "gpt-6-2-astra"),
    "google/gemini-4-hydrogen": ("Gemini 4 Hydrogen", "gemini-4-hydrogen"),
    "anthropic/claude-sonnet-5.5": ("Claude Sonnet 5.5", "sonnet-5-5"),
    "qwen/qwen3.5-122b-a10b": ("Qwen 3.5 122B-A10B", "qwen3-5-122b"),
    "anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", "haiku-4-5"),
    "deepseek/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "deepseek-v4-1-flash"),
    "openai/gpt-4.1": ("GPT-4.1", "gpt-4-1"),
}


def meta(model_id: str, arm: str, effort: str) -> dict:
    name, key = PAPER.get(model_id, (model_id, re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower())))
    return {"model": name, "key": key, "arm": arm, "effort": effort, "model_id": model_id}


def run_dir(root: Path, model_id: str, arm: str, effort: str) -> Path:
    return root / meta(model_id, arm, effort)["key"] / f"{arm}-{effort}"


def sm_latest_checkpoint(workspace: Path) -> dict | None:
    """The newest step or iteration-end checkpoint Agentomics-ML committed: iteration, step and commit time."""
    if not (workspace / ".git").exists():
        return None
    log = subprocess.run(["git", "log", "--format=%ct %s", "--all"], cwd=workspace, capture_output=True, text=True).stdout
    for line in log.splitlines():
        when, _, message = line.partition(" ")
        m = re.match(r"^agentomics/[^/]+/(\d{4})/(\S+)$", message.strip())
        if m:
            return {"iteration": int(m.group(1)), "step": m.group(2), "time": float(when)}
    return None


def sm_finished_iterations(workspace: Path) -> list[int]:
    """Iterations of an Agentomics-ML workspace that reached their end checkpoint."""
    if not (workspace / ".git").exists():
        return []
    log = subprocess.run(["git", "log", "--format=%s", "--all"], cwd=workspace, capture_output=True, text=True).stdout
    return sorted({int(m.group(1)) for m in re.finditer(r"^agentomics/[^/]+/(\d{4})/end$", log, re.M)})


def state_of(rdir: Path, arm: str) -> tuple[str, str]:
    """('complete' | 'resumable' | 'incomplete' | 'new', a short description)."""
    if (rdir / "result.json").exists():
        result = read_json(rdir / "result.json") or {}
        return "complete", f"done: test AU-PRC {result.get('test_auprc')}"
    tries = results.attempts(rdir)
    if not tries:
        return "new", "not run yet"
    last = tries[-1]
    if (read_json(last / "events.json") or {}).get("status") == "done":
        return "resumable", f"{last.name} finished its iterations; will continue with its scoring"
    if arm == "sm":
        checkpoint = sm_latest_checkpoint(last / "workspace")
        if checkpoint:
            return "resumable", (f"{last.name} stopped after the {checkpoint['step']} step of iteration "
                                 f"{checkpoint['iteration']}; will continue from there")
    return "incomplete", f"{last.name} did not finish; will start again"


def new_attempt(rdir: Path) -> Path:
    tries = results.attempts(rdir)
    attempt = rdir / f"attempt-{int(tries[-1].name.split('-')[1]) + 1 if tries else 1}"
    attempt.mkdir(parents=True)
    return attempt


def _finish(rdir: Path, attempt: Path, events: dict, arm: str, model_id: str, effort: str,
            on_step: Callable[[str], None]) -> dict:
    from harness.scoring import score_skill_iterations, score_sm_iterations

    on_step("scoring every iteration on test")
    (score_skill_iterations if arm == "skill" else score_sm_iterations)(attempt / "workspace", attempt / "scores")
    on_step("writing result")
    result = results.build(rdir, meta(model_id, arm, effort))
    write_json(rdir / "result.json", result)
    return result


def execute(rdir: Path, arm: str, model_id: str, effort: str, api_key: str, resume: bool,
            on_event: Callable[[str, str], None], on_step: Callable[[str], None]) -> dict:
    """Runs (or continues) one run to its result.json and returns the result."""
    from harness.agentomics_runner import run_agentomics
    from harness.dataset_prep import prep_skill_workspace
    from harness.skill_agent import run_skill

    rdir.mkdir(parents=True, exist_ok=True)
    tries = results.attempts(rdir)
    previous = tries[-1] if tries else None

    # an attempt that finished its iterations and test step needs only its scoring
    if previous and (read_json(previous / "events.json") or {}).get("status") == "done":
        return _finish(rdir, previous, read_json(previous / "events.json"), arm, model_id, effort, on_step)

    # a stopped Agentomics-ML attempt continues from its newest checkpoint: a step's or an iteration's end
    fork = sm_latest_checkpoint(previous / "workspace") if resume and arm == "sm" and previous else None

    attempt = new_attempt(rdir)
    workspace = attempt / "workspace"
    usage = UsageLog(results.usage_path(attempt))
    events: dict = {"attempt": attempt.name, "status": "running", "start": time.time()}
    if fork:
        events.update(fork_from=previous.name, fork_iteration=fork["iteration"], fork_step=fork["step"],
                      fork_point=fork["time"])
    write_json(attempt / "events.json", events)

    try:
        if arm == "skill":
            prep_skill_workspace(workspace)
            outcome = run_skill(model_id, workspace, api_key, effort, usage, on_event=on_event)
        else:
            outcome = run_agentomics(model_id, workspace, api_key, effort, results.usage_path(attempt).parent,
                                     iterations=ITERATIONS,
                                     fork_from=previous / "workspace" if fork else None,
                                     on_event=lambda line: on_event("", line))
        events.update(outcome["events"], status="done")
        write_json(attempt / "events.json", events)
    except BaseException as error:
        events.update(status="failed", error=str(error) or type(error).__name__, end=time.time())
        write_json(attempt / "events.json", events)
        raise
    return _finish(rdir, attempt, events, arm, model_id, effort, on_step)

