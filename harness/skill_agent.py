"""A general-purpose LangGraph agent with the ml-research skill available, as in the paper's skill condition.

Its system prompt names its tools and the skills it can load, and nothing about machine learning; the skill reaches it
only when it loads it. Its commands run in the same container image as Agentomics-ML.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from harness import protocol
from harness.dataset_prep import UPLOAD_NAME, deliver_test_inputs
from harness.sandbox import Sandbox
from harness.usage import UsageCallback, UsageLog

# LangGraph counts a model call and the tool calls it asks for as two steps
RECURSION_LIMIT = 2 * protocol.SKILL_ROUNDS + 1
COMMAND_TIMEOUT = 600   # SKILL.md: "A command has 10 minutes"
SKILL_DIR = Path(__file__).resolve().parent.parent / "skill"


def _description(skill_md: Path) -> str:
    match = re.search(r"^description:\s*(.+)$", skill_md.read_text(), re.M)
    return match.group(1).strip() if match else ""


SYSTEM_PROMPT = f"""You are a general-purpose agent working in a workspace, a folder you share with the person.

Your tools:
- execute: run a shell command in the workspace
- read_file, write_file, edit_file: read and change files in the workspace
- ls, glob: list and find files in the workspace
- load_skill: load a skill's instructions

Skills you can load:
- ml-research: {_description(SKILL_DIR / "SKILL.md")}

When a skill covers the person's task, load it with load_skill and follow it."""

TASK_MESSAGE = (f"The dataset is in uploads/{UPLOAD_NAME}. Use the ml-research skill to train a model that tells whether "
                "a microRNA binds a target site (label 1) or not (label 0). Run 3 iterations, judged on AUPRC.")


def build_tools(workspace: Path, sandbox: Sandbox) -> list:
    """Shell and file tools bound to a workspace; the shell runs in the sandbox container."""
    root = workspace.resolve()

    def inside(path: str) -> Path | None:
        p = (root / path).resolve()
        return p if p == root or root in p.parents else None

    @tool
    def execute(command: str) -> str:
        """Run a shell command in the workspace. Returns its combined output."""
        try:
            result = sandbox.run(command, COMMAND_TIMEOUT)
        except Exception as error:
            return f"[command did not finish: {error}]"
        out = result.stdout + (("\n[stderr]\n" + result.stderr) if result.stderr else "")
        if result.returncode != 0:
            out += f"\n[exit {result.returncode}]"
        return out or "(no output)"

    @tool
    def read_file(path: str) -> str:
        """Read a file from the workspace."""
        p = inside(path)
        if p is None or not p.is_file():
            return f"[file not found: {path}]"
        return p.read_text(errors="replace")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file in the workspace (creates directories as needed)."""
        p = inside(path)
        if p is None:
            return f"[outside the workspace: {path}]"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"written {len(content)} bytes to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of `old` with `new` in a workspace file."""
        p = inside(path)
        if p is None or not p.is_file():
            return f"[file not found: {path}]"
        text = p.read_text(errors="replace")
        if old not in text:
            return f"[string not found in {path}]"
        p.write_text(text.replace(old, new, 1))
        return f"edited {path}"

    @tool
    def ls(path: str = ".") -> str:
        """List files in a workspace directory."""
        p = inside(path)
        if p is None or not p.is_dir():
            return f"[directory not found: {path}]"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        return "\n".join(("  " if e.is_file() else "d ") + e.name for e in entries) or "(empty)"

    @tool
    def glob(pattern: str) -> str:
        """Find files in the workspace. Returns workspace-relative paths, one per line."""
        found = sorted(m for m in root.glob(pattern) if inside(str(m.relative_to(root))))
        return "\n".join(str(m.relative_to(root)) for m in found) or "(no matches)"

    @tool
    def load_skill(name: str) -> str:
        """Load a skill's instructions by name."""
        if name != "ml-research":
            return f"[no skill named {name}]"
        return (f"Skill ml-research. Its files are in .skills/ml-research/ (read them with read_file).\n\n"
                + (SKILL_DIR / "SKILL.md").read_text())

    return [execute, read_file, write_file, edit_file, ls, glob, load_skill]


def build_agent(model_id: str, workspace: Path, sandbox: Sandbox, api_key: str, reasoning_effort: str):
    llm = ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=protocol.TEMPERATURE,
        reasoning_effort=reasoning_effort,
    )
    return create_react_agent(llm, build_tools(workspace, sandbox), prompt=SYSTEM_PROMPT)


def _read_iter_scores(workspace: Path) -> list[float | None]:
    """Per-iteration validation AU-PRC from results/metrics.json, written by check_iteration.py."""
    metrics_path = workspace / "results" / "metrics.json"
    if not metrics_path.exists():
        return []
    try:
        data = json.loads(metrics_path.read_text())
        iterations = data.get("iterations", [])
        out = []
        for it in sorted(iterations, key=lambda x: x.get("iteration", 0)):
            v = (it.get("validation") or {}).get("AUPRC")
            out.append(round(float(v), 4) if v is not None else None)
        return out
    except Exception:
        return []


TEST_MESSAGE = ('Here is the test set: its inputs are in uploads/input (a data.csv, no labels). '
                'Follow the "Here is the test set" section of the ml-research skill.')


def _current_iteration(workspace: Path) -> int:
    folders = [int(p.name) for p in (workspace / "iterations").glob("*") if p.is_dir() and p.name.isdigit()]
    return max(folders, default=0)


def run_skill(
    model_id: str,
    workspace: Path,
    api_key: str,
    reasoning_effort: str,
    usage_log: UsageLog,
    on_event: Callable[[str, str], None] | None = None,
) -> dict:
    """Run the skill agent: one turn for the iterations, then the test inputs in the same conversation.

    Iteration k ends when the agent first writes into iteration k+1's folder; the last ends when the first turn does.
    Returns the run's timing events and the test predictions' path.
    """
    events: dict = {"start": time.time(), "iteration_ends": {}}
    stop = threading.Event()

    def watch_iterations() -> None:
        while not stop.wait(1):
            for k in range(_current_iteration(workspace)):
                events["iteration_ends"].setdefault(str(k), time.time())

    with Sandbox(workspace) as sandbox:
        agent = build_agent(model_id, workspace, sandbox, api_key, reasoning_effort)
        callback = UsageCallback(usage_log, lambda: _current_iteration(workspace))
        config = {"callbacks": [callback], "recursion_limit": RECURSION_LIMIT}

        def run_turn(messages: list) -> list:
            final = messages
            for state in agent.stream({"messages": messages}, config=config, stream_mode="values"):
                final = state["messages"]
                last = final[-1]
                content = getattr(last, "content", "")
                if on_event and isinstance(content, str) and content.strip():
                    on_event(type(last).__name__.replace("Message", "").lower(), content[:2000])
            return final

        threading.Thread(target=watch_iterations, daemon=True).start()
        try:
            history = run_turn([HumanMessage(content=TASK_MESSAGE)])
            stop.set()
            events["last_iteration_end"] = time.time()
            for k in range(_current_iteration(workspace)):
                events["iteration_ends"].setdefault(str(k), events["last_iteration_end"])

            deliver_test_inputs(workspace)
            if on_event:
                on_event("person", TEST_MESSAGE)
            run_turn(history + [HumanMessage(content=TEST_MESSAGE)])
        finally:
            stop.set()
            events["end"] = time.time()

    pred = workspace / "final" / "test_predictions.csv"
    return {
        "scores": _read_iter_scores(workspace),
        "events": events,
        "predictions_path": pred if pred.exists() else None,
    }
