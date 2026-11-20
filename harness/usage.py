"""Per-call usage log: one JSON line per model call, written as the call returns, so an interrupted attempt keeps its cost."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler


def usage_fields(usage: dict) -> dict:
    """Cost and token counts from an OpenRouter usage block."""
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    return {
        "cost": float(usage.get("cost") or 0.0),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(prompt.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion.get("reasoning_tokens") or 0),
    }


class UsageLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, start: float, end: float, iteration: int, model: str, usage: dict) -> None:
        line = {"start": start, "end": end, "iteration": iteration, "model": model, **usage_fields(usage)}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")


def read_usage(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class UsageCallback(BaseCallbackHandler):
    """Logs every chat-model call of a LangChain agent, stamped with the iteration it starts in."""

    def __init__(self, log: UsageLog, iteration: Callable[[], int]):
        self.log = log
        self.iteration = iteration
        self._started: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = (time.time(), self.iteration())

    def on_llm_end(self, response, *, run_id, **kwargs):
        start, iteration = self._started.pop(run_id, (time.time(), self.iteration()))
        output = response.llm_output or {}
        self.log.record(start, time.time(), iteration, output.get("model_name", ""), output.get("token_usage") or {})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._started.pop(run_id, None)
