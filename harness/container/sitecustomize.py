"""Mounted into the Agentomics-ML container: sets the run's reasoning effort on every chat completion, logs each call's
usage to a file outside the agent's workspace, and sets the step budget, retries and tools the harness asks for."""
import json
import os
import time

_effort = os.environ.get("HARNESS_REASONING_EFFORT")
_log = os.environ.get("HARNESS_USAGE_LOG")
_iteration_file = "/workspace/run/current_iteration/runtime_info/iteration_metadata.json"


def _iteration():
    try:
        with open(_iteration_file) as f:
            return int(json.load(f)["iteration"])
    except Exception:
        return -1


def _record(start, iteration, kwargs, response):
    usage = getattr(response, "usage", None)
    usage = usage.model_dump() if usage is not None else {}
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    line = {
        "start": start, "end": time.time(), "iteration": iteration, "model": kwargs.get("model", ""),
        "cost": float(usage.get("cost") or 0.0),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(prompt.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion.get("reasoning_tokens") or 0),
    }
    with open(_log, "a") as f:
        f.write(json.dumps(line) + "\n")


if _effort or _log:
    try:
        from openai.resources.chat.completions import AsyncCompletions

        _create = AsyncCompletions.create

        async def _create_logged(self, *args, **kwargs):
            if _effort:
                kwargs["reasoning_effort"] = _effort
            # Anthropic rejects a forced tool call while the model reasons; the model then picks the tool itself
            if (_effort != "none" and str(kwargs.get("model", "")).startswith("anthropic/")
                    and kwargs.get("tool_choice") not in (None, "auto", "none")):
                kwargs["tool_choice"] = "auto"
            start, iteration = time.time(), _iteration()
            response = await _create(self, *args, **kwargs)
            if _log and not kwargs.get("stream"):
                try:
                    _record(start, iteration, kwargs, response)
                except Exception:
                    pass
            return response

        AsyncCompletions.create = _create_logged
    except ImportError:
        pass


# Agentomics-ML builds its Config (fresh or loaded for a fork) after this file runs, so the limits are applied when the
# module defining it is first imported.
_max_steps = os.environ.get("HARNESS_MAX_STEPS")
_max_retries = os.environ.get("HARNESS_MAX_VALIDATION_RETRIES")
_no_tools = [t for t in os.environ.get("HARNESS_DISABLED_TOOLS", "").split(",") if t]

if _max_steps or _max_retries or _no_tools:
    import importlib.abc
    import sys

    def _patch(module):
        config = module.Config
        original = config.__init__

        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if _max_steps:
                self.max_steps = int(_max_steps)
            if _max_retries:
                self.max_validation_retries = int(_max_retries)
            if _no_tools:
                self.tool_ids = [t for t in self.tool_ids if t not in _no_tools]

        config.__init__ = __init__

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "agentomics.utils.config":
                return None
            sys.meta_path.remove(self)
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None:
                return None
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
