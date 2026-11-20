"""Runs Agentomics-ML's own Docker launcher with the run's reasoning effort set, and every model call logged, inside the
agent's container.

Usage: python -m harness.sm_launch <reasoning-effort> <usage-log-dir> <agentomics-run arguments...>
The usage log directory is mounted at /harness-log, outside the agent's /workspace.
"""
import sys
from pathlib import Path

from agentomics.cli import run as launcher

from harness import protocol

SITE_PACKAGES = "/opt/conda/envs/agentomics-env/lib/python3.12/site-packages"
HOOK = Path(__file__).resolve().parent / "container" / "sitecustomize.py"


def main() -> int:
    effort = sys.argv.pop(1)
    log_dir = Path(sys.argv.pop(1)).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    environment_arguments = launcher._docker_environment_arguments

    def with_harness(agent_id: str) -> list[str]:
        return environment_arguments(agent_id) + [
            "-e", f"HARNESS_REASONING_EFFORT={effort}",
            "-e", "HARNESS_USAGE_LOG=/harness-log/usage.jsonl",
            "-e", f"HARNESS_MAX_STEPS={protocol.ROUNDS_PER_STEP}",
            "-e", f"HARNESS_MAX_VALIDATION_RETRIES={protocol.RETRIES}",
            "-e", "HARNESS_DISABLED_TOOLS=paper_fetch",
            "--mount", f"type=bind,src={HOOK},dst={SITE_PACKAGES}/sitecustomize.py,readonly",
            "--mount", f"type=bind,src={log_dir},dst=/harness-log",
        ]

    launcher._docker_environment_arguments = with_harness
    return launcher.main()


if __name__ == "__main__":
    sys.exit(main())
