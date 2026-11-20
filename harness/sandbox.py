"""A container for the skill agent's commands: the same image Agentomics-ML runs in, with the workspace at /workspace."""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

IMAGE = "biogemt/agentomics:1.2.0"
LABEL = "ago2-harness=skill"
DOCKER = shutil.which("docker") or "/usr/local/bin/docker"


def _user() -> list[str]:
    return ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]


class Sandbox:
    """A long-lived container; commands run in it with `docker exec`, as the agent's shell."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.name = f"ago2-skill-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> "Sandbox":
        subprocess.run(
            [DOCKER, "run", "-d", "--rm", "--name", self.name, "--label", LABEL, *_user(),
             "--mount", f"type=bind,src={self.workspace},dst=/workspace", "-w", "/workspace",
             "--entrypoint", "sleep", IMAGE, "infinity"],
            check=True, capture_output=True, text=True)
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run([DOCKER, "rm", "-f", self.name], capture_output=True)

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run([DOCKER, "exec", "-w", "/workspace", self.name, "sh", "-c", command],
                              capture_output=True, text=True, timeout=timeout)


def run_offline(mounts: dict[Path, str], command: list[str], workdir: str, timeout: int) -> subprocess.CompletedProcess:
    """One command in a fresh container of the same image, without network."""
    args = [DOCKER, "run", "--rm", "--network", "none", "--label", LABEL, *_user(), "-w", workdir]
    for source, target in mounts.items():
        args += ["--mount", f"type=bind,src={source.resolve()},dst={target}"]
    return subprocess.run([*args, "--entrypoint", command[0], IMAGE, *command[1:]],
                          capture_output=True, text=True, timeout=timeout)
