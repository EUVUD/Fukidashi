"""Facts about an eval run that make it reproducible (OUTLINE 4.8): data hash and git state."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo: Path) -> dict:
    """HEAD commit and whether the working tree has uncommitted or untracked files; None when unknown."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": None if commit is None else commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
    }
