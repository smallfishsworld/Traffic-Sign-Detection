"""Provenance of a result file: which code produced it, with which command and environment.

Every raw result (benchmarks, serving runs, evaluation and parity reports, profiles) embeds
:func:`provenance` so that a number can be traced to a commit and a command line. ``dirty`` means
that tracked code (``src/``, ``bench/*.py``, ``pyproject.toml``, ``uv.lock``) differs from the
recorded commit or that untracked files exist there; result files under ``bench/results`` and the
documentation are not code and are ignored.

A timing run's runner wraps each measurement in ``flock <lock>`` and passes the lock's path in
``TSD_BENCH_LOCK_HELD``; :func:`provenance` then records that path and whether the lock was held
while the result was written (``bench_lock``).
"""

from __future__ import annotations

import fcntl
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import tsd

CODE_PATHS = ("src", "bench", "pyproject.toml", "uv.lock")
ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "PYTHONPATH",
    "TSD_PARENT_COMMAND",
)


LOCK_ENV = "TSD_BENCH_LOCK_HELD"


def lock_is_held(path: Path) -> bool | None:
    """Whether some process holds an exclusive ``flock`` on ``path`` (under ``flock <path> cmd``
    that is the wrapping ``flock``): a non-blocking attempt on a fresh descriptor fails. ``None``
    if the file cannot be opened."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # also releases the lock if the attempt succeeded
    return False


class DirtyTreeError(RuntimeError):
    """Measurements were requested from uncommitted code."""


def repo_root() -> Path | None:
    """The checkout that holds the imported ``tsd`` package (``src/tsd`` -> repository)."""
    root = Path(tsd.__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def git_state(root: Path | None = None) -> dict[str, Any]:
    """Commit, dirty flag and the first changed code paths of ``root`` (default: this checkout)."""
    root = root or repo_root()
    if root is None:
        return {"commit": None, "dirty": None, "note": "not a git checkout"}

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "note": head.stderr.strip()[:200]}
    status = git(
        "status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS,
        ":(exclude)bench/results",
    )  # fmt: skip
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(changed) or status.returncode != 0,
        "dirty_paths": changed[:20],
        "checked_paths": list(CODE_PATHS),
    }


def provenance(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Git state, the exact command line, relevant environment variables and, optionally, the
    effective runtime ``settings`` of the process that writes a result."""
    out: dict[str, Any] = {
        "git": git_state(),
        "command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "tsd_source": str(Path(tsd.__file__).resolve().parent),
        "env": {k: os.environ[k] for k in ENV_VARS if k in os.environ},
        "recorded_unix": round(time.time(), 1),
    }
    lock = os.environ.get(LOCK_ENV)
    if lock:
        out["bench_lock"] = {"path": lock, "held": lock_is_held(Path(lock))}
    if settings is not None:
        out["runtime_settings"] = settings
    return out


def require_clean(allow_dirty: bool = False) -> dict[str, Any]:
    """The git state; raises :class:`DirtyTreeError` for uncommitted code unless
    ``allow_dirty`` (then the result files say ``dirty: true``)."""
    state = git_state()
    if state.get("dirty") and not allow_dirty:
        raise DirtyTreeError(
            "uncommitted code changes (" + ", ".join(state.get("dirty_paths", [])[:5]) + "); "
            "commit first or pass --allow-dirty"
        )
    return state


def warn_if_dirty() -> None:
    state = git_state()
    if state.get("dirty"):
        print(
            "warning: uncommitted code changes; the report records dirty=true "
            f"({', '.join(state.get('dirty_paths', [])[:5])})",
            file=sys.stderr,
        )
