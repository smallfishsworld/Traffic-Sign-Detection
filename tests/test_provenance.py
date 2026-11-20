"""Result-file provenance: git commit, dirty flag, command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tsd.provenance import DirtyTreeError, git_state, provenance, require_clean


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        check=True,
        capture_output=True,
    )


def test_dirty_means_uncommitted_code_not_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "bench/results").mkdir(parents=True)
    (repo / "src/a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    state = git_state(repo)
    assert state["dirty"] is False
    assert len(state["commit"]) == 40
    (repo / "bench/results/r.json").write_text("{}")  # results are not code
    (repo / "README.md").write_text("docs")
    assert git_state(repo)["dirty"] is False
    (repo / "src/a.py").write_text("x = 2\n")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["dirty_paths"] == ["src/a.py"]


def test_provenance_records_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench/x.py", "--model", "a b.xml"])
    doc = provenance({"threads": 4})
    assert doc["command"] == "bench/x.py --model 'a b.xml'"
    assert doc["runtime_settings"] == {"threads": 4}
    assert "commit" in doc["git"]


def test_require_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    import tsd.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda: {"dirty": True, "dirty_paths": ["src/x.py"]})
    with pytest.raises(DirtyTreeError, match=r"src/x\.py"):
        require_clean()
    assert require_clean(allow_dirty=True)["dirty"] is True


def test_the_bench_lock_is_recorded_with_whether_it_is_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fcntl

    from tsd.provenance import LOCK_ENV, lock_is_held

    lock = tmp_path / "bench.lock"
    lock.touch()
    monkeypatch.delenv(LOCK_ENV, raising=False)
    assert "bench_lock" not in provenance()
    monkeypatch.setenv(LOCK_ENV, str(lock))
    assert provenance()["bench_lock"] == {"path": str(lock), "held": False}
    with lock.open("rb") as holder:  # what `flock <lock> cmd` does for the command's lifetime
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert provenance()["bench_lock"] == {"path": str(lock), "held": True}
    assert lock_is_held(lock) is False
    assert lock_is_held(tmp_path / "missing.lock") is None
