# Local simulation of the CI test job

Before pushing, I ran the `checks` job of `.github/workflows/ci.yml` locally. This directory holds
that local simulation for commit
`dcc2d5b` on both of its Python versions, made with `ci_simulation.sh.txt` (run as
`bash ci_simulation.sh.txt dcc2d5b 3.12`, then with `3.13`; uv 0.11.32, the version the
workflow pins):

1. `git archive dcc2d5b` into an empty directory (no models, no data, no untracked files);
2. `uv sync --locked --extra onnx --extra openvino --extra eval --extra serve` into a virtual
   environment in a directory that does not exist before the sync (the log checks and says so),
   synced exactly to the lock, with the machine's interpreters (3.12.3 and 3.13.18; no
   interpreter download) and without `torch` and `ultralytics` (checked and printed). Packages
   came from the local uv cache where it had them, as the workflow's `enable-cache` would
   give them on GitHub;
3. the workflow's steps, `ruff check .`, `ruff format --check .`, `mypy` (strict, `src/tsd` and
   `bench`) and `pytest -q -rs`, plus `tsd --help` as an import smoke test, all under
   `nice -n 19` with 2 threads. Each step's exit status is the command's own
   (`${PIPESTATUS[0]}`, under `set -o pipefail`), not that of the `tail` that shortens its output.

| log | Python | ruff check | ruff format | mypy | pytest | `tsd --help` |
|---|---|---|---|---|---|---|
| `ci_simulation_dcc2d5b_py3.12.log` | 3.12.3 | 0 errors | 96 files formatted | no issues in 45 files | 263 passed, 24 skipped | exit 0 |
| `ci_simulation_dcc2d5b_py3.13.log` | 3.13.18 | 0 errors | 96 files formatted | no issues in 45 files | 263 passed, 24 skipped | exit 0 |

Every skip prints its reason: 5 tests need the model weights (not in a clean checkout), 5 need
`torchvision` and 14 need `ultralytics`. The commits after `dcc2d5b` change documentation,
result files and the memory limit in `docker-compose.yml` (`fff697c`, which no CI step reads):
`git diff --stat dcc2d5b -- src tests bench/*.py pyproject.toml uv.lock Dockerfile .dockerignore
.github` is empty.

`dev_checks_dcc2d5b.log` (`bash dev_checks.sh.txt dcc2d5b`) runs the same checks in the full
development environment (Python 3.12.3 with PyTorch and Ultralytics, the cached `yolo11n.pt`
through `TSD_TEST_WEIGHTS`) from a clean worktree of the same commit, under `nice -n 19` with 2
threads: ruff and the format check clean, mypy clean (45 files), **287 passed, 0 skipped**
(263 + 24: the tests the simulation skips all run here), and no tracked file changed.

Not simulated: the `docker` job as written (Buildx with the GitHub Actions cache). The image was
built locally with `docker build` from a clean worktree, and the job's two checks (`tsd --help`
and the service imports) ran against it in `../docker/smoke_test.txt`. Also
different from GitHub: the runner (my VM, not `ubuntu-24.04`) and the interpreters'
patch versions.

The earlier simulation of `9f772b4` (committed in `a57eb66`, removed here) is superseded: its
script took each step's exit status after a pipe, so a failing step could still have printed 0,
and it synced into an environment that already existed.
