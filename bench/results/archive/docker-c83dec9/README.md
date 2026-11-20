# The first Docker image (working tree that became commit c83dec9)

The first image was built from the working tree that was committed as `c83dec9` after the build;
neither file records a commit or whether the tree was clean.
`build_and_size.txt` holds its build log (72 s, no cached steps), `docker images` (201 MB content
size), `docker history` (474 MB virtual environment) and package sizes; `smoke_test.txt` holds
its smoke test (healthy, a detection, a non-image upload, metrics, 534.8 MiB of memory).

That image predates the service's request gate, adaptive wait and disconnect handling (commit
`6d1a565`: its `/healthz` has no `max_inflight`, and the lone request in its smoke test waited
5.3 ms in the queue) and the Dockerfile changes of `ca14ec5` and `baf9fe4` (base image pinned by
digest, the dependency layer built from `pyproject.toml` and `uv.lock` only, `uv sync --locked`,
sequential downloads). It is superseded by the build and smoke test in `../../docker/`, and kept
only as the record the earlier documentation cited. The image figures in the header of
`../../docker/dev_env_size.txt` (452 MiB by `du -sm`, 474 MB in `docker history`) are this
image's.
