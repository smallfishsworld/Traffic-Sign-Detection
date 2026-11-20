# Result files and the commands that produced them

Everything here was measured in this repository's container in 2026 (4 shared vCPUs
of an Intel Xeon @ 2.10 GHz with AVX-512 VNNI and AMX, KVM, no GPU; `benchmarks/lscpu.txt`).
Nothing here comes from the 2024 course project. Long CPU jobs ran under `nice -n 19` with 2
threads. The timing measurements ran under `flock /home/user/work/bench.lock`: the benchmark,
profile and serving runners take the lock per configuration or per serving session; the
batch-shape probe in `serving/` was run under a lock taken by hand (its file does not record
that; the probe takes the lock itself since `6a2c9e6`, and it was not run again), and the Docker
build and each of the two smoke tests in `docker/` ran inside its own `flock` call (each file's
first line says so).

**Provenance.** Every raw result file written in the fix round embeds `provenance`: the git
commit, a dirty flag for the code paths (`src/`, `bench/`, `pyproject.toml`, `uv.lock`), the
exact command line (`command`, `argv`; a runner's own command as `env.TSD_PARENT_COMMAND`) and
the thread variables. The runtime's effective settings (`runtime_settings`) are recorded only in
the benchmark raw files and the profiles; the serving files record the CPU affinity only, and
the evaluation reports none. Every run was made from a clean git worktree with
`PYTHONPATH=<worktree>/src`, so that editing could continue in the main checkout (`cwd` in the
provenance shows the worktree):

* the benchmark matrix and the profile at commit `cfff803`;
* the serving comparison at `c6e3f1b` (the warm-up fix; the run at `cfff803` is in
  `archive/serving-cfff803/`);
* the evaluation detections at `bb7ea72`, and every evaluation report re-scored from those saved
  detections at `31263cc` (six-decimal metrics; the detection files' SHA-256 values are listed
  in each report);
* the batch-shape probe and the duplicate probe at `bb7ea72`;
* the parity check of the deployed models, the augmentation probes, the 200-scenario
  pycocotools comparison and the benchmark tables' re-aggregation at `8d47bd4`;
* the Docker image from a clean worktree at `a45e982` (`docker/build.txt`); what goes into the
  image (`.dockerignore`: `pyproject.toml`, `uv.lock`, `LICENSE`, `src/`, and `README.md`) is
  the same at every later commit except `README.md`;
* the CI simulation and the development-environment checks at `dcc2d5b` (`ci/`), and the
  warm-up memory probe at `dcc2d5b` (`docker/warmup_rss.txt`; its repeat,
  `docker/warmup_rss_main_checkout.txt`, ran in the main checkout at `ed409e6`, whose `src/` is
  the same).

The code paths that changed after `cfff803` do not touch the timed inference path
(`git diff cfff803 -- src/tsd/infer src/tsd/benchmark.py`: the export parity gate, the
aggregation and its headline, what a raw result records, how attempts are numbered, and a
docstring). The runners refuse a dirty tree unless `--allow-dirty`.

**Raw files written before `6a2c9e6`** (all of the committed benchmark, profile and serving
files) do not have what the harness records since then: the lock path and whether it was held
(`provenance.bench_lock`), every timed call (`samples_ms`), the model files' SHA-256
(`model_sha256`), the attempt number counted over measured sessions only with busy exits
separately (`busy_skips`; in the committed files `attempt` also counts busy exits, so
`attempt` 3 can come with fewer than two `*.attempt<k>.json` files), and, for serving, the
server's `/healthz` with its runtime settings, the model hashes and the image names. The
benchmark raw files do record each model's path and size, and `parity/check_deployed_files.txt`
confirms that the deployed files match those sizes and every SHA-256 the evaluation and parity
reports record.

The first round's results (one run per configuration, no provenance) are kept in
[`archive/v1-0c6c533/`](archive/v1-0c6c533/README.md) for comparison, including its parity and
quantization reports.

| file | what | command |
|---|---|---|
| `benchmarks/raw/<config>.run<k>.json` | one file per configuration and round (`*.attempt<j>.json`: rejected, contaminated attempts) | `python bench/run_benchmarks.py --runs 3` |
| `benchmarks/benchmarks.{json,md}`, `benchmarks/lscpu.txt` | the matrix: medians over the rounds, ranges, paired speedups, joined with the accuracy report; `run_to_run`: spread over the rounds and the fastest runtime per round | written by the same command; rebuilt with `python bench/run_benchmarks.py --aggregate-only` after the evaluation |
| `benchmarks/profile/<model>_<config>.json`, `benchmarks/profile_yolo11{n,s}.{json,md}` | OpenVINO per-node kernel profile of FP32, BF16, INT8 and INT8 + BF16 hint | `python bench/profile_ops.py` |
| `serving/<setting>_c<level>.run<k>.json`, `serving/serving.{json,md}` | `tsd serve` under closed-loop load: batching off, fixed wait, adaptive wait; 3 runs per level (commit `c6e3f1b`) | `python bench/serve_compare.py --runs 3` |
| `archive/serving-cfff803/` | the same comparison before the warm-up fix (the concurrency-8 tail) | the same command at commit `cfff803` |
| `serving/batch_shape_probe.json` | first call at a new batch size vs repeated calls (the concurrency-8 tail) | `flock /home/user/work/bench.lock python bench/batch_shape_probe.py` at `bb7ea72` (the probe now takes the lock itself: `python bench/batch_shape_probe.py`) |
| `eval/coco_classes.{json,md}` | accuracy of the 14 model/runtime combinations | `tsd eval` with the 14 models listed in `docs/EVALUATION.md`, `--out bench/results/eval/coco_classes.json` |
| `eval/coco_classes_strict.{json,md}` | the same, unverified detections counted as false positives | the same with `--count-unverified --out bench/results/eval/coco_classes_strict.json` |
| `eval/coco_classes_with_negatives.{json,md}` | the 380-image variant with the verified-negative images | `tsd eval <12 models> --data-dir data/openimages-traffic-negatives --detections-dir bench/results/eval/detections_with_negatives --out bench/results/eval/coco_classes_with_negatives.json` |
| `eval/tiling.{json,md}` | sliced inference vs plain, merged at the detector's NMS IoU (0.7) | `tsd eval models/yolo11n_openvino_fp32/yolo11n.xml models/yolo11n_openvino_fp32/yolo11n.xml@tiles=512 models/yolo11s_openvino_fp32/yolo11s.xml models/yolo11s_openvino_fp32/yolo11s.xml@tiles=512 --out bench/results/eval/tiling.json` |
| `eval/tiling_merge_iou_0.5.{json,md}` | control: the same with the old merge IoU | the same with `--tile-merge-iou 0.5` |
| `eval/tiling_no_border_filter.{json,md}` | the first tiling version's settings (no border filter, merge 0.5) | the same with `--no-tile-border-filter --tile-merge-iou 0.5` |
| `eval/detections/*.json.gz`, `eval/detections_with_negatives/*.json.gz` | every run's saved detections (boxes, scores, classes per image); each report lists the file and its SHA-256 per run | written by `tsd eval` |
| `augment_probes.json` | geometry of the perspective boxes cut by the border (before/after the polygon clip) and sign replacement in four cases (same outline, and disc/triangle crossed) | `nice -n 19 python bench/augment_probes.py --code-label "HEAD (this worktree)"` |
| `augment_replacement_cfff803.json` | the same replacement cases and metric against the code before shape-aware replacement | `git archive cfff803 src \| tar -x -C <dir>`, then `PYTHONPATH=<dir>/src nice -n 19 python bench/augment_probes.py --only replacement --code-label "cfff803 (git archive of src/)" --out bench/results/augment_replacement_cfff803.json`, from the clean worktree at `8d47bd4` (the file's `cwd`; its `git` field is empty because the package under test is not a git checkout) |
| `augment_replacement_before_cfff803.txt` | the replacement probe from my re-check (another metric: pixels unchanged) run against the old code | the command in the file |
| `dedup_copy_sim.json` | dHash distances of simulated resize + JPEG copies and of unrelated pairs; old vs current duplicate rule | `nice -n 19 python bench/dedup_copy_sim.py` |
| `letterbox_rect_share.txt` | pixel share of Ultralytics' rectangular letterbox on the benchmark images | the command in the file |
| `network/openimages_download.log` | download log of the 219 evaluation images (sizes, timestamps) | `uv run tsd -v openimages --reuse-dir /home/user/work/oi 2> bench/results/network/openimages_download.log` |
| `network/openimages_negatives_download.log` | download log of the 161 verified-negative images | `OMP_NUM_THREADS=1 nice -n 19 tsd -v openimages --out data/openimages-traffic-negatives --reuse-dir /home/user/work/oi --include-verified-negatives --manifest-copy-dir data/manifests/openimages-traffic-negatives 2> bench/results/network/openimages_negatives_download.log` (after hard-linking the 219 existing images into the new directory) |
| `network/negative_candidates.txt` | box-less images with a verified-negative label, and how many also have a positive one (dropped by the builder) | the command in the file (script: `negative_candidates.py.txt`) |
| `network/train_csv_head.txt` | size of the Open Images train box CSV (for the P3 decision) | `curl -sS -I https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv` |
| `parity/parity_yolo11{n,s}.json` | the export parity gate on the deployed files: PyTorch vs ONNX Runtime vs OpenVINO FP32 on 8 validation images, tolerances, pass/fail and every file's SHA-256 | `OMP_NUM_THREADS=2 nice -n 19 tsd parity models/yolo11n.pt --threads 2 --report bench/results/parity/parity_yolo11n.json` (same for yolo11s) |
| `parity/check_deployed_files.txt` | the deployed model files against every SHA-256 (evaluation and parity reports) and size (benchmark raw files) recorded for them | `python3 -I bench/results/parity/check_deployed_files.py.txt` |
| `eval/pycocotools_agreement_200.txt` | the evaluator against pycocotools on 200 random scenarios | the command in the file |
| `archive/v1-0c6c533/{parity,quantize}_*.json` | the first round's export parity and INT8 quantization reports (commit `8c6088f`, no provenance, no tolerance or pass fields, no hashes); the deployed INT8 models were written in that round, and their SHA-256 values are recorded by the evaluation reports | `tsd export... --report`, `tsd quantize... --report` at `8c6088f` |
| `ci/ci_simulation_dcc2d5b_py3.1{2,3}.log` | the CI test job without PyTorch, simulated for commit `dcc2d5b` on Python 3.12 and 3.13 (later commits change no code that CI runs; `docker-compose.yml` changed) | `bash bench/results/ci/ci_simulation.sh.txt dcc2d5b 3.12` (and `3.13`); see `ci/README.md` |
| `ci/dev_checks_<commit>.log` | ruff, the format check, mypy and pytest in the full development environment | the commands in the file |
| `docker/build.txt`, `docker/build.log` | the image build from a clean worktree at `a45e982`: commit, `git status --porcelain` (empty), exit status and duration; the full BuildKit log | `flock /home/user/work/bench.lock timeout 900 bash build.sh.txt <worktree> <out>`, which runs `docker build --progress plain --network host --build-arg HTTPS_PROXY=$HTTPS_PROXY --secret id=ca_bundle,src=/root/.ccr/ca-bundle.crt -t tsd-serve:0.1.0 -t tsd-serve:a45e982 <worktree>` (the proxy settings are only needed behind this environment's proxy); the script writes its record to `build_and_size.txt`, committed here as `build.txt` |
| `docker/smoke_test.txt` | the image's ID and sizes (`docker save` piped to gzip and as stored, every layer decompressed and counted by `layer_sizes.py.txt`, `docker history`, `docker images`, package sizes), the CI job's two checks, and the service under the compose settings (2 GiB): health with its settings, a detection, an upload over 1 MB, a non-image upload, 411, 413, 429 with `Retry-After` at the admission limit, metrics, memory and its peak | `flock /home/user/work/bench.lock timeout 900 bash smoke.sh.txt tsd-serve:a45e982 <out>`; every command is printed before its output |
| `docker/smoke_test_1g_oom.txt` | the same smoke test with the first compose file's 1 GiB: neither container became healthy and the port never answered (this file does not record why; the OOM kills are recorded in `oom_1g*.txt`) | `flock... bash smoke_1g.sh.txt tsd-serve:a45e982 <out>`; the script writes `smoke_test.txt`, committed here under this name |
| `docker/oom_1g.txt`, `docker/oom_1g_run2.txt` | 5 + 5 starts of the image under 1 GiB with the compose security settings: state (`OOMKilled`, exit code, start and finish times) and logs of each. A single start before them became healthy; the first 5-start run overwrote its record | `bash oom_1g.sh.txt tsd-serve:a45e982 <out> 5` (not under the lock: no timing; lowest CPU weight) |
| `docker/memory_probe.txt` | the service's start-up memory in the container with a 3 GiB limit (cgroup peak and usage after the warm-up and after one request) | `bash memory_probe.sh.txt tsd-serve:a45e982 <out>` (not under the lock: no timing; lowest CPU weight) |
| `docker/warmup_rss.txt`, `docker/warmup_rss_main_checkout.txt` | the detector's resident memory on the host as each batch size is warmed up, in three orders (two runs: a clean worktree at `dcc2d5b`, and the main checkout at `ed409e6` with the same `src/`) | the commands in the files (script: `warmup_rss.py.txt`) |
| `docker/dev_env_size.txt` | size of the development environment (with PyTorch), for comparison; the image figures in its header are the first image's | the commands in the file |
| `archive/docker-c83dec9/` | the first image (built from the working tree that became `c83dec9`, before the request gate and the Dockerfile review changes) | see its README |
