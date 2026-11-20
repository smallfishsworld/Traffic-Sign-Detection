# Engineering notes

My notes on the decisions I made while rebuilding the 2024 course project, each written as
*problem → measurement → fix → trade-off*. Unless a number is marked as historical (2024, A100,
TT100K, from our course report), I measured it on my VM (4 shared vCPUs of an Intel Xeon with
AVX-512 VNNI and AMX, no GPU).

## 0. Ground rules for numbers

* **Historical** (2024, A100, TT100K, from the course report): run 3 → 4 precision
  0.757 → 0.800, mAP50 0.701 → 0.724. Nothing else from 2024 survives (see
  [`2024-course-project/`](2024-course-project/README.md)). These numbers cannot be reproduced
  here: I couldn't reach the TT100K download servers and had no GPU.
* **Measured here**: everything in `bench/results/`, each with the command that produced it.
  Tables in the README are copied from those files.

## 1. Data

### 1.1 TT100K converter (`tsd.data.tt100k`)

*Problem.* The 2024 conversion script (report appendix A) loaded the whole annotation JSON,
wrote TT100K `test` into the YOLO `val` directory (so checkpoint selection and the reported
metrics used the same split), kept every class, and wrote boxes as given; Ultralytics drops an
entire image when one label falls outside [0, 1].

*Fix.* Streams the JSON with `ijson`, reads image sizes from file headers in a 2-process pool,
clips boxes to the image (dropping those under 2 px), keeps classes by an explicit list or a
minimum instance count, re-indexes them densely, carves a deterministic hash-based validation
split out of TT100K *train* (salted SHA-256 of the image id) and keeps TT100K *test* held out.

*Trade-off.* Without the TT100K data the converter is verified only on a synthetic
TT100K-format fixture (`tests/test_tt100k.py`), not on the real data.

### 1.2 Evaluation data: Open Images instead of TT100K

*Problem.* TT100K could not be downloaded, so a public, licensed substitute was needed for
measuring accuracy.

*Fix.* `tsd openimages` selects every Open Images V7 validation/test image with a Traffic sign,
Stop sign or Traffic light box: 219 images (52 validation, 167 test), all CC BY 2.0 (manifest
with SHA-256, licence, author and source URL in `data/manifests/openimages-traffic/`). Images are
used as stored: none has an EXIF orientation tag, and the two whose `Rotation` metadata is 270
are stored sideways *with boxes that match the stored pixels* (checked by drawing the boxes), so
rotating them would have broken the labels. The protocol (verified labels, group-of boxes,
excluded Traffic sign class) is in [`EVALUATION.md`](EVALUATION.md).

*Trade-off.* Open Images "Traffic sign" is a generic class, unlike TT100K's 201 fine-grained
classes (2024 report, section 4.2.4), and COCO-pretrained models have no traffic-sign class at
all, so only Stop sign and Traffic light are evaluated. 142 of the 219 images contain a stop
sign or traffic light; the other 77 have only generic Traffic sign boxes and count through their
false positives.

*Selection effect, and a second variant.* The set is selected by box presence, so it keeps only
the human-verified *negative* images that also have a target box: 49 of the 170 images verified
to contain no stop sign and 6 of the 61 verified to contain no traffic light. An Open Images
challenge-style evaluation scores every verified image, and the left-out negatives can only add
false positives, so the main set's absolute AP is optimistic. `tsd openimages
--include-verified-negatives` builds the variant with them: 380 images, the 219 plus 161
box-less images (121 verified without a stop sign, 55 without a traffic light, 15 of them
both; 40 validation, 121 test), manifest in `data/manifests/openimages-traffic-negatives/`. The
161 extra images (43.2 MB) came over one connection at 1.85 MB/s
(`bench/results/network/openimages_negatives_download.log`); the 219 already present were
hard-linked first, so nothing was downloaded twice. Both variants are reported (section 4.4).

### 1.3 Polite downloader (`tsd.net`)

*Problem.* The metadata CSVs are large (the test bounding-box file is 77 MB) and the machine and
its network are shared.

*Fix.* One connection by default (hard maximum 2), a shared token-bucket cap of 2 MB/s,
resumable `.part` files via HTTP `Range`, size checks, MD5 checks against pinned hashes or the
server's `x-goog-hash` / S3 single-part ETag, skip-if-present, retries with exponential backoff
only for transient errors (honouring `Retry-After`), a generic User-Agent. Downloaded files are
treated as untrusted (own directory; image dimensions read from headers).

*Measurement.* The 219 evaluation images (66,518,842 bytes) arrived in 35.8 s from the first to
the last completed file: 1.85 MB/s after the initial burst, under the 2 MB/s cap
(`bench/results/network/openimages_download.log`).

### 1.4 Near-duplicates between calibration and evaluation images

*Problem.* INT8 calibration uses COCO128; COCO and Open Images were both collected from Flickr,
so "different data set" does not guarantee different photos.

*Measurement* (`bench/dedup_copy_sim.py`, raw results in `bench/results/dedup_copy_sim.json`).
Unrelated pairs, all 128 × 219 = 28,032 (calibration, evaluation) pairs: on the 64-bit dHash the
closest is at 9 bits, the only candidate within 10 (a tennis court and a "Dead End" sign, both
mostly smooth sky), and pairs with matching aspect ratios are at least 12 bits apart; on the
256-bit dHash every pair is at least 53 bits apart (63 with matching aspect ratios). Simulated
copies of all 219 evaluation images (longest side 640 px, JPEG quality 70, with OpenCV area,
bilinear and bicubic and PIL Lanczos and bilinear resizing, plus re-encoding alone): at most 5
bits on the 64-bit hash for every recipe, but up to **41 bits** on the 256-bit hash (27 for
OpenCV bilinear, 34-35 for re-encoding alone and Lanczos; 99th percentile 17-22), always on
the same low-texture image. The first version said copies stay within 25 bits on the 256-bit
hash; that number had no raw file and does not reproduce (found when I re-checked my results).

*Fix.* Two stages: 64-bit candidates (within 10 bits) are confirmed if the aspect ratio matches
within 2% and either the 64-bit match is strong (at most 5 bits, which every simulated copy
achieves) or the 256-bit dHash is within 40 bits. The first version required the 256-bit hash
to confirm, so it missed one copy in three of the six recipes (218 of 219 flagged); the current
rule flags 219 of 219 in all six and none of the 28,032 unrelated pairs. `tsd quantize` refuses
to calibrate on a confirmed duplicate, and fails instead of skipping the check when the
evaluation images it should compare against are missing; the report is stored with each
quantization (the first round's reports: `bench/results/archive/v1-0c6c533/quantize_*.json`).

*Limits.* The margin on the 64-bit hash is thin: copies reach 5 bits, the strong-match
threshold, and the nearest unrelated pair with a matching aspect ratio is at 12. A copy that is
cropped (not only rescaled) changes the aspect ratio and is not caught. The check covers the
calibration images only; the models' own training set (COCO train2017) was not compared with
the evaluation images (section 1.2, `docs/EVALUATION.md`).

### 1.5 Sign replacement must cover the old sign (`tsd.augment.replacement`)

*Problem* (found when I re-checked my results). The 2024 copy-paste augmentation replaces a sign with a
template of another class and relabels the box. The first rebuild checked only the template
canvas's aspect ratio, so a disc could be replaced by a triangle: the probe from my re-check, re-run
here against the old code (commit cfff803), left 34.6% of a replaced disc's pixels unchanged
(max 37.5%, 50 seeds) under the new triangle label, and a triangle replaced by a disc 5.5%
(`bench/results/augment_replacement_before_cfff803.txt`; that probe counts pixels left
unchanged, the measurement below counts a marker colour). A detector trained on that learns
that half-visible discs are warning signs. The same review found that the rarest classes could
themselves be overwritten, and that the alpha feathering did nothing for opaque templates.

*Fix.* Templates carry an outline (`circle`, `triangle`, `inverted_triangle`, `octagon`,
`rect`; `shape_for_class` maps TT100K class names), and a sign is only replaced by a class of
the same outline. Before pasting, the new template's silhouette must cover
`min_coverage` (95%) of the old sign's outline (`silhouette_mask` of that shape in the old box)
or the replacement is skipped; templates are cropped to their silhouette first, so a template
with transparent margins still fills the box. Only signs of a class more common than the
pasted one are replaced, so the rarest classes are never overwritten; the feathering blurs with
a zero border so opaque canvases get soft edges too, and an opaque template triggers a warning.

*Measurement* (`bench/augment_probes.py`; the old sign drawn in a colour no template contains,
50 seeds, templates of two classes, the old sign's and a rarer one; the share of the old sign's
pixels that still show the colour after a replacement). The same probe and metric ran against
the current code (`bench/results/augment_probes.json`) and against the code before the change
(`git archive` of cfff803, `bench/results/augment_replacement_cfff803.json`):

| case | cfff803: replaced, old sign visible (mean / max) | now: replaced, visible |
|---|---|---|
| disc by disc | 50/50, 0% / 0% | 50/50, 0% / 0% |
| triangle by triangle | 50/50, 0% / 0% | 50/50, 0% / 0% |
| disc by triangle | 50/50, 34.2% / 36.9% | 0/50 (refused) |
| triangle by disc | 50/50, 3.1% / 4.7% | 0/50 (refused) |

So the change that matters is refusing a paste of another outline; a same-outline replacement
covered the old sign completely in both versions. Before the silhouette crop was added, the
synthetic warning triangle's transparent margins (8 px above and 7 below on its 128 px canvas)
kept it from covering a box-filling triangle, and the coverage check refused most triangle
replacements
(`tests/test_augment_replacement.py::test_templates_with_transparent_margins_are_cropped_before_stretching`).

*Trade-off.* The outline of a real TT100K sign is inferred from its class prefix, not from the
pixels; a sign whose box is not tight to its outline (occlusion, motion blur) can still show
around the new one. The augmentation is a standalone transform, not wired into a training run
(section 9).

### 1.6 Perspective warp: boxes cut by the image border (`tsd.augment.perspective`)

*Problem* (found when I re-checked my results). A homography maps a box to a quadrilateral; the label is that
quadrilateral's enclosing box. When the quadrilateral crosses the image border, the first
version clipped the *enclosing box* to the image and computed the visible share on that box.
Both are wrong for a tilted quadrilateral: the visible part of a tilted edge is shorter than the
clipped box suggests, so the label extends past the warped pixels.

*Fix.* The quadrilateral itself is clipped to the image (Sutherland-Hodgman, `clip_polygon`);
the label is the clipped polygon's enclosing box and the visible share is the clipped area over
the full quadrilateral's area (`polygon_area`). The crossing point is placed exactly on the
border, so floating-point error cannot leave a coordinate slightly outside the image.

*Measurement* (`bench/augment_probes.py`, `bench/results/augment_probes.json`; 2,000 seeded
homographies per corner shift on a 640 × 480 image, a filled 6-40 px box within 30 px of the
left or right border, the label compared with the extent of the warped pixels; the old rule is
re-implemented in the script and run on the same seeds):

| corner shift | rule | boxes kept | max label error | labels off by > 1.5 px | label / pixel area, p99 |
|---|---|---|---|---|---|
| 0.08 | before | 1,766 | 2.45 px | 0.68% | 1.159 |
| 0.08 | after | 1,758 | 0.73 px | 0 | 1.137 |
| 0.15 | before | 1,444 | 5.57 px | 1.94% | 1.163 |
| 0.15 | after | 1,436 | 0.93 px | 0 | 1.152 |

After the fix the label is within one pixel of the warped pixels (nearest-neighbour
rasterization alone explains that much). Visible share and size are now smaller for tilted
boxes, so the box filter drops 8 more boxes per 2,000 at both shifts. The area ratio stays
above 1 at the 99th percentile because a 6 px box gains a large relative area from one pixel of
rasterization. Like the replacement, the warp is a standalone transform, not used in a training
run here.

## 2. Environment and dependencies

### 2.1 PyTorch wheels and the CUDA libraries

*Problem.* The PyPI `torch` wheel (2.15.0+cu132) declares the CUDA libraries as hard ELF
dependencies (`libtorch_cuda.so` and `libtorch_python.so` list cudart, cuBLAS, cuDNN, NCCL as
`DT_NEEDED`), so installing it without them makes `import torch` fail even on a CPU-only
machine; `download.pytorch.org` (CPU wheels) is blocked here.

*Fix.* Torch is installed once with its CUDA dependencies (the virtual environment is 6.3 GiB,
`bench/results/docker/dev_env_size.txt`).
The service never needs it: the Docker image installs only the `openvino` and `serve` extras,
and CI runs without torch (tests that need it are skipped with a reason).

### 2.2 OpenVINO telemetry

*Problem.* With the `openvino-telemetry` package present, `ov.convert_model()` sends Google
Analytics events (no consent file is treated as consent). I noticed this during my first
exports (unexpected outbound requests) before I removed it.

*Fix.* `openvino-telemetry` is excluded from the lock with a never-true marker override in
`pyproject.toml`; afterwards an export spawned no telemetry processes.

### 2.3 OpenVINO runs "FP32" models in BF16 on AMX CPUs

*Problem.* On CPUs with AMX, OpenVINO's default `INFERENCE_PRECISION_HINT` is BF16, so an FP32
IR is silently executed in BF16, which would have mislabelled every "FP32" latency.

*Fix.* The OpenVINO backend pins the hint to `f32`; BF16 is measured as its own configuration
(`openvino-bf16`). Its accuracy on the evaluation set: ΔmAP50-95 vs PyTorch +0.0001
[−0.002, +0.003] (YOLO11n) and −0.0015 [−0.004, +0.001] (YOLO11s), 95% paired bootstrap
intervals (`bench/results/eval/coco_classes.json`). The IR is also saved with
`compress_to_fp16=False` (the default stores FP16 weights).

## 3. Export and quantization

### 3.1 ONNX / OpenVINO export with a parity gate

`tsd export` uses the TorchScript ONNX exporter (opset 18, dynamic batch axis, fixed square
input, 640 by default) and converts ONNX to an OpenVINO IR. A parity check runs PyTorch and each
export on the same 8 images and compares the raw head outputs and the final detections.
Measured on the deployed files with the enforced gate (`tsd parity`, commit 8d47bd4,
`bench/results/parity/parity_yolo11{n,s}.json`; their SHA-256 values match the evaluation
reports', `parity/check_deployed_files.txt`): largest box difference 0.037 px (YOLO11n, ONNX
Runtime), largest score difference 3.3e-5 (YOLO11s, ONNX Runtime), every detection matched in
both directions (52/52 and 64/64), so both models pass. The first round's reports (commit
8c6088f, `bench/results/archive/v1-0c6c533/parity_*.json`, no provenance) gave the same
largest differences. The evaluation then shows identical mAP for PyTorch, ONNX Runtime and
OpenVINO FP32.

*Bug I found when re-checking: the "gate" never gated.* The first version computed and printed the
parity report but exported anyway, the README's export command skipped it, matching went one
way only (extra detections of an export were never counted), and the raw-output comparison
assumed a 640 × 640 input, so `--imgsz 512` with parity images failed. Now `ParityReport` has
explicit tolerances (`PARITY_TOLERANCE`: boxes within 0.5 px, scores within 1e-3, at least 98%
of the detections matched in *both* directions) and `failures()`; the raw outputs are compared
at the actual input size. `tsd export` writes both exports into a staging directory inside the
output directory and moves them into place only when both pass; otherwise nothing is installed
and the exit status is 1 (`--skip-parity` to bypass, `--parity-images` to choose the images),
so a later `tsd quantize` or `tsd serve` cannot pick up an export that failed. `tsd parity`
runs the same gate on exports already in place. `tests/test_parity_gate.py` checks that a
report outside the tolerances fails, `tests/test_cli_export.py` drives `tsd export` with fake
exporters and runtimes into the failing path (exit status 1, nothing left in the output
directory) and the passing one, and `tests/test_export_e2e.py` checks that the real exports
pass with two-way matching.

### 3.2 INT8 with a floating-point box decoder

*Problem.* YOLO11's detect head ends with box decoding: a softmax over 16 distance bins (DFL),
anchor-grid arithmetic that produces pixel coordinates up to 640, and a sigmoid. Quantizing those
ops rounds coordinates.

*Fix.* NNCF post-training quantization (MIXED preset: symmetric weights, asymmetric
activations, which suits SiLU's skewed outputs) with the decoding ops in an ignored scope
(28 ops); the head's convolution branches are still quantized. 128 COCO128 calibration images go
through the same letterbox as inference.

*Measurement* (paired bootstrap, `bench/results/eval/coco_classes.json`):

| model | INT8, decoder in FP32 | the same with the BF16 hint | INT8, decoder quantized (ablation) |
|---|---|---|---|
| YOLO11n | ΔmAP50-95 −0.0024 [−0.014, +0.008] | −0.0030 [−0.019, +0.011] | −0.0633 [−0.081, −0.044] |
| YOLO11s | +0.0001 [−0.005, +0.007] | +0.0014 [−0.005, +0.010] | −0.0723 [−0.086, −0.056] |

mAP50 drops much less than mAP50-95 in the ablation (YOLO11n 0.814 → 0.786), i.e. the damage is
localization precision, which matches the explanation. (The first round's values, in
`bench/results/archive/v1-0c6c533/eval/`, differ by less than 0.0005.)

*Trade-off.* The FP32 decoder costs a few dequantize/quantize transitions at the end of the
network; the latency effect is part of the INT8 numbers in the benchmark.

## 4. Evaluation

### 4.1 A pycocotools-equivalent implementation, tested against pycocotools

*Problem.* A 95% bootstrap interval needs ~1000 re-evaluations per run; re-running pycocotools
(matching included) each time is slow, and 12+ runs are compared.

*Fix.* `tsd.eval.metrics` caches the COCO matching per (image, class, size bucket) once; a
bootstrap replicate only concatenates cached rows and re-accumulates. One image resample per
replicate is shared by all runs, so backend differences get **paired** intervals.

*Measurement.* Agreement with `pycocotools.COCOeval`: largest difference 2.2e-16 on 200
random scenarios (crowd regions, all size buckets, duplicates, false positives;
`python tests/test_eval_metrics.py 200`, output in
`bench/results/eval/pycocotools_agreement_200.txt`; the test suite runs 6 of them) and 3.3e-16
over all 52 committed real runs (each report stores its runs' `pycocotools.max_abs_diff`; the
largest is YOLO11s tiled in `tiling_merge_iou_0.5.json`). The randomized test caught one real
discrepancy: rounding the IoU/recall thresholds to two decimals. pycocotools uses raw
`np.linspace` values (its 0.70 recall point is 0.7000000000000001), which changes AP when a
class's recall is exactly 7/10.

### 4.2 Open Images is not exhaustively labelled

*Problem.* Counting every detection of a class nobody verified in an image as a false positive
penalizes correct detections of unlabelled objects.

*Fix.* The Open Images rule: detections of a class neither verified present nor absent in an
image (and without boxes there) are dropped; verified-absent classes still count as false
positives. A strict variant (`--count-unverified`) is reported alongside: it lowers YOLO11n
mAP50-95 from 0.669 to 0.663 and YOLO11s from 0.702 to 0.697, and leaves the backend comparison
unchanged within its intervals.

### 4.3 NMS restricted to the evaluated classes

*Problem.* Ultralytics' validation NMS keeps at most 300 detections per image across all 80
COCO classes, so detections of irrelevant classes could crowd out stop signs and traffic lights.

*Fix.* Evaluation runs the same NMS settings (confidence 0.001, IoU 0.7, multi-label, 300) with
the candidates restricted to the evaluated classes, i.e. the detector restricted to those
classes. Detections are saved (boxes to 0.001 px, scores to 1e-6) and every score is computed
from the saved file; a first version scored in-memory detections on fresh runs and saved ones
on reruns, which differed in the 6th decimal.

*Committed detections*. The first round kept the detections
outside the repository and reused a saved file whenever the model files matched, even if the
NMS or tiling settings had changed. Now every run's detections are committed as gzipped JSON
(`bench/results/eval/detections/`, 0.10-0.36 MB per run; byte-identical for identical
detections), the file name carries a hash of the NMS, tiling and precision-hint settings, a
saved file is reused only if model files, settings and image ids all match, and each report
records the file and SHA-256 it scored. *Bug found while re-running the evaluation:* the reuse
check compared the settings read back from JSON (lists) with the live values (tuples), so
nothing was ever reused and the strict rerun silently rewrote the files the main report had
just recorded. Fixed (`can_reuse`, tested); the strict variant now reuses all 14 runs.

### 4.4 The evaluation set without its verified negatives (selection effect)

*Problem* (found when I re-checked my results). The set is selected by box presence (section 1.2), so the
images Open Images verified to contain no stop sign or no traffic light are left out unless
they happen to have another target box: 49 of 170 stop-sign negatives and 6 of 61
traffic-light negatives are in it. On the left-out images every detection of those classes is
a false positive, so the main set's absolute AP is optimistic relative to the Open Images
protocol, which scores every verified image.

*Fix.* `tsd openimages --include-verified-negatives` builds a second variant with them: 380
images, the 219 plus 161 box-less ones (121 verified without a stop sign, 55 without a traffic
light, 15 both; my first count, "176", counts those 15 twice). The 219 existing images were
hard-linked into the new directory and the 161 new ones (43.2 MB) downloaded with the same
limits (one connection, 2 MB/s cap, 1.85 MB/s achieved; log in
`bench/results/network/openimages_negatives_download.log`). The builder also drops a box-less
image that has a positive verified label for a data-set class (an object without a box could
never be matched); no candidate here has one (`bench/results/network/negative_candidates.txt`).

*Measurement* (`bench/results/eval/coco_classes_with_negatives.json`, 12 runs; scored from
`bench/results/eval/detections_with_negatives/`): mAP50-95 drops from 0.669 to 0.653 for
YOLO11n PyTorch FP32 and from 0.702 to 0.697 for YOLO11s (mAP50 0.814 to 0.793 and 0.848 to
0.841); per class, YOLO11n's Stop sign AP50-95 goes from 0.958 to 0.932 and Traffic light from
0.380 to 0.373. Every runtime's paired difference still has an interval that contains zero;
the INT8 point estimates move inside their intervals (YOLO11n -0.0024 to +0.0047
[-0.011, +0.024], YOLO11s +0.0001 to -0.0005 [-0.005, +0.006]). The conclusions about the
runtimes do not depend on the variant; the 219-image numbers stay the headline because the
earlier reports and the tiling study use them, and both are reported.

## 5. CPU benchmarks (`bench/run_benchmarks.py`, `tsd.benchmark`)

### 5.1 What is timed

One call = letterbox + inference + NumPy NMS + box rescaling for an image that is already
decoded in memory (JPEG decoding is excluded: it is identical for every backend). The
inference-only time is recorded next to it. 16 evaluation images are used round-robin; 20
warm-up calls, then 200 timed calls (batch 8: 3 + 30). Every configuration runs in a fresh
process with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and the runtime's own thread setting
(`torch.set_num_threads`, ONNX Runtime `intra_op_num_threads`, OpenVINO
`INFERENCE_NUM_THREADS`) equal to the thread count, and OpenCV limited to the same number. The
configurations: 1 and 4 threads at batch 1, batch 8 at 4 threads; plus sliced inference
(section 7). Model size is the size of the files the runtime loads (`.pt`; `.onnx`; `.xml` +
`.bin`).

*What the baseline is.* Every runtime, PyTorch included, gets the same fixed 640 × 640
letterbox, so all of them do the same work. The PyTorch baseline is the fused Ultralytics
`DetectionModel` in eager mode, the network Ultralytics' `predict` runs, but not its pipeline:
`predict` pads `.pt` inputs only to the smallest stride-32 rectangle, 73.4% of the square's
pixels on average on the 16 benchmark images (50-85%; `bench/results/letterbox_rect_share.txt`),
so it would be somewhat faster than this baseline. Speedups here are "same network, same input".

*Recorded settings and thread pinning.* Every raw result embeds the git commit and dirty flag,
the exact command lines (the matrix runner's as `TSD_PARENT_COMMAND`), the thread variables and
the runtime's *effective* settings read back after compilation (`provenance.runtime_settings`:
for OpenVINO `INFERENCE_NUM_THREADS`, `INFERENCE_PRECISION_HINT`, `PERFORMANCE_HINT`,
`NUM_STREAMS`, `ENABLE_CPU_PINNING`,...). OpenVINO pins its inference threads to cores by
default; PyTorch and ONNX Runtime do not. The `-nopin` rows (OpenVINO FP32 and INT8+BF16 at
4 threads, batch 1) measure OpenVINO with pinning off, so the comparison can be read either way.

*Precision labels.* `openvino-fp32` and `openvino-int8` run with the `f32` precision hint (the
non-quantized ops in FP32); `openvino-bf16` is the FP32 IR with the `bf16` hint;
`openvino-int8-bf16` is the INT8 IR with the `bf16` hint, i.e. the non-quantized ops in BF16,
which is what OpenVINO does by default on this AMX CPU when no hint is given.

### 5.2 A shared machine: measuring contamination instead of hoping

*Problem.* The 4 vCPUs are shared with other tenants (other agents' builds, tests and
benchmarks, a database). A benchmark that only reads the clock cannot tell a slow backend from
a busy machine. (The first pass of the matrix, before the fix below, was discarded without
being kept, so its numbers are not quoted here.)

*Fix.* (a) Every timed loop records the CPU time used by **other** processes: machine-wide busy
time from `/proc/stat` minus the benchmark process's own CPU time (`os.times()`, all threads),
divided by the wall time, in cores (steal time is recorded separately). (b) `bench_one.py`
checks for 3 s *before* loading the model and exits with status 75 if the machine is busy, so
the lock is released while waiting. (c) A result is accepted only if the other processes used
at most max(0.3, (CPUs − threads) / 2) cores (0.3 cores for 4-thread runs, 1.5 for 1-thread
runs); otherwise it is kept as `<id>.run<k>.attempt<j>.json` (never overwritten) and
re-measured after 30 s, up to `--max-attempts` times (30 for the matrix, 20 for the profiler,
10 for the serving comparison), after which the least contaminated attempt is used and
flagged with `*` in the tables (`tsd.benchmark.measure_with_retries`, shared by the matrix, the
profiler and the serving comparison). Each attempt holds `/home/user/work/bench.lock` only for
its own configuration: the timed loop is capped by `--max-seconds`, and the longest matrix
configurations (PyTorch YOLO11s at 1 thread, 20 + 200 calls) spend at most 68.3 s in warm-up
and timed calls (`seconds` in each raw file), plus the 3 s quiet check, imports and model
loading while holding the lock.

*Trade-off.* The metric sees CPU time only. Other tenants' use of memory bandwidth, shared caches
or turbo headroom on the same host is invisible to `/proc/stat`, so a clean result is not proof
of an undisturbed one; section 5.3 measures how much clean runs still differ.

### 5.3 Run-to-run spread, and why every number is a median of three rounds

*Problem* (found when I re-checked my results). The first round measured each configuration once and repeated
only six YOLO11n configurations; the headline speedups came from run 1, whose PyTorch baseline
was the slowest of its three runs (134.1 vs 119.0 and 110.3 ms at 1 thread), and the YOLO11s
"4.00x" was never repeated (`bench/results/archive/v1-0c6c533/benchmarks/repeats.md`).

*Fix.* `bench/run_benchmarks.py --runs 3` measures the whole matrix three times in rounds
(round 1 = every configuration once, then round 2,...), so slow drifts of the shared machine
spread over all configurations. Tables report the median over the rounds with the range of the
per-run p50, and a speedup is the median of the three *paired* per-round ratios (run *k* of a
configuration against run *k* of PyTorch at the same setting), with their range; the ratio of
the medians is stored next to it (`benchmarks.json`, `ratio_of_medians`).

*Measurement* (`bench/results/benchmarks/benchmarks.json`, 126 runs, all at commit cfff803 from
a clean tree, all accepted by the contamination check; 14 rejected attempts kept as
`*.attempt<k>.json`; the statistics below are `run_to_run` in `benchmarks.json`). The spread
of a configuration's p50 over its three runs, (max - min) / median, has a median of 6.2% over
the 42 configurations and is above 10% for 8 of them, up to
30% (YOLO11s BF16 at batch 8: 252.2, 193.5 and 194.0 ms per batch); clean contamination numbers
do not rule such outliers out (section 5.2's trade-off). The fastest pinned runtime was
OpenVINO BF16 in 16 of the 18 (model, setting, round) cases; in round 1 of YOLO11s at 4 threads
(batch 1 and batch 8) OpenVINO INT8 was faster (`fastest_per_round`).

### 5.4 Results

Source: `bench/results/benchmarks/benchmarks.md`. p50 end-to-end latency at 4 threads, batch 1,
median of three rounds (paired speedup vs PyTorch FP32, median and range over the rounds;
AMX/VNNI CPU, so the INT8 and BF16 gains are larger than on a CPU without those units):

| runtime | YOLO11n | YOLO11s |
|---|---|---|
| PyTorch FP32 | 66.0 ms | 121.9 ms |
| ONNX Runtime FP32 | 40.8 ms (1.64x, 1.60-1.67) | 78.1 ms (1.55x, 1.48-1.57) |
| OpenVINO FP32 | 33.7 ms (1.87x, 1.77-2.13) | 68.7 ms (1.68x, 1.67-1.78) |
| OpenVINO BF16 (FP32 IR, bf16 hint) | **22.3 ms (2.84x, 2.83-3.13)** | **30.6 ms (3.85x, 3.19-3.98)** |
| OpenVINO INT8 (f32 hint) | 25.7 ms (2.58x, 2.34-2.65) | 37.1 ms (3.29x, 3.10-3.43) |
| OpenVINO INT8 + BF16 hint | 27.4 ms (2.49x, 2.26-2.50) | 40.1 ms (2.94x, 2.89-3.11) |

* At 4 threads the FP32 exports alone are 1.5-1.9x faster than eager PyTorch with identical
  detections (section 3.1): a large part of the speedup is graph optimisation, not precision.
* **BF16 is the fastest configuration on this CPU** at every setting by the medians (YOLO11n
  batch 8: 61.2 images/s against 53.5 for INT8 and 51.5 for INT8 + BF16; 1 thread: 44.4 against
  53.5 and 54.4 ms), although INT8 halves the bytes per weight and activation again.
* **Where the time goes, per the profile** (`bench/results/benchmarks/profile_yolo11{n,s}.md`;
  OpenVINO `PERF_COUNT`, batch 1, 4 threads, one session of 100 inferences per configuration,
  median per-node times in ms per inference over the 15 most expensive node types):

  | | YOLO11n BF16 | INT8 | INT8 + BF16 hint | YOLO11s BF16 | INT8 | INT8 + BF16 hint |
  |---|---|---|---|---|---|---|
  | Convolution | 9.06 | 11.07 | 11.36 | 19.72 | 20.93 | 19.15 |
  | FakeQuantize | - | 1.08 | 0.99 | - | 2.06 | 1.08 |
  | Reorder | 0.82 | 1.18 | 3.12 | 0.87 | 1.25 | 4.11 |
  | all other types | 3.16 | 3.09 | 2.94 | 4.98 | 5.04 | 4.12 |
  | sum of node times | 13.06 | 16.42 | 18.45 | 25.52 | 29.28 | 28.44 |
  | benchmark p50 (end to end) | 22.3 | 25.7 | 27.4 | 30.6 | 37.1 | 40.1 |

* **Why INT8 does not beat BF16** (by the profile): INT8's extra node time over BF16 (YOLO11n
  +3.4 ms, YOLO11s +3.8 ms) is mostly convolution (+2.0 and +1.2 ms: the AMX INT8 kernels,
  `brgconv_avx512_amx*_I8`, are not faster than the BF16 ones for these models) and the
  FakeQuantize nodes (+1.1 and +2.1 ms); Reorder adds 0.4 ms and all other node types together
  cost the same within 0.1 ms. As a description of the graphs: the INT8 IR executes 54 of its
  178 nodes (YOLO11n; YOLO11s 57 of 182) in FP32 kernels, among them the box decoder kept in
  floating point on purpose (section 3.2), and those FP32 nodes take 3.8 and 5.8 ms; the
  profile does not show that they cause the gap.
* **The BF16 hint does not help the INT8 model.** With the hint, OpenVINO runs the INT8 IR's
  non-quantized ops in BF16 (its default on this CPU when no hint is given), and it got slower:
  27.4 vs 25.7 ms (YOLO11n) and 40.1 vs 37.1 ms (YOLO11s) at 4 threads, about equal at 1
  thread. With the hint, 44 (YOLO11n) and 43 (YOLO11s) nodes run in BF16 kernels and 34
  instead of 54 and 57 in FP32 ones, and the graph grows from 178 to 227 executed nodes
  (YOLO11n; YOLO11s 182 to 225). For YOLO11n the profile accounts for the slowdown: Reorder
  time (the precision changes between the BF16 and INT8 parts) rises from 1.18 to 3.12 ms, and
  the node times sum to 2.0 ms more. For YOLO11s it does not: Reorder rises from 1.25 to
  4.11 ms, but convolution (-1.8 ms), FakeQuantize (-1.0 ms) and the other node types (-0.9 ms)
  fall by more, and the node times sum to 0.8 ms *less* (28.4 vs 29.3 ms) while the benchmark is
  3.0 ms slower. With one profiling session per configuration, that discrepancy is unexplained
  (per-node times need not add up to the request's wall time, which is recorded next to them).
* **Thread pinning is not the reason OpenVINO is faster**: with OpenVINO's pinning switched off
  the 4-thread p50 stayed within the run-to-run range (FP32: YOLO11n 34.0 vs 33.7 ms, YOLO11s
  69.0 vs 68.7 ms; INT8 + BF16: 27.9 vs 27.4 and 39.0 vs 40.1 ms).
* Batching pays off more for the low-precision runtimes: at 4 threads, batch 8 raises YOLO11n
  throughput over batch 1 by 36% for BF16 (44.9 -> 61.2 images/s, medians), 38% for INT8 (38.8
  -> 53.5) and 41% for INT8 + BF16, against 15% for OpenVINO FP32 and 22% for PyTorch; for
  YOLO11s FP32 batching gains nothing (OpenVINO -1%, PyTorch -10%) while INT8 gains 33%. This is
  what the service's micro-batching exploits.

*Decision: the service defaults to the INT8 model with the f32 hint, not BF16.* BF16's lower
latency here (22.3 vs 25.7 ms for YOLO11n) depends on native BF16 units (AMX or AVX512-BF16),
which many laptop and older server CPUs lack, while the INT8 kernels need only VNNI, which is
more widely available. INT8 is also 3.1x smaller than the FP32 IR (3.5 vs 10.9 MB for YOLO11n)
and loses no measurable accuracy (section 3.2). The service pins the f32 hint because the
measurement above shows that OpenVINO's own default for the INT8 model on this CPU (BF16 for
the non-quantized ops) is slower. On an AMX host, `TSD_MODEL_PATH=<FP32 IR>` with
`TSD_OPENVINO_PRECISION=bf16` selects the fastest variant measured here.

## 6. HTTP service (`tsd.serve`)

### 6.1 Design

*Problem.* A detector behind HTTP must not block the event loop, should use batching when
requests arrive together, and must degrade predictably when overloaded.

*Fix.* The model is loaded and warmed up (every batch size up to the maximum, section 6.3) in
the FastAPI lifespan before `/healthz` reports ready. Every `/v1/detect` request first passes a
pure ASGI gate, **before its body is read**: a missing `Content-Length` gets 411, one above
`TSD_MAX_UPLOAD_BYTES` 413, and admission control (`TSD_MAX_INFLIGHT`, by default queue size +
batch size + pre-processing threads = 74) answers 429 with `Retry-After: 1` once that many
requests are already being received, decoded, queued or run. Admitted requests: the image size
is read from the file header with Pillow before decoding (413 above `TSD_MAX_IMAGE_PIXELS`,
Pillow's decompression-bomb error included; 400 for images under 8 px on a side), and decoding
+ letterboxing run on a small thread pool (`TSD_PREPROCESS_THREADS`). The prepared image goes
into an `asyncio.Queue` (`TSD_QUEUE_SIZE`, 64; full: 429). One worker task collects up to
`TSD_MAX_BATCH_SIZE` (8) images and runs the batch on a single dedicated inference thread (one
model instance; NMS included). With `TSD_BATCH_WAIT=adaptive` (default) it waits up to
`TSD_MAX_WAIT_MS` (5 ms) for more images only while other requests are already in the queue
(a request still being decoded does not count), so a lone request is never delayed; `fixed`
always waits, counted from the first image's arrival in the queue. A client that disconnects
while waiting is detected, its request is dropped from the batch if it has not started, and it
is logged with status 499.
`/metrics` (own Prometheus registry) exposes request counts and durations by status as seen by
the gate (upload and parsing included), the handler time, decode, queue wait, inference, batch
sizes, queue depth, rejections by reason (`admission`, `queue_full`) and client disconnects.
Settings are `pydantic-settings` fields with the `TSD_` prefix.

*Bug I found when re-checking: 429 came too late.* The first version checked the queue only after
Starlette had read and parsed the whole multipart upload and the image had been decoded, so an
overload still cost a full upload and decode per rejected request, and the decode pool's
backlog was unbounded (the queue bounded only decoded images). The gate now rejects before
reading: `tests/test_serve.py::test_admission_rejects_before_reading_or_decoding` checks that a
request beyond the limit gets 429 before a byte of its upload is read or anything is decoded,
and `test_decode_bound_overload_returns_429` floods a service whose inference is blocked.

*Trade-off.* One inference thread means one batch at a time: throughput is bounded by the
model's batch throughput at `TSD_THREADS` threads, and concurrency beyond the batch size only
adds queueing. A second model instance would need its own threads on the same 4 vCPUs.

### 6.2 Bug found by a test: requests hung on shutdown

The first batcher's `stop()` cancelled the worker and failed the requests still in the queue,
but not the batch the worker had already taken from the queue (being collected or running).
Those requests' futures were never resolved, so their HTTP handlers waited forever. The test
that shuts the batcher down with a batch in flight hung. Fix: the worker records the batch it
holds (`_inflight`) and `stop()` fails those futures too; the test asserts that every
submitted request finishes (with a result or an error) within a timeout.

### 6.3 Measurement: micro-batching off vs on (`bench/results/serving/serving.md`)

Setup (`bench/serve_compare.py`): `tsd serve` with OpenVINO INT8 YOLO11n, 2 inference threads
and 1 pre-processing thread, pinned to CPUs 0-2; the closed-loop load tester pinned to CPU 3;
16 evaluation JPEGs; 20 s per concurrency level after 3 s of warm-up. Three settings: `off`
(batch size 1), `fixed` (up to 8 images, always waits up to 5 ms after the first request
arrived) and `adaptive` (the same, but no wait when nothing else is queued; the default). One
session = one server lifetime for one setting, all four levels, under the lock; sessions run
in rounds, 3 runs per level, every level accepted only below 0.3 cores of other load (rejected
attempts are kept as `*.attempt<k>.json`: 5 at commit cfff803, 12 at c6e3f1b). No request
failed in any accepted run.

*First repeated measurement (commit cfff803, `bench/results/archive/serving-cfff803/`).* The
review had found that the first round measured each level once and quoted the best level only
(claims-3). Repeated, the picture at concurrency 1, 4 and 16 held, but at concurrency 8 both
batching settings had a far worse p99 than no batching: 554.7 ms (range 488.6-558.2, adaptive)
and 592.7 ms (536.4-890.5, fixed) against 337.8 ms, although their p50 was lower (224.8 and
231.2 vs 277.6 ms). The per-request breakdown of the slowest 1% showed where: inference of
their batch took 283-538 ms at mean batch sizes of 4.1-7.0, while full batches of 8 at
concurrency 16 took 217-287 ms. A larger batch was faster than a smaller one, so the time was
not the batch's work.

*Cause.* The service warmed up batch sizes 1 and 8 only. At concurrency 8 the batcher forms
batches of 2-7, and the first inference at a shape the runtime has not run yet is slow.
`bench/batch_shape_probe.py` measured it directly (same model, threads and CPUs as the
service, under the lock; `bench/results/serving/batch_shape_probe.json`): with sizes 1 and 8
warmed, the first call at sizes 2-7 took 354-494 ms against 60-184 ms for the next five
(2.7-5.9x); on a fresh detector warmed at every size from 1 to 8, the first call was 0.95-1.05x
the repeats. *Fix* (commit c6e3f1b): the start-up warms every batch size up to the maximum
(8 sizes x 3 calls). Its cost is memory: OpenVINO keeps memory for each new shape, and in the
container the service used 953.4 MiB after the smoke test's requests, against 534.8 MiB for the
first image, which warmed only sizes 1 and 8 (section 8).

*Measurement after the fix* (commit c6e3f1b), median (range) over 3 runs:

| concurrency | setting | req/s | p50 ms | p99 ms | mean batch |
|---|---|---|---|---|---|
| 1 | off | 22.5 (22.3-23.3) | 42.9 (41.6-43.0) | 61.9 (61.6-64.9) | 1.00 |
| 1 | fixed | 20.1 (19.1-20.5) | 48.4 (47.8-50.8) | 64.5 (62.8-70.5) | 1.00 |
| 1 | adaptive | 21.8 (20.2-23.1) | 44.2 (41.5-48.5) | 63.9 (61.4-72.0) | 1.00 |
| 4 | off | 28.0 (26.7-28.9) | 139.9 (135.5-146.0) | 177.4 (175.2-207.3) | 1.00 |
| 4 | fixed | 30.6 (28.8-30.7) | 129.5 (128.7-133.6) | 161.0 (157.4-213.6) | 2.00 |
| 4 | adaptive | 30.7 (30.0-31.1) | 128.4 (126.3-131.1) | 160.2 (157.3-177.9) | 2.00 |
| 8 | off | 27.7 (25.4-27.8) | 284.2 (283.7-308.1) | 360.1 (339.6-408.2) | 1.00 |
| 8 | fixed | 33.9 (31.0-33.9) | 234.6 (233.9-255.8) | 292.2 (272.0-323.6) | 3.98 |
| 8 | adaptive | 33.7 (33.1-35.3) | 234.0 (224.5-238.7) | 290.7 (282.4-305.7) | 3.98 |
| 16 | off | 27.6 (26.5-27.9) | 576.3 (567.2-593.2) | 656.1 (646.6-731.3) | 1.00 |
| 16 | fixed | 37.4 (35.4-37.7) | 424.1 (417.2-441.8) | 504.1 (494.3-555.8) | 7.93 |
| 16 | adaptive | 36.9 (36.4-37.5) | 426.1 (420.6-439.6) | 517.9 (494.2-561.8) | 7.91 |

* The concurrency-8 p99 with batching dropped from 554.7 to 290.7 ms (adaptive) and from 592.7
  to 292.2 ms (fixed), now below the 360.1 ms without batching; the slowest 1% spend 130-200 ms
  in inference at mean batch sizes of 4.0-5.6. At the other levels the two commits agree within
  their ranges, as expected from a change that only affects the first call at each size.
* Without batching the service saturates at about 28 requests/s from concurrency 4 on, and in a
  closed loop the latency then grows in proportion to the concurrency. With batching,
  throughput keeps rising with the batch size: +34% at concurrency 16 (adaptive), with p50 and
  p99 26% and 21% lower.
* The fixed wait costs 5.5 ms of p50 and 11% of throughput at concurrency 1 (5.1 ms at commit
  cfff803), where every request waits for company that never comes. The adaptive wait does not
  wait there (mean queue wait 0.05-0.06 ms per run, as with `off`); its p50 was 1.3 ms (here)
  and 0.6 ms (cfff803) above `off`, inside the run-to-run ranges, and here its inference time
  was 1.3 ms higher too (median of the per-run means 35.7 vs 34.4 ms; `breakdown_ms` in the raw
  files). From concurrency 4 on, adaptive and fixed are within each other's ranges.

*Trade-off.* The adaptive rule only looks at the queue when a batch starts: two requests that
arrive 1 ms apart at low load are run as two batches. That is the intended behaviour at low
load (latency first); at high load the queue is never empty and the rule behaves like the fixed
wait. Warming every size costs start-up time (24 calls instead of 6 at batch sizes up to 8) and
does not cover shapes other than the batch size (the input is always letterboxed to 640 x 640).

## 7. Sliced inference for small objects (`tsd.infer.tiling`)

### 7.1 Design

*Problem.* Small objects are where the models fail: YOLO11n's AP50-95 on small traffic lights
(≤ 32² px) is 0.17 against 0.74 on large objects (section 4, `bench/results/eval/coco_classes.json`).
The detector sees a 640 × 640 letterbox of the whole image, so an object shrinks by the
image's long side / 640 (1.6x for a 1024 px image).

*Fix.* SAHI-style slicing: overlapping 512 px crops of the original image (20% overlap, tile
positions spread evenly so the last tile ends at the border), each letterboxed to 640 (so
objects are enlarged by 1.25x), plus the usual full-image pass so that large objects no tile
contains whole are still found; boxes are shifted back to image coordinates and merged by
class-wise NMS at the detector's own NMS IoU (`TilingConfig.merge_iou=None`; 0.7 in
evaluation); batches of 8 views. Images whose long side is at most 512 px get no tiles.
`TiledDetector` has the same `predict` interface, so the evaluator, the benchmark and the CLI
(`model@tiles=512`) use it unchanged.

*Bug I found when re-checking: the tiling ablation changed two things.* The first versions merged at
a fixed IoU 0.5 while the plain detector's NMS used 0.7 (the evaluation setting), so "tiled vs
plain" also compared two NMS thresholds: a stricter merge suppresses more neighbouring boxes
even when the tiles add nothing. `tests/test_tiling.py::test_merge_nms_uses_the_detectors_iou_unless_set`
checks that the merge adds no suppression of its own (two boxes at IoU 0.54 survive a 0.7
detector, tiled or not), and `tsd eval --tile-merge-iou` reruns the old 0.5 merge as a control
(section 7.4).

### 7.2 Bug: the class offset of batched NMS

Class-wise NMS is implemented the usual way: every box is shifted by `class_id x offset` so
that boxes of different classes cannot overlap, then one NMS runs. The offset was a constant
7,680 px (enough for a 640 px letterbox). Tiling runs NMS in original-image coordinates, where
boxes can exceed 7,680 px; a class-1 box shifted by 7,680 could then overlap a class-0 box and
suppress it. (A collision needs coordinates above 7,680 px in *both* axes, so the 8,100 x 3,888
px image in the evaluation set could not trigger it, but nothing limited the input size.) Fix:
the offset is max(7,680, largest absolute coordinate + 1), and a test builds the collision.

### 7.3 First measurement: tiling made YOLO11n much worse

*Measurement* (first round:
`bench/results/archive/v1-0c6c533/eval/tiling_v1_no_border_filter.json`; re-run with committed
detections: `bench/results/eval/tiling_no_border_filter.json`; plain SAHI-style merging at
IoU 0.5):
YOLO11n mAP50-95 fell from 0.669 to 0.355 (paired Δ −0.314 [−0.358, −0.243]). By size bucket the
picture was split: small objects improved (AP50-95 0.174 → 0.244, Δ +0.070 [+0.017, +0.110];
AR@100 0.34 → 0.41), medium ones were unchanged, and large ones collapsed (0.738 → 0.369).
Stop signs, all of them large, went from AP50-95 0.958 to 0.370 with AP75 0.98 → 0.32 and
*recall* 0.98 → 0.77: correct full-image boxes had disappeared. YOLO11s behaved the same way
(0.702 → 0.446).

*Diagnosis.* A tile that cuts through a large stop sign still "detects" the visible part, with
a score as high as the full-image detection, and that truncated box overlaps the complete box
with an IoU above the 0.5 merge threshold. NMS keeps one of them: often the truncated one, so
the image loses its correctly localized box (the recall and AP75 drop); when the overlap is
below 0.5, the truncated box survives as an extra false positive.
`tests/test_tiling.py::test_truncated_boxes_at_inner_tile_borders_are_dropped` reproduces it
with a synthetic detector.

*Fix.* Drop a tile detection that reaches a tile border lying inside the image (within 2% of
the tile size, 10 px for 512 px tiles); borders on the image edge do not count. Neighbouring
tiles overlap by at least 102 px (20%), so an object narrower than the overlap minus twice the
margin (about 80 px) lies inside some tile and clear of its inner borders, and is detected
there without truncation; larger objects come from the full-image pass.

*Refinement after re-checking.* Dropping every border-touching box can lose an object completely
when the full-image pass misses it too (an object larger than the overlap that is too small in
the full-image letterbox). A border-touching tile box is therefore dropped only if a same-class
box that does not touch an inner border (another tile's, or the full image's) covers at least
half of it (intersection over the tile box's area, `rescue_ioa=0.5`); otherwise it is kept as
the only evidence of that object
(`tests/test_tiling.py::test_object_seen_only_by_tiles_survives_the_border_filter`: a 140 px
object in an 8,192 px wide image, invisible to the full-image pass, in the overlap band of two
tiles).

### 7.4 Measurement (`bench/results/eval/tiling.json`)

512 px tiles, 20% overlap, merged at the detector's NMS IoU (0.7), conditional border filter;
paired against plain inference with the same OpenVINO FP32 model:

| | YOLO11n plain -> tiled | Δ [95% CI] | YOLO11s plain -> tiled | Δ [95% CI] |
|---|---|---|---|---|
| mAP50-95, all | 0.669 -> 0.684 | +0.015 [+0.002, +0.029] | 0.702 -> 0.711 | +0.010 [+0.000, +0.020] |
| AP50-95, small (≤ 32²) | 0.175 -> 0.262 | +0.087 [+0.035, +0.131] | 0.233 -> 0.314 | +0.080 [+0.040, +0.124] |
| AR@100, small | 0.345 -> 0.455 | +0.110 | 0.390 -> 0.470 | +0.080 |
| AP50-95, medium | 0.487 -> 0.529 | +0.041 [+0.011, +0.073] | 0.548 -> 0.568 | +0.019 [-0.006, +0.044] |
| AR@100, medium | 0.637 -> 0.700 | +0.063 | 0.684 -> 0.709 | +0.025 |
| AP50-95, large | 0.738 -> 0.721 | -0.017 [-0.034, +0.002] | 0.775 -> 0.765 | -0.0095 [-0.022, +0.003] |
| AR@100, large | 0.896 -> 0.894 | -0.002 | 0.881 -> 0.876 | -0.005 |

(Small and medium objects are all traffic lights; the AR deltas have no interval because the
bootstrap covers AP only.) Small-object AP50-95 rises by 50% (YOLO11n) and 34% (YOLO11s) and
the overall mAP improves; the loss on large objects is small and its interval now includes
zero.

*Old-merge control* (`tiling_merge_iou_0.5.json`, the same with the first round's merge IoU
0.5, while the plain run keeps its NMS at 0.7): the stricter merge alone costs recall that the
tiles had found. YOLO11n: overall Δ +0.013
[+0.000, +0.027] instead of +0.015, AR@100 small +0.074 instead of +0.110, AR@100 large -0.022
instead of -0.002, AP50-95 large -0.019 [-0.038, +0.001]; YOLO11s: +0.0084 [-0.0015, +0.0195],
AP50-95 large -0.014 [-0.028, -0.001]. So the first round's tiling numbers (merge 0.5,
`archive/v1-0c6c533/eval/tiling.json`) understated the recall gain and overstated the
large-object loss; the small-object AP gain (+0.084 to +0.087 for YOLO11n) does not depend on
the merge IoU. The first version's settings (no border filter, merge 0.5;
`tiling_no_border_filter.json`) reproduce section 7.3's mAP50-95 values with the current code
to within 0.0002 (YOLO11n 0.669 -> 0.355, Δ -0.314 [-0.358, -0.243], the same number of scored
detections; YOLO11s 0.702 -> 0.446, 9304 scored detections against the first round's 9299),
now with committed detections; the mAP50 values differ from the first round's by up to
0.0013.

*Latency cost* (`bench/results/benchmarks/benchmarks.md`, 4 threads, one image per call, median
of three rounds, the benchmark's NMS (confidence 0.25, IoU 0.5), 16 benchmark images with 6.8
inferences each on average): YOLO11n 203.0 ms p50 against 33.7 ms plain (6.0x; per round
5.1-7.0x), YOLO11s 455.6 ms against 68.7 ms (6.6x; 6.1-6.9x), from `slowdown_vs_untiled` in
`benchmarks.json`. The cost grows with the
image size (an 8,100 x 3,888 px evaluation image needs 201 inferences), so tiling is an opt-in
mode (`TiledDetector`, `model@tiles=512` in `tsd eval`), not the service default.

## 8. Docker image

*Problem.* The development environment is 6.3 GiB because of the CUDA-linked PyTorch wheel
(section 2.1); the service needs none of it. The build machine also sits behind a
TLS-intercepting proxy whose CA certificate must not end up in an image layer.

*Fix.* Two stages on `python:3.12-slim-bookworm`, pinned by digest: the build stage installs
`uv` and runs `uv sync --locked --no-dev` with only the `openvino` and `serve` extras, first
the dependencies alone from bind-mounted `pyproject.toml` and `uv.lock` (so a README or source
edit does not invalidate that layer; the uv download cache is a cache mount), then the project
itself; the runtime stage copies only
the virtual environment, runs as an unprivileged user (uid 10001), and has a `HEALTHCHECK` on
`/healthz`. The proxy CA is a BuildKit secret (`--mount=type=secret,id=ca_bundle,required=false`),
which exists only during the `RUN` steps that need it; without the secret the build is a plain
public build (CI). The model is not baked in: compose mounts `./models` read-only, publishes the
port on 127.0.0.1 only, and runs the container with a read-only root file system, a `/tmp`
tmpfs, all capabilities dropped, `no-new-privileges`, 2 CPUs and 2 GB of memory (1 GB at first:
see the measurement below).

`nice -n 19 docker build` only lowers the priority of the Docker client: BuildKit runs the
steps under the shared `dockerd`, so the build's own parallelism is capped in the Dockerfile
(`UV_CONCURRENT_DOWNLOADS=1`, one package download at a time; `UV_CONCURRENT_INSTALLS=2`), and
the build runs under the benchmark lock like the timing runs.

*Measurement* (`bench/results/docker/`). The image was rebuilt from a clean worktree at commit
a45e982 (`build.txt`: the commit, an empty `git status --porcelain`, exit status 0 after 38 s;
`build.log` shows the packages downloaded one at a time), under the benchmark lock, and
smoke-tested under the lock in separate holds (first with 1 GiB, below, then with 2 GiB). What
goes into the image (`.dockerignore`: `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`,
`src/`) is the same at every later commit except `README.md`. Image ID
`sha256:9a02567e0be4...`; 201 MB compressed (its 7 gzip layers in `docker save`: 201,179,084
bytes; gzip does not shrink the saved tar further, 199,929,753 bytes, because this Docker's
containerd image store saves the layers compressed) and 595 MB
uncompressed (595,143,680 bytes decompressed; the virtual environment layer is 467 MB of it,
`docker history` gives 474 MB for it). Inside it, `du -sm` puts the environment at 452 MiB:
OpenVINO 178 MiB, OpenCV (`cv2` and its bundled libraries) 172 MiB and NumPy 58 MiB, against
6.3 GiB for the development environment.

Smoke test (`smoke_test.txt`, the compose settings on the command line): the CI job's two
checks (`tsd --help` and the service imports) pass; the service becomes healthy, runs as uid
10001 with a read-only root file system (writing `/app` fails, `/tmp` is a writable tmpfs);
`/healthz` reports `max_inflight` 74, the adaptive wait and the runtime settings; the stop sign
in a validation image is detected (score 0.957); a 1,133,053-byte upload, which Starlette
spools to the tmpfs, gets 200; a non-image upload 400; `bench/http_checks.py`, which writes the
requests by hand on a socket, gets 411 for a chunked upload without `Content-Length` and 413 for
a `Content-Length` over the limit, both answered with no body byte sent. With
`TSD_MAX_INFLIGHT=1`, a second request gets 429 with `Retry-After: 1` while a first upload is
half sent, the first upload then completes with 200 and the next request is admitted (200);
`/metrics` counts the rejection under `reason="admission"`. Memory after these requests:
953.4 MiB.

*Bug found by the smoke test: the memory limit.* With the first compose file's 1 GiB, the rebuilt
service did not become healthy in the smoke test (`smoke_test_1g_oom.txt`), and of 10 further starts
under 1 GiB, 9 were OOM-killed 8.1-9.6 s after starting, during the warm-up (`oom_1g.txt`,
`oom_1g_run2.txt`: exit code 137, `OOMKilled` true); one started. A single start before these 10
also became healthy; the first 5-start run overwrote its record, so it is not in the
files. The start-up warm-up of every batch size 1-8 (section 6.3) peaked at 1,156,116,480 bytes
(1.08 GiB) with a 3 GiB limit (`memory_probe.txt`, cgroup v1 `memory.max_usage_in_bytes`) and at
1,048,621,056 bytes in the passing smoke test, against 1,073,741,824 bytes for 1 GiB. After the
smoke test's requests the service used 953.4 MiB, the first image (which warmed only sizes 1 and 8)
534.8 MiB. On the host the detector alone grows from 117 MiB after loading to 1030-1074 MiB when the
sizes are warmed in ascending order (`warmup_rss.txt`, `warmup_rss_main_checkout.txt`: OpenVINO
keeps memory for each new, larger shape). Compose now allows 2 GiB. Uploads over 1 MB are spooled to
the `/tmp` tmpfs, whose pages count against the same limit. The CI workflow builds the image and
runs the two import checks, but it has never run (section 10).

*Earlier bug found by the smoke test (first image).* The first `HEALTHCHECK` probed a fixed
port 8000; the smoke test runs the service on `TSD_PORT=8799`, and the container stayed in
`starting`: it could never have become healthy on any port but 8000. The probe now reads
`TSD_PORT`. (The failing run's output was not kept; commit c83dec9 records the observation and
the fix.)

*Trade-off.* OpenCV is only used for decoding and resizing; Pillow could replace it and save
172 MiB of the 452 MiB environment, but the letterbox would then have to be re-validated against
the exports' parity results (different interpolation code). Not done. Warming the largest batch
size first would leave the detector at 779-832 MiB instead of 1030-1074 MiB on the host (the
same two runs); not done either, because the serving results (section 6.3) were measured with
the ascending warm-up.

## 9. Not done: fine-tuning on the Open Images Traffic sign class (P3, optional)

*Decision.* Skipped, and the COCO-pretrained models are evaluated without the Traffic sign class
(section 1.2). Reasons, measured where possible:

* Data: the evaluation set is built from validation + test, so training data would have to come
  from the Open Images *train* split. Its box annotations are one 2,258,447,590-byte CSV
  (`curl -I`, `bench/results/network/train_csv_head.txt`); at the downloader's 2 MB/s cap that
  is at least 1,129 s for the metadata alone (size divided by the cap), before any image.
* Compute: CPU only, and the etiquette for long jobs on this shared machine is `nice -n 19` with
  at most 2 threads. One YOLO11n image already takes 122.6 ms end to end on one thread
  (PyTorch FP32, p50, median of three rounds, section 5.4); training adds the backward pass for
  every image of every epoch.
* Value for this repository: a model trained for minutes on a CPU would mostly demonstrate the
  training loop. The engineering questions this repository answers (export parity, quantization
  accuracy with intervals, latency under contamination control, serving behaviour) do not need
  it, and a fine-tuned class would not be comparable with the 2024 TT100K results either.

What exists for a later training run: the TT100K converter writes an Ultralytics dataset with
`data.yaml` (`tsd.data.tt100k`), and the two 2024 augmentations are implemented and tested as
standalone transforms (`tsd.augment`); hooking them into Ultralytics' training data pipeline is
not done.

## 10. CI without PyTorch

*Problem.* CI installs only the `onnx`, `openvino`, `eval` and `serve` extras: the PyTorch
wheel with its CUDA libraries is several GB (section 2.1). Code and tests must therefore work,
type-check and skip cleanly without `torch` and `ultralytics`.

*Measurement (first round, not logged).* A CI simulation of the first round (a clean
`git archive` of the commit, an environment with exactly the CI extras, no models or data)
passed ruff and pytest (183 passed, 20 skipped with a reason: missing weights,
`torch`/`ultralytics`/`torchvision`) but **failed strict mypy**: without Ultralytics installed,
`import ultralytics` resolved to `Any` (`no-any-return`), and a `type: ignore` needed only when
Ultralytics is installed became an "unused ignore" error. Those counts come from that round's
notes; its output was not kept.

*Fix.* Ultralytics is imported through `importlib.import_module` (typed as `ModuleType` either
way), and the one ignore lists `unused-ignore`. mypy now gives the same result in both
environments (then 203 tests in the full environment, also unlogged). The logged runs are in
`bench/results/ci/`: the CI simulation of commit `dcc2d5b` on Python 3.12.3 and 3.13.18 (ruff,
the format check and strict mypy over `src/tsd` and `bench` clean; 263 passed and 24 skipped on
each, every skip with its reason) and the full development environment's checks of the same
commit (287 passed, 0 skipped; `dev_checks_dcc2d5b.log`).

*Trade-off.* The CI job never exercises the PyTorch path or the real-model tests; those run
locally (`pytest` with the cached weights).

*Status.* Before the first push I checked the workflow with a local simulation of its test job.
It runs on pushes to `main`
and on pull requests (not twice for a pull request's pushes), cancels superseded runs, tests
Python 3.12 and 3.13, installs with `uv sync --locked` (fails if the lock is stale, where
`--frozen` would silently use it) and smoke-tests the built image (`tsd --help`, no PyTorch in
it). The simulation of the test job on both Python versions is in `bench/results/ci/`; the
Docker job was not simulated as written (Buildx with the GitHub cache), but its two checks ran
against the image built locally from a clean worktree (section 8).
