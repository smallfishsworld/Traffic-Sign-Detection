# Evaluation protocol

All numbers in this document were measured in this repository's container (CPU only) on Open
Images data. They are unrelated to the historical 2024 TT100K/A100 results in
[`2024-course-project/`](2024-course-project/README.md).

## Data

The evaluation set is built by `tsd openimages` (see the README for its download limits):
every Open Images V7 **validation and test** image with at least one box of *Traffic sign*,
*Stop sign* or *Traffic light*: **219 images** (52 validation, 167 test). The image list with
SHA-256 checksums, licences and authors, and the ground truth, are committed in
[`data/manifests/openimages-traffic/`](../data/manifests/openimages-traffic/).

| class | images | boxes | group-of boxes | boxes < 1% of the image area |
|---|---|---|---|---|
| Traffic sign | 114 | 212 | 1 | 117 |
| Stop sign | 47 | 47 | 1 | 0 |
| Traffic light | 95 | 233 | 7 | 171 |

(From `summary.json`: validation + test.) 142 of the 219 images contain a stop sign or a
traffic light; the other 77 have only generic Traffic sign boxes, which are not evaluated
(below), so they contribute only false positives.

**Selection effect.** The set is selected by *box presence*. Open Images also has images that
a human verified to contain no stop sign (170 in validation + test) or no traffic light (61);
the set keeps only those that happen to have a box of another target class (49 and 6). The
Open Images challenge scores every verified image, and the left-out negatives can only add
false positives, so absolute AP on this set is an upper bound relative to that protocol. The
**verified-negatives variant** adds them: 380 images, the 219 plus 161 box-less images (121
verified without a stop sign, 55 without a traffic light, 15 both), built with `tsd openimages
--include-verified-negatives` (manifest in `data/manifests/openimages-traffic-negatives/`).
Both variants are reported below; the main tables use the 219-image set.

**Overlap with other data.** None of these images is used for calibration: the INT8
calibration images (COCO128) were checked for near-duplicates of them with a two-stage
difference hash, and none was found (closest pair 9/64 bits, rejected on its 256-bit hash and
aspect ratio; the first round's `bench/results/archive/v1-0c6c533/quantize_*.json`; the
margins of the rule are measured in `bench/results/dedup_copy_sim.json`). The models themselves
are Ultralytics' COCO-pretrained weights, trained on COCO train2017; whether any evaluation
image (or a copy of it) is in COCO train2017 was **not checked** (both collections come from
Flickr; train2017 is 118k images), so a small train/test overlap cannot be ruled out.

## Classes

A model is evaluated on the data-set classes it also has, matched by case-insensitive name.
For COCO-pretrained YOLO11 this means:

| Open Images class | COCO class | evaluated |
|---|---|---|
| Stop sign | 11 `stop sign` | yes |
| Traffic light | 9 `traffic light` | yes |
| Traffic sign | none | **excluded** (COCO has no generic traffic-sign class) |

mAP is the mean over the evaluated classes (COCO convention: a class without ground truth in a
size bucket is left out of that bucket's mean).

## Images and classes that are not annotated

Open Images is not exhaustively labelled: a missing box does not mean a missing object.
Following the Open Images challenge protocol, a class is **evaluated in an image** only if a
human verified it present or absent there (image-level labels, expanded over the class
hierarchy: present propagates to ancestors, absent to descendants) or the image has boxes of it.

* Boxes are expanded up the hierarchy as well: a Stop sign box is also a Traffic sign box (all
  47 stop-sign images are Traffic-sign positive, but only 8 of them have a Traffic sign box of
  their own). This matters only for the Traffic sign class, which is excluded for the COCO
  models (below), so it changes none of the reported numbers.
* Stop sign is evaluated in 96 of the 219 images, Traffic light in 101.
* Detections of a class that is not evaluated in an image are **dropped** before matching
  (neither true nor false positives). For YOLO11n (PyTorch) that removes 514 stop-sign and
  1,075 traffic-light detections (at confidence ≥ 0.001).
* Detections of a class verified *absent* count as false positives.
* As a sensitivity check, `--count-unverified` scores every detection (unverified ones become
  false positives): [`bench/results/eval/coco_classes_strict.json`](../bench/results/eval/coco_classes_strict.json).
  It lowers YOLO11n mAP50-95 from 0.669 to 0.663 and YOLO11s from 0.702 to 0.697; backend
  differences are the same within their intervals.

Other flags:

* **Group-of** boxes (one box around several objects) become COCO *crowd* regions: detections
  inside them are ignored and the region is never a miss. (The Open Images challenge instead
  counts one true positive or one miss per group-of box; with 1 + 7 such boxes the choice
  barely matters here.)
* Depicted, occluded and truncated objects are ordinary ground truth, as in Open Images.

## Predictions

Images are decoded with OpenCV, EXIF orientation applied (none needs a rotation: none of the
219 images has an orientation tag, and the one tag among the 161 added negatives is 1,
upright), letterboxed to 640 × 640, and passed through the same post-processing for every
backend: Ultralytics' validation settings (confidence ≥ 0.001, class-wise NMS at IoU 0.7,
multi-label, at most 300 per image), restricted to the evaluated classes so that detections of
other COCO classes cannot use up the per-image cap. Detections are saved as gzipped JSON under
`bench/results/eval/detections/` (boxes rounded to 0.001 px, scores to 1e-6; committed,
0.10-0.36 MB per run), named by the run and a key of everything that changes them (NMS settings,
tiling settings, OpenVINO precision hint). **Metrics are always computed from the saved
files**, and each report lists, per run, the file it was scored from and its SHA-256. With
`--reuse` (default) a saved file is used instead of running the model only if the model files'
SHA-256, the NMS and tiling settings, the precision hint and the evaluation image ids all match.
The detection file's own SHA-256 in a report is for audit (it names the exact file that was
scored); reuse does not check it against an earlier report. The reported protocol (`protocol`
in each report) is generated from the constants the code uses (NMS, IoU thresholds, area
ranges), not written by hand.

## Metrics

COCO box AP: greedy matching of score-sorted detections at IoU 0.50:0.05:0.95, 101-point
interpolated precision, at most 100 detections per image and class, size buckets by box area in
original-image pixels (small ≤ 32², medium 32²–96², large ≥ 96²; the bounds are inclusive as in
pycocotools, so the one traffic-light box of exactly 96² px counts in both). Reported: mAP50,
mAP50-95, mAP75, AR@100, per class and per size bucket.

`tsd.eval.metrics` re-implements `pycocotools.COCOeval` so that matching can be cached per image
(needed for a fast bootstrap). Its largest difference from pycocotools is 2.2e-16 on 200 random
scenarios (`python tests/test_eval_metrics.py 200`, output in
`bench/results/eval/pycocotools_agreement_200.txt`; the test suite runs 6) and 3.3e-16 over all
52 committed real runs: each report entry carries the pycocotools numbers and the largest
difference (`pycocotools.max_abs_diff`). A size bucket
without ground truth is undefined in both (pycocotools reports -1, read as NaN): a metric that
is undefined on one side only is a disagreement (`nan_mismatch` lists it and `max_abs_diff`
becomes infinite). The first version took Python's `max()` over differences that could be NaN
(which `max()` skips unless it comes first) and returned 0 instead of NaN for every bucket when
a run had no detections, so a disagreement on an empty bucket could be reported as
`max_abs_diff = 0` (found when I re-checked my results). A first version
rounded the IoU and recall thresholds to two decimals; pycocotools uses the raw `np.linspace`
values (its 0.70 recall point is 0.7000000000000001), which changed AP when a class had a
recall of exactly 7/10. The randomized test found this.

Ground-truth boxes per size bucket (non-crowd): Stop sign 0 small / 0 medium / 46 large;
Traffic light 69 small / 104 medium / 54 large.

## Confidence intervals

95% **percentile bootstrap over images**: 1,000 replicates (seed 0), each drawing 219 images
with replacement. Every replicate scores *all* runs on the same resample, so the interval of a
backend's difference to PyTorch FP32 is **paired**: image-to-image variation shared by both runs
cancels, which is why a difference can be significant while the per-run intervals overlap.

Limitations: 219 images and two classes give wide per-run intervals (about ±0.04 mAP50-95);
images are treated as independent units; the intervals cover sampling of images, not the choice
of classes, thresholds or data source.

## Commands

```bash
uv run tsd eval models/yolo11n.pt models/yolo11n.onnx \
  models/yolo11n_openvino_fp32/yolo11n.xml models/yolo11n_openvino_fp32/yolo11n.xml@bf16 \
  models/yolo11n_openvino_int8/yolo11n.xml models/yolo11n_openvino_int8/yolo11n.xml@bf16 \
  models/yolo11n_openvino_int8_fullq/yolo11n.xml \
... (the same seven for yolo11s) \
  --out bench/results/eval/coco_classes.json          # add --count-unverified for the strict variant
```

`@bf16` runs an OpenVINO IR with the BF16 inference-precision hint: the FP32 IR entirely in
BF16 (`openvino-bf16`), the INT8 IR with its non-quantized ops in BF16 (`openvino-int8-bf16`).
`--data-dir data/openimages-traffic-negatives` evaluates the verified-negatives variant.

## Results

[`bench/results/eval/coco_classes.md`](../bench/results/eval/coco_classes.md) (full numbers in the
JSON next to it):

| run | mAP50 | 95% CI | mAP50-95 | 95% CI | Δ mAP50-95 vs PyTorch FP32 [95% CI] |
|---|---|---|---|---|---|
| yolo11n/pytorch-fp32 | 0.814 | [0.770, 0.860] | 0.669 | [0.634, 0.709] | (baseline) |
| yolo11n/onnxruntime-fp32 | 0.814 | [0.770, 0.860] | 0.669 | [0.634, 0.709] | +0.0000 [0.000, 0.000] |
| yolo11n/openvino-fp32 | 0.814 | [0.770, 0.860] | 0.669 | [0.634, 0.709] | +0.0000 [-0.000, 0.000] |
| yolo11n/openvino-bf16 | 0.814 | [0.772, 0.860] | 0.669 | [0.634, 0.710] | +0.0001 [-0.002, 0.003] |
| yolo11n/openvino-int8 | 0.809 | [0.765, 0.856] | 0.667 | [0.629, 0.710] | -0.0024 [-0.014, 0.008] |
| yolo11n/openvino-int8-bf16 | 0.807 | [0.763, 0.855] | 0.666 | [0.628, 0.712] | -0.0030 [-0.019, 0.011] |
| yolo11n/openvino-int8-fullq | 0.786 | [0.741, 0.834] | 0.606 | [0.570, 0.650] | -0.0633 [-0.081, -0.044] |
| yolo11s/pytorch-fp32 | 0.848 | [0.804, 0.892] | 0.702 | [0.664, 0.744] | (baseline) |
| yolo11s/onnxruntime-fp32 | 0.848 | [0.804, 0.892] | 0.702 | [0.664, 0.744] | -0.0000 [-0.000, 0.000] |
| yolo11s/openvino-fp32 | 0.848 | [0.804, 0.892] | 0.702 | [0.664, 0.744] | -0.0000 [-0.000, 0.000] |
| yolo11s/openvino-bf16 | 0.848 | [0.803, 0.892] | 0.700 | [0.663, 0.743] | -0.0015 [-0.004, 0.001] |
| yolo11s/openvino-int8 | 0.847 | [0.806, 0.889] | 0.702 | [0.664, 0.745] | +0.0001 [-0.005, 0.007] |
| yolo11s/openvino-int8-bf16 | 0.850 | [0.808, 0.892] | 0.703 | [0.665, 0.744] | +0.0014 [-0.005, 0.010] |
| yolo11s/openvino-int8-fullq | 0.820 | [0.777, 0.865] | 0.629 | [0.594, 0.671] | -0.0723 [-0.086, -0.056] |

Reading it:

* ONNX Runtime and OpenVINO FP32 reproduce PyTorch exactly at this precision (export parity of
  the deployed files: `bench/results/parity/parity_*.json`, largest box difference 0.037 px on
  8 images, the gate passed).
* INT8 with the box-decoding head kept in floating point (`openvino-int8`) shows no detectable
  loss: both paired intervals contain zero, with the f32 and with the BF16 hint.
* Quantizing the decoding head as well (`int8-fullq`, an ablation) costs 0.063–0.072 mAP50-95,
  with intervals far from zero, while mAP50 drops less: rounding the decoded coordinates hurts
  localization at strict IoU thresholds. This is why the default keeps the head in FP32.
* Per class (YOLO11n, PyTorch): Stop sign AP50-95 0.958, Traffic light 0.380 (traffic lights
  are small: 69 of their 226 boxes are at most 32² px, where mAP50-95 is 0.18, and 104 more at
  most 96²).
* Compared with the first round (`bench/results/archive/v1-0c6c533/eval/`), mAP50 moved by at
  most 0.0015 and mAP50-95 and the paired deltas by less than 0.0005 over the 12 runs both rounds
  have: for YOLO11n PyTorch FP32 the detections are identical in count, class and score, and
  boxes moved by up to 2.2 px in original-image pixels, because they are now mapped back with
  the exact per-axis resize gains (`bench/results/archive/v1-0c6c533/eval/compare_detections.txt`).

### Verified-negatives variant (380 images)

[`bench/results/eval/coco_classes_with_negatives.md`](../bench/results/eval/coco_classes_with_negatives.md):
the 219 images plus 161 box-less images verified to contain no stop sign and/or no traffic
light (section Data). Same models, protocol and bootstrap; the INT8 ablation with a quantized
decoder is left out (12 runs). Detections: `bench/results/eval/detections_with_negatives/`.

| run | mAP50 | mAP50-95 [95% CI] | Δ mAP50-95 vs PyTorch FP32 [95% CI] | same Δ on the 219 images |
|---|---|---|---|---|
| YOLO11n PyTorch FP32 | 0.793 | 0.653 [0.609, 0.694] | (baseline) | (baseline) |
| YOLO11n ONNX Runtime / OpenVINO FP32 | 0.793 | 0.653 [0.609, 0.694] | +0.0000 [-0.000, 0.000] | +0.0000 |
| YOLO11n OpenVINO BF16 | 0.793 | 0.654 [0.611, 0.694] | +0.0009 [-0.002, 0.005] | +0.0001 |
| YOLO11n OpenVINO INT8 | 0.796 | 0.657 [0.615, 0.700] | +0.0047 [-0.011, 0.024] | -0.0024 |
| YOLO11n OpenVINO INT8 + BF16 hint | 0.795 | 0.658 [0.616, 0.702] | +0.0052 [-0.016, 0.027] | -0.0030 |
| YOLO11s PyTorch FP32 | 0.841 | 0.697 [0.658, 0.738] | (baseline) | (baseline) |
| YOLO11s ONNX Runtime / OpenVINO FP32 | 0.841 | 0.697 [0.658, 0.738] | -0.0000 [-0.000, 0.000] | -0.0000 |
| YOLO11s OpenVINO BF16 | 0.840 | 0.695 [0.656, 0.737] | -0.0015 [-0.004, 0.000] | -0.0015 |
| YOLO11s OpenVINO INT8 | 0.840 | 0.696 [0.659, 0.737] | -0.0005 [-0.005, 0.006] | +0.0001 |
| YOLO11s OpenVINO INT8 + BF16 hint | 0.843 | 0.698 [0.661, 0.739] | +0.0010 [-0.005, 0.009] | +0.0014 |

Absolute mAP50-95 is 0.016 (YOLO11n) and 0.005 (YOLO11s) lower: the added images contribute
only false positives, mostly on Stop sign (YOLO11n 0.958 to 0.932 AP50-95; Traffic light 0.380
to 0.373). Recall is unchanged. Apart from the quantized-decoder ablation (evaluated on the 219
images only), no runtime's paired difference has an interval that excludes zero in either
variant, so the runtime comparison does not depend on the choice.

## Sliced inference

`model@tiles=512` evaluates the same model with sliced inference (`tsd.infer.tiling`). A tiled
run is paired with the *untiled* run of the same backend and weights, and its tile and
full-image detections are merged at the detector's own NMS IoU (0.7 here), so its Δ is the
effect of the extra views alone (`--tile-merge-iou` sets another merge IoU for a control);
when a report contains a tiled run, paired intervals are also computed for each size bucket
(`by_size_delta_vs_baseline`). Results:
[`bench/results/eval/tiling.md`](../bench/results/eval/tiling.md) (JSON next to it), discussed in
[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md#74-measurement-benchresultsevaltilingjson)
section 7.
