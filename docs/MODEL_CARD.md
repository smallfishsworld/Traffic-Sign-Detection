# Model card: YOLO11n / YOLO11s CPU deployment variants

All numbers on this card were measured in this repository's container; none comes from the 2024
course project (whose TT100K-trained YOLO11x weights are lost, see
[`2024-course-project/`](2024-course-project/README.md)).

## Model details

| | |
|---|---|
| Base models | Ultralytics YOLO11n and YOLO11s, COCO-pretrained, **not fine-tuned here** |
| Weights | `yolo11n.pt` SHA-256 `0ebbc80d…644ee1`, `yolo11s.pt` `85a76fe8…d502d5` (pinned in `tsd.assets`) |
| Licence | AGPL-3.0 (Ultralytics), hence this repository's licence |
| Input | RGB, letterboxed to 640 × 640 (grey 114 padding), values in [0, 1] |
| Output | boxes in original-image pixels, scores, 80 COCO class ids; NMS in NumPy |
| Variants | PyTorch FP32 (`.pt`); ONNX FP32 (opset 18, dynamic batch); OpenVINO IR FP32 (run with the FP32 or the BF16 precision hint); OpenVINO IR INT8 (run with the FP32 hint, the service default, or the BF16 hint for its non-quantized ops) |
| INT8 recipe | NNCF post-training quantization, MIXED preset, 128 COCO128 calibration images (checked for near-duplicates of the evaluation images), box-decoding ops of the detect head kept in floating point (28 ops) |
| Model files | YOLO11n: `.pt` 5.6 MB, ONNX 10.7 MB, OpenVINO FP32 10.9 MB, OpenVINO INT8 3.5 MB (`bench/results/benchmarks/benchmarks.json`) |

## Intended use

* Demonstrating and measuring CPU deployment of a small detector: export, quantization,
  evaluation with confidence intervals, an HTTP service with micro-batching.
* Detecting **stop signs and traffic lights** (two of the 80 COCO classes) in street photos.

Out of scope: any driving or safety decision (ADAS, autonomous driving), recognising
*which* traffic sign is shown (the COCO models have no traffic-sign classes besides stop sign),
and traffic-light *state* (red/green).

## Evaluation

Data: 219 Open Images V7 validation/test images with Traffic sign, Stop sign or Traffic light
boxes (142 contain a stop sign or traffic light); COCO mAP with Open Images verified-label
filtering (detections of classes nobody verified in an image are ignored) and hierarchy
expansion, group-of boxes scored as COCO crowd regions; 95% percentile-bootstrap intervals over
images.
Details: [`EVALUATION.md`](EVALUATION.md). Report: `bench/results/eval/coco_classes.json`
(detections committed in `bench/results/eval/detections/`).

| model / runtime | mAP50 | mAP50-95 [95% CI] | Stop sign AP50-95 | Traffic light AP50-95 | AR@100 |
|---|---|---|---|---|---|
| YOLO11n PyTorch FP32 | 0.814 | 0.669 [0.634, 0.709] | 0.958 | 0.380 | 0.784 |
| YOLO11n OpenVINO INT8 | 0.809 | 0.667 [0.629, 0.710] | 0.945 | 0.388 | 0.785 |
| YOLO11s PyTorch FP32 | 0.848 | 0.702 [0.664, 0.744] | 0.948 | 0.455 | 0.800 |
| YOLO11s OpenVINO INT8 | 0.847 | 0.702 [0.664, 0.745] | 0.955 | 0.448 | 0.803 |

The set is selected by box presence, so it keeps only 49 of the 170 images verified to contain
no stop sign and 6 of the 61 verified to contain no traffic light. On the 380-image variant that
adds the other 161 (box-less) verified-negative images, mAP50-95 is 0.653 [0.609, 0.694] for
YOLO11n PyTorch FP32 and 0.657 [0.615, 0.700] for its INT8 export, 0.697 [0.658, 0.738] and
0.696 [0.659, 0.737] for YOLO11s; Stop sign AP50-95 drops most (YOLO11n 0.958 to 0.932), and
AR@100 is unchanged because the added images have no boxes
(`bench/results/eval/coco_classes_with_negatives.json`).

By object size (YOLO11n PyTorch FP32; all small and medium objects are traffic lights, every
stop sign is large):

| size (box area, px) | GT boxes | mAP50-95 | mAP50 | AR@100 |
|---|---|---|---|---|
| small (≤ 32²) | 69 | 0.1753 | 0.4385 | 0.3449 |
| medium (32²–96²) | 104 | 0.4873 | 0.7939 | 0.6365 |
| large (≥ 96²) | 100 | 0.7378 | 0.8650 | 0.8958 |

(COCO bucket bounds are inclusive: the one traffic-light box of exactly 96² px is counted as
medium and as large.)

Quantization: the INT8 model's paired mAP50-95 difference to PyTorch FP32 is −0.0024
[−0.014, +0.008] (YOLO11n) and +0.0001 [−0.005, +0.007] (YOLO11s); run with the BF16 hint for
its non-quantized ops, −0.0030 [−0.019, +0.011] and +0.0014 [−0.005, +0.010]; quantizing the box
decoder too costs −0.063 and −0.072 (see `ENGINEERING_NOTES.md`). ONNX Runtime and OpenVINO FP32
are numerically equivalent to PyTorch: the deployed exports pass the parity gate with a largest
box difference of 0.037 px on the 8 parity images (`bench/results/parity/`, checked at commit
8d47bd4; the first round's reports gave the same figure without provenance).
These numbers differ from the first round's in the third decimal at most because boxes are now
mapped back with the exact per-axis resize gains (`tsd.infer.preprocess`).

## CPU latency

YOLO11n, one image, 4 threads, p50 end-to-end, median of three rounds
(`bench/results/benchmarks/benchmarks.md`): PyTorch FP32 66.0 ms, OpenVINO INT8 25.7 ms,
OpenVINO INT8 with the BF16 hint 27.4 ms, OpenVINO BF16 22.3 ms (the fastest configuration
measured; 66.0 / 22.3 is 2.96x). The speedups over PyTorch, each the median of the three paired
per-round ratios (with their range), are 2.58x (2.34-2.65x) for INT8, 2.49x (2.26-2.50x) with
the BF16 hint and 2.84x (2.83-3.13x) for BF16. Measured on 4 shared
vCPUs of an Intel Xeon with AVX-512 VNNI and AMX; INT8 and BF16 gains on a typical laptop CPU
without AMX are smaller.

## Limitations and risks

* **Small objects**: traffic lights under 32 × 32 px are mostly missed (AR@100 0.34). Sliced
  inference (512 px tiles, opt-in) raises their AP50-95 from 0.175 to 0.262 (paired Δ +0.087
  [+0.035, +0.131]) and AR@100 to 0.46, at 6.0x the latency (203.0 vs 33.7 ms p50, 4 threads,
  OpenVINO FP32, median of three rounds; `bench/results/eval/tiling.json`,
  `bench/results/benchmarks/benchmarks.md`).
* **Small, narrow evaluation set**: 219 images, two classes, mostly daytime Flickr photos; the
  intervals are about ±0.04 mAP50-95 wide and cover image sampling only. Night, rain, glare and
  non-US sign designs are not specifically evaluated.
* **No generic traffic-sign detection**: COCO has no such class; Open Images "Traffic sign" is
  excluded from evaluation and is never predicted.
* Calibration and evaluation images are both Flickr photos; they were checked for near-duplicate
  pairs (none found), not for overlap of scenes or photographers. Whether evaluation images are
  in COCO train2017, on which the pretrained weights were trained, was not checked.
* **Optimistic absolute AP**: the evaluation set is selected by box presence and misses most
  verified-negative images (above).
* Detections can be confidently wrong; scores are not calibrated probabilities.
* Street photos contain people and licence plates. The service neither stores nor logs uploaded
  images; uploads over 1 MB are spooled to a temporary file (Starlette's default) for the
  duration of the request.
