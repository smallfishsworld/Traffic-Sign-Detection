# The 2024 course project (historical record)

In 2024 this project was a machine-learning course project: traffic-sign detection on
TT100K (Tsinghua-Tencent 100K: 2048x2048 street-view images, mostly very small signs) with
Ultralytics YOLO11, trained on one NVIDIA A100 GPU (report section 5). The full report is not
part of this repository (it contains the students' personal details); the facts below are
taken from it, with its section numbers.

* **Classes** (section 4.2.4): TT100K has 201 traffic-sign classes, and their distribution is
  very imbalanced: 84 classes have fewer than 10 image instances, 72 have 10-100 and 45 more than
  100. That long tail is why the project added the replacement augmentation.

## What it did

* Converted the TT100K annotation JSON to YOLO label files (report appendix A; rewritten as
  `tsd.data.tt100k`, see `docs/ENGINEERING_NOTES.md` for what the rewrite changes).
* Added two augmentations of its own (section 4.2), on top of Ultralytics' training with default
  hyper-parameters:
  * a perspective (homography) transform (section 4.2.3), and
  * a "replacement" transform that pasted standard sign templates over existing signs so that
    rare classes appear more often (section 4.2.4; only `pl*` and `il*` had templates,
    section 7).
* Trained five combinations (section 5, Table 2; the report's "No augmentation" means no
  augmentation beyond Ultralytics' defaults: every run used default hyper-parameters, section
  6.2):

| Run | Model   | Epochs | Augmentation added                    |
|-----|---------|--------|---------------------------------------|
| 1   | YOLO11m | 50     | none (default hyper-parameters)       |
| 2   | YOLO11x | 50     | none (default hyper-parameters)       |
| 3   | YOLO11x | 200    | none (default hyper-parameters)       |
| 4   | YOLO11x | 200    | perspective                           |
| 5   | YOLO11x | 200    | perspective + replacement             |

## Results stated in the report (2024, A100, TT100K)

| Comparison (section 6.2.2) | Precision      | mAP50          |
|----------------------------|----------------|----------------|
| run 3 -> run 4             | 0.757 -> 0.800 | 0.701 -> 0.724 |

This is the effect of the perspective transform alone (run 4 adds it to run 3). The report
gives no other numbers in its text: runs 1 vs 2 are compared only through result figures
(section 6.2.1: "the precision and the mAP50 of YOLOv11x are larger than those of YOLOv11m"),
and run 5 (replacement added) vs run 4 only through example detections (section 6.2.3: on the
rare `pl15` speed-limit sign "the agent performs much better"). The replacement transform's
effect is therefore qualitative only, and none of these statements can be checked any more.

Caveat on these numbers: the 2024 conversion script wrote the TT100K `test` split into the
dataset's `val` directory, so the split used for checkpoint selection during training is also the
split the metrics were reported on.

## Surviving files (unchanged copies)

* `draw.py`: plots the loss, precision/recall/mAP and learning-rate columns of an Ultralytics
  `results.csv` with matplotlib (it reads a hard-coded Windows path).
* `results.csv`: 5 epochs of one Ultralytics training log. The report does not say which run it
  belongs to; its last row (epoch 5: mAP50 0.01328; the cumulative `time` column grows by 374-406 s
  per epoch) shows a run in its first epochs, far from the final scores above.

Everything else (training code, augmentation code, weights, the converted dataset) was lost.
