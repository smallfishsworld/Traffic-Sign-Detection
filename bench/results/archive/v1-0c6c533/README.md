# Archived results (commit 0c6c533, first round)

The benchmark matrix, serving comparison and evaluation reports as committed in `0c6c533`,
kept so that the numbers I re-checked (and the "before" column of the fix round) stay
traceable. They were superseded in the fix round:

* `benchmarks/`: one run per configuration (plus two repeats of six YOLO11n configurations in
  `repeats/`); raw files without git provenance; INT8 measured only with the f32 hint.
* `serving/`: one 20 s run per concurrency level, fixed 5 ms batching wait.
* `eval/`: no committed detections; tiling merged at IoU 0.5 while the detector's NMS used 0.7;
  `tiling_v1_no_border_filter.*` is the first tiling version (no inner-border filter).
* `parity_yolo11{n,s}.json`, `quantize_yolo11{n,s}{,_fullq}.json`: the export parity and INT8
  quantization reports, written at commit `8c6088f` (moved here from `bench/results/`). They
  have no provenance, no file hashes, and the parity files have no tolerance, pass/fail or
  two-way matching fields (the gate was added in `4c0103f`). The deployed exports were checked
  against the enforced gate later: `bench/results/parity/`.

Current results: `bench/results/benchmarks/`, `bench/results/serving/`, `bench/results/eval/`.
