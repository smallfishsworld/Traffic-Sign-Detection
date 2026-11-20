# CPU inference benchmarks

CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 vCPUs (KVM); flags: avx2, avx512f, avx512_vnni, avx_vnni, avx512_bf16, amx_tile, amx_int8, amx_bf16.

> Speedups were measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units); INT8 and BF16 gains on a typical laptop CPU without AMX are smaller.

Latency = end-to-end per image (letterbox + inference + NMS, decoded image in memory), p50 / p95 / p99 in ms. Batch-8 = images/s for batches of 8 at 4 threads (from the median batch time). Speedups are against PyTorch FP32 at the same setting. mAP50-95: Open Images stop sign + traffic light; in brackets the paired difference to PyTorch FP32 with its 95% bootstrap interval. Every measurement records the CPU use of other processes on this shared machine (benchmarks.json, `other_load_cores`) and was repeated until it stayed below the limit; `*` marks one that never did.

| model | backend | size (MB) | 1 thread | 4 threads | batch-8 img/s | speedup 1t / 4t / b8 | mAP50-95 (Δ vs PyTorch [95% CI]) |
|---|---|---|---|---|---|---|---|
| yolo11n | pytorch-fp32 | 5.6 | 134.1 / 154.7 / 175.6 | 68.0 / 84.0 / 96.3 | 18.2 | 1.00x / 1.00x / 1.00x | 0.669 |
| yolo11n | onnxruntime-fp32 | 10.7 | 81.9 / 98.2 / 106.6 | 38.9 / 50.9 / 56.6 | 28.8 | 1.64x / 1.75x / 1.59x | 0.669 (+0.0000 [+0.0000, +0.0000]) |
| yolo11n | openvino-fp32 | 10.9 | 86.6 / 102.5 / 114.9 | 30.6 / 44.0 / 52.4 | 36.9 | 1.55x / 2.22x / 2.03x | 0.669 (+0.0000 [-0.0000, +0.0000]) |
| yolo11n | openvino-bf16 | 10.9 | 44.5 / 53.7 / 63.4 | 21.7 / 27.1 / 30.6 | 63.3 | 3.01x / 3.13x / 3.48x | 0.669 (+0.0001 [-0.0020, +0.0028]) |
| yolo11n | openvino-int8 | 3.5 | 53.1 / 58.2 / 63.5 | 25.9 / 36.7 / 44.4 | 55.0 | 2.53x / 2.62x / 3.03x | 0.666 (-0.0024 [-0.0137, +0.0082]) |
| yolo11s | pytorch-fp32 | 19.3 | 296.8 / 328.8 / 361.5 | 119.9 / 163.3 / 232.9 | 7.6 | 1.00x / 1.00x / 1.00x | 0.702 |
| yolo11s | onnxruntime-fp32 | 38.0 | 210.9 / 244.1 / 267.0 | 76.3 / 99.7 / 112.6 | 13.9 | 1.41x / 1.57x / 1.83x | 0.702 (-0.0000 [-0.0000, +0.0000]) |
| yolo11s | openvino-fp32 | 38.2 | 214.2 / 240.8 / 258.8 | 69.4 / 87.0 / 102.0 | 15.4 | 1.39x / 1.73x / 2.03x | 0.702 (-0.0000 [-0.0000, +0.0000]) |
| yolo11s | openvino-bf16 | 38.2 | 76.0 / 88.9 / 105.4 | 30.0 / 36.4 / 46.8 | 40.6 | 3.90x / 4.00x / 5.35x | 0.700 (-0.0018 [-0.0045, +0.0002]) |
| yolo11s | openvino-int8 | 10.3 | 85.4 / 98.5 / 106.0 | 37.1 / 46.2 / 50.8 | 38.5 | 3.48x / 3.23x / 5.07x | 0.702 (+0.0005 [-0.0043, +0.0072]) |

## Headline (single image, 4 threads)

- yolo11n: openvino-bf16 21.7 ms vs PyTorch FP32 68.0 ms p50 = **3.13x** (mAP50-95 +0.0001 (95% CI [-0.0020, +0.0028])). AMX/VNNI CPU: INT8/BF16 speedups on a typical laptop are smaller.
- yolo11s: openvino-bf16 30.0 ms vs PyTorch FP32 119.9 ms p50 = **4.00x** (mAP50-95 -0.0018 (95% CI [-0.0045, +0.0002])). AMX/VNNI CPU: INT8/BF16 speedups on a typical laptop are smaller.

## Sliced inference: latency cost (single image, 4 threads)

One call = all tiles of one image plus the full-image pass, merged; the inference count is the mean over the benchmark images (it grows with the image size).

| model | backend | tile (px) / overlap | inferences per image | p50 / p95 / p99 ms | untiled p50 ms | slowdown |
|---|---|---|---|---|---|---|
| yolo11n | openvino-fp32 | 512 / 20% | 6.8 | 190.7 / 230.2 / 238.7 | 30.6 | 6.2x |
| yolo11s | openvino-fp32 | 512 / 20% | 6.8 | 462.5 / 663.7 / 676.6 | 69.4 | 6.7x |
