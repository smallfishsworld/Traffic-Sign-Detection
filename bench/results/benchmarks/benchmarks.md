# CPU inference benchmarks

CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 vCPUs (KVM); flags: avx2, avx512f, avx512_vnni, avx_vnni, avx512_bf16, amx_tile, amx_int8, amx_bf16.

> Speedups were measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units); INT8 and BF16 gains on a typical laptop CPU without AMX are smaller.

Measured at commit(s) cfff803426aa (clean tree).

Latency = end-to-end per image (fixed 640x640 letterbox for every runtime + inference + NMS, decoded image in memory), p50 / p95 / p99 in ms: the median over the repeated runs, in brackets the range of the per-run p50. Batch-8 = images/s for batches of 8 at 4 threads (from the median batch time). Speedups are against PyTorch FP32 at the same setting: the median of the per-run paired ratios (run k of a configuration against run k of PyTorch, measured in the same round), in brackets their range. mAP50-95: Open Images stop sign + traffic light; in brackets the paired difference to PyTorch FP32 with its 95% bootstrap interval. Every measurement records the CPU use of other processes on this shared machine (benchmarks.json, `other_load_cores`) and was repeated until it stayed below the limit; `*` marks one that never did.

| model | backend | size (MB) | 1 thread | 4 threads | batch-8 img/s | speedup 1t / 4t / b8 | mAP50-95 (Δ vs PyTorch [95% CI]) |
|---|---|---|---|---|---|---|---|
| yolo11n | pytorch-fp32 | 5.6 | 122.6 / 138.1 / 160.8 (122.0-124.6) | 66.0 / 97.6 / 152.9 (63.1-68.3) | 18.5 (17.6-19.5) | 1.00x (1.00-1.00) / 1.00x (1.00-1.00) / 1.00x (1.00-1.00) | 0.669 |
| yolo11n | onnxruntime-fp32 | 10.7 | 82.4 / 100.6 / 111.2 (81.6-84.0) | 40.8 / 57.1 / 84.4 (38.4-41.3) | 29.9 (29.1-30.4) | 1.50x (1.45-1.51) / 1.64x (1.60-1.67) / 1.64x (1.53-1.65) | 0.669 (+0.0000 [+0.0000, +0.0000]) |
| yolo11n | openvino-bf16 | 10.9 | 44.4 / 49.4 / 60.3 (43.1-44.7) | 22.3 / 29.8 / 37.2 (21.8-23.2) | 61.2 (59.9-63.0) | 2.75x (2.74-2.89) / 2.84x (2.83-3.13) / 3.41x (3.07-3.47) | 0.669 (+0.0001 [-0.0019, +0.0028]) |
| yolo11n | openvino-fp32 | 10.9 | 89.2 / 102.5 / 115.8 (87.7-89.9) | 33.7 / 51.1 / 57.7 (31.0-38.6) | 34.1 (31.7-34.6) | 1.40x (1.36-1.40) / 1.87x (1.77-2.13) / 1.85x (1.62-1.96) | 0.669 (+0.0000 [-0.0000, +0.0000]) |
| yolo11n | openvino-fp32-nopin | 10.9 | - | 34.0 / 47.9 / 54.1 (29.2-35.3) | - | - / 1.94x (1.86-2.26) / - | - |
| yolo11n | openvino-int8 | 3.5 | 53.5 / 62.7 / 67.5 (53.0-54.3) | 25.7 / 34.7 / 40.7 (25.5-27.0) | 53.5 (52.7-58.4) | 2.28x (2.26-2.35) / 2.58x (2.34-2.65) / 2.99x (2.74-3.16) | 0.667 (-0.0024 [-0.0137, +0.0082]) |
| yolo11n | openvino-int8-bf16 | 3.5 | 54.4 / 64.1 / 70.8 (53.8-56.5) | 27.4 / 34.9 / 42.2 (26.5-28.0) | 51.5 (50.7-54.0) | 2.24x (2.17-2.32) / 2.49x (2.26-2.50) / 2.74x (2.64-3.07) | 0.666 (-0.0030 [-0.0191, +0.0106]) |
| yolo11n | openvino-int8-bf16-nopin | 3.5 | - | 27.9 / 36.3 / 39.4 (26.4-28.2) | - | - / 2.44x (2.24-2.50) / - | - |
| yolo11s | pytorch-fp32 | 19.3 | 305.6 / 344.7 / 361.2 (304.5-305.9) | 121.9 / 174.9 / 226.8 (115.6-123.5) | 7.4 (7.1-7.6) | 1.00x (1.00-1.00) / 1.00x (1.00-1.00) / 1.00x (1.00-1.00) | 0.702 |
| yolo11s | onnxruntime-fp32 | 38.0 | 212.2 / 242.4 / 264.5 (211.9-214.7) | 78.1 / 107.5 / 137.3 (77.5-79.9) | 13.8 (13.2-14.0) | 1.44x (1.42-1.44) / 1.55x (1.48-1.57) / 1.89x (1.73-1.95) | 0.702 (-0.0000 [-0.0000, +0.0000]) |
| yolo11s | openvino-bf16 | 38.2 | 75.8 / 91.3 / 107.9 (72.9-80.9) | 30.6 / 44.0 / 56.9 (30.0-38.7) | 41.2 (31.7-41.3) | 4.04x (3.77-4.19) / 3.85x (3.19-3.98) / 5.57x (4.16-5.83) | 0.700 (-0.0015 [-0.0042, +0.0005]) |
| yolo11s | openvino-fp32 | 38.2 | 217.8 / 244.2 / 266.4 (215.7-220.1) | 68.7 / 105.6 / 135.9 (68.5-74.1) | 14.4 (13.9-15.5) | 1.40x (1.38-1.42) / 1.68x (1.67-1.78) / 2.03x (1.83-2.09) | 0.702 (-0.0000 [-0.0000, +0.0000]) |
| yolo11s | openvino-fp32-nopin | 38.2 | - | 69.0 / 96.6 / 115.3 (67.2-69.4) | - | - / 1.79x (1.67-1.81) / - | - |
| yolo11s | openvino-int8 | 10.3 | 84.8 / 101.1 / 109.6 (84.5-98.7) | 37.1 / 48.9 / 59.8 (36.0-37.3) | 36.0 (35.6-38.0) | 3.60x (3.08-3.62) / 3.29x (3.10-3.43) / 5.08x (4.67-5.12) | 0.702 (+0.0001 [-0.0045, +0.0068]) |
| yolo11s | openvino-int8-bf16 | 10.3 | 83.8 / 97.1 / 109.0 (82.4-83.8) | 40.1 / 55.3 / 63.1 (39.2-42.0) | 35.4 (33.2-36.0) | 3.65x (3.65-3.69) / 2.94x (2.89-3.11) / 4.85x (4.35-5.01) | 0.703 (+0.0014 [-0.0045, +0.0096]) |
| yolo11s | openvino-int8-bf16-nopin | 10.3 | - | 39.0 / 50.4 / 54.4 (38.0-40.0) | - | - / 3.17x (2.89-3.21) / - | - |

## Headline (single image, 4 threads)

- yolo11n: openvino-bf16 22.3 ms vs PyTorch FP32 66.0 ms median p50 (ratio of the medians 2.96x); speedup **2.84x**, the median of 3 paired per-round ratios (range 2.83-3.13x; mAP50-95 +0.0001 (95% CI [-0.0019, +0.0028])). AMX/VNNI CPU: INT8/BF16 speedups on a typical laptop are smaller.
- yolo11s: openvino-bf16 30.6 ms vs PyTorch FP32 121.9 ms median p50 (ratio of the medians 3.98x); speedup **3.85x**, the median of 3 paired per-round ratios (range 3.19-3.98x; mAP50-95 -0.0015 (95% CI [-0.0042, +0.0005])). AMX/VNNI CPU: INT8/BF16 speedups on a typical laptop are smaller.

## Sliced inference: latency cost (single image, 4 threads)

One call = all tiles of one image plus the full-image pass, merged; the inference count is the mean over the benchmark images (it grows with the image size).

| model | backend | tile (px) / overlap | inferences per image | p50 / p95 / p99 ms | untiled p50 ms | slowdown |
|---|---|---|---|---|---|---|
| yolo11n | openvino-fp32 | 512 / 20% | 6.8 | 203.0 / 269.2 / 314.1 (197.9-217.5) | 33.7 | 6.0x |
| yolo11s | openvino-fp32 | 512 / 20% | 6.8 | 455.6 / 547.2 / 586.2 (451.2-476.9) | 68.7 | 6.6x |
