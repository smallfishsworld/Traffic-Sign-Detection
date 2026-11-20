# Run-to-run spread (yolo11n, single image, batch 1, e2e p50 ms)

| backend | threads | run 1 | run 2 | run 3 | min-max | spread |
|---|---|---|---|---|---|---|
| pytorch-fp32 | 1 | 134.1 | 119.0 | 110.3 | 110.3-134.1 | 20% |
| pytorch-fp32 | 4 | 68.0 | 64.7 | 64.4 | 64.4-68.0 | 5% |
| onnxruntime-fp32 | 4 | 38.9 | 37.5 | 38.7 | 37.5-38.9 | 4% |
| openvino-fp32 | 4 | 30.6 | 33.4 | 36.6 | 30.6-36.6 | 18% |
| openvino-bf16 | 4 | 21.7 | 20.6 | 23.3 | 20.6-23.3 | 12% |
| openvino-int8 | 4 | 25.9 | 27.2 | 24.4 | 24.4-27.2 | 11% |

Spread = (max - min) / median. Speedup over PyTorch FP32 at 4 threads, pairing run k of
each configuration with run k of the baseline (measured minutes apart):

| backend | run 1 | run 2 | run 3 | min-max |
|---|---|---|---|---|
| onnxruntime-fp32 | 1.75x | 1.73x | 1.66x | 1.66x-1.75x |
| openvino-fp32 | 2.22x | 1.94x | 1.76x | 1.76x-2.22x |
| openvino-bf16 | 3.13x | 3.14x | 2.77x | 2.77x-3.14x |
| openvino-int8 | 2.62x | 2.38x | 2.64x | 2.38x-2.64x |

Measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units); INT8 and
BF16 gains on a typical laptop CPU without AMX are smaller.
