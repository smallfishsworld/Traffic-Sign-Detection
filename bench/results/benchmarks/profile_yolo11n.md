# OpenVINO per-node profile (yolo11n, batch 1, 4 threads)

Sum of per-node `real_time` (median over the profiled runs), the infer request's wall
time (median; per-node times are the plugin's and need not add up to it), and the split
of node time by the precision of the kernel that ran each node. Profiling overhead
included.

| configuration | node sum (ms) | request wall (ms) | time by kernel precision | nodes by precision | other load (cores) |
|---|---|---|---|---|---|
| openvino-fp32 | 24.89 | 26.64 | F32 100% | F32 162 | 0.04 |
| openvino-bf16 | 13.06 | 15.17 | BF16 95%, F32 5% | BF16 150, F32 13 | 0.08 |
| openvino-int8 | 16.41 | 17.85 | I8 77%, F32 23% | F32 54, I8 124 | 0.16 |
| openvino-int8-bf16 | 18.45 | 20.28 | I8 74%, BF16 15%, F32 11% | BF16 44, F32 34, I8 149 | 0.07 |

## Time by node type (ms per inference)

| node type | openvino-fp32 | openvino-bf16 | openvino-int8 | openvino-int8-bf16 |
|---|---|---|---|---|
| Add | 0.38 | 0.33 | 0.18 | 0.19 |
| Concat | 1.47 | 0.95 | 0.77 | 0.75 |
| Convolution | 19.15 | 9.06 | 11.07 | 11.36 |
| FakeQuantize | - | - | 1.08 | 0.99 |
| GroupConvolution | 0.55 | 0.56 | 0.56 | 0.55 |
| MatMul | 0.30 | 0.13 | 0.27 | 0.31 |
| Reorder | 1.26 | 0.82 | 1.18 | 3.12 |
| Sigmoid | 0.18 | 0.16 | 0.16 | 0.17 |
| Softmax | 0.22 | 0.25 | 0.20 | 0.22 |
| VariadicSplit | 0.70 | 0.47 | 0.65 | 0.44 |

## openvino-fp32: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_f32` | 11.17 | 45% |
| `brgconv_avx512_1x1_f32` | 7.98 | 32% |
| `ref_f32` | 2.17 | 9% |
| `jit_avx512_f32` | 1.10 | 4% |
| `reorder_brgemm_f32` | 0.83 | 3% |
| `brgconv_avx512_dw_f32` | 0.55 | 2% |
| `reorder_f32` | 0.39 | 2% |
| `brgemm_avx512_f32` | 0.30 | 1% |

## openvino-bf16: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_bf16` | 4.64 | 36% |
| `brgconv_avx512_amx_1x1_bf16` | 4.42 | 34% |
| `ref_bf16` | 1.17 | 9% |
| `jit_avx512_bf16` | 0.77 | 6% |
| `jit_uni_bf16` | 0.58 | 4% |
| `brgconv_avx512_dw_bf16` | 0.56 | 4% |
| `ref_f32` | 0.25 | 2% |
| `jit_uni_f32` | 0.24 | 2% |

## openvino-int8: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_I8` | 6.48 | 39% |
| `brgconv_avx512_amx_1x1_I8` | 4.09 | 25% |
| `jit_avx512_f32` | 1.49 | 9% |
| `ref_f32` | 0.95 | 6% |
| `jit_avx512_I8` | 0.58 | 4% |
| `reorder_f32` | 0.47 | 3% |
| `ref_I8` | 0.45 | 3% |
| `reorder_brgemm_f32` | 0.45 | 3% |

## openvino-int8-bf16: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_I8` | 6.76 | 37% |
| `brgconv_avx512_amx_1x1_I8` | 4.12 | 22% |
| `jit_uni_bf16` | 1.87 | 10% |
| `jit_avx512_f32` | 1.19 | 6% |
| `jit_uni_I8` | 0.84 | 5% |
| `jit_avx512_I8` | 0.59 | 3% |
| `ref_I8` | 0.46 | 3% |
| `ref_bf16` | 0.46 | 2% |

Measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units).
