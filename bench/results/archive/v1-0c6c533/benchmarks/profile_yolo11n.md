# OpenVINO per-node profile (yolo11n, batch 1, 4 threads)

Sum of per-node `real_time` (median over the profiled runs) and its split by the
precision of the kernel that ran each node. Profiling overhead included.

| configuration | sum of node times (ms) | time by kernel precision | nodes by precision |
|---|---|---|---|
| openvino-fp32 | 25.6 | F32 100% | F32 162 |
| openvino-bf16 | 13.5 | BF16 95%, F32 5% | BF16 150, F32 12 |
| openvino-int8 | 15.4 | I8 76%, F32 24% | F32 54, I8 124 |

## openvino-fp32: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_f32` | 11.87 | 46% |
| `brgconv_avx512_1x1_f32` | 8.31 | 32% |
| `ref_f32` | 2.21 | 9% |
| `jit_avx512_f32` | 1.09 | 4% |
| `reorder_brgemm_f32` | 0.82 | 3% |
| `brgconv_avx512_dw_f32` | 0.57 | 2% |
| `reorder_f32` | 0.41 | 2% |
| `brgemm_avx512_f32` | 0.32 | 1% |

## openvino-bf16: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_bf16` | 4.74 | 35% |
| `brgconv_avx512_amx_1x1_bf16` | 4.62 | 34% |
| `ref_bf16` | 1.19 | 9% |
| `jit_avx512_bf16` | 0.79 | 6% |
| `brgconv_avx512_dw_bf16` | 0.59 | 4% |
| `jit_uni_bf16` | 0.59 | 4% |
| `jit_uni_f32` | 0.25 | 2% |
| `ref_f32` | 0.24 | 2% |

## openvino-int8: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_I8` | 6.03 | 39% |
| `brgconv_avx512_amx_1x1_I8` | 3.84 | 25% |
| `jit_avx512_f32` | 1.42 | 9% |
| `ref_f32` | 0.91 | 6% |
| `jit_avx512_I8` | 0.58 | 4% |
| `reorder_f32` | 0.44 | 3% |
| `ref_I8` | 0.43 | 3% |
| `reorder_brgemm_f32` | 0.43 | 3% |

Measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units).
