# OpenVINO per-node profile (yolo11s, batch 1, 4 threads)

Sum of per-node `real_time` (median over the profiled runs), the infer request's wall
time (median; per-node times are the plugin's and need not add up to it), and the split
of node time by the precision of the kernel that ran each node. Profiling overhead
included.

| configuration | node sum (ms) | request wall (ms) | time by kernel precision | nodes by precision | other load (cores) |
|---|---|---|---|---|---|
| openvino-fp32 | 66.89 | 65.89 | F32 100% | F32 169 | 0.09 |
| openvino-bf16 | 25.52 | 25.06 | BF16 97%, F32 3% | BF16 153, F32 12 | 0.11 |
| openvino-int8 | 29.28 | 29.06 | I8 80%, F32 20% | F32 57, I8 125 | 0.07 |
| openvino-int8-bf16 | 28.44 | 29.97 | I8 79%, BF16 13%, F32 8% | BF16 43, F32 34, I8 148 | 0.05 |

## Time by node type (ms per inference)

| node type | openvino-fp32 | openvino-bf16 | openvino-int8 | openvino-int8-bf16 |
|---|---|---|---|---|
| Add | 0.74 | 0.47 | 0.33 | 0.26 |
| Concat | 3.07 | 1.60 | 1.02 | 1.11 |
| Convolution | 57.57 | 19.72 | 20.93 | 19.15 |
| FakeQuantize | - | - | 2.06 | 1.08 |
| GroupConvolution | 1.16 | 0.97 | 1.04 | 0.83 |
| Interpolate | 0.36 | 0.20 | 0.15 | 0.14 |
| MatMul | 0.45 | 0.21 | 0.50 | 0.42 |
| Reorder | 1.43 | 0.87 | 1.25 | 4.11 |
| Softmax | 0.31 | 0.30 | 0.34 | 0.27 |
| VariadicSplit | 1.28 | 0.76 | 1.19 | 0.68 |

## openvino-fp32: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_f32` | 32.67 | 49% |
| `brgconv_avx512_1x1_f32` | 24.90 | 37% |
| `ref_f32` | 4.35 | 6% |
| `jit_avx512_f32` | 1.83 | 3% |
| `brgconv_avx512_dw_f32` | 1.16 | 2% |
| `reorder_brgemm_f32` | 1.01 | 2% |
| `brgemm_avx512_f32` | 0.45 | 1% |
| `reorder_f32` | 0.38 | 1% |

## openvino-bf16: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_bf16` | 10.34 | 40% |
| `brgconv_avx512_amx_1x1_bf16` | 9.38 | 37% |
| `ref_bf16` | 2.11 | 8% |
| `jit_avx512_bf16` | 1.16 | 5% |
| `brgconv_avx512_dw_bf16` | 0.97 | 4% |
| `jit_uni_bf16` | 0.61 | 2% |
| `jit_uni_f32` | 0.26 | 1% |
| `ref_f32` | 0.25 | 1% |

## openvino-int8: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_I8` | 11.82 | 40% |
| `brgconv_avx512_amx_1x1_I8` | 8.26 | 28% |
| `jit_avx512_f32` | 2.62 | 9% |
| `ref_f32` | 1.48 | 5% |
| `jit_avx512_I8` | 0.98 | 3% |
| `jit_avx512_amx_1x1_I8` | 0.77 | 3% |
| `ref_I8` | 0.72 | 2% |
| `brgconv_avx512_dw_I8` | 0.55 | 2% |

## openvino-int8-bf16: top kernels

| kernel | ms | share |
|---|---|---|
| `brgconv_avx512_amx_I8` | 11.11 | 39% |
| `brgconv_avx512_amx_1x1_I8` | 7.33 | 26% |
| `jit_uni_bf16` | 2.42 | 8% |
| `jit_avx512_f32` | 1.29 | 5% |
| `jit_uni_I8` | 1.24 | 4% |
| `jit_avx512_I8` | 0.83 | 3% |
| `ref_I8` | 0.80 | 3% |
| `ref_bf16` | 0.68 | 2% |

Measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units).
