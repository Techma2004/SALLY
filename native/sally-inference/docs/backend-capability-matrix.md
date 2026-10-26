# Backend capability matrix

Backend: CPU, scalar kernels, correctness first. No SIMD, no GPU, no linked C
library. Rows are dequantized into an f32 scratch buffer and dotted in f32;
rows are split across `std::thread::scope` workers in contiguous chunks.
Results are bit-identical for any thread count.

## Supported tensor types (matvec and dequant_row)

| ggml type | id | block (elements / bytes) | layout verified against ggml |
|-----------|----|--------------------------|------------------------------|
| F32       | 0  | 1 / 4                    | plain little-endian f32 |
| F16       | 1  | 1 / 2                    | IEEE binary16, own converter (subnormals, inf, NaN) |
| Q4_0      | 2  | 32 / 18                  | `block_q4_0`, `dequantize_row_q4_0` |
| Q8_0      | 8  | 32 / 34                  | `block_q8_0`, `dequantize_row_q8_0` |
| Q4_K      | 12 | 256 / 144                | `block_q4_K`, `get_scale_min_k4`, `dequantize_row_q4_K` |
| Q6_K      | 14 | 256 / 210                | `block_q6_K`, `dequantize_row_q6_K` |

## Verified against

Files fetched from `raw.githubusercontent.com/ggml-org/llama.cpp/master/` on
2026-10-26 (the `master` branch head reported by `git ls-remote` at that time
was `63b4b303bd87456adc2f09d56c7e1f74e9e26673`; the files were fetched from the
`master` ref, not pinned to that commit):

- `ggml/src/ggml-common.h` (sha256 `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`):
  `block_q4_0`, `block_q8_0`, `block_q4_K`, `block_q6_K`, `QK_K = 256`, `K_SCALE_SIZE = 12`.
- `ggml/src/ggml-quants.c` (sha256 `5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab`):
  `dequantize_row_q4_0`, `dequantize_row_q8_0`, `dequantize_row_q4_K`,
  `dequantize_row_q6_K`, `get_scale_min_k4`.

The kernels are a Rust re-implementation of those dequantization routines; no
ggml source is copied. Verification was by reading the C source and by unit
tests that build block bytes by hand and compute expected values analytically
(`tests/kernels.rs`). No real model file has been tested.

## Not supported

Any other tensor type returns `unsupported_architecture` naming the type id:
Q4_1, Q5_0, Q5_1, Q8_1, Q2_K, Q3_K, Q5_K, Q8_K, IQ* quants, BF16, I8/I16/I32,
TQ*, MXFP4 and others. Also not supported: mixture-of-experts routing, MLA
(multi-head latent attention), sparse attention, GPU backends, SIMD kernels,
fused quantized dot products (activations stay f32).
