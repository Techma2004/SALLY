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

## Backend decision

Decided in Task 4 (M2). Three options were considered for running the model graph.

| Option | Pros | Cons |
|--------|------|------|
| Own Rust CPU kernels (chosen) | Self-contained core; no native build dependency; every unsafe/untrusted-input boundary is ours and testable; small, auditable, works on Linux and Termux with plain `cargo` | We write and verify every kernel; slower than tuned SIMD/GPU backends until optimised |
| Link llama.cpp via FFI | Mature, fast, broad model and quant coverage | Spec section 2 says a native dependency needs its own documented license, build and compatibility boundary; adds a C++ toolchain and ABI surface to Termux/Android builds; moves loading, memory policy and cancellation out of Rust control |
| candle / burn crates | Rust-native, existing tensor ops | Large dependency trees, fast-moving APIs, would conflict with the small-dependency constraint; their GGUF and quantised support would still need validation against our limits |

Chosen: own Rust CPU kernels, Llama-family dense models first (RMSNorm, RoPE,
grouped-query attention, SwiGLU). Reasons: it keeps the core self-contained,
satisfies the spec section 2 rule that any native dependency needs its own
documented license/build/compatibility boundary (we avoid needing one), and it
matches SALLY's default model, Llama 3.2 1B.

Explicitly not supported by this decision: MLA (multi-head latent attention),
sparse attention, and mixture-of-experts routing. These belong to Task 11 and
no document or test may claim support for them before then.
