# Model qualification

A model is "supported" only after its integration test passes on the target
machine. Nothing here claims support for any specific real model yet.

## Tokenizer

Source: `src/tokenizer.rs`, built from GGUF metadata.

Supported:

- `tokenizer.ggml.model == "gpt2"` (byte-level BPE) with `tokenizer.ggml.pre`
  equal to `"llama-bpe"` or absent (absent is treated as llama-bpe; do not rely
  on that for non-Llama-3 models). Any other value returns
  `unsupported_architecture` naming it.
- Merge ranks from `tokenizer.ggml.merges`; pre-tokens that are whole vocabulary
  entries are emitted directly (llama.cpp's `ignore_merges` for llama-bpe).
- Control tokens (`token_type == 3`) are matched literally in input text and
  decode to empty bytes unless explicitly requested. `encode` (raw text) parses
  them, as llama.cpp does with parse_special. Chat prompts must be built with
  `encode_chat`, which inserts structural tokens as ids and encodes role names
  and message content with `encode_plain` (no special-token parsing), so user text
  such as `<|eot_id|>` stays literal text. `apply_chat` (string output) is for
  display/debugging only.
- Vocabularies lacking any of the 256 byte-level tokens are rejected as
  `invalid_model`.
- Streaming output goes through `Utf8Stream`, which holds incomplete trailing
  UTF-8 and replaces genuinely invalid bytes with U+FFFD.

Caveats:

- Unicode approximation: the Llama 3 pre-tokenizer regex uses `\p{L}`, `\p{N}`
  and `\s`. Rust std has no general-category tables (no new dependencies are
  allowed), so the hand-written splitter uses `char::is_alphabetic` (minus
  numerics), `char::is_numeric` and `char::is_whitespace`. Exact for ASCII and
  nearly all common text; may differ for combining marks and a few number or
  symbol categories, which changes token boundaries only for rare scripts.
- Chat template: a Llama 3 instruct template is hardcoded
  (`<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` per message,
  then an open assistant header). The model's Jinja `tokenizer.chat_template`
  is not read or interpreted. Models without those header tokens return
  `unsupported_architecture`; roles other than system/user/assistant/tool return
  `invalid_request`.
- Fidelity against llama.cpp is unverified on a real model. The unit tests use
  a synthetic vocabulary. The test `real_model_tokenizer_matches_known_ids` in
  `tests/tokenizer.rs` runs only when `SALLY_TEST_MODEL` points at a Llama 3
  GGUF (it asserts `encode("Hello world") == [9906, 1917]` and round-trips
  several strings). It has not been run; it must be run on the target machine
  before the tokenizer is considered qualified.

## Llama architecture adapter

Status: **implemented, verified only against a synthetic reference; NOT yet run on a real model.**

### Supported
- `general.architecture == "llama"`, dense only.
- RMSNorm, grouped-query attention (any `head_count` that is a multiple of `head_count_kv`), SwiGLU MLP.
- RoPE as ggml's `GGML_ROPE_TYPE_NORMAL` (adjacent pairs), with the optional `rope_freqs.weight` factors that Llama 3.x GGUFs carry.
- Tied embeddings (no `output.weight`: `token_embd.weight` is reused for the output projection).
- Weight types F32, F16, Q4_0, Q8_0, Q4_K, Q6_K (norm vectors must be F32 or F16).

### Refused at load time (typed error, never ignored)
- Any other `general.architecture` (`unsupported_architecture`, names the value), including `glm_moe_dsa`.
- Mixture-of-experts: tensors containing `_exps` or `ffn_gate_inp`, or `llama.expert_count > 0`.
- Weight biases (any tensor ending `.bias`).
- `llama.rope.scaling.type` other than `none` (yarn, linear, longrope...). Llama 3.x GGUFs normally express their scaling only through `rope_freqs.weight`, which is supported. If a real Llama 3.x file is refused for this reason, `inspect` will show the key; it means the scaling is not implemented, and ignoring it would change the model.
- Per-layer head counts (array metadata), differing key/value head sizes.
- Missing required tensors or metadata, shapes that disagree with the metadata (`invalid_model`, names the tensor).
- Unsupported tensor types (`unsupported_architecture`, names the tensor), checked at load so generation never fails halfway.
- Context beyond the model's trained `context_length` (`resource_exhausted`).

### How correctness was established
- `tests/generation.rs` compares the engine's logits with an independent naive f64 forward pass written in the test file (no shared code) over a 6-token sequence, for: GQA, plain multi-head attention, `rope_freqs.weight`, tied embeddings and Q8_0 weights. Tolerance 1e-4 absolute plus 1e-4 relative.
- The comparison was mutation-checked: flipping the RoPE sign, dropping the freq-factor division, breaking the GQA head mapping and changing the attention scale each make the reference tests fail.
- RoPE semantics were read from ggml source (master on 2026-10-27): `LLM_ARCH_LLAMA` maps to `LLAMA_ROPE_TYPE_NORM` in `src/llama-model.cpp`; `ggml/src/ggml-cpu/ops.cpp` `ggml_rope_cache_init` divides `theta` by `freq_factors[i0/2]` and multiplies by `theta_scale = freq_base^(-2/n_dims)` per pair; `rotate_pairs` with offset 1 rotates `(x[i], x[i+1])`.
- **Not verified:** agreement with llama.cpp logits on a real checkpoint. `real_model_smoke` (set `SALLY_TEST_MODEL=/path/to/llama-3.2-1b-instruct-q4_k_m.gguf`) must be run on the target machine; it asserts that greedy generation for "The capital of France is" repeats exactly and contains "Paris".

### Speed on the target machine

| Model | Quant | Threads | Prefill tok/s | Decode tok/s | Peak RSS |
|---|---|---|---|---|---|
| llama-3.2-1b-instruct | Q4_K_M | - | NOT MEASURED YET | NOT MEASURED YET | NOT MEASURED YET |

Known performance limits of this first version (correctness first): scalar kernels, rows dequantized into scratch before each dot product, threads spawned per matrix-vector product, and prefill processes one token at a time.

## Memory policy (spec section 7)

| Resource | Policy | Enforced by |
|---|---|---|
| Model weights | Memory-mapped, file-backed. **Reported, not charged** to the budget: the OS pages them in and out, which costs speed, not heap. `generate` prints the mapped size and warns when it exceeds `MemAvailable`. | `tensor_store`, CLI warning |
| KV cache | f32, `2 * layers * ctx * kv_heads * head_dim * 4` bytes. Charged to the budget before allocation; a refusal says how many tokens of context would fit. | `memory::open_cache` |
| Scratch buffers | Allocated once at model load, reused; charged to the budget. | `Architecture::scratch_bytes` |
| Context | A request whose cached + prompt + output tokens exceed `min(cache, model context)` is refused with `resource_exhausted`, never truncated. | `generate::generate` |
| Concurrency | One generation at a time (scheduler arrives with the API). | Task 7 |
| Failure | Typed `resource_exhausted` with the numbers, no repeated OOM attempts. | `memory::Budget::reserve` |

The budget is `--memory-mb` (default 2048). It bounds the engine's own working memory only. Peak RSS (`VmHWM`) is printed after every run so the real figure is always visible next to the budget; if RSS is far above the budget, the difference is paged-in weights.

Honest limits: the budget does not cover allocator overhead, thread stacks or the OS page cache, and a model whose weights do not fit in RAM will run, but from disk, slowly. Nothing is swapped to hide this.

Verified by tests: reservations are released on drop and are atomic across threads; an oversized KV cache is refused naming the bytes needed and the context that would fit; RSS tracks a real 64 MiB allocation; 20 repeated requests grow RSS by less than 5% after warm-up (checked to fail when a 256 KiB/request leak is injected); logs carry no request text.

## Streaming and cancellation (measured)

`stream: true` returns Server-Sent Events over chunked transfer encoding (`data: {chunk}` events ending with `data: [DONE]`); errors found before generation starts (validation, full queue, context overflow) are still ordinary JSON errors with their normal status. When the client disconnects, the write fails, the job is cancelled between tokens and its KV reservation is released.

Measured on the synthetic tiny model with an injected 40 ms per forward step (test `disconnect_cancels_and_frees_kv_and_the_next_request_works`): the budget returned to zero 187 ms after the client closed the socket (about one to four token steps), and the next request succeeded. With a real model the latency is one token step (the loop checks cancellation between tokens), so it scales with decode time per token; prefill is also token-by-token, so a cancel during a long prompt takes effect after the current token. Not yet measured on the real 1B model.

Known gap: a streamed request that times out mid-stream ends with an error event and `[DONE]`; this path is implemented but has no dedicated test (the non-streaming timeout path does).
