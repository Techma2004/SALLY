# SALLY Rust-Native Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A loopback-only Rust inference service that loads a known-compatible small GGUF model, generates repeatable text under measured memory limits, and serves it to SALLY and Hermes through an OpenAI-compatible local API.

**Architecture:** One crate, `sally-inference` (library plus binary), layered exactly as spec section 5: GGUF reader, tensor store (mmap), CPU backend (own minimal kernels), architecture adapter (Llama-family first), tokenizer, memory budget, scheduler, HTTP API. Unsupported architectures, tensor types or oversize requests fail with typed errors; nothing is silently degraded.

**Tech Stack:** Stable Rust (cargo 1.97 available). Dependencies, each justified: `memmap2` (mmap), `serde` + `serde_json` (config, API), `tiny_http` (small sync HTTP server, no async runtime). Everything else (GGUF parsing, dequant kernels, BPE tokenizer, sampling, arg parsing) is plain std Rust.

**Spec:** `docs/inference-engine-specification.pdf` (SALLY Rust-Native Multi-Model Inference Engine, 26 Oct 2026). Section numbers below refer to it.

## Global Constraints

- Rust-first core: loading, lifecycle, scheduling, memory policy, API, config, logging are Rust (spec 1).
- Stable Rust, small dependency set; unsafe isolated, documented, tested (spec 8).
- Checked arithmetic for every untrusted offset, count and size; typed errors, no panics on malformed input (spec 8, 10).
- Service binds `127.0.0.1` by default; never LAN/internet by default (spec 9).
- One active generation at a time initially (spec 5 scheduler).
- Unsupported architecture or tensor type is an error, never a fallback (spec 6).
- A model is "supported" only after its integration test passes; no test or doc may claim the 27B target is supported until M8 passes (spec 12).
- Structured logs must not dump full prompts by default (spec 10).
- Error codes: `unsupported_architecture`, `invalid_model`, `resource_exhausted`, `cancelled`, `internal` (spec 10).
- Model soup is excluded (spec 13). Backend is Rust-native: custom CPU kernels, no linked C library (spec 2 note; recorded in Task 4's decision doc).
- Repo is private; keep SALLY's private license untouched. If any third-party code is copied, record it in `THIRD_PARTY_NOTICES.md`.

## Review Focus

1. A GGUF with a huge metadata count or tensor dimension product must return `invalid_model` before any allocation (no OOM, no panic).
2. A prompt plus `max_tokens` exceeding the configured context must return `resource_exhausted` over the API, not truncate silently.
3. Invalid UTF-8 produced by byte-level BPE mid-token must not break streaming or JSON responses; incomplete bytes are held until completable.
4. Client disconnect or `cancel` mid-generation must release the KV cache and leave the loaded model usable for the next request.
5. A second request arriving while one is generating must queue up to the limit, then return `resource_exhausted`, never run concurrently.

---

## File Structure (path: `native/sally-inference/`)

- `Cargo.toml` — crate manifest, dependency list above.
- `src/error.rs` — `EngineError` enum with `code() -> &'static str` mapping.
- `src/config.rs` — `Config` (JSON file + CLI overrides), all limits validated.
- `src/gguf/{mod,reader,metadata,tensor_index}.rs` — parsing and validation only.
- `src/tensor_store.rs` — mmap handle, `TensorView`, bounded row reads.
- `src/backend/{mod,dequant,matmul}.rs` — CPU kernels.
- `src/architecture/{mod,llama}.rs` — model graph, RoPE, GQA attention, SwiGLU, KV cache.
- `src/tokenizer.rs` — BPE built from GGUF metadata, chat template.
- `src/sampling.rs` — temperature, top-k, top-p, seeded RNG.
- `src/model_registry.rs` — capability entries, allow-list.
- `src/memory.rs` — `Budget`, accounting, limit checks.
- `src/scheduler.rs` — single-flight queue, cancellation token.
- `src/api/{mod,types}.rs` — endpoints and schemas.
- `src/telemetry.rs` — counters, timings, RSS reader.
- `src/main.rs` — subcommands `inspect`, `generate`, `serve`, `qualify`, `bench`.
- `tests/` — integration tests; `tests/common/gguf_builder.rs` builds synthetic GGUF files.
- `docs/{backend-capability-matrix,model-qualification}.md`, `scripts/{inspect-gguf,smoke-test-local-api}`.

The SALLY repository was not available in this session, so Task 1 starts the crate standalone at `native/sally-inference/`, matching spec section 14, so it can be dropped into the repo's `native/` folder without moving files.

---

### Task 1: Audit and crate scaffold (M0)

**Files:**
- Create: `native/sally-inference/Cargo.toml`, `src/{lib,main,error,config}.rs`, `docs/audit.md`
- Test: `native/sally-inference/src/config.rs` (unit tests)

**Interfaces:**
- Produces: `EngineError` with variants `UnsupportedArchitecture(String)`, `InvalidModel(String)`, `ResourceExhausted(String)`, `Cancelled`, `Internal(String)` and `fn code(&self) -> &'static str` returning the spec 10 codes. `Config { model_path: PathBuf, context_len: usize, max_output_tokens: usize, max_body_bytes: usize, queue_len: usize, request_timeout_secs: u64, bind: SocketAddr, memory_budget_mb: usize, threads: usize }` with `Config::load(path: Option<&Path>, overrides: &[(String,String)]) -> Result<Config, EngineError>` and `Config::validate(&self) -> Result<(), EngineError>`.

- [ ] **Step 1: Write `docs/audit.md`.** Record: SALLY repo not present in this session, existing Python stack (llama-cpp-python, `sally-native` PyO3 crate) to be audited on the target machine, the Hermes provider config still to be checked against the installed version.
- [ ] **Step 2: Write failing tests** in `config.rs`: `default_bind_is_loopback` asserts `Config::default().bind.ip().is_loopback()`; `rejects_zero_context` asserts `Config { context_len: 0, ..Default::default() }.validate()` returns an error whose `code()` is `"resource_exhausted"`; `override_applies` asserts `context_len=512` override is reflected.
- [ ] **Step 3: Run** `cargo test -p sally-inference config` — Expected: FAIL (not defined).
- [ ] **Step 4: Implement** `error.rs` and `config.rs` (JSON via serde, defaults: bind `127.0.0.1:8088`, context 2048, max output 512, body 1 MiB, queue 2, timeout 120s, budget 2048 MB).
- [ ] **Step 5: Run** the same command — Expected: PASS. Then `cargo build` clean.
- [ ] **Step 6: Commit** `feat: scaffold sally-inference crate, errors, config`.

### Task 2: GGUF reader and `inspect` CLI (M1)

**Files:**
- Create: `src/gguf/{mod,reader,metadata,tensor_index}.rs`, `tests/common/gguf_builder.rs`, `tests/gguf_parse.rs`, `scripts/inspect-gguf`
- Modify: `src/main.rs` (add `inspect <path>`)

**Interfaces:**
- Consumes: `EngineError`.
- Produces: `GgufFile::open(path: &Path) -> Result<GgufFile, EngineError>`; `GgufFile.version: u32`, `.metadata: Metadata`, `.tensors: Vec<TensorInfo>`, `.data_offset: u64`; `Metadata::get_str/get_u32/get_u64/get_f32/get_str_array(&self, key: &str) -> Option<…>`; `TensorInfo { name: String, dims: Vec<u64>, dtype: GgmlType, offset: u64, n_bytes: u64 }`; `enum GgmlType { F32, F16, Q4_0, Q8_0, Q4_K, Q6_K, Other(u32) }`; `GgufFile::report(&self) -> String`.
- Builder (tests only): `GgufBuilder::new().kv_str(k,v).kv_u32(k,v).tensor(name, dims, GgmlType, bytes).build() -> Vec<u8>`.

- [ ] **Step 1: Write failing tests** in `tests/gguf_parse.rs`: `parses_valid_minimal_file` (metadata `general.architecture="llama"`, one F32 tensor, asserts name, dims, offset); `rejects_bad_magic`; `rejects_truncated_header`; `rejects_tensor_range_past_eof`; `rejects_impossible_tensor_count` (count `u64::MAX`); `rejects_dim_product_overflow`; `rejects_unsupported_version` (version 1). Each malformed test asserts `err.code() == "invalid_model"`.
- [ ] **Step 2: Run** `cargo test --test gguf_parse` — Expected: FAIL.
- [ ] **Step 3: Implement the reader** with a bounds-checked cursor (`checked_add`, `checked_mul`; cap counts against remaining file bytes before allocating Vecs). Tensor `n_bytes` computed from dtype block size, validated `offset + n_bytes <= file_len - data_offset`. Alignment from `general.alignment` (default 32).
- [ ] **Step 4: Run** the same tests — Expected: PASS.
- [ ] **Step 5: Add `inspect` subcommand** printing `report()`; exit code 2 for invalid or unsupported files. `scripts/inspect-gguf` wraps `cargo run --release -- inspect "$@"`. Verify on a synthetic file written by a test helper binary or on the user's `llama-3.2-1b-instruct-q4_k_m.gguf` (record architecture, tensor types, size).
- [ ] **Step 6: Add a fuzz-style test** `random_bytes_never_panic` feeding 2,000 seeded random buffers to the parser; Expected: all return `Err`/`Ok`, no panic.
- [ ] **Step 7: Commit** `feat: checked GGUF reader and inspect CLI`.

### Task 3: Tensor store and dequantization kernels

**Files:**
- Create: `src/tensor_store.rs`, `src/backend/{mod,dequant,matmul}.rs`, `tests/kernels.rs`

**Interfaces:**
- Consumes: `GgufFile`, `TensorInfo`, `GgmlType`.
- Produces: `TensorStore::open(file: &GgufFile, path: &Path) -> Result<TensorStore, EngineError>` (mmap via memmap2, the only `unsafe` in the crate, wrapped in one documented function); `TensorStore::view(&self, name: &str) -> Result<TensorView<'_>, EngineError>`; `TensorView { dtype: GgmlType, rows: usize, cols: usize, data: &[u8] }`; `backend::matvec(w: &TensorView, x: &[f32], out: &mut [f32]) -> Result<(), EngineError>` supporting F32, F16, Q8_0, Q4_K, Q6_K (any other dtype returns `UnsupportedArchitecture` naming the dtype); `backend::dequant_row(w: &TensorView, row: usize, out: &mut [f32])`.

- [ ] **Step 1: Write failing tests** in `tests/kernels.rs`: for each supported dtype, build a small tensor with a Rust reference quantizer inside the test, compute `matvec` and compare to an f32 reference within 1e-2 relative (1e-5 for F32); `unsupported_dtype_errors` asserts `Other(99)` yields code `"unsupported_architecture"`; `view_rejects_unknown_tensor`.
- [ ] **Step 2: Run** `cargo test --test kernels` — Expected: FAIL.
- [ ] **Step 3: Implement** block layouts exactly as in ggml (Q8_0: 32 values, f16 scale; Q4_K and Q6_K: 256-value super-blocks). Verify layouts against `ggml-common.h` in `ggml-org/llama.cpp` before writing; note the source in `docs/backend-capability-matrix.md`. Use scalar loops first, `std::thread::scope` row-parallelism over `config.threads`.
- [ ] **Step 4: Run** the same tests — Expected: PASS.
- [ ] **Step 5: Commit** `feat: mmap tensor store and CPU dequant kernels`.

### Task 4: Backend decision, tokenizer, sampling (M2)

**Files:**
- Create: `docs/backend-capability-matrix.md`, `src/tokenizer.rs`, `src/sampling.rs`, `tests/tokenizer.rs`, `tests/sampling.rs`

**Interfaces:**
- Consumes: `Metadata` (keys `tokenizer.ggml.model`, `.tokens`, `.merges`, `.token_type`, `.bos_token_id`, `.eos_token_id`, `tokenizer.chat_template`).
- Produces: `Tokenizer::from_gguf(m: &Metadata) -> Result<Tokenizer, EngineError>`; `encode(&self, text: &str, add_bos: bool) -> Vec<u32>`; `decode_piece(&self, id: u32) -> Vec<u8>`; `Utf8Stream::push(&mut self, bytes: &[u8]) -> String` (holds incomplete trailing bytes); `Tokenizer::apply_chat(&self, msgs: &[ChatMessage]) -> String` (Llama 3 header format; other templates return `UnsupportedArchitecture`). `Sampler::new(temperature: f32, top_k: usize, top_p: f32, seed: u64)`; `Sampler::sample(&mut self, logits: &mut [f32]) -> u32`; temperature 0 means argmax.

- [ ] **Step 1: Write the decision doc.** Backend options table (custom Rust kernels, link llama.cpp via FFI, candle/burn crates), the chosen option (custom kernels, Llama-family dense first, because it keeps the core self-contained and matches the default SALLY model), and what is explicitly not supported (MLA, sparse attention, MoE routing; those are Task 11).
- [ ] **Step 2: Write failing tests**: `encode_decode_roundtrip` on a synthetic vocab built in the test; `utf8_stream_holds_partial_codepoint` (feed the first 2 bytes of "€", expect `""`, then the third, expect `"€"`); `argmax_when_temperature_zero`; `same_seed_same_sequence`; `top_k_one_equals_argmax`; and `real_model_tokenizer_matches_known_ids` gated on env `SALLY_TEST_MODEL` (skipped with a printed reason when unset).
- [ ] **Step 3: Run** `cargo test --test tokenizer --test sampling` — Expected: FAIL.
- [ ] **Step 4: Implement** byte-level BPE with merge ranks from GGUF metadata, special-token handling, and a seeded xorshift RNG (no `rand` crate).
- [ ] **Step 5: Run** — Expected: PASS (gated test skipped or passing).
- [ ] **Step 6: Commit** `feat: tokenizer, sampling, backend decision doc`.

### Task 5: Llama architecture and first generation (M3)

**Files:**
- Create: `src/architecture/{mod,llama}.rs`, `src/model_registry.rs`, `tests/generation.rs`
- Modify: `src/main.rs` (add `generate --model P --prompt S --max-tokens N --temperature T --seed S`)

**Interfaces:**
- Consumes: `TensorStore`, `backend::matvec`, `Tokenizer`, `Sampler`, `Config`.
- Produces: `trait Architecture { fn forward(&mut self, token: u32, pos: usize, state: &mut KvCache) -> Result<&[f32], EngineError>; fn context_len(&self) -> usize; fn kv_bytes(&self, ctx: usize) -> usize; }`; `LlamaModel::load(store: TensorStore, meta: &Metadata) -> Result<LlamaModel, EngineError>` (reads `llama.block_count`, `.embedding_length`, `.attention.head_count`, `.attention.head_count_kv`, `.rope.freq_base`, `.attention.layer_norm_rms_epsilon`; requires every expected tensor name, otherwise `InvalidModel`); `KvCache::new(layers, kv_heads, head_dim, ctx) -> KvCache`; `registry::lookup(arch: &str) -> Result<&'static Capability, EngineError>` where only `"llama"` is in the allow-list and anything else returns `UnsupportedArchitecture`.

- [ ] **Step 1: Write failing tests**: `registry_rejects_unknown_arch` (`"glm_moe_dsa"` → code `unsupported_architecture`); `missing_tensor_is_invalid_model` (synthetic llama GGUF lacking `output_norm.weight`); `tiny_synthetic_model_is_deterministic` (hand-built 1-layer model with fixed weights; two runs with the same seed yield identical token ids, and logits match precomputed values within 1e-4); and `real_model_smoke` gated on `SALLY_TEST_MODEL`: prompt "The capital of France is", temperature 0, 8 tokens, asserts output contains "Paris" and is identical across two runs.
- [ ] **Step 2: Run** `cargo test --test generation` — Expected: FAIL.
- [ ] **Step 3: Implement** RMSNorm, RoPE (with `rope.freq_base`, handling Llama 3 scaling metadata if present, otherwise refuse with `UnsupportedArchitecture`), grouped-query attention against the cache, SwiGLU MLP, final norm and output projection (tied embeddings if `output.weight` is absent).
- [ ] **Step 4: Run** the tests — Expected: PASS; run `generate` on `llama-3.2-1b-instruct-q4_k_m.gguf` and record prefill and decode tokens/s in `docs/model-qualification.md`.
- [ ] **Step 5: Commit** `feat: llama architecture adapter and CLI generation`.

### Task 6: Memory controls and telemetry (M4)

**Files:**
- Create: `src/memory.rs`, `src/telemetry.rs`, `tests/memory.rs`
- Modify: `src/architecture/llama.rs` (scratch buffers allocated once), `src/main.rs`

**Interfaces:**
- Produces: `Budget::new(limit_bytes: usize) -> Budget`; `Budget::reserve(&self, what: &'static str, bytes: usize) -> Result<Reservation, EngineError>` (error `ResourceExhausted` naming the resource and numbers); `Reservation` releases on drop; `Budget::used(&self) -> usize`. `telemetry::rss_bytes() -> Option<u64>` (parse `/proc/self/status` `VmRSS`); `Metrics { prompt_tokens, generated_tokens, first_token_ms, tokens_per_sec, load_ms, peak_rss, errors }` and `Metrics::log_json(&self)` (no prompt text).

- [ ] **Step 1: Write failing tests**: `reserve_over_limit_fails_with_resource_exhausted`; `drop_releases_reservation`; `kv_cache_reservation_rejects_oversized_context` (context requiring more KV bytes than budget is refused at session start with a message that names KV bytes needed vs available); `rss_reader_returns_nonzero`.
- [ ] **Step 2: Run** `cargo test --test memory` — Expected: FAIL.
- [ ] **Step 3: Implement** budget accounting; reserve KV cache (`Architecture::kv_bytes`) and scratch before generation; refuse requests where `prompt_tokens + max_tokens > min(context_len, config.context_len)`.
- [ ] **Step 4: Run** — Expected: PASS. Run a 20-request loop in a test and assert RSS growth stays under 5% after warm-up.
- [ ] **Step 5: Commit** `feat: memory budget, limits, telemetry`.

### Task 7: HTTP API and scheduler (M5)

**Files:**
- Create: `src/scheduler.rs`, `src/api/{mod,types}.rs`, `tests/api.rs`
- Modify: `src/main.rs` (add `serve`)

**Interfaces:**
- Consumes: `Config`, `LlamaModel`, `Tokenizer`, `Sampler`, `Budget`, `Metrics`.
- Produces: `Scheduler::new(queue_len: usize) -> Scheduler`; `Scheduler::submit(&self, job: Job) -> Result<JobHandle, EngineError>` (`ResourceExhausted` when queue full); `CancelToken::cancel(&self)` / `is_cancelled(&self) -> bool`. Endpoints from spec 9: `GET /health` (`{"status":"ok","model_loaded":bool}`), `GET /v1/models` (only the loaded model id), `POST /v1/chat/completions` (fields `model`, `messages`, `max_tokens`, `temperature`; unknown fields are listed in the response header `x-sally-ignored-fields`), non-streaming. Errors as `{"error":{"code":<spec code>,"message":...}}` with HTTP 400/404/413/429/500/503 mapped from the codes. Admin endpoints are not built.

- [ ] **Step 1: Write failing tests** (start the server on port 0 in-process): `health_ok`; `models_lists_only_loaded_model`; `valid_chat_returns_choice` (synthetic tiny model); `invalid_json_is_400`; `unknown_model_is_404`; `oversize_body_is_413`; `excess_context_is_429_resource_exhausted` (Review Focus 2); `queue_full_returns_429` (Review Focus 5, with a slow job); `binds_loopback_by_default`.
- [ ] **Step 2: Run** `cargo test --test api` — Expected: FAIL.
- [ ] **Step 3: Implement** `tiny_http` accept loop on a thread, inference on one dedicated worker thread fed by the scheduler, a request timeout, and body limits from `Config`. Logging is structured JSON with token counts only.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat: local OpenAI-compatible API and scheduler`.

### Task 8: Hermes integration and smoke test (M6)

**Files:**
- Create: `scripts/smoke-test-local-api`, `docs/hermes-integration.md`

- [ ] **Step 1: Read the installed Hermes version's provider docs** (`NousResearch/hermes-agent` and the local install) to find the exact setting names for a custom OpenAI-compatible endpoint; do not guess them.
- [ ] **Step 2: Write `scripts/smoke-test-local-api`** (shell with curl): checks `/health`, reads the model id from `/v1/models`, sends a short prompt, exits non-zero on any failure or empty answer; also exercises the four failure cases from spec 9 step 6 (engine stopped, unsupported model, memory limit, cancelled request).
- [ ] **Step 3: Run it** against `serve` with the real model — Expected: all checks print PASS.
- [ ] **Step 4: Document** the verified Hermes configuration in `docs/hermes-integration.md`, and run one end-to-end Hermes prompt. If Hermes is not installed in the environment, say so in the doc and leave the end-to-end step explicitly unchecked.
- [ ] **Step 5: Commit** `docs: hermes integration and smoke test`.

### Task 9: Streaming and cancellation (M7)

**Files:**
- Modify: `src/api/mod.rs`, `src/scheduler.rs`, `src/architecture/llama.rs`
- Test: `tests/streaming.rs`

**Interfaces:**
- Consumes: `CancelToken`, `Utf8Stream`.
- Produces: `stream: true` support returning SSE chunks `data: {…}\n\n` ending with `data: [DONE]`; generation loop checks `CancelToken` between tokens; client disconnect triggers cancel.

- [ ] **Step 1: Write failing tests**: `stream_emits_incremental_chunks_then_done`; `multibyte_text_never_split_across_chunks` (Review Focus 3); `disconnect_cancels_and_frees_kv` (Review Focus 4, asserts `Budget::used()` returns to baseline); `next_request_works_after_cancel`.
- [ ] **Step 2: Run** `cargo test --test streaming` — Expected: FAIL.
- [ ] **Step 3: Implement** SSE writer, per-token cancel check, and cleanup on every exit path.
- [ ] **Step 4: Run** — Expected: PASS. Record cancellation latency in `docs/model-qualification.md`.
- [ ] **Step 5: Commit** `feat: streaming and cancellation`.

### Task 10: Qualification tool and 27B+ model gate (M8)

**Files:**
- Create: `src/main.rs` (subcommand `qualify <path>`), `tests/qualify.rs`, extend `docs/model-qualification.md`

**Interfaces:**
- Consumes: `GgufFile`, `registry::lookup`, `Budget`, `Architecture::kv_bytes`.
- Produces: `qualify(path, &Config) -> QualificationReport { architecture: String, supported: bool, unsupported_ops: Vec<String>, weight_bytes: u64, dense_bytes: Option<u64>, kv_bytes_at_ctx: u64, fits_in_budget: bool, verdict: String }` and exit code 0 only when `supported && fits_in_budget`.

- [ ] **Step 1: Write failing tests** with synthetic GGUFs: `unknown_architecture_reports_unsupported`; `moe_expert_tensors_flagged_unsupported` (names containing `ffn_gate_exps`); `oversize_model_reports_does_not_fit` (budget smaller than weights without streaming); `supported_small_model_passes`.
- [ ] **Step 2: Run** `cargo test --test qualify` — Expected: FAIL.
- [ ] **Step 3: Implement** the report. For MoE or MLA or sparse-attention architectures it lists exactly which required operations are missing; it never recommends a fallback.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Document** the GLM 5.2 audit from this session in `docs/model-qualification.md` with its sources: about 744B total and about 40B active parameters, 78 layers (3 dense, 75 MoE), 256 routed experts, 8 routed per token, MLA with sparse attention; smallest GGUF about 217 GB. Verdict: not qualified on a 3.7 GiB machine (dense part and per-token expert reads exceed what 4 GB and a laptop disk can serve); mainline llama.cpp also lacks sparse attention for it. Mark these figures as secondhand and to be re-verified with `inspect` on the real file.
- [ ] **Step 6: Commit** `feat: qualify command and model qualification doc`.

### Task 11: Benchmarks and tuning (M9)

**Files:**
- Create: `benches/run.rs` (or `bench` subcommand), extend `docs/model-qualification.md`

- [ ] **Step 1: Implement `bench`** recording the spec 7 matrix: load time, first-token latency, warm tokens/s, peak RSS for short prompt, long prompt, repeated generation, and cancellation, printed as one JSON object per row.
- [ ] **Step 2: Run on the target Debian machine** with the real model and paste the numbers into the doc (cold and warm separately). If run elsewhere, label the hardware.
- [ ] **Step 3: Tune one stated metric at a time** (for example SIMD in `matvec` or thread count); keep a change only if the stated metric improves and Task 3 and Task 5 tests still pass.
- [ ] **Step 4: Commit** `perf: benchmarks and measured tuning`.

---

## Not in this plan (need their own spec, then plan)

The spec does not cover these items from your request, so I left them out rather than guess: expert streaming for MoE models (spec 4 says to add it after correctness, on a verified MoE adapter), weight converter to a smaller local format (`llama.cpp` quantization tools cover GGUF requantization for now), grammar-constrained JSON output, speculative decoding, persistent KV cache across restarts, and desktop/web front ends. Each builds on the Task 5 `Architecture` and `KvCache` interfaces.

## Self-Review Notes

- Spec coverage: sections 1–2 (Tasks 1, 7), 3 and 15 (Tasks 6, 10), 4 (tensor store design, expert streaming deferred above), 5–6 (Tasks 2–5), 7 (Tasks 6, 11), 8 (Global Constraints, Task 3), 9 (Tasks 7–8), 10 (Tasks 6, 7), 11 (all milestones M0–M9 mapped), 12 (tests in every task), 13 (excluded), 14 (File Structure).
- Type consistency checked: `EngineError`, `Config`, `GgufFile`, `TensorStore`/`TensorView`, `Architecture`/`KvCache`, `Tokenizer`/`Utf8Stream`, `Sampler`, `Budget`, `Scheduler`/`CancelToken` are each defined once and reused with the same signatures.
