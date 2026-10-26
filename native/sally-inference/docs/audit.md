# M0 Audit — SALLY repo vs. the inference engine spec

Audited 2026-10-26 against `main` @ `b062cdb` (branch `feat/sally-inference`).

## Existing code to reuse or keep untouched
- `native/` is the `sally-native` PyO3 `cdylib` (pyo3 0.29, sysinfo, battery, chrono). It has no Cargo workspace.
  `native/sally-inference/` is a separate, independent crate; do not add a workspace root or touch `sally-native`.
- `core/llm.py` wraps `llama-cpp-python` (lazy load, `stream_chat` with `json_mode`, `timeout_s`, `cancel` Event, a generation lock).
  The Rust engine does not replace it. It will be reachable later as an OpenAI-compatible provider (roadmap M1 provider interface).
- `core/config.py` holds `settings.llm` (model path, n_ctx, n_threads, n_batch). The engine has its own config; no coupling yet.
- Default model: `models/llama-3.2-1b-instruct-q4_k_m.gguf` (gitignored, not in CI or this sandbox).

## Avoid duplicating
- Do not re-implement SALLY routing, sessions, memory or tools in Rust. The engine only does model loading, generation and the local API.
- Python-side grammar/JSON mode (`json_mode`) has no Rust equivalent yet; out of scope for this plan.

## Environment limits found
- crates.io reachable; Hugging Face blocked in the build sandbox, so tests that need a real GGUF are gated on `SALLY_TEST_MODEL` and must be run on the target Debian machine.
- Hermes provider setting names are not verified yet (Task 8).
