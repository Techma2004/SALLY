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
