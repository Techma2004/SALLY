# Hermes integration

Status: **configuration documented from Hermes' docs; not tested end-to-end.**
Hermes is not installed in the build environment, so no Hermes prompt has been run
against this server. `scripts/smoke-test-local-api` verifies the API itself.

## What Hermes needs (from its docs, https://hermes-agent.nousresearch.com/docs/integrations/providers)

Any server with an OpenAI-compatible `/v1/chat/completions`. In `config.yaml`:

```yaml
model:
  provider: custom
  default: <model id from GET /v1/models>   # for the tiny engine: the GGUF file name without .gguf
  base_url: http://127.0.0.1:8088/v1
  api_key: ""                               # local server, no key
  context_length: <the --context value passed to serve>
```

Or run `hermes model`, choose the custom endpoint option, and enter the same base URL
and model name. `LLM_MODEL` is removed in current Hermes; `OPENAI_BASE_URL` applies
only to the `openai-api` provider.

## Known incompatibilities today (honest status)

| Hermes expectation (per its docs) | This engine |
|---|---|
| At least 64,000 tokens of context for agent use; smaller windows are rejected at startup | Default context is 2048 and is bounded by memory (KV cache is f32). Llama-3.2-1B at 64k needs roughly 4 GiB of KV cache, far over a 3.7 GiB machine's budget. |
| Tool calling | Not implemented; `tools` is refused with a 400 naming the parameter. |
| Streaming | Implemented (SSE, chunked); not tested with Hermes itself. |

So Hermes **cannot currently use this server for its agent mode**. What works today is
plain and streaming chat completions (for example via `curl` or any simple OpenAI
client). Closing the gap requires tool-call support and a way to
reach a 64k context in the available memory (quantised KV cache); none of these are
done, and none are promised by this document.

## Verifying the server

```sh
sally-inference serve --model models/llama-3.2-1b-instruct-q4_k_m.gguf --context 2048
scripts/smoke-test-local-api            # defaults to http://127.0.0.1:8088
```

Checks: health, model id, a non-empty completion, unsupported-model refusal (404),
context/output limit refusal (400 `resource_exhausted`), the server staying usable after
a client abandons a request, and a clear failure when the engine is not running.
It was run against the synthetic tiny model: all PASS. It has not yet been run against
the real 1B model, and with the tiny model the abandoned request completes before the
client gives up, so the cancel check there proves only that the server stays healthy;
server-side cancellation itself is covered by `tests/api.rs`
(`a_request_that_runs_too_long_times_out_and_is_cancelled`).

## End-to-end Hermes prompt

- [ ] Install Hermes, apply the configuration above, send one prompt. Blocked by the
      incompatibilities listed above.
