# Plan: route mitcfu-rag's LLM calls through glyph-gate instead of vLLM directly

## Background

`AgentStreamingGenerator` (`src/mitcfu_rag/rag/generators/agent_streaming_generator.py`) currently
talks straight to a vLLM deployment via `AsyncOpenAI(base_url=MITCFU_VLLM_URL, api_key="unused")`.
Available models are a static, deploy-time list (`MITCFU_VLLM_MODELS` env var), validated in
`service/endpoints.py` against `app.state.available_models`. This is the only place an LLM is
called — embeddings (`rag/retrievers/indexes/multilinguale5.py`) and the reranker
(`rag/validators/ms_marco_minilm_validator.py`) run as local HF/torch models, not via an API.

glyph-gate is an OpenAI-compatible gateway (`llm.dbc.dk`) that proxies to model backends (vLLM,
Chatterbox TTS, Triton), enforcing auth/usage limits via a companion service, **llm-access**. Its
source (`myprompts/glyph-gate-main/`) was reviewed directly to confirm the details below, and a
real token was tested live against `/v1/models` and `/v1/chat/completions` (both non-streaming and
streaming) — see "Verified live" below. The only remaining unknowns are pure ops tasks (final
prod token, deploy config).

## Verified live (real token, 2026-09-28)

- `GET /v1/models` and `POST /v1/chat/completions` both work end-to-end once the token's app
  identity is allowed for the model (see "Model id mismatch" below for the id used).
- **Streaming confirmed for real.** `stream: true` on `/v1/chat/completions` returns genuine
  incremental SSE chunks: `delta.content` pieces arrive one at a time, terminated by a
  `finish_reason: "stop"` chunk, then a final usage-only chunk (`choices: []` + `usage`), then
  `data: [DONE]`. This is exactly the shape `AsyncOpenAI`'s streaming client already expects — the
  earlier concern about the OpenAPI spec's "non-streaming" wording is fully resolved.
- **Reasoning field name differs from what the current code reads.** The model streams reasoning
  content as **`delta.reasoning`**, not `delta.reasoning_content` as
  `agent_streaming_generator.llm_generate` currently reads via
  `getattr(delta, "reasoning_content", None)`. Today this just means reasoning is silently never
  logged in debug mode (harmless), but the field name should be corrected to `reasoning` as part
  of the backend swap.
- **Model id mismatch between the registry and the backend's own name.** Requests must use the
  glyph-gate/llm-access *registry* id, `google/gemma-4-26B-A4B-internal` — not
  `google/gemma-4-26B-A4B-it` (that 400s with `model_not_found`/empty `/v1/models`... actually it's
  simply not a registered id). The response's own `model` field then reports back
  `google/gemma-4-26B-A4B-it`, vLLM's own served name for the same backend — that's just an echo,
  not what to request. **`MITCFU_VLLM_MODELS` in the Dockerfile needs to become
  `google/gemma-4-26B-A4B-internal`** when this migration lands.
- **`app_not_allowed_for_model` is not a request-side fixable error.** It's llm-access denying
  based on the token's associated `app_id` vs. the model card's `allowed_apps` restriction — not a
  field you can add to the request. Confirmed by testing: an initially-provisioned token got this
  error and an empty `/v1/models` list; it started working once the token/model's app allowlist was
  sorted out on the llm-access side.
- **Model card for `google/gemma-4-26B-A4B-internal`**: capabilities `chat_completions`,
  `responses`, `tools` (no `embeddings` — expected, it's an LLM not an embedder);
  `input_modalities: [text, image]`; `output_modalities: [text]`; `reasoning_mode: optional`.

## Confirmed from the glyph-gate source

- **Streaming is real, not just accepted-and-ignored** — confirmed live (see above).
  `VLLMAdapter.stream_chat` (`src/glyph_gate/adapters/adapter_vllm.py`) opens a streaming HTTP call
  to vLLM and proxies the raw SSE bytes through **unmodified** — there's no re-parsing into a
  response schema on that path. So the chunk shape mitcfu-rag receives is vLLM's own native
  OpenAI-compatible delta, byte-for-byte, modulo the `reasoning` field-name detail above. (The
  endpoint description text "Create a non-streaming chat completion" in the OpenAPI spec is
  misleading boilerplate — the README's compatibility table lists streaming as supported for both
  `/v1/chat/completions` and `/v1/responses`.)
- **`/v1/responses` streaming does rewrite frames** (fixing up `content_index` for
  `response.output_text.delta`/`.done` events), but that only matters if we adopt `/v1/responses`
  instead of `/v1/chat/completions` — not needed for this migration.
- **Auth is bearer-token, checked per request against llm-access.** Missing/malformed
  `Authorization: Bearer ...` → 401. Token present but denied by policy or missing the resource's
  capability → 403. Rate-limited → 429. Errors come back in the same `OpenAIErrorResponse` shape
  (`error.message/type/param/code`) that `endpoints.py`'s `openai.APIError` handling already
  expects.
- **`/v1/models` is per-token filtered**, via two independent filters: a model-side
  `allowed_apps` restriction and a token-side `allowed_models` restriction returned by
  llm-access's `/v1/authorize`. `resource="model.list"` is auto-allowed for any active token — no
  separate capability grant needed just to list models.
- **Local/dev testing needs no token at all**: running glyph-gate with `LLM_ACCESS_ENABLED=false`
  disables the llm-access integration entirely (every request allowed, nothing metered) — useful
  for integration-testing mitcfu-rag's client code against a local glyph-gate instance before a
  real token exists.
- **Capability gating is per-endpoint-resource.** Chat completions needs
  `inference.chat.completions`; if we ever want the `extra_body` passthrough escape hatch, that's
  a *separate* capability (`inference.extra_body`) granted independently.
- **Model availability is not automatic.** A model must exist in llm-access's Model Registry with
  a `model_card` (capabilities, `api_base`, adapter type, etc.) before glyph-gate will serve it —
  see the "Model cache" section of `myprompts/glyph-gate-main/README.md` and its linked Confluence
  deployment guide. Our target model needs to be registered there (or confirmed already present)
  independently of any mitcfu-rag code change.

## Phases

1. **Auth plumbing.** Add a token source (env var, e.g. `MITCFU_LLM_GATEWAY_TOKEN`, or a mounted
   secret file) and a gateway base URL (`MITCFU_LLM_GATEWAY_URL`, e.g. `https://llm.dbc.dk/v1`),
   replacing `MITCFU_VLLM_URL`. Never log the token; redact it in any error output.

2. **Swap the backend in `agent_streaming_generator.py`.** Point
   `AsyncOpenAI(base_url=..., api_key=...)` at glyph-gate instead of vLLM. Because it's
   OpenAI-compatible and streaming chat-completions frames pass through vLLM unmodified, this is
   close to a drop-in swap of `base_url`/`api_key`. One small fix needed in `llm_generate`: change
   `getattr(delta, "reasoning_content", None)` to `getattr(delta, "reasoning", None)` to match the
   field name glyph-gate/this vLLM version actually streams.

3. **Replace static model config with `GET /v1/models`.** Instead of `MITCFU_VLLM_MODELS`, fetch
   the model list from the gateway at startup (glyph-gate already filters this to what our token
   can see), cache it on `app.state`/`ModelBackend`, and use the first (or an explicitly
   configured default) as the fallback model. This replaces `served_model_names()` with a gateway
   call and keeps `endpoints.py`'s existing "reject unknown model" behavior almost unchanged.

4. **Error handling — done.** `endpoints.py` maps `openai.APIStatusError` 401/403/429 to
   `authentication_error`/`permission_denied`/`rate_limit_exceeded` with matching HTTP status
   (`_upstream_error_payload`), falling back to a generic 502 `upstream_error` for anything else.
   Also wrapped the graph's `ainvoke()` call itself (the router's own LLM call, which runs before
   any response starts) — previously an upstream error there would have been an unhandled
   exception, not the nice JSON error body. Mid-stream errors (after SSE has already sent a 200)
   still only get the improved error-body `type`, not a changed status code, since HTTP status
   can't change once streaming has started. Covered by 6 new tests in `tests/test_service.py`
   (`TestChatCompletionsUpstreamErrors`).

5. **Dockerfile / deploy config.** Drop `MITCFU_VLLM_MODELS`/vLLM URL, add the gateway URL and
   wire the token in as a secret (not a plain `ENV`). Set the default model to
   `google/gemma-4-26B-A4B-internal` (the registry id — see "Model id mismatch" above), not the
   current `google/gemma-4-26B-A4B-it`.

6. **Tests.** Update `tests/test_service.py` to mock the gateway client (models endpoint + chat
   completions) instead of `served_model_names`; add cases for token-missing/invalid and for the
   dynamic model list.

7. **Separate follow-up: embeddings, with a full re-index.** glyph-gate's `/v1/embeddings` example
   model is literally `intfloat/multilingual-e5-large-instruct`, the same model run locally today
   in two places: `multilinguale5.py`'s `e5multilingualEmbedder` (used by `index_vector_db.py` /
   `create-faiss-index` to *build* the FAISS index from Kafka) and
   `streaming_multilingual_retriever.py`'s `EmbeddingRetriever` (used at *query* time by the running
   service). Decision: don't try to verify bit-for-bit equivalence with glyph-gate's embedding
   output — assume it differs and rebuild the index rather than risk a silent, undetectable
   retrieval-quality regression from mixing embedding sources.

   This makes it a bigger, riskier piece of work than the chat-completions swap, so it's tracked as
   its own follow-up, not bundled into finishing this branch:
   - **7a.** Add a glyph-gate-backed embedder (reusing the `MITCFU_LLM_GATEWAY_URL`/
     `MITCFU_LLM_GATEWAY_TOKEN` plumbing from phase 1) that calls `POST /v1/embeddings`.
   - **7b.** Swap `index_vector_db.py` (`create-faiss-index`) to use it.
   - **7c.** Swap `EmbeddingRetriever`'s query-time embedding to use it too, dropping the local
     `AutoModel`/`AutoTokenizer` load for the embedding model specifically. The `ms-marco-MiniLM`
     cross-encoder reranker (also loaded inside `EmbeddingRetriever`) has no equivalent in the
     glyph-gate spec (no rerank capability), so it stays local regardless.
   - **7d.** Rebuild the full FAISS index from scratch via a full Kafka reingestion, using the new
     gateway-backed embedder. Deploy the new query-time code and the rebuilt index **together** —
     old-index-vectors + new-query-vectors (or vice versa) would silently corrupt retrieval with no
     error, so this can't be rolled out incrementally.
   - **7e.** Update `Jenkinsfile-update-vector-db` (the nightly rebuild job) to use the gateway
     embedder too, and provision it with a token.
   - **7f.** Requires the `inference.embeddings` capability granted on the token(s) used for
     indexing/query, separate from `inference.chat.completions`.

## Remaining open items

- **Prod vs. staging tokens.** A working token exists for at least one environment now
  (capability `inference.chat.completions`, meters `input_tokens`/`output_tokens` — see prior
  discussion). Confirm whether staging and prod need separate tokens/quotas, and who owns
  renewal/rotation.
- **~~Model registration~~ — resolved.** `google/gemma-4-26B-A4B-internal` is registered and
  working; card has `capabilities: [chat_completions, responses, tools]`,
  `reasoning_mode: optional`. Just need the Dockerfile's model id updated to match (phase 5).
