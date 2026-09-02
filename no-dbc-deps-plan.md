# Plan: lessen dependency on dbc-pyutils / dbc-data

Based on `no-dbc-deps-research.md` (research + resolved decisions). Two
independent work streams (A: `dbc-data`/indexing, B: tornado→FastAPI +
gated `dbc_pyutils`) plus shared verification.

## Phase 1 — `dbc-data` group is already correct; fix the one CI gap

No `pyproject.toml` change needed for `dbc-data` placement (research §1
correction: the `dbc` dependency group is already excluded from a plain
`uv sync` since `[tool.uv] default-groups` is unset and defaults to
`["dev"]`).

1. `Jenkinsfile-update-vector-db`: change `uv sync --frozen` (line 41) to
   `uv sync --frozen --group dbc` so the nightly indexing job keeps
   installing `dbc-data`/`dbc_pyutils` for `create-faiss-index`. Everything
   else in that file is unchanged.
2. No changes to `src/mitcfu_rag/rag/index_vector_db.py` — its `dbc_data`/
   `dbc_pyutils.Time` imports stay hard imports (indexing already requires
   DBC hardware/Kafka; decision: keep bundled in the `dbc` group).

## Phase 2 — README: prebuilt vector DB fetch instructions

Add a new subsection to `README.md`, next to "Create faiss embeddings",
documenting the no-Kafka-access path:

- Fetch the nightly-built artifacts from Artifactory:
  `mitcfu_faiss_index.tgz` and `mitcfu_faiss_index_file.json` at
  `[MORTY_URL_HERE]` (placeholder — real URL doesn't exist yet, per
  decision).
- `tar -xzvf mitcfu_faiss_index.tgz` to get the `mitcfu_faiss_index/`
  directory consumed by `streaming-service-mitcfu`/`Dockerfile`.
- Keep `create-faiss-index` documented as-is, but note it additionally now
  requires `uv sync --group dbc` (DBC network + Kafka access) since that
  group is no longer installed by default.

## Phase 3 — Drop `tornado` from top-level deps, add FastAPI stack

`pyproject.toml`:
- Remove `"tornado"` from `dependencies`.
- Add `"fastapi"`, `"uvicorn"`, `"httpx"` to `dependencies` (matches the
  dbc-pytools starter template; `httpx` for the eventual dummy-request-style
  smoke helper / test client transport).
- `pydantic` already present — keep.
- `dbc = ["dbc-data", "dbc_pyutils"]` group unchanged (decision: no split).
- `[project.scripts]`: point `streaming-service-mitcfu` at the new module —
  `streaming-service-mitcfu = "mitcfu_rag.service.start:main"` (keeps the
  script name so `Dockerfile`/README invocations don't change).
- Run `uv lock` after editing to refresh `uv.lock` (fastapi/uvicorn/httpx
  resolve from public PyPI, no VPN needed; `tornado` drops out except as a
  transitive dep of `dbc_pyutils` inside the `dbc` group, which is fine).

## Phase 4 — New `service` package (replaces `service.py`)

Delete `src/mitcfu_rag/service.py`; add `src/mitcfu_rag/service/` following
the dbc-pytools starter layout, adapted to this app's existing behavior
(preserve `/v1/chat/completions`, `/status`, `/metrics`, CLI args exactly —
this is a lift-and-port, not a redesign).

### `service/_dbc_optional.py` (new — the single gating seam)
- At import time: `try: import dbc_pyutils as _dbc_pyutils` (plus the
  specific symbols) `except ImportError: ...`; expose module-level
  `DBC_AVAILABLE: bool` and best-effort references, `None` when
  unavailable, for: `install_base_handler`, `PrometheusMiddleware`,
  `metrics_endpoint`, `setup_logging`, `build_info`, `create_instance_id`,
  `Statistics`.
- Plain `try/except ImportError`, not `importlib.util.find_spec` —
  simpler, standard, and every one of these is a normal top-level import in
  `dbc_pyutils.__init__`.
- This module is the seam tests monkeypatch to simulate "not installed"
  (see Phase 6) without needing to actually uninstall the package.

### `service/schemas.py` (new)
- Pydantic request/response models for the OpenAI-chat-style contract
  currently hand-built as dicts in `GlyphGateHandler.post` (chat message,
  chat-completion response, non-stream `choices`/`message` shape) — only if
  this doesn't complicate preserving the exact current JSON shape; otherwise
  keep the dict-building as-is and skip schemas for this endpoint (the
  contract is dictated by `streamlit_ui.py`/OpenAI-client consumers, not by
  FastAPI's own validation needs, so schemas are optional polish here, not
  required).

### `service/endpoints.py` (new — no `dbc_pyutils` imports at all)
- `router = APIRouter()`.
- `POST /v1/chat/completions` — same logic as `GlyphGateHandler.post`:
  read `request.app.state.agentic_graph`, `request.app.state.default_model`;
  build `messages` the same way; on `stream=True` return
  `StreamingResponse(chunk_generator(), media_type="text/event-stream")`
  where the generator re-yields `result["output"]` chunks (already
  `"data: ...\n\n"`-formatted per `AgentStreamingGenerator`) then
  `"data: [DONE]\n\n"`; on non-stream, build and return the same
  `chat.completion` JSON body via `JSONResponse`.
- `GET /status` — reads `request.app.state.{dbc_available, instance_id,
  build_info, stats, ab_id}` (all populated in `create_app`, see below) and
  returns the same payload shape as the old `StatusHandler` when
  `dbc_available` is true; returns the starter template's bare
  `{"status": "ok"}` when false (decision).
- No `/metrics` route here — mounted conditionally in `start.py` only
  (decision: omitted entirely, not reimplemented, when `dbc_pyutils` is
  absent).

### `service/start.py` (new)
- `create_app(args) -> FastAPI`:
  - Build domain objects exactly as `main(args)` does today: `AgenticRAG(...)`
    then `AgenticGraph(type=args.graph_type, model=model)`.
  - If `_dbc_optional.DBC_AVAILABLE`: call `_dbc_optional.setup_logging()`;
    `instance_id = _dbc_optional.create_instance_id(num_digits=8)`;
    `info = _dbc_optional.build_info.get_info("mitcfu_rag")`;
    `stats = {"query": _dbc_optional.Statistics(name="query")}`.
    Else: `logging.basicConfig(level=logging.INFO)`; `instance_id = None`;
    `info = None`; `stats = {}`.
  - `app = FastAPI(title="mitcfu-rag service")`; stash
    `app.state.agentic_graph`, `app.state.default_model = DEFAULT_MODEL`,
    `app.state.dbc_available`, `app.state.instance_id`,
    `app.state.build_info`, `app.state.stats`, `app.state.ab_id = args.ab_id`.
  - `app.include_router(router)`.
  - If `_dbc_optional.DBC_AVAILABLE`:
    `_dbc_optional.install_base_handler(app)`;
    `app.add_middleware(_dbc_optional.PrometheusMiddleware,
    excluded_paths={"/metrics", "/status"})`;
    `app.add_api_route("/metrics", _dbc_optional.metrics_endpoint,
    methods=["GET"])`.
  - Return `app`.
- `main()`: parse args (see below), call `create_app(args)`, then
  `uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=None)`
  (`log_config=None` only makes sense once `setup_logging`/JSON formatting
  is wired above; fine either branch since we always call one of the two
  logging setups first in `create_app`).
- `parse_args()`: identical arg set to today's `cli()` (this **is** the
  merge decided in research — the starter template only contributes
  `-p/--port`, which already exists here): `embedding_model_path`,
  `faiss_path`, `--article_index_path`, `--validator-model-path`,
  `--graph-type` (default `"service"`), `--use-ceph` (flag), `-a/--ab-id`
  (default 1), `-p/--port` (default 5000), `-v/--verbose`.

### `service/__init__.py` (new)
- `from .endpoints import router` / `from .start import create_app` /
  `__all__ = ["router", "create_app"]` (matches template).

## Phase 5 — Dockerfile

- Confirm `CMD` still invokes `streaming-service-mitcfu` with the same
  positional/flag args (unchanged by this refactor since CLI shape is
  preserved) — only the implementation behind that script name changes
  from tornado to uvicorn/FastAPI.
- No `dbc` group install needed in the Docker image — the runtime service
  never imports `dbc_data`, and `dbc_pyutils` is optional (image build uses
  `uv sync --no-dev --frozen`, no `--group dbc`, so the service runs in its
  "not installed" fallback mode in this image *unless* a decision is made
  to add `--group dbc` here too so cluster deployments get the enriched
  `/status`/`/metrics`). **Needs an explicit call**: research assumed
  cluster images install the `dbc` group; today's `Dockerfile` does not.
  Recommend adding `--group dbc` to the Docker build's `uv sync` so
  `/metrics` + k8s prometheus scrape annotations keep working in
  ai-staging/ai-prod — flag this for confirmation when implementing, since
  it reintroduces the internal devpi index as a hard requirement for
  building the *service* image (acceptable: the Dockerfile already isn't
  buildable outside DBC network today, per its `FROM
  docker-dbc.artifacts.dbccloud.dk/...` base image and Artifactory `wget`s).

## Phase 6 — Tests

- New `tests/test_service.py` (or `tests/service/test_start.py`), mirroring
  `test_service_start.py.j2`'s use of `fastapi.testclient.TestClient`:
  - Build `create_app(args)` with a stub/minimal `AgenticRAG`/`AgenticGraph`
    (monkeypatch or lightweight fake — avoid loading real embedding/torch
    models in tests) and assert `/status` returns 200 with the expected
    shape.
  - Test both gate branches by monkeypatching
    `mitcfu_rag.service._dbc_optional.DBC_AVAILABLE` (and the symbol
    attributes) to `False`/back to real values around separate
    `create_app(args)` calls, asserting: `/status` → bare `{"status": "ok"}`
    and `/metrics` → 404 when unavailable; enriched `/status` and `/metrics`
    → 200 with prometheus text content-type when available (this repo's dev
    `.venv` already has `dbc_pyutils` installed, so the "available" branch
    runs for real without extra setup).
  - `POST /v1/chat/completions` non-streaming happy path against a faked
    graph, asserting the `chat.completion` response shape stays intact
    (regression check for the tornado→FastAPI port).
- No changes needed to `tests/test_knn_searcher.py` /
  `tests/test_generic_parser.py`.

## Phase 7 — README service docs

- Update "Starting the streaming service" section: same
  `streaming-service-mitcfu ...` invocation (unchanged CLI), note it's now
  FastAPI/uvicorn-based, and that interactive docs are available at
  `/docs`/`/redoc`.
- Update "How to start the service with DOCKER" section only if Phase 5's
  `--group dbc` decision changes the build command.
- Add a short "Optional `dbc_pyutils` integration" note: what's
  present/absent depending on whether `dbc_pyutils` is installed
  (`/metrics`, enriched `/status` with build/instance/query-stat info,
  structured JSON logging vs. plain stdlib logging) — so external
  contributors understand the degraded-but-working local experience.

## Phase 8 — Cleanup

- Remove the now-dead `service.py` (superseded by `service/` package) and
  any stray `__pycache__`/`.pyc` for it.
- Remove `MetricsApp`/`GlyphGateHandler`/tornado-specific classes entirely
  (no aliases/back-compat shims — clean cutover per engineering conventions).
- Double-check nothing else imports `mitcfu_rag.service:cli`/`:make_app`
  (only `pyproject.toml`'s script entry and `Dockerfile`'s `CMD` reference
  the script name, not the module path directly) — `lsp references` on the
  old `cli`/`make_app` symbols before deleting `service.py`, to be sure.

## Verification (run once implementation phases land)

1. `uv run ruff check` / `uv run ruff format --check` (pre-commit's own
   hooks).
2. `uv run pytest` — full suite including the new service tests, run
   without `--group dbc` explicitly excluded (default `uv sync` already
   excludes it) to prove the service works without DBC packages installed
   is *not* directly provable in this dev venv (dbc_pyutils is already
   installed here) — instead rely on the monkeypatch-based gate tests in
   Phase 6 to exercise the "absent" branch deterministically.
3. Manual smoke test: `uv run streaming-service-mitcfu <embedding-model>
   <faiss-index> -p 5000` (or a minimal fixture/mocked model path) and hit
   `/status`, `/metrics`, `/docs`, and a real `/v1/chat/completions` request
   (both `stream: true` and `false`) with `curl`, comparing response shape
   against the current tornado behavior documented in the research doc.
4. `uv run streamlit run src/mitcfu_rag/streamlit_ui.py` against the new
   service to confirm the SSE contract (`STREAMING_ENDPOINT`) still streams
   correctly end-to-end.
5. `uv lock` succeeds without DBC network access (proves `fastapi`/
   `uvicorn`/`httpx` resolve from public PyPI and the `dbc` group stays
   opt-in).
