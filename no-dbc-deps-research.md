# Research: lessening dependency on dbc-pyutils / dbc-data

Scope note: this is a research/scoping pass only (facts + touchpoints), not an
implementation plan.

## Decisions (resolved 2026-09-02)

1. **`Time` import in `index_vector_db.py`** stays bundled in the single
   `dbc` dependency group alongside `dbc-data` — no split.
2. **FastAPI service's optional-`dbc_pyutils` gate reads off that same
   `dbc` group** (no separate group for the service-only pieces) — one
   on/off switch, installed on cluster hardware/CI, absent everywhere else.
3. **Fallback when `dbc_pyutils` is absent:** `/status` degrades to the
   starter template's bare `{"status": "ok"}` (no reimplemented build
   info/instance id/stats), and `/metrics` is omitted entirely (no local
   reimplementation of `PrometheusMiddleware`/`metrics_endpoint`).
   Prometheus metrics are not considered vital for local development.
4. **CLI args:** merge the current domain args (`embedding_model_path`,
   `faiss_path`, `--article_index_path`, `--validator-model-path`,
   `--use-ceph`, `--graph-type`, `-a/--ab-id`, `-v/--verbose`) with the
   starter template's `-p/--port`, into one `parse_args()`.
5. **README Artifactory URL:** doesn't exist yet — document the fetch
   section with a `[MORTY_URL_HERE]` placeholder rather than a real link.


## 1. Current dependency wiring (pyproject.toml)

```toml
dependencies = [
    "langchain-core", "langchain-text-splitters", "langchain-community",
    "langchain-unstructured", "langgraph", "streamlit", "scikit-learn",
    "numpy", "nltk", "sentence_transformers", "rich", "transformers",
    "torch", "accelerate", "tornado", "pydantic", "aiohttp", "requests",
    "faiss-cpu>=1.14.2", "jq",
]

[dependency-groups]
dev = ["pre-commit", "pytest", "ruff", "pip-audit"]
dbc = ["dbc-data", "dbc_pyutils"]
```

`dbc-data` and `dbc_pyutils` are already grouped under a non-`dev`
`dependency-groups.dbc` entry, and (correction, verified against `uv`
0.11.7 docs/behavior) `uv sync`'s default groups are `["dev"]` unless
`[tool.uv] default-groups` says otherwise — this `pyproject.toml` has no
such override, so a plain `uv sync`/`uv run pytest` today already does
**not** install `dbc-data`/`dbc_pyutils`; only `uv sync --group dbc` (or
`--all-groups`) does. So the dependency-group placement for `dbc-data`
itself is *already correct* — no `pyproject.toml` restructuring needed
there. `tornado` is the one remaining hard top-level dependency to remove.
One consequence: `Jenkinsfile-update-vector-db`'s `uv sync --frozen` (no
`--group` flag, see §5) does not currently request the `dbc` group either —
i.e. as the file stands today, the nightly indexing job's `uv sync` would
not install `dbc-data`, and `create-faiss-index` would fail on import. This
looks like a pre-existing gap from grouping `dbc-data` out of default deps
(pyproject.toml's last edit predates this conversation) rather than
something this cleanup introduces — but it now needs `--group dbc` added to
that Jenkins stage's `uv sync` call for the job to keep working. Both
packages resolve only from the internal index:

```toml
[[tool.uv.index]]
name = "dbc-devpi"
url = "https://devpi.dbccloud.dk/dbc/packages/+simple/"
```

Locally installed versions: `dbc-pyutils==0.2.354` (deps: alembic, fastapi,
httpx, pandas, parameterized, prometheus-client, psycopg2-binary, redis,
requests, requests_futures, simplejson, sqlalchemy, tornado, tqdm, uvicorn),
`dbc-data==0.3.758` (deps: asyncpg, confluent-kafka, dbc_pyutils, deepdiff,
jsonlines, lxml, marisa_trie, numpy, orjson, pandas, psycopg2-binary,
python-dateutil, rdflib, requests, sqlitedict, tqdm). `dbc-data` itself
depends on `dbc_pyutils`, so `dbc_pyutils` can't be dropped entirely as long
as `dbc-data` stays a dependency.

## 2. `dbc-data` usage — confined to one file

Only consumer: `src/mitcfu_rag/rag/index_vector_db.py` (entry point script
`create-faiss-index = mitcfu_rag.rag.index_vector_db:main`).

- `from dbc_data import kafka`
- `kafka.get_consumer_beginning(options=options, topics=[kafka_topic])` (line 179)
- `kafka.iterate_consumer(kafka_consumer, stop_at_current_end_offset=True, close_when_done=True)` (line 186)

Also in the same file: `from dbc_pyutils import Time` — used as a timing
context manager around indexing work. This import has nothing to do with the
tornado→FastAPI migration; it's DBC-hardware-only tooling exactly like
`dbc-data`, so it's part of "indexing" scope, not "service" scope.

Nothing else in `src/` imports `dbc_data` (verified via repo-wide grep).

### How the vector DB is currently built/published (for README + Docker context)
- `Jenkinsfile-update-vector-db`: nightly Jenkins job (`aibuild-p03` worker,
  `build-env` docker image) that:
  - runs `create-faiss-index` (needs Kafka + `dbc-data`) to build/update the
    FAISS index and JED document JSON,
  - tars the index (`mitcfu_faiss_index.tgz`) and uploads both
    `mitcfu_faiss_index.tgz` and `mitcfu_faiss_index_file.json` to Artifactory
    at `${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/mitcfu-rag/`.
- `Dockerfile` downloads those same two artifacts (plus a reranker model
  tarball `ms-marco-MiniLM-L-6-v2.tgz`) via `wget` from the same Artifactory
  path at image build time — i.e. the runtime service image never invokes
  `dbc-data`/Kafka itself, only the nightly indexing job does.
- These Artifactory URLs are internal (`docker-dbc.artifacts.dbccloud.dk` /
  `${ARTIFACTORY_URL}`) — reachable on DBC network/VPN, not public. README
  needs to document these fetch URLs/paths for people with DBC access rather
  than assume public reachability.

## 3. `dbc_pyutils` usage in the service — full symbol inventory

`src/mitcfu_rag/service.py` (tornado app, entry point
`streaming-service-mitcfu = mitcfu_rag.service:cli`):

| Import | Defined in (dbc_pyutils) | What it does | DBC-environment-only? |
|---|---|---|---|
| `create_instance_id` | `json_formatter.py` | `uuid4().hex` chunked into dash-separated groups | No — pure stdlib |
| `Statistics` | `stat_collector.py` | in-memory pandas/numpy-based rolling stats collector (used for `STATS["query"]`) | No — pure pandas/numpy |
| `build_info` (module) / `build_info.get_info("mitcfu_rag")` | `build_info.py` | reads a generated `<pkg>._build_info` module (written by `dbc_pytools.make_build_info_file` in CI) for build/git/version metadata; falls back to `{"build_number": "not available", ...}` if absent | No at runtime — degrades gracefully already, but still requires the `dbc_pyutils` import |
| `setup_logging` | `setup_dbc_logging.py` | root logger + `JSONFormatter` (structured JSON logs) | No — pure stdlib/json |
| `StatusHandler` | `status_handler.py` | Tornado `RequestHandler` subclass serving `/status` (ab_id, build info, instance id, stats) | Tornado-specific, must be replaced regardless |
| `PrometheusMixIn` | `metrics_handler.py` | Tornado `Application` mixin tracking request durations/counts | Tornado-specific, must be replaced |
| `MetricsHandler` | `metrics_handler.py` | Tornado handler serving `/metrics` in prometheus text format | Tornado-specific, must be replaced |
| `BaseHandler` | `base_handler.py` | Tornado `RequestHandler` with JSON error handling (parent class of `GlyphGateHandler`) | Tornado-specific, must be replaced |

None of these actually require network/cluster access to *import* — they're
all pure-Python/pandas/prometheus-client — but they only add value (build
metadata, prometheus scraping conventions, consistent JSON log format) when
running inside the DBC cluster/CI, which matches the user's framing.

### Tornado surface being replaced (functional inventory to preserve)
- `GlyphGateHandler.post` → `POST /v1/chat/completions`: OpenAI-chat-style
  request/response, supports `stream: true` (SSE-like chunked `self.write` +
  `flush`, terminated with `data: [DONE]\n\n`) and non-streaming JSON
  (`chat.completion` object shape). Drives `AgenticGraph`/`AgenticRAG` and
  `async_gen_wrapper` (`tools/llm_formatting.py`).
- `MetricsApp(PrometheusMixIn, tw.Application)` + `MetricsHandler` → `GET /metrics`.
- `StatusHandler` (mounted at `/status` with `ab_id=1`, `info`, `instance_id`,
  `statistics=list(STATS.values())`) → `GET /status`.
- `make_app(model, graph_type)` wires handlers + `build_info.get_info`.
- `main(args)` → `app.listen(args.port)` + `asyncio.Event().wait()` (tornado
  serving loop) — becomes `uvicorn.run(...)`.
- `cli()` → argparse: `embedding_model_path`, `faiss_path`,
  `--article_index_path`, `--validator-model-path`, `--graph-type`,
  `--use-ceph` (flag), `-a/--ab-id`, `-p/--port`, `-v/--verbose`. These are
  positional/domain args (model paths), distinct from the generic
  `-p/--port` the FastAPI starter template expects — need to be merged into
  the new `parse_args()`.
- Consumers of the running service: `streamlit_ui.py` posts to
  `STREAMING_ENDPOINT` (`MITCFU_UI_STREAM_URL` env, default
  `http://ai-p301:5009/v1/chat/completions`) with `stream: true` and consumes
  SSE-style `data: ...` lines — response contract (`/v1/chat/completions`,
  SSE framing) must be preserved by the FastAPI port.

## 4. Internal FastAPI project-starter template (`dbc-pytools`)

`dbc-pytools` (1.2.31, installed locally) ships a Jinja project-starter
template at `dbc_pytools/project_starter_template/` (generated via
`dbc_pytools.project_starter`, invoked e.g. as a `pip install dbc-pytools`
CLI/module — it is *not* itself a runtime dependency of any generated
project; not currently referenced in `mitcfu-rag`'s `pyproject.toml`).

Relevant files for the "web" service variant:
- `pyproject.toml.j2` — when `web`: `dependencies = ["fastapi", "uvicorn",
  "httpx", "pydantic", "dbc-pyutils"]` (hard dep on `dbc-pyutils`, no
  gating), script entry `{{name}}-service = "{{namepath}}.service.start:main"`.
- `src/service/__init__.py.j2` — re-exports `router`, `create_app`.
- `src/service/start.py.j2`:
  - `from dbc_pyutils.base_handler_fastapi import install_base_handler, with_timelogged`
  - `from dbc_pyutils import PrometheusMiddleware, metrics_endpoint`
  - `from dbc_pyutils import JSONFormatter, build_info, create_instance_id, setup_logging`
  - `create_app(args)` builds `FastAPI(...)` + lifespan + `include_router`.
  - `main()` calls `install_base_handler(app)`,
    `app.add_middleware(PrometheusMiddleware, excluded_paths={"/metrics", "/status"})`,
    then `uvicorn.run(app, host="0.0.0.0", port=..., log_config=None)`.
  - `parse_args()` — generic `-p/--port` only.
- `src/service/endpoints.py.j2`:
  - `GET /status` → `{"status": "ok"}` (plain, no build info/instance id used
    here in the template, unlike the tornado `StatusHandler`).
  - `GET /metrics` → `await metrics_endpoint(request)` (from
    `dbc_pyutils.metric_handler_fastapi`).
  - `POST /hello` demo endpoint using `schemas.py.j2` pydantic models.
- `src/service/schemas.py.j2` — plain pydantic `BaseModel`s, no dbc coupling.
- `tests/test_service_start.py.j2` — imports `start`, builds `create_app(args)`,
  drives it with `fastapi.testclient.TestClient`, asserts on `/status` and
  `/hello`. This test exercises `create_app`, but `main()`/`install_base_handler`
  and the Prometheus middleware are only wired in `main()`, not in
  `create_app()` — i.e. the template's own test never imports the
  dbc_pyutils-touching code path at all currently. That's the seam the
  requested "gate on install" behavior needs to formalize explicitly rather
  than rely on accidentally.
- Underlying `dbc_pyutils` FastAPI modules used by the template
  (`base_handler_fastapi.py`: `install_base_handler`, `with_timelogged`,
  `TimeLogMiddleware` — global exception handlers + request timing/logging
  middleware; `metric_handler_fastapi.py`: `PrometheusMiddleware`,
  `metrics_endpoint` — ASGI middleware + `/metrics` responder backed by
  `prometheus_client`) are themselves plain-Python/Starlette/prometheus_client
  code with no DBC-network calls — they just aren't publishable to PyPI
  as-is since they live inside the internal `dbc-pyutils` package.
- No existing "optional dependency"/`importlib.util.find_spec` gating pattern
  exists anywhere in the starter template today — the requested
  install-or-ignore gating is new, not something to copy from a template
  example.
- Non-web template scaffolding also present (not directly relevant here):
  `Dockerfile.j2` (`pip install .`, `ENTRYPOINT ["python", "-m", "src.{{namepath}}.service.start"]`),
  `deploy/`, `deploy_info/` (Kubernetes manifests + `DEPLOY_README.md.j2`),
  `Jenkinsfile.j2`, `README.md.j2` (documents `docker build`/`docker run`,
  and says "Always use `dbc_pyutils.http.dbc_requests`" for the demo request
  helper — another spot where the doc pattern assumes dbc_pyutils presence).

## 5. Other repo touchpoints found

- `tornado` also appears only in `service.py` (`import tornado.web as tw`)
  and as a transitive dep pulled in by `dbc-pyutils` itself — dropping the
  hard `tornado` dependency and the tornado import is sufficient; no other
  file imports tornado.
- `MANIFEST.in` only packages `faktalink_icon.png`; no changes implied.
- No `_build_info.py` is currently generated/present in this repo (confirmed
  via glob) — `build_info.get_info()` today always falls back to the
  `"devel"`/`"not available"` default locally; the enriched build metadata
  only ever gets populated when the internal Jenkins/`dbc_pytools
  make_build_info_file` step runs. This is a concrete example of "only
  useful in our environment" behavior beyond just import-gating.
- Tests directory (`tests/test_knn_searcher.py`, `tests/test_generic_parser.py`)
  has zero references to tornado/dbc_pyutils/dbc_data today — the service
  layer currently has no test coverage at all, so porting to FastAPI is also
  the first opportunity to add a `TestClient`-based test (mirroring the
  starter template's `test_service_start.py.j2` pattern) without needing any
  internal package installed.
- `Jenkinsfile` / `Jenkinsfile-update-vector-db` are internal Groovy CI files
  using `@Library('ai')` (DBC shared Jenkins library) and DBC-internal
  worker labels/registries (`aibuild-p03`, `docker-dbc.artifacts.dbccloud.dk`,
  `is.dbc.dk`) — out of scope for public-GitHub cleanliness in terms of
  Python deps in general, but `Jenkinsfile-update-vector-db`'s
  `uv sync --frozen` call (line 41) needs a `--group dbc` flag added (see §1
  correction) so the indexing job keeps getting `dbc-data`/`dbc_pyutils`;
  this file is also where the Artifactory vector-db URLs and
  `create-faiss-index` invocations used for the README documentation live.
- README currently has no section on fetching a prebuilt vector DB — section
  43-56 ("Create faiss embeddings") only documents building/updating via
  Kafka + `create-faiss-index`, and assumes checkout on `ai-p301` with the
  models "currently the only place where the files/models are" — this is the
  section that needs the new "fetch prebuilt index from URL" alternative.

## 6. Summary of concrete touchpoints for the two work items

### A. `dbc-data` → dbc-specific group
- `pyproject.toml`: `dbc-data` + `dbc_pyutils`'s `Time` usage already live in
  the non-default `dbc` group (decision above: keep bundled, no split) and,
  per the §1 correction, that group is *already* excluded from a plain
  `uv sync`/`uv run pytest` — no further `pyproject.toml` change needed here.
- `src/mitcfu_rag/rag/index_vector_db.py`: only file importing `dbc_data`/`kafka`
  and the `Time` helper — no code changes needed beyond dependency-group
  placement, since it's already isolated behind the `create-faiss-index`
  entry point (not imported by the service or by `streamlit_ui.py`).
- `README.md`: needs a new subsection alongside "Create faiss embeddings"
  documenting fetching the nightly-built `mitcfu_faiss_index.tgz` /
  `mitcfu_faiss_index_file.json` from the Artifactory path used by
  `Jenkinsfile-update-vector-db`/`Dockerfile`, as the default path for
  contributors without DBC/Kafka access, with `create-faiss-index`
  (requiring `dbc` group + Kafka access) documented as the DBC-internal
  alternative for updating/rebuilding it. The URL doesn't exist yet
  (decision above) — write the section with a `[MORTY_URL_HERE]`
  placeholder rather than a real link.

### B. tornado → FastAPI, gated `dbc_pyutils`
- Replace `src/mitcfu_rag/service.py` (tornado) with a `service` package
  following the `dbc-pytools` starter layout
  (`service/__init__.py`, `service/start.py`, `service/endpoints.py`,
  optionally `service/schemas.py`), preserving: `/v1/chat/completions`
  (streaming + non-streaming, same JSON/SSE contract consumed by
  `streamlit_ui.py`); CLI args merged per decision above (model/index paths
  + `--use-ceph` + `--graph-type` + `-a/--ab-id` + `-p/--port` +
  `-v/--verbose`).
- `/status` and `/metrics` gated per decision above: with `dbc_pyutils`
  installed, keep current behavior (`install_base_handler`/
  `TimeLogMiddleware`, `PrometheusMiddleware`/`metrics_endpoint`,
  `setup_logging`/`JSONFormatter`, `build_info.get_info`,
  `create_instance_id` — richer `/status` with build info/instance
  id/stats, `/metrics` mounted); without it, `/status` falls back to the
  starter template's bare `{"status": "ok"}` and `/metrics` is not mounted
  at all (no local reimplementation of the prometheus middleware). This
  needs an explicit `importlib.util.find_spec`/try-except gate around each
  `dbc_pyutils` import in `start.py`/`endpoints.py`, since — per the
  template review in §4 — nothing does this today.
- `pyproject.toml`: drop hard `tornado` dependency; add `fastapi`, `uvicorn`,
  `httpx` (per starter template) to top-level deps; keep `dbc_pyutils` out of
  top-level deps (only in the `dbc` group, alongside `dbc-data`) since it
  must be optional at import time in the service.
- `Dockerfile`: entrypoint currently calls `streaming-service-mitcfu` with
  positional CLI args (model/index paths) — template's
  `uvicorn.run`/`-p/--port`-only pattern needs reconciling with those
  existing positional args per the merged CLI decision above. Cluster/CI
  images install the `dbc` group, so the built Docker image always gets the
  enriched `/status` + `/metrics`; the Kubernetes prometheus scrape
  annotations in `DEPLOY_README.md.j2` keep working unchanged there — the
  fallback path only matters for local dev/tests run without the `dbc`
  group.
- New/updated tests: a `TestClient`-based service test (no internal package
  required), mirroring `test_service_start.py.j2`.
- `README.md`: update "Starting the streaming service"/"How to start the
  service with DOCKER" sections for the FastAPI entry point and document the
  optional-`dbc_pyutils` behavior (what changes/is unavailable when it's not
  installed: build metadata, prometheus scrape format details, structured
  JSON logging).
