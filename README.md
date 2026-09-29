# MitCFU-RAG

RAG-løsning til MitCFU 2025

## How to start the service from the command line
The easiest way to test is to start two services: the streaming service and the streamlitui service
Before starting them, make sure you have done the following:

Checkout the project on `ai-p301`. This is currently the only place where the files/models are
Run `uv sync` from the root of the project (add `--group dbc` for the enriched `dbc_pyutils`
integration described below; `faiss-cpu` is a regular dependency, no separate conda install needed)

### Required environment variables
The streaming service calls LLMs through [glyph-gate](https://llm.dbc.dk), DBC's internal
OpenAI-compatible LLM gateway, instead of a hardcoded backend. It fails fast at startup if these
aren't set:

- `MITCFU_LLM_GATEWAY_URL` — the glyph-gate base URL, e.g. `https://llm.dbc.dk`.
- `MITCFU_LLM_GATEWAY_TOKEN` — a bearer token issued by llm-access for this app. Never commit this;
  inject it as a secret (the Docker image does not set it — see the Dockerfile). You can use the 
  `llm-access-skolegpt-prod-token` that has been set up in Bitwarden for skoleGPT, for example.

The available models are discovered at startup via `GET /v1/models` on the gateway (filtered to
whatever the token is authorized for), not a fixed list — so no separate "which model" env var is
needed.

Optional tuning:
- `MITCFU_MAX_TOKENS` — cap on generated tokens; unset/empty means no limit.
- `MITCFU_LLM_TIMEOUT_SECONDS` — HTTP timeout for LLM calls, default `60`.

### Starting the streaming service
Start the service with the following parameters
`streaming-service-mitcfu /data/mitCFU-models/multilingual-e5-large /data/mitCFU/faiss-indexes/mitcfu_faiss_index_with_pedagogical_note/ -p 5011 --article_index_path /data/mitCFU/faiss-indexes/mitcfu_faiss_index_file_with_pedagogical_note`

The service is FastAPI/uvicorn-based; interactive API docs are available at
`/docs` (Swagger UI) and `/redoc` once it's running.

### Starting the streamlitui service
Start the service with the following parameters.
`streamlit run src/mitcfu_rag/streamlit_ui.py --server.port 8111`
NOTE: If you did not run your streaming service on port 5011, set the `MITCFU_UI_STREAM_URL` env
var to point at it (default `http://localhost:5000/v1/chat/completions`) instead of editing the
code. The UI's sidebar model picker is populated from the streaming service's `GET /v1/models`.

### Optional `dbc_pyutils` integration
The service works fully without `dbc_pyutils` installed (the default for a
plain `uv sync`/`pip install -e .`). What differs depending on whether it's
present (installed via `uv sync --group dbc`):

- `GET /status` — bare `{"status": "ok"}` when absent; enriched with build
  info, instance id, ab-id, memory usage and query statistics when present.
- `GET /metrics` — not mounted at all when absent; Prometheus-format
  scrape endpoint when present.
- Logging — plain stdlib logging (`logging.basicConfig`) when absent;
  structured JSON logging when present.

The Docker image (see below) always installs the `dbc` group, so cluster
deployments get the enriched behavior; local/external contributors run in
the degraded-but-fully-functional mode by default.

## How to start the service with DOCKER
Build the docker image from the Dockerfile:

`docker build -t $USER/<my-service-name>:test -f Dockerfile .`

Run the docker image (see "Required environment variables" above — `MITCFU_LLM_GATEWAY_TOKEN`
isn't baked into the image and must be passed in):

`docker run -v /data/mitCFU-models/multilingual-e5-large:/data/mitcfu-rag-1-0 -e MITCFU_LLM_GATEWAY_TOKEN=<token> -p 5011:5000 -it $USER/mitcfu-rag`

`-e LOG_FORMAT=text` gives you log output in text instead of json

`--rm` ensures the docker container is closed down properly after use

If you for example have started the service on the server ai-p301 you can reach the service via this url
`http://ai-p301:<PORT_NUMBER>`

or locally via this url: 

`localhost:<PORT_NUMBER>`

## Create faiss embeddings
`create-faiss-index` reads CFU documents from Kafka and indexes them into a FAISS
vector database. This additionally requires DBC network + Kafka access, so
install with `uv sync --group dbc` (the plain `uv sync` does not pull in
`dbc-data`/`dbc_pyutils`, since indexing is the only consumer of those
packages).

To build a new index from scratch:
`create-faiss-index --index-output-path mitcfu_faiss_index_file.json --kafka-topic cisterne-work-jed-1-3 --kafka-group-id <your-group-id> --batch-size 100`

To update an existing index:
`create-faiss-index --database-input-path mitcfu_faiss_index --index-input-path mitcfu_faiss_index_file.json --kafka-topic cisterne-work-jed-1-3 --kafka-group-id <your-group-id> --batch-size 100`

See `Jenkinsfile-update-vector-db` for the exact invocations used in the nightly
production rebuild, and `create-faiss-index --help` for the full set of options
(Kafka bootstrap servers, embedding model, limit, etc.).

## Fetching a prebuilt vector index
If you don't have DBC network/Kafka access, skip `create-faiss-index`
entirely and fetch the nightly-built artifacts instead:

- `mitcfu_faiss_index.tgz` and `mitcfu_faiss_index_file.json` from
  `[MORTY_URL_HERE]` (Artifactory path used by `Jenkinsfile-update-vector-db`
  and `Dockerfile`).
- `tar -xzvf mitcfu_faiss_index.tgz` to get the `mitcfu_faiss_index/`
  directory the streaming service (and `Dockerfile`) consume directly.

`create-faiss-index` (documented above) remains the DBC-internal path for
building or updating this index from scratch.

## How to run tests for this project
### Unit tests
Run `pytest` (or `uv run pytest`) from the root of the project. Tests live under
`tests/` and currently cover `GenericParser` (`tools/generic_parser.py`),
`KNNSearch` (`tools/knn_searcher.py`), the FastAPI streaming service including
its glyph-gate upstream-error handling (`test_service.py`), and the
`GET /v1/models` model-discovery call (`test_agent_streaming_generator.py`).
These all run without real embedding/torch models or a live glyph-gate
connection — no `MITCFU_LLM_GATEWAY_*` env vars are needed to run the suite.


### Sanity checks before Merge Request
- `uv run pytest` — must pass.
- `uv run ruff format`
- `uv run ruff check`

## Artifacts built in this project
- A Docker image for the streaming service, built from `Dockerfile` and tagged
  per branch/build number (see `Jenkinsfile`), deployed to Kubernetes.
- The FAISS vector index, its embeddings/labels, and the JED document metadata
  index (`mitcfu_faiss_index.tgz`, `mitcfu_faiss_index_file.json`), rebuilt or
  updated nightly by `Jenkinsfile-update-vector-db` and published to Artifactory.