FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

ARG MODEL_PATH=${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/mitcfu-rag/ms-marco-MiniLM-L-6-v2.tgz
ARG FAISS_PATH=${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/mitcfu-rag/mitcfu_faiss_index.tgz
ARG INDEX_PATH=${ARTIFACTORY_URL}/${AI_DOCKER_LAYERS}/mitcfu-rag/mitcfu_faiss_index_file.json

RUN useradd -m python
USER python
WORKDIR /home/python

ENV PATH=/home/python/.local/bin:$PATH

COPY --chown=python src src
COPY --chown=python pyproject.toml pyproject.toml
COPY --chown=python uv.lock uv.lock

RUN wget -nv --no-check-certificate ${MODEL_PATH} -O ms-marco-MiniLM-L-6-v2.tgz && \
    wget -nv --no-check-certificate ${FAISS_PATH} -O mitcfu_faiss_index.tgz && \
    wget -nv --no-check-certificate ${INDEX_PATH} -O mitcfu_jed_documents.json && \
    tar -xzvf ms-marco-MiniLM-L-6-v2.tgz && \
    tar -xzvf mitcfu_faiss_index.tgz && \
    rm ms-marco-MiniLM-L-6-v2.tgz && \
    rm mitcfu_faiss_index.tgz && \
    uv sync --no-dev --frozen --group dbc

# Ensure uv env is on path
ENV PATH="/home/python/.venv/bin:$PATH"
# glyph-gate base URL; the model list (and its default, the first entry) is
# fetched from GET /v1/models at startup instead of being pinned here.
ENV MITCFU_LLM_GATEWAY_URL="http://glyph-gate-1-0.ai-prod.svc.cloud.dbc.dk"
# MITCFU_LLM_GATEWAY_TOKEN (the glyph-gate bearer token) must be injected at
# deploy time as a k8s secret -- never bake a bearer token into the image.
# /data/mitcfu-rag-1-0 is a symlink to the model on the k8s volume mount
# temporarily use non-symlinked version while switching embedding models
CMD ["streaming-service-mitcfu", "/data/multilingual-e5-large-instruct", "mitcfu_faiss_index", "--article_index_path", "mitcfu_jed_documents.json", "--validator-model-path", "ms-marco-MiniLM-L-6-v2", "--port", "5000"]

EXPOSE 5000
