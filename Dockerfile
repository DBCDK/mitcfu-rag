FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

RUN apt-get update && apt-get install -y --no-install-recommends wget

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
    uv sync --no-dev --frozen

# Ensure uv env is on path
ENV PATH="/home/python/.venv/bin:$PATH"
# /data/mitcfu-rag-1-0 is a symlink to the model on the k8s volume mount
# temporarily use non-symlinked version while switching embedding models
CMD ["streaming-service-mitcfu", "/data/multilingual-e5-large-instruct", "mitcfu_faiss_index", "--article_index_path", "mitcfu_jed_documents.json", "--validator-model-path", "ms-marco-MiniLM-L-6-v2", "--port", "5000"]

EXPOSE 5000
