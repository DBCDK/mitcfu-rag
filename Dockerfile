FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    conda install -y -c conda-forge faiss

ARG MODEL_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/mitcfu-rag/ms-marco-MiniLM-L-6-v2.tgz
ARG FAISS_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/mitcfu-rag/mitcfu_faiss_index.tgz
#ARG INDEX_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/mitcfu-rag/mitcfu_jed_documents.tgz

RUN useradd -m python
USER python
WORKDIR /home/python

ENV PATH=/home/python/.local/bin:$PATH

COPY --chown=python src src
COPY --chown=python setup.py setup.py

RUN wget -nv --no-check-certificate ${MODEL_PATH} -O ms-marco-MiniLM-L-6-v2.tgz && \
    wget -nv --no-check-certificate ${FAISS_PATH} -O mitcfu_faiss_index.tgz && \
    #wget -nv --no-check-certificate ${INDEX_PATH} -O mitcfu_jed_documents.tgz && \
    tar -xzvf ms-marco-MiniLM-L-6-v2.tgz && \
    tar -xzvf mitcfu_faiss_index.tgz && \
    #tar -xzvf mitcfu_jed_documents.tgz && \
    rm ms-marco-MiniLM-L-6-v2.tgz && \
    rm mitcfu_faiss_index.tgz && \
    #rm mitcfu_jed_documents.tgz && \
    pip install --user pip && \
    pip install --user .

# /data/e5-mistral-7b-instruct is on the k8s volume mount
CMD ["streaming-service-mitcfu", "/data/mitcfu-rag-1-0", "mitcfu_faiss_index", "--validator-model-path", "ms-marco-MiniLM-L-6-v2", "--port", "5000"]

EXPOSE 5000
