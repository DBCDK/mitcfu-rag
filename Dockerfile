FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    conda install -y -c conda-forge faiss

ARG MODEL_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/mitcfu-rag/ms-marco-MiniLM-L-6-v2.tgz
ARG FAISS_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/science-rag/pdf_test_index.tgz
ARG INDEX_PATH=${ARTIFACTORY_URL}/${AI_PRODUCTION}/science-rag/science_docs_list_jed_for_index.json

RUN useradd -m python
USER python
WORKDIR /home/python

ENV PATH=/home/python/.local/bin:$PATH

COPY --chown=python src src
COPY --chown=python setup.py setup.py

RUN wget -nv --no-check-certificate ${MODEL_PATH} -O ms-marco-MiniLM-L-6-v2.tgz && \
    wget -nv --no-check-certificate ${FAISS_PATH} -O pdf_test_index.tgz && \
    wget -nv --no-check-certificate ${INDEX_PATH} -O science_docs_list_jed_for_index.json && \
    tar -xzvf ms-marco-MiniLM-L-6-v2.tgz && \
    tar -xzvf pdf_test_index.tgz && \
    rm ms-marco-MiniLM-L-6-v2.tgz && \
    rm pdf_test_index.tgz && \
    pip install --user pip && \
    pip install --user .

# /data/mitcfu-rag-1-0 is a symlink to the model on the k8s volume mount
# temporarily use non-symlinked version while switching embedding models
CMD ["streaming-service-science-rag", "/data/multilingual-e5-large-instruct", "pdf_test_index", "--article_index_path", "science_docs_list_jed_for_index.json", "--validator-model-path", "ms-marco-MiniLM-L-6-v2", "--port", "5000", "--use-ceph"]
EXPOSE 5000
