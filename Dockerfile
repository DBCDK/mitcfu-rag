FROM docker-dbc.artifacts.dbccloud.dk/dbc-python3:latest

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    conda install -y -c conda-forge faiss

ARG ARTIFACT_FOLDER=${ARTIFACTORY_URL}/${AI_NON_PRODUCTION}/fakta-chat
ARG MODEL_PATH=${ARTIFACT_FOLDER}/ms-marco-MiniLM-L-6-v2.tar.zst
ARG FAISS_PATH=${ARTIFACT_FOLDER}/e5_mistral_instruct_embeddings_faiss_index.tar.zst
ARG INDEX_PATH=${ARTIFACT_FOLDER}/index.json

RUN useradd -m python
USER python
WORKDIR /home/python

ENV PATH=/home/python/.local/bin:$PATH

COPY --chown=python src src
COPY --chown=python setup.py setup.py

RUN wget -nv --no-check-certificate ${MODEL_PATH} -O ms-marco-MiniLM-L-6-v2.tar.zst && \
    wget -nv --no-check-certificate ${FAISS_PATH} -O e5_mistral_instruct_embeddings_faiss_index.tar.zst && \
    wget -nv --no-check-certificate ${INDEX_PATH} -O index.json && \
    tar -xvf ms-marco-MiniLM-L-6-v2.tar.zst && \
    tar -xvf e5_mistral_instruct_embeddings_faiss_index.tar.zst && \
    rm ms-marco-MiniLM-L-6-v2.tar.zst && \
    rm e5_mistral_instruct_embeddings_faiss_index.tar.zst && \
    pip install --user pip && \
    pip install --user .

# /data/e5-mistral-7b-instruct is on the k8s volume mount
CMD ["streaming-service", "/data/e5-mistral-7b-instruct", "e5_mistral_instruct_embeddings_faiss_index", "index.json", "--port", "5000"]

EXPOSE 5000
