# MitCFU-RAG

RAG-løsning til MitCFU 2025

## How to start the service from the command line
The easiest way to test is to start two services: the streaming service and the streamlitui service
Before starting them, make sure you have done the following:

Checkout the project on `ai-p301`. This is currently the only place where the files/models are
Run `pip install -e .` from the root of the project on
Run `conda install -c conda-forge faiss` to install faiss

### Starting the streaming service
Start the service with the following parameters
`streaming-service-mitcfu /data/mitCFU-models/multilingual-e5-large /data/mitCFU/faiss-indexes/mitcfu_faiss_index_with_pedagogical_note/ -p 5011 --article_index_path /data/mitCFU/faiss-indexes/mitcfu_faiss_index_file_with_pedagogical_note`

### Starting the streamlitui service
Start the service with the following parameters.
`streamlit run src/mitcfu_rag/streamlit_ui.py --server.port 8111`
NOTE: If you did not run your streaming service on port 5011, you will have to manually change the endpoint by editing
the variable `STREAMING_ENDPOINT` in `streamlit_ui.py`

## How to start the service with DOCKER
Build the docker image from the Dockerfile:

`docker build -t $USER/<my-service-name>:test -f Dockerfile .`

Run the docker image:

`docker run -v /data/mitCFU-models/multilingual-e5-large:/data/mitcfu-rag-1-0 -p 5011:5000 -it $USER/mitcfu-rag`

`-e LOG_FORMAT=text` gives you log output in text instead of json

`--rm` ensures the docker container is closed down properly after use

If you for example have started the service on the server ai-p301 you can reach the service via this url
`http://ai-p301:<PORT_NUMBER>`

or locally via this url: 

`localhost:<PORT_NUMBER>`

## Create faiss embeddings
`create-faiss-index` reads CFU documents from Kafka and indexes them into a FAISS
vector database. Run `pip install -e .` and `conda install -c conda-forge faiss`
first, as above.

To build a new index from scratch:
`create-faiss-index --index-output-path mitcfu_faiss_index_file.json --kafka-topic cisterne-work-jed-1-3 --kafka-group-id <your-group-id> --batch-size 100`

To update an existing index:
`create-faiss-index --database-input-path mitcfu_faiss_index --index-input-path mitcfu_faiss_index_file.json --kafka-topic cisterne-work-jed-1-3 --kafka-group-id <your-group-id> --batch-size 100`

See `Jenkinsfile-update-vector-db` for the exact invocations used in the nightly
production rebuild, and `create-faiss-index --help` for the full set of options
(Kafka bootstrap servers, embedding model, limit, etc.).

## How to run tests for this project
### Unit tests
Run `pytest` (or `uv run pytest`) from the root of the project. Tests live under
`tests/` and currently cover `GenericParser` (`tools/generic_parser.py`) and
`KNNSearch` (`tools/knn_searcher.py`).


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