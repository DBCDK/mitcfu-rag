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
`streamlit run src/science_rag/streamlit_ui.py --server.port 8111`
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
### How to run <my_command>
Run `pip install -e .` from the root of the project.
Run `conda install -c conda-forge faiss`

First run `touch mitcfu-index-file-path`
Then run `create-faiss-index --path_to_db mitcfu-faiss-db --path_to_folder /data/mitcfu-rag/jed-docs --path_to_index_file mitcfu-index-file-path --batch_size 10 --create_new_index_extract`

This will start the indexing of the documents in the folder "--path_to_folder" and save the FAISS index and labels in the path specified after "--path_to_db".
"""

## How to run tests for this project
### Unit tests 

### Validation tests

### Performance tests

### Sanity checks before Merge Request


## Artifacts built in this project


## Related Jenkins jobs on is.dbc.dk


## Related artifacts from Artifactory


## Related repositories


## Production version of the service
 
 
## Documentation on confluence
