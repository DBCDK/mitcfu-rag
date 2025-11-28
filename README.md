# Science-RAG

RAG-solution for PDFs from CFU. The repository uses the same RAG-structure as MitCFU-RAG, but instead of getting data from
MitCFU Marc entries (which are then converted to JED), the PDFs are loaded using LangChain Documentloaders through the script
tools/GenericParser.py. This script can handle common filetypes (such as .pdf, .txt, .json) and has the option to try to read
other filetypes. Then, a list of LangChain Document objects are created. From this, we can use the metadata and page_content
to create FAISS indexes for similarity search. For now, we simply extract page_content and create a JED-like structure to enable
the rest of the MitCFU-pipeline to do the hard work. If you want to do this with your own documents, see the description below
in "So you want to index your own documents ..."

## So you want to index your own documents ...
On ai-p301:
1. Place all your documents (pdfs, text files, etc.) in a directory of your choice (I have mine locally under git/science-rag/data)
2. Load a list of all the document file paths.
3. Use the GenericParser to parse the documents into LangChain Document objects.
4. Convert the LangChain document objects into dictionary structure (JED), which is compatible with the MitCFU RAG pipeline.
Here, it is important that page_content --> abstract, and that you create a form of unique_id for each document. 
5. Save a list of these JED documents as a json file (this will be used later when starting the service)
6. Create a FAISS index from the JED documents using e.g. multilingual-e5 embeddings. You can upload these to artifactory or save them
locally or on ai-p301.
7. When starting the RAG service, point to the location of the json file list and the FAISS index (as well as embedding/validator models).
8. You can start the service using: 

`streaming-service-science-rag /data/mitCFU-models/multilingual-e5-large /data/mitCFU/science-RAG/faiss-indexes/pdf-test-index 
--article_index_path ./src/science_rag/data/parsed_science_docs.json --validator-model-path /data/mitCFU-models/ms-marco-MiniLM-L-6-v2 
--verbose --use-ceph --port 5009`

9. You can then start Streamlit using:
streamlit run src/science_rag/streamlit_ui.py --server.port 8111

10. Hooray! You should now be able to query your own documents using RAG.

If you have further questions about the process, ask rani for his notebook example for the PDFs from CFU.

## NOTE: The rest of the documentation is from MitCFU-RAG.
Many of commands should be analogous, but see the guide above for greater clarity.

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
MitCFU-RAG (mitcfu-rag).


## Production version of the service
 
 
## Documentation on confluence
