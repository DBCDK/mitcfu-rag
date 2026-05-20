from science_rag.tools.generic_parser import GenericParser
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from science_rag.tools.knn_searcher import KNNSearch
from science_rag.tools.embedder import Embedder
from science_rag.rag.retrievers.indexes.multilinguale5 import index_paragraph_docs_GPU_batches
from transformers import AutoTokenizer, AutoModel
import random
import logging
import os
import json

logger = logging.getLogger(__name__)

# Temporary map to show some examples of why links work/don't work
WEBPDF_MAP = {
    "zoo-aarsberetning-2024.pdf": "https://content.zoo.dk/media/fk1nlps0/zoo-aarsberetning-2024.pdf",
    "aarsberetning-2023.pdf": "https://content.zoo.dk/media/whqkk2vs/aarsberetning-2023.pdf",
    "FORVALTNING AF DYREBESTAND.pdf": "https://www.zoo.dk/om-zoo/dyrene-i-zoo/forvaltning-af-dyrebestanden",
    "Computational Thinking integreret i matematikundervisningen.pdf": "https://doi.org/10.5281/zenodo.19255073",
    "Fight the Bite.pdf": "https://undervisning.life.dk/fb",
}


def get_science_rag_document_paths(path_to_folder):
    # temporary list of documents to ignore
    documents_to_ignore = {
        "Samling af datakilder til RAG.docx",
        "links til kilder.docx",
        "Webhenvisning fra zoo.docx",
        "speciale-henvisninger.docx",
    }

    science_rag_doc_paths = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            if file in documents_to_ignore:
                continue
            if file.endswith(".pdf"):
                science_rag_doc_paths.append(os.path.join(root, file))
    return science_rag_doc_paths


def get_docling_chunks(input_file):
    # parse document
    converter = DocumentConverter()
    doc = converter.convert(input_file).document

    # chunk document
    chunker = HybridChunker()
    chunks = [chunk for chunk in chunker.chunk(dl_doc=doc)]

    # convert to expected json format
    jedish_docs = []
    for i, chunk in enumerate(chunks):
        source = WEBPDF_MAP.get(chunk.meta.origin.filename, chunk.meta.origin.filename)
        jedish_json = {
            f"{source}_side{chunk.meta.doc_items[0].prov[0].page_no}_chunk{i}": {
                "abstract": chunker.contextualize(chunk=chunk)
            }
        }
        jedish_docs.append(jedish_json)

    return jedish_docs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_science_rag_folder",
        metavar="path-to-science-rag-folder",
        help="path to folder containing science rag documents",
        type=str,
    )
    parser.add_argument(
        "document_index_file_path", metavar="document-index-file-path", help="path to save the indexed chunks", type=str
    )
    parser.add_argument(
        "faiss_db_directory",
        metavar="faiss-db-directory",
        help="path to directory to save faiss index and embeddings",
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        metavar="batch-size",
        help="batch size for embedding chunks",
        default=50,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Getting paths to science rag documents from {args.path_to_science_rag_folder}")
    science_rag_doc_paths = get_science_rag_document_paths(args.path_to_science_rag_folder)
    science_rag_chunks = []

    logger.info(f"Reading and chunking {len(science_rag_doc_paths)} science rag documents")
    for file_path in science_rag_doc_paths:
        science_rag_chunks.extend(get_docling_chunks(file_path))

    logger.info(f"Saving {len(science_rag_chunks)} science rag chunks")
    with open(args.document_index_file_path, "w") as f:
        json.dump(science_rag_chunks, f)

    logger.info(
        f"Embedding {len(science_rag_chunks)} science rag chunks and saving FAISS db to {args.faiss_db_directory}"
    )
    index_paragraph_docs_GPU_batches(
        path_to_index_file=args.document_index_file_path, path=args.faiss_db_directory, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
