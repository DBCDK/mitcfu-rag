import argparse
import json
import logging
import os

import pandas as pd
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from science_rag.preprocessing.astra_df_to_chunked_docs import astra_df_to_docling_chunks
from science_rag.preprocessing.astra_preprocessor import AstraPreprocessor
from science_rag.rag.retrievers.indexes.multilinguale5 import index_paragraph_docs_GPU_batches

logger = logging.getLogger(__name__)

# Standard columns to keep for metadata/exclude from the CSV files
aktiviteter_metadata_cols = [
    "ID",
    "Title",
    "[Manchet] Varighed",
    "[Manchet] Niveau",
    "Fag",
    "Klassetrin",
    "Emneord",
    "Kategori",
]
aktiviteter_exclude_cols = ["Which tabs to show", "URL", "page_content_raw", "page_content"]
aktiviteter_exclude_col_if_contains = []

forlob_metadata_cols = ["ID", "Title", "Varighed", "Partnere", "Tilknyttede aktiviteter"]
forlob_exclude_col_if_contains = ["download_or_link", "pdf_link"]
forlob_exclude_cols = [
    "Hvilke faner skal vises",
    "Video url",
    "Sidebar email_acf_education_material_sidebar_boxes_email_header",
    "Sidebar email_acf_education_material_sidebar_boxes_email_content",
    "URL",
    "page_content_raw",
    "page_content",
]

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
    parser.add_argument(
        "--aktiviteter-csv", type=str, required=False, help="(Optional) Path to the Aktiviteter CSV file."
    )
    parser.add_argument("--forlob-csv", type=str, required=False, help="(Optional) Path to the Forløb CSV file.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Getting paths to science rag documents from {args.path_to_science_rag_folder}")
    science_rag_doc_paths = get_science_rag_document_paths(args.path_to_science_rag_folder)

    # a science_rag_chunk must - in order (to work with index_paragraph_docs_GPU_batches - contain:
    # ID as the key to a dictionary with at least "abstract" as a key, and the value of "abstract"
    # should be the text to embed
    science_rag_chunks = []

    logger.info(f"Reading and chunking {len(science_rag_doc_paths)} science rag documents")
    for file_path in science_rag_doc_paths:
        science_rag_chunks.extend(get_docling_chunks(file_path))

    # For now, adding hardcoded .csv files for astra csv's (should we add .csv handling?)
    if args.aktiviteter_csv and args.forlob_csv:
        logger.info(f"Processing Aktiviteter CSV from {args.aktiviteter_csv} and Forløb CSV from {args.forlob_csv}")
        preprocessor = AstraPreprocessor()

        # Loading the CSV files into dataframes
        aktiviteter_df = pd.read_csv(args.aktiviteter_csv, sep=",", encoding="utf-8")
        forlob_df = pd.read_csv(args.forlob_csv, sep=",", encoding="utf-8")

        # Preprocessing and chunking the dataframes into docling chunks
        aktiviteter_list_of_jedish_docs = astra_df_to_docling_chunks(
            aktiviteter_df,
            preprocessor=preprocessor,
            metadata_cols=aktiviteter_metadata_cols,
            exclude_cols=aktiviteter_exclude_cols,
            exclude_col_if_contains=aktiviteter_exclude_col_if_contains,
        )

        forlob_list_of_jedish_docs = astra_df_to_docling_chunks(
            forlob_df,
            preprocessor=preprocessor,
            metadata_cols=forlob_metadata_cols,
            exclude_cols=forlob_exclude_cols,
            exclude_col_if_contains=forlob_exclude_col_if_contains,
        )

        science_rag_chunks.extend(aktiviteter_list_of_jedish_docs)
        science_rag_chunks.extend(forlob_list_of_jedish_docs)

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
