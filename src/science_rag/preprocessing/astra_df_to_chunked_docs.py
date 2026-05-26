import pandas as pd
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from science_rag.preprocessing.astra_preprocessor import AstraPreprocessor
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def _df_to_json_safe_dict(value):
    """
    Small helper function to make sure metadata is JSON serializable
    and does not contain e.g. NaNs or Pandas-specific types.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def astra_df_to_docling_chunks(
    df: pd.DataFrame,
    preprocessor: AstraPreprocessor,
    metadata_cols: list[str],
    exclude_cols: list[str],
    exclude_col_if_contains: list[str],
):
    """
    Convert a DataFrame to Docling chunks in a jedish-compatible format. This format is a list of
    dictionaries, where each dictionary has a chunk ID_chunk_idx as keys and values are dictionaries
    containing 'abstract' (text to be embedded) and 'metadata' - including at least the Title and URL
    in this metadata is important for downstream tasks.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be processed (should contain Title and URL column).
        preprocessor (AstraPreprocessor): An instance of AstraPreprocessor or similar to preprocess the text.
        metadata_cols (list[str]): List of column names to keep as metadata and not use in abstract.
        exclude_cols (list[str]): List of column names to exclude from the abstract.
        exclude_col_if_contains (list[str]): List of substrings; any column containing these are excluded from abstract.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary has a chunk ID as key and a
        value that is another dictionary with 'abstract' (text for embedding) and 'metadata'.
    """
    df_raw = preprocessor.concatenate_page_content_raw(
        df,
        metadata_cols=metadata_cols,
        exclude_cols=exclude_cols,
        exclude_col_if_contains=exclude_col_if_contains,
    )
    preprocessed_data = df_raw.copy()
    preprocessed_data["page_content"] = preprocessed_data["page_content_raw"].apply(preprocessor.preprocess)

    # Chunking logic needs to be added here
    converter = DocumentConverter()
    chunker = HybridChunker()

    jedish_docs = []
    for row_idx, row in preprocessed_data.iterrows():
        page_content = str(row["page_content"] or "").strip()

        if not page_content:
            continue

        if row["metadata"] is None or not isinstance(row["metadata"], dict):
            logger.warning(f"Row {row_idx} has invalid metadata: {row['metadata']}. Skipping.")
            continue

        # Base metadata is all metadata columns (see astra_csv_cols_config.py).
        # Should contain at least URL and Title, which are used downstream during generation.
        base_metadata = {k: _df_to_json_safe_dict(v) for k, v in row["metadata"].items()}
        doc_name = base_metadata.get("Title", "No_Title").replace(" ", "_").replace(",", "")
        doc = converter.convert_string(content=page_content, format=InputFormat.MD, name=doc_name).document

        chunks = [chunk for chunk in chunker.chunk(dl_doc=doc)]
        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": chunk_idx,
                "chunk_id": f"{doc_name}_chunk{chunk_idx}",
                "docling_doc_name": doc_name,
            }

            jedish_json = {
                f"{doc_name}_chunk{chunk_idx}": {
                    "abstract": chunker.contextualize(chunk=chunk),
                    "metadata": chunk_metadata,
                }
            }
            jedish_docs.append(jedish_json)

    return jedish_docs
