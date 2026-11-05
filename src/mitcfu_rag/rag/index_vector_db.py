#!/usr/bin/env python
"""
:mod:` embeds all cfu-documents and stores in a vector-database

=============
IndexVectorDB
=============

Uses the specified embedding-model to create embeddings of the documents and
storing them in a FAISS db
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dbc_data import kafka
from dbc_pyutils import Time
from mitcfu_rag.tools.embedding_factory import create_embedder, DEFAULT_MODEL
from mitcfu_rag.tools.knn_searcher import KNNSearch

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    check_args(args)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.index_input_path:
        document_dict = load_document_dict(args.index_input_path)
        logger.info("Loaded %s documents from %s", len(document_dict), args.index_input_path)
    else:
        document_dict = {}

    new_documents, documents_to_delete = read_kafka_topic(
        document_dict_keys=set(document_dict.keys()),
        kafka_topic=args.kafka_topic,
        kafka_group_id=args.kafka_group_id,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        limit=args.limit,
    )

    # Instantiate the embedder only if there is anything to encode. Deletions-only
    # runs don't require an embedding service or local model to be available.
    if new_documents:
        embedder = create_embedder(
            backend=args.embedding_backend,
            model=args.embedding_model,
            local_model_path=args.local_model_path,
        )
        logger.info("Embedding via backend %s, model %s", args.embedding_backend, embedder.name)
        new_embeddings, new_labels = generate_embeddings_and_labels(new_documents, embedder, batch_size=args.batch_size)
    else:
        new_embeddings = np.empty((0, 0), dtype=np.float32)
        new_labels = np.empty((0,), dtype=str)

    # Do not write updated documents.json if the embedding/FAISS write fails.
    create_faiss_database(
        args.database_input_path,
        new_embeddings,
        new_labels,
        documents_to_delete,
        output_path=args.database_output_path,
    )

    index_path = args.index_output_path or args.index_input_path
    if index_path is not None:
        save_document_dict(document_dict, new_documents, documents_to_delete, index_path)
        logger.info("Saved document dictionary to %s", index_path)


def parse_args():
    KAFKA_BOOTSTRP_SERVERS = (
        "kafkadata-prod-dc1.dbccloud.dk, kafkadata-prod-dc2.dbccloud.dk, kafkadata-prod-dc3.dbccloud.dk"
    )
    default_kafka_group = f"ai-dev-mitcfu-vector-db-{datetime.now().isoformat()}"
    parser = argparse.ArgumentParser(
        description="Reads CFU documents from Kafka and indexes them in a FAISS vector database"
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("api", "local"),
        default=os.getenv("EMBEDDING_BACKEND", "api"),
        help="Embedding implementation (default: EMBEDDING_BACKEND or api)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        help="Deployed embedding model ID for API mode",
    )
    parser.add_argument(
        "--local-model-path",
        default=None,
        help="Path to local E5 model; defaults to LOCAL_EMBEDDING_MODEL_PATH",
    )
    parser.add_argument(
        "--database-input-path",
        type=str,
        help="Directory of an existing FAISS database (embeddings + labels.npy)",
    )
    parser.add_argument(
        "--database-output-path",
        type=str,
        default=None,
        help="Directory to save FAISS index; defaults to input directory or current directory",
    )
    parser.add_argument("--index-input-path", type=str, default=None)
    parser.add_argument("--index-output-path", type=str, default=None)
    parser.add_argument("--kafka-topic", default="cisterne-work-jed-1-3")
    parser.add_argument("--kafka-group-id", default=default_kafka_group)
    parser.add_argument("--kafka-bootstrap-servers", default=KAFKA_BOOTSTRP_SERVERS)
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of harvested CFU documents")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Maximum text chunks per embedding API call (default: 32)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def load_document_dict(path_to_mitcfu_documents: str):
    logger.info(f"Reading index file from {path_to_mitcfu_documents}")
    with open(path_to_mitcfu_documents, "r") as file:
        mitcfu_documents = json.load(file)

    mitcfu_dict = {k: v for _dict in mitcfu_documents for k, v in _dict.items()}
    return mitcfu_dict


def save_document_dict(
    mitcfu_documents_dict: dict, new_documents: dict, documents_to_delete: list, path_to_mitcfu_documents: str
):
    mitcfu_documents_dict.update(new_documents)
    for doc in documents_to_delete:
        if doc in mitcfu_documents_dict:
            mitcfu_documents_dict.pop(doc)
    mitcfu_documents = [{k: v} for k, v in mitcfu_documents_dict.items()]
    with open(path_to_mitcfu_documents, "w") as file:
        json.dump(mitcfu_documents, file)


def read_kafka_topic(
    document_dict_keys: set,
    kafka_topic: str,
    kafka_group_id: str,
    kafka_bootstrap_servers: str,
    limit: int | None = None,
):
    # Setup Kafka consumer
    options = {
        "bootstrap.servers": kafka_bootstrap_servers,
        "default.topic.config": {
            "auto.offset.reset": "earliest",
        },
        "group.id": kafka_group_id,
        "enable.auto.commit": "false",
        "max.poll.interval.ms": 1_800_000,  # 30 minutes, default is 5 minutes
    }
    logger.info(f"Connecting to Kafka with options: {options}")
    kafka_consumer = kafka.get_consumer_beginning(options=options, topics=[kafka_topic])
    new_document_dict = {}
    keys_for_deletion = set()
    key2mitcfu_id = {}
    with Time("Kafka harvesting took", level="info"):
        logger.info("Starting kafka harvesting")
        message_counter = 0
        for key_bytes, message in kafka.iterate_consumer(
            kafka_consumer, stop_at_current_end_offset=True, close_when_done=True
        ):
            try:
                key = key_bytes.decode("utf-8")
                if message is not None and len(message) > 0:
                    if key in keys_for_deletion:
                        keys_for_deletion.remove(key)
                    js = json.loads(message.decode("utf8"))
                    pworkid = js.get("workId")
                    if "work-of:875080-cfu" in pworkid:
                        mitcfu_id = _get_mitcfu_id(js)
                        key2mitcfu_id[key] = mitcfu_id  # used to handle tombstones
                        if mitcfu_id not in document_dict_keys:
                            new_document_dict[key] = js
                        message_counter += 1
                        if limit and message_counter >= limit:
                            logger.info(f"Reached limit of {limit} cfu-documents. Stopping consumer.")
                            kafka_consumer.close()
                            break
                else:  # hit tombstone
                    if key in document_dict_keys:
                        keys_for_deletion.add(key)
                    if key in new_document_dict:
                        del new_document_dict[key]

            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding message '{message.decode('utf8')}' for key '{key}': {e}",
                )

    kafka_consumer.close()
    logger.info(
        f"Finished consuming {kafka_topic}. Found {len(new_document_dict)} new documents and {len(keys_for_deletion)} tombstones."
    )
    mitcfu_ids_for_deletion = [key2mitcfu_id[key] for key in keys_for_deletion if key in key2mitcfu_id]
    new_document_dict_mitcfu_ids = {_get_mitcfu_id(doc): doc for k, doc in new_document_dict.items()}
    return new_document_dict_mitcfu_ids, mitcfu_ids_for_deletion


def _get_mitcfu_id(jed_doc: dict) -> str:
    local_id = jed_doc["manifestations"]["bestRepresentations"][0]
    return local_id.split(":")[1]


def generate_embeddings_and_labels(
    new_documents: dict, embedder, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Split abstracts and embed in bounded batches; preserve chunk/label alignment."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    batch_texts: list[str] = []
    batch_labels: list[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        embedded = np.asarray(embedder.embed_documents(batch_texts), dtype=np.float32)
        if embedded.ndim != 2 or embedded.shape[0] != len(batch_labels):
            raise ValueError(f"Embedder returned shape {embedded.shape} for {len(batch_labels)} chunks")
        if not np.isfinite(embedded).all():
            raise ValueError("Embedder returned invalid (NaN/Inf) embeddings")
        vectors.append(embedded)
        labels.extend(batch_labels)
        batch_texts.clear()
        batch_labels.clear()

    logger.info("Embedding abstracts from %s documents (max %s chunks/request)", len(new_documents), batch_size)
    for doc_id, doc in new_documents.items():
        if not validate_abstract(doc):
            continue
        abstract = " ".join(doc["abstract"])
        for i, chunk in enumerate(text_splitter.split_text(abstract)):
            batch_texts.append(chunk)
            batch_labels.append(f"{doc_id}_chunk{i}")
            # Flush *inside* the chunk loop; a long document cannot exceed the cap.
            if len(batch_texts) == batch_size:
                flush_batch()
    flush_batch()
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=str)
    return np.vstack(vectors).astype(np.float32), np.asarray(labels, dtype=str)


def create_faiss_database(
    faiss_database_path: str | None,
    embeddings: np.ndarray,
    labels: np.ndarray,
    documents_to_delete: list[str],
    *,
    output_path: str | None = None,
) -> None:
    """Update or build the KNNSearch index; support deletion-only and empty runs."""
    if embeddings.ndim != 2 or len(labels) != embeddings.shape[0]:
        raise ValueError("Embedding matrix/label count mismatch")
    if not faiss_database_path and embeddings.shape[0] == 0:
        if documents_to_delete:
            raise ValueError("Cannot apply deletions without an existing FAISS database")
        logger.info("No embeddings to index; skipping empty FAISS build")
        return
    if not documents_to_delete and embeddings.shape[0] == 0:
        logger.info("No new vectors or deletions; skipping FAISS update")
        return

    if faiss_database_path:
        faiss_database = load_faiss_database(faiss_database_path)
        if embeddings.shape[0]:
            faiss_database.update(embeddings, labels)
    else:
        faiss_database = build_faiss_database(embeddings, labels)

    if documents_to_delete:
        faiss_database.delete_by_prefix(documents_to_delete)

    destination = Path(output_path or faiss_database_path or ".")
    destination.mkdir(parents=True, exist_ok=True)
    faiss_database.save(
        index_path=str(destination / "embeddings"),
        labels_path=str(destination / "labels"),
    )
    logger.info("Saved FAISS index and labels to %s", destination)


def load_faiss_database(faiss_database_path: str) -> KNNSearch:
    path = Path(faiss_database_path)
    return KNNSearch.load(str(path / "embeddings"), str(path / "labels.npy"))


def build_faiss_database(embeddings: np.ndarray, labels: np.ndarray):
    return KNNSearch.build(embeddings, labels)


def validate_abstract(document: dict) -> bool:
    abstract_list = document.get("abstract")
    if not isinstance(abstract_list, list) or not all(isinstance(item, str) for item in abstract_list):
        logger.debug("Skipping document with invalid abstract: %s", document.get("workId"))
        return False
    return len(" ".join(abstract_list)) >= 150


def check_args(args: argparse.Namespace):
    if not (args.index_input_path or args.index_output_path):
        raise ValueError("Specify --index-input-path and/or --index-output-path")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")


if __name__ == "__main__":
    main()
