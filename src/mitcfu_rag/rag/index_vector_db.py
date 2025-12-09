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
import numpy as np
import sys

from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dbc_data import kafka
from dbc_pyutils import Time
from mitcfu_rag.rag.retrievers.indexes.multilinguale5 import e5multilingualEmbedder
from mitcfu_rag.tools.knn_searcher import KNNSearch

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    __check_args(args)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.index_input_path and len(args.index_input_path) > 1:
        document_dict = load_document_dict(args.index_input_path)
        logger.info(f"Loaded {len(document_dict)} documents from {args.index_input_path}")
    else:
        document_dict = {}
        logger.info(f"Created empty document dict.")


    new_documents, documents_to_delete = read_kafka_topic(
        document_dict_keys=set(document_dict.keys()),
        kafka_topic=args.kafka_topic,
        kafka_group_id=args.kafka_group_id,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        limit=args.limit,
    )

    # create embeddings and labels for the new documents
    new_embeddings, new_labels = generate_embeddings_and_labels(
        new_documents, args.embedding_model, batch_size=args.batch_size
    )

    # create/update the faiss database
    create_faiss_database(
        args.database_input_path,
        new_embeddings,
        new_labels,
        documents_to_delete,
    )

    # save the updated document_dict
    if args.index_output_path and len(args.index_output_path) > 0:
        logger.info(f"Writing new index file to {args.index_output_path}")
        save_document_dict(document_dict, new_documents, documents_to_delete, args.index_output_path)
    else:
        logger.info(f"Overwriting existing index file at {args.index_input_path}")
        save_document_dict(document_dict, new_documents, documents_to_delete, args.index_input_path)


def parse_args():
    KAFKA_BOOTSTRP_SERVERS = "kafkadata-prod-dc1.dbccloud.dk, kafkadata-prod-dc2.dbccloud.dk, kafkadata-prod-dc3.dbccloud.dk"
    default_kafka_group = f"ai-dev-mitcfu-vector-db-{datetime.now().isoformat()}"
    parser = argparse.ArgumentParser(
        description="Reads cfu-documents from kafka and indexes them in FAISS vector database"
    )
    parser.add_argument(
        "--embedding-model",
        help="huggingface repo/name of the model to use. Documents are formatted to be optimized for intfloat/multilingual-e5-large-instruct",
        default="intfloat/multilingual-e5-large-instruct",
    )
    parser.add_argument(
        "--database-input-path",
        type=str,
        help="Path to existing faiss database",
    )
    parser.add_argument(
        "--index-input-path",
        type=str,
        default=None,
        help="The path to the existing index file.",
    )
    parser.add_argument(
        "--index-output-path",
        type=str,
        default=None,
        help="Path to save the index file. Must be supplied if not updating existing index file.",
    )
    parser.add_argument(
        "--kafka-topic",
        help="Which kafka-topic to consume",
        default="cisterne-work-jed-1-3",
    )
    parser.add_argument(
        "--kafka-group-id",
        help="what to call this group of consumers.",
        default=default_kafka_group,
    )
    parser.add_argument(
        "--kafka-bootstrap-servers",
        help="where to fetch kafka broker info",
        default=KAFKA_BOOTSTRP_SERVERS,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        dest="limit",
        help="if set, limits the number of harvested documents",
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=False,
        help="The batch size for the embedding process. If not specified, it will be set to the length of the data.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to DEBUG"
    )
    return parser.parse_args()


def load_document_dict(path_to_mitcfu_documents: str):
    logger.info(f"Reading index file from {path_to_mitcfu_documents}")
    with open(path_to_mitcfu_documents, "r") as file:
        mitcfu_documents = json.load(file)

    mitcfu_dict = {k: v for _dict in mitcfu_documents for k, v in _dict.items()}
    return mitcfu_dict


def save_document_dict(mitcfu_documents_dict: dict, new_documents: dict, documents_to_delete: list, path_to_mitcfu_documents: str):
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
                        key2mitcfu_id[key] = mitcfu_id # used to handle tombstones
                        if mitcfu_id not in document_dict_keys:
                            new_document_dict[key] = js
                        message_counter += 1
                        if limit and message_counter >= limit:
                            logger.info(
                                f"Reached limit of {limit} cfu-documents. Stopping consumer."
                            )
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
    new_document_dict_mitcfu_ids = {
        _get_mitcfu_id(doc): doc for k, doc in new_document_dict.items()
    }
    return new_document_dict_mitcfu_ids, mitcfu_ids_for_deletion


def _get_mitcfu_id(jed_doc: dict) -> str:
    local_id = jed_doc["manifestations"]["bestRepresentations"][0]
    return local_id.split(":")[1]


def generate_embeddings_and_labels(
    new_documents: dict, embedding_model_path: str, batch_size: int|None=None
):
    embedding_model = e5multilingualEmbedder(embedding_model_path)
    logger.info(f"Loaded {embedding_model_path} on device: {embedding_model.device}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    if not batch_size or batch_size <= 0:
        batch_size = len(new_documents)
        logger.info(f"Batch size not specified, setting it to {batch_size}")

    embeddings = []
    labels = []
    abstracts_to_embed_batch = []
    labels_for_batch = []
    logger.info(f"Embedding {len(new_documents)} documents in batches of {batch_size}")
    for doc_id, doc in new_documents.items():
        if not validate_abstract(doc):
            continue

        abstract_list = doc.get("abstract")
        abstract = " ".join(abstract_list)
        text = f"{abstract}"

        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            abstracts_to_embed_batch.append(chunk)
            labels_for_batch.append(f"{doc_id}_chunk{i}")

        # Check if the batch size is reached, so we can start embedding the current batch. Otherwise, we continue filling the batch
        if len(abstracts_to_embed_batch) >= batch_size:
            embeddings_for_this_batch = embedding_model.embed_documents(
                abstracts_to_embed_batch
            )
            embeddings.extend(embeddings_for_this_batch)
            labels.extend(labels_for_batch)

            # Reset the batch for both embeddings and labels
            abstracts_to_embed_batch.clear()
            labels_for_batch.clear()

    # Check if there are any remaining embeddings in the (final) batch
    if abstracts_to_embed_batch:
        embeddings_for_this_batch = embedding_model.embed_documents(
            abstracts_to_embed_batch
        )
        embeddings.extend(embeddings_for_this_batch)
        labels.extend(labels_for_batch)

    return np.array(embeddings), np.array(labels)


def create_faiss_database(
    faiss_database_path: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    documents_to_delete: list,
):
    # if updating existing database, load it here
    if faiss_database_path and len(faiss_database_path) > 0:
        faiss_database = load_faiss_database(faiss_database_path)

        # update db with new embeddings and labels
        faiss_database.update(embeddings, labels)
    else:
        faiss_database = build_faiss_database(embeddings, labels)

    # delete documents marked for deletion
    faiss_database.delete_by_prefix(documents_to_delete)

    faiss_database.save(index_path="embeddings", labels_path="labels")


def load_faiss_database(faiss_database_path: str) -> KNNSearch:
    faiss_database = KNNSearch.load(
        faiss_database_path + "/embeddings", faiss_database_path + "/labels.npy"
    )
    return faiss_database


def build_faiss_database(embeddings, labels):
    faiss_database = KNNSearch.build(embeddings, labels)
    return faiss_database


def validate_abstract(document: dict) -> bool:
    abstract_list = document.get("abstract")
    abstract = " ".join(abstract_list)
    # before appending, check if the text is non-string type
    if not isinstance(abstract, str):
        logger.debug(
            f"Abstract is not a string: {abstract} in doc {document} with id {document['workId']}"
        )
        return False

    if len(abstract) < 150:
        return False

    return True


def __check_args(args: argparse.Namespace):
    if (args.index_input_path is None or len(args.index_input_path) < 1) and (
        args.index_output_path is None
        or len(args.index_output_path) < 1
    ):
        logger.error("No index input or output file specified. Exiting.")
        sys.exit(1)
