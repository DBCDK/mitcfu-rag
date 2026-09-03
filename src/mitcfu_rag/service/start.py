#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.service.start` -- entry point for streaming RAG service

==================
Streaming Endpoint
==================

FastAPI/uvicorn service exposing the agentic RAG graph. `dbc_pyutils`
integration (`/metrics`, enriched `/status`, structured JSON logging) is
mounted only when installed -- see `_dbc_optional`.
"""

import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI

from mitcfu_rag.config import DEFAULT_MODEL
from mitcfu_rag.rag.agent_streaming_rag import AgenticRAG
from mitcfu_rag.rag.langgraph_graphs import AgenticGraph
from mitcfu_rag.service import _dbc_optional
from mitcfu_rag.service.endpoints import router

logger = logging.getLogger(__name__)


def create_app(args) -> FastAPI:
    """Builds the FastAPI application for the given parsed CLI args."""
    logger.info("Loading model")
    model = AgenticRAG(
        embedding_model=args.embedding_model_path,
        faiss_index=args.faiss_path,
        jed_document_path=args.article_index_path,
        validator_model=args.validator_model_path,
        use_ceph=args.use_ceph,
    )
    agentic_graph = AgenticGraph(type=args.graph_type, model=model)

    if _dbc_optional.DBC_AVAILABLE:
        _dbc_optional.setup_logging()
        instance_id = _dbc_optional.create_instance_id(num_digits=8)
        info = _dbc_optional.build_info.get_info("mitcfu_rag")
        stats = {"query": _dbc_optional.Statistics(name="query")}
    else:
        logging.basicConfig(level=logging.INFO)
        instance_id = None
        info = None
        stats = {}

    app = FastAPI(title="mitcfu-rag service")
    app.state.agentic_graph = agentic_graph
    app.state.default_model = DEFAULT_MODEL
    app.state.dbc_available = _dbc_optional.DBC_AVAILABLE
    app.state.instance_id = instance_id
    app.state.build_info = info
    app.state.stats = stats
    app.state.ab_id = args.ab_id
    app.include_router(router)

    if _dbc_optional.DBC_AVAILABLE:
        _dbc_optional.install_base_handler(app)
        app.add_middleware(_dbc_optional.PrometheusMiddleware, excluded_paths={"/metrics", "/status"})
        app.add_api_route("/metrics", _dbc_optional.metrics_endpoint, methods=["GET"])

    return app


def parse_args(argv=None):
    """Commandline interface"""
    port = 5000
    parser = argparse.ArgumentParser(description="query related subject")
    parser.add_argument(
        "embedding_model_path",
        metavar="embedding-model-path",
        help="path to embedding model",
    )
    parser.add_argument(
        "faiss_path",
        metavar="faiss-path",
        help="path to faiss index",
    )
    parser.add_argument(
        "--article_index_path",
        metavar="article-index-path",
        help="path to article index",
        default=None,
    )
    parser.add_argument(
        "--validator-model-path",
        dest="validator_model_path",
        help="path to validator model",
        default=None,
    )
    parser.add_argument(
        "--graph-type",
        dest="graph_type",
        help="type of langgraph graph to use. default is service.",
        default="service",
    )
    parser.add_argument(
        "--use-ceph",
        dest="use_ceph",
        action="store_true",
        help="Set this flag if running on Ceph or in dockerfile",
    )
    parser.add_argument("-a", "--ab-id", dest="ab_id", help="ab id of service. default is 1", default=1)
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        help=f"port to expose service on. Default is {port}",
        default=port,
    )
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")
    return parser.parse_args(argv)


async def _serve(args) -> None:
    # `create_app` constructs `AgenticRAG`, which opens an `aiohttp.ClientSession` --
    # that requires a running event loop, so app construction happens here, inside
    # the loop `main()` drives, rather than before `uvicorn.run()` starts one.
    app = create_app(args)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"Starting endpoint at port {args.port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_config=None)
    server = uvicorn.Server(config)
    await server.serve()


def main():
    args = parse_args()
    asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
