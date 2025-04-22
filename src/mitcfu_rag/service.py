#!/usr/bin/env python3

"""
:mod:`fakta_chat.streaming_endpoint` -- endpoint for streaming RAG

==================
Streaming Endpoint
==================

Endpoint for streaming RAG
"""

import logging
import asyncio
import datetime
import json
import tornado.web as tw
from dbc_pyutils import create_instance_id
from dbc_pyutils import Statistics
from dbc_pyutils import build_info
from dbc_pyutils import JSONFormatter
from dbc_pyutils import StatusHandler
from dbc_pyutils import PrometheusMixIn
from dbc_pyutils import MetricsHandler
from dbc_pyutils import BaseHandler

from mitcfu_rag.config import RAG

INSTANCE_ID = create_instance_id(num_digits=8)
STATS = {"query": Statistics(name="query")}
logger = logging.getLogger(__name__)


class StreamingHandler(BaseHandler):
    """
    JudgeTheCoverHandler
    """

    def initialize(self, model, info, stat_collector):
        """
        Initializes handler
        """
        self.info = info
        self.stat_collector = stat_collector
        self.static_header_content = {
            "build": self.info["build_number"],
            "git": self.info["git"],
            "version": self.info["version"],
        }
        self.model = model

    async def post(self):
        self.set_header("Content-Type", "text/plain; charset=utf-8")
        body = json.loads(self.request.body.decode("utf8"))
        self.version = body.get("version", "v1")
        messages = body.get("messages", [])

        self.flush()

        result = await self.generate_response(messages)
        async for chunk in result:
            self.write(chunk)
            await self.flush()

    def __stream_response(self, messages):
        response_stream = self.model.stream_response_with_validator(messages, self.version)
        return response_stream

    async def generate_response(self, messages):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.__stream_response(messages))


class MetricsApp(PrometheusMixIn, tw.Application):
    pass


def make_app(model):
    info = build_info.get_info("mitcfu_rag")
    handlers = [
        (r"/", StreamingHandler, dict(model=model, info=info, stat_collector=STATS["query"])),
        (r"/metrics", MetricsHandler),
        ("/status", StatusHandler, dict(ab_id=1, info=info, instance_id=INSTANCE_ID, statistics=list(STATS.values()))),
    ]
    return MetricsApp(handlers)


async def main(args):
    # logger.info("Loading model")
    model = RAG(
        args.enable_validator,
    )  # args.embedding_model_path, args.faiss_path, args.article_index_path are not used currently
    logging.info(f"Starting endpoint at port {args.port}")
    app = make_app(model)
    app.listen(args.port)
    await asyncio.Event().wait()


def cli():
    """Commandline interface"""
    import argparse

    port = 5000
    parser = argparse.ArgumentParser(description="query related subject")
    # parser.add_argument("embedding_model_path", metavar="embedding-model-path", help="path to embedding model")
    # parser.add_argument("faiss_path", metavar="faiss-path", help="path to faiss index")
    # parser.add_argument("article_index_path", metavar="article-index-path", help="path to article index")
    parser.add_argument(
        "--validator-model", dest="enable_validator", help="Enable the validator model in ms_marco_minilm_validator.py", action="store_true")
    parser.add_argument("-a", "--ab-id", dest="ab_id", help="ab id of service. default is 1", default=1)
    parser.add_argument(
        "-p", "--port", dest="port", type=int, help=f"port to expose service on. Default is {port}", default=port
    )
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")

    args = parser.parse_args()
    if args.verbose:
        print("Setting logging level to DEBUG in service.py cli()")
        level = logging.DEBUG
    else:
        print("Setting logging level to INFO in service.py cli()")
        level = logging.INFO
    logger = logging.getLogger("")
    logger.setLevel(level)

    asyncio.run(main(args))
