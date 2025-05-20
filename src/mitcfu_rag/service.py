#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.streaming_endpoint` -- endpoint for streaming RAG

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
from dbc_pyutils import setup_logging
from dbc_pyutils import StatusHandler
from dbc_pyutils import PrometheusMixIn
from dbc_pyutils import MetricsHandler
from dbc_pyutils import BaseHandler

from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict

from mitcfu_rag.config import RAG, AGENTIC
from mitcfu_rag.config import (
    RAG_TEMPLATE,
    SIMPLE_TEMPLATE,
    ROUTER_TEMPLATE,
    FALLBACK_TEMPLATE,
    FOLLOW_UP_TEMPLATE,
    REFORMULATE_TEMPLATE
)

INSTANCE_ID = create_instance_id(num_digits=8)
STATS = {"query": Statistics(name="query")}
logger = setup_logging()

path_to_embeddings = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/embeddings"
path_to_labels = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds-e5-multilingual-instruct-faiss-index/labels.npy"
path_to_JEDs = "/data/rani/mitcfu-data/10plus-abstract-77295-jeds"


class AgentState(TypedDict):
    input: str
    output: str
    agent: str
    prompt_template: str


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
        self.route_template = ROUTER_TEMPLATE()
        self.simple_template = SIMPLE_TEMPLATE
        self.rag_template = RAG_TEMPLATE
        self.fallback_template = FALLBACK_TEMPLATE
        self.follow_up_template = FOLLOW_UP_TEMPLATE
        self.reformulate_template = REFORMULATE_TEMPLATE
        self.graph = self.create_graph()

    def create_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("route", self.route_response)
        workflow.add_node("rag_agent", self.rag_response)
        workflow.add_node("simple_agent", self.simple_response)
        workflow.add_node("fallback_agent", self.fallback_response)
        workflow.add_node("follow_up_agent", self.follow_up_response)

        workflow.add_conditional_edges(
            "route",
            lambda x: x["agent"],
            {
                "RAG": "rag_agent",
                "SIMPLE": "simple_agent",
                "FALLBACK": "fallback_agent",
                "FOLLOW_UP": "follow_up_agent",
            },
        )

        workflow.set_entry_point("route")
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("simple_agent", END)
        workflow.add_edge("fallback_agent", END)
        workflow.add_edge("follow_up_agent", END)
        #workflow.add_edge("summarizer_agent", END)

        return workflow.compile()

    async def post(self):
        self.set_header("Content-Type", "text/plain; charset=utf-8")
        body = json.loads(self.request.body.decode("utf8"))
        self.version = body.get("version", "v1")
        messages = body.get("messages", [])

        self.flush()

        result = await self.graph.ainvoke({"input": messages})
        async for chunk in result["output"]:
            self.write(chunk)
            await self.flush()

    async def route_response(self, messages):
        messages["agent"] = "ROUTER"
        route_result_stream = await self.stream_response(messages, self.route_template)
        raw_response = [r async for r in self.gen_wrapper(route_result_stream)]
        route_result = "".join(raw_response)
        print(route_result)
        logger.info(f"Router result:\n{route_result}\n")
        if set("RAG").issubset(set(route_result)):
            return {"agent": "RAG"}
        elif set("SIMPLE").issubset(set(route_result)):
            return {"agent": "SIMPLE"}
        elif set("FOLLOW_UP").issubset(set(route_result)):
            return {"agent": "FOLLOW_UP"}
        else:
            return {"agent": "FALLBACK"}  # create fallback here

    async def simple_response(self, messages):
        result = await self.stream_response(messages, self.simple_template)
        return {"output": result}

    async def rag_response(self, messages):
        messages["agent"] = "REFORMULATOR"
        reformulated_response = await self.stream_response(messages, self.reformulate_template)
        raw_response = [r async for r in self.gen_wrapper(reformulated_response)]
        reformulate_output = "".join(raw_response).replace('json', '').replace("```", "")
        logger.info(f"Reformulated response:{reformulate_output}")
        try:
            json_response = json.loads(reformulate_output)
            messages["reformulated_queries"] = json_response.get("søgninger", [])
        except:
            logger.info("Unable to ")
            messages["reformulated_queries"] = []
        messages["agent"] = "RAG"
        result = await self.stream_response(messages, self.rag_template)
        return {"output": result}

    async def fallback_response(self, messages):
        result = await self.stream_response(messages, self.fallback_template)
        return {"output": result}

    async def follow_up_response(self, messages):
        messages["agent"] = "REFORMULATOR"
        reformulated_response = await self.stream_response(
            messages, self.reformulate_template
        )
        raw_response = [r async for r in self.gen_wrapper(reformulated_response)]
        reformulate_output = (
            "".join(raw_response).replace("json", "").replace("```", "")
        )
        logger.info(f"Reformulated response:{reformulate_output}")
        try:
            json_response = json.loads(reformulate_output)
            messages["reformulated_queries"] = json_response.get("søgninger", [])
        except:
            logger.info("Unable to ")
            messages["reformulated_queries"] = []
        messages["agent"] = "FOLLOW_UP"
        result = await self.stream_response(messages, self.follow_up_template)
        return {"output": result}

    async def stream_response(self, messages, template):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.__generate(messages, self.model, template)
        )

    def __generate(self, messages, model, template):
        response_stream = model.stream_response(messages, template)
        return response_stream

    async def gen_wrapper(self, stream):
        def decode(input):
            try:
                if isinstance(input, str):
                    return input
                else:
                    return input.decode("utf-8")
            except UnicodeDecodeError as e:
                return input.decode("utf-8", errors="ignore")

        # for item in stream.iter_content(chunk_size=None, decode_unicode=True):
        #     for i in item:
        #         yield i

        async for item in stream:
            decoded_item = decode(item)
            obj = json.loads(decoded_item.replace("data:", ""))
            for choice in obj.get("choices", []):
                if token := choice.get("delta", {}).get("content", ""):
                    if not token == "<end_of_turn>":
                        yield token


class MetricsApp(PrometheusMixIn, tw.Application):
    pass


def make_app(model):
    info = build_info.get_info("mitcfu_rag")
    handlers = [
        (
            r"/",
            StreamingHandler,
            dict(model=model, info=info, stat_collector=STATS["query"]),
        ),
        (r"/metrics", MetricsHandler),
        (
            "/status",
            StatusHandler,
            dict(
                ab_id=1,
                info=info,
                instance_id=INSTANCE_ID,
                statistics=list(STATS.values()),
            ),
        ),
    ]
    return MetricsApp(handlers)


async def main(args):
    logger.info("Loading model")
    model = AGENTIC(
        args.embedding_model_path,
        args.faiss_path,
        args.article_index_path,
        args.validator_model_path,
    )
    logger.info(f"Starting endpoint at port {args.port}")
    app = make_app(model)
    app.listen(args.port)
    await asyncio.Event().wait()


def cli():
    """Commandline interface"""
    import argparse

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
        default=path_to_embeddings,
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
        "-a", "--ab-id", dest="ab_id", help="ab id of service. default is 1", default=1
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        help=f"port to expose service on. Default is {port}",
        default=port,
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="verbose output"
    )

    args = parser.parse_args()
    level = logging.INFO
    logger.setLevel(level)

    asyncio.run(main(args))
