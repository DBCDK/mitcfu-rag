#!/usr/bin/env python
import argparse

import json
from rich.console import Console, OverflowMethod
from rich.prompt import Prompt
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from mitcfu_rag.config import RAG


# Config of colors
rag_output_color = "green3"
input_debug_color = "purple"
header_color = "bold dark_red"


### SETUP
RAG = RAG()
history = []


console = Console()
console.clear()
cprompt = Prompt(console=console)


def display_query_debug(query):
    console.print(
        Align(
            Panel(
                f"[{input_debug_color}]" + "\n".join(query) + f"[/{input_debug_color}]",
                title="query",
                title_align="left",
                expand=False,
                style=input_debug_color,
            ),
            align="right",
            pad=True,
        )
    )


def display_rag(response):
    console.print(
        f"\n :robot_face: [{rag_output_color}]>: "
        + response
        + f"[/{rag_output_color}]\n",
        markup=None,
    )


def display_stream(stream, response):
    console.print(
        f"\n :robot_face: [{rag_output_color}]> [/{rag_output_color}]: ", end=""
    )
    for word in stream:
        console.print(f"[{rag_output_color}]{word}[/{rag_output_color}]", end="")
        response.append(word)
    console.print("")


def chat(user_cfg: dict | None = None):
    cfg = {"display_rag_input": False}
    cfg = {"stream": True}
    if user_cfg:
        cfg.update(user_cfg)

    ### MAIN LOOP
    logo = """
    Hov, jeg er indtil videre stadig bare en anden version af:
      _____     _    _         ____ _           _
     |  _____ _| | _| |_ __ _ / ___| |__   __ _| |_
     | |_ / _` | |/ | __/ _` | |   | '_ \ / _` | __|
     |  _| (_| |   <| || (_| | |___| | | | (_| | |_
     |_|  \__,_|_|\_\\__\__,_|\____|_| |_|\__,_|\__|
    """

    console.print(f"[{header_color}]{logo}[/{header_color}]\n")
    console.print(f"[{header_color}]'    Faktalink chatbot'[/{header_color}]\n")
    console.print("    spørg mig om noget :smiley:")

    while True:
        console.print(" :nerd_face: >", end="")
        user_input = cprompt.ask(f"").strip()

        if not user_input or b"[" in user_input.encode():
            continue

        history.append({"role": "user", "content": user_input})
        query = [json.dumps(h) for h in history]
        if cfg["display_rag_input"]:
            display_query_debug(query)

        if cfg["stream"]:
            response = []
            stream = RAG.stream_response(history)
            display_stream(stream, response)
        else:
            response = RAG.get_response(query)
            display_rag(response)

        history.append({"role": "assistant", "content": "".join(response)})


def cli():
    """Commandline interface"""

    port = 5000
    parser = argparse.ArgumentParser(description="term faktachat")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument(
        "-s",
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Dont stream output",
    )

    args = parser.parse_args()

    print(args)
    cfg = {"display_rag_input": args.verbose, "stream": args.stream}
    chat(cfg)


if __name__ == "__main__":
    cli()
