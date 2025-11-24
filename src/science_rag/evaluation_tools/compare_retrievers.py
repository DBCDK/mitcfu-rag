import logging
import argparse
from tqdm.auto import tqdm
from typing import Optional
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from mitcfu_rag.rag.rag import Reference
from mitcfu_rag.evaluation_tools.evaluation import sample_evaluation_questions

logger = logging.getLogger(__name__)

console = Console()


def get_retrieval_models():
    retrieval_model_instances = []
    for Model in Compare_Retrievers:
        retrieval_model_instances += [Model()]
    return retrieval_model_instances


def evaluate_retrieval(golden_refs, model_refs):
    # print(f'golden refs: {golden_refs}')
    # print()
    # print([ref.article_link for ref in model_refs])
    # print()
    golden_refs_in_retrieval = 0
    golden_refs_not_in_retrieval = 0
    retrieval_ranking_score = 0
    if golden_refs[0] in model_refs:
        golden_refs_in_retrieval += 1
        ref_rank = model_refs.index(golden_refs[0])
        retrieval_ranking_score += -ref_rank
    # TODO: does not consider more than 1 reference yet
    else:
        golden_refs_not_in_retrieval += 1
    return np.array(
        [
            golden_refs_not_in_retrieval,
            golden_refs_in_retrieval,
            retrieval_ranking_score,
        ]
    )


def main(
    limit: int,
    question_type: str,
):
    # Get references
    evaluation_file = sample_evaluation_questions(int(limit), question_type)
    retrievers = get_retrieval_models()

    model2refs = {}
    model2refs["index"] = [
        "golden_refs_not_in_retrieval",
        "golden_refs_in_retrieval",
        "diff_retrieval_rank_avg",
    ]
    for line in tqdm(evaluation_file, desc="Retrieving relevant documents"):
        query = line["query"]
        message = [{"role": "user", "content": query}]
        # print(f'query: {query}')
        golden_refs = [
            Reference("", "", line["link_to_answer_1"], line["text_snippet_1"]),
            Reference("", "", line["link_to_answer_2"], line["text_snippet_2"]),
        ]
        for model in retrievers:
            scores, refs = model.retrieve(message)
            # print(model.__class__.__name__)
            model2refs[model.__class__.__name__] = model2refs.get(
                model.__class__.__name__, np.array([0, 0, 0])
            ) + evaluate_retrieval(golden_refs, refs)

    df = pd.DataFrame.from_dict(model2refs)
    df.set_index("index", inplace=True)
    df.loc["diff_retrieval_rank_avg"] = (
        df.loc["diff_retrieval_rank_avg"] / df.loc["golden_refs_in_retrieval"]
    ).round(2)

    # Initiate a Table instance to be modified
    table = Table(show_header=True, header_style="bold magenta")

    # Modify the table instance to have the data from the DataFrame
    table = df_to_table(df, table)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    console.print(table)


# https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938
def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in zip(
        pandas_dataframe.index.tolist(), pandas_dataframe.values.tolist()
    ):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


# TODO
def compare_top_5(question: str = None):
    pass


def cli():
    """Command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n-questions", default=30, help="number of evaluation_questions"
    )
    parser.add_argument(
        "-q", "--question-type", default="all", help="Type of question."
    )
    # TODO
    parser.add_argument(
        "--compare-top-5",
        action="store_true",
        default=False,
        help="to compare top 5 retrieved references to questions from the testset.",
    )
    # TODO
    parser.add_argument(
        "--question",
        type=str,
        help="Optional a question string can be given to compare top 5 retrieved references. \
                        Obs, can just be used with --compare-top-5 flag.",
    )
    return parser.parse_args()


def run():
    """Entrypoint for running these methods using the CLI"""
    args = cli()
    if args.compare_top_5:
        compare_top_5(args.question)
    else:
        main(args.n_questions, args.question_type)


if __name__ == "__main__":
    run()
