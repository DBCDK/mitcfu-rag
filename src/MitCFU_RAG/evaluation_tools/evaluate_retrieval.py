import argparse
from typing import Optional
import pandas as pd
from tqdm import tqdm
from rich import box
from rich.console import Console
from rich.table import Table

from fakta_chat.config import RAG, ComparisonRAG
from fakta_chat.evaluation_tools.evaluation import sample_evaluation_questions

console = Console()

def main(limit : int, question_type : str,):
    # Get references
    participant_model = RAG()
    comparison_model = ComparisonRAG()
    evaluation_file = sample_evaluation_questions(int(limit), question_type)
    
    for line in tqdm(evaluation_file, desc="Generating Predictions"):
        query = line['query']
        refs_1, ans_1 = participant_model.evaluate(query)
        refs_2, ans_2 = comparison_model.evaluate(query)
    
        print('\n\n')
        print(f'QUERY: """{query}"""')
        print()
        print('TOP REFERENCES WITH ANSWERS:\n')
        df = pd.DataFrame()
        df["MODEL: " + str(participant_model.__class__.__name__) + "\n\n" + f'GENERATOR: """{ans_1.strip()}"""' + "\n\n"] = \
                ["\n\n".join(["id: " + ref.id, "title: " + ref.article_headline, "snippet: \"\"\"" + ref.text + "\"\"\""]) + "\n\n" for ref in refs_1]
        df["MODEL: " + str(comparison_model.__class__.__name__) + "\n\n" + f'GENERATOR: """{ans_1.strip()}"""' + "\n\n"] = \
                ["\n\n".join(["id: " + ref.id, "title: " + ref.article_headline, "snippet: \"\"\"" + ref.text + "\"\"\""]) + "\n\n" for ref in refs_2]

        # Initiate a Table instance to be modified
        table = Table(show_header=True, header_style="bold magenta")

        # Modify the table instance to have the data from the DataFrame
        table = df_to_table(df, table)

        # Update the style of the table
        table.row_styles = ["none", "dim"]
        table.box = box.SIMPLE_HEAD

        console.print(table)

#https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938     
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

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table

def cli():
    """Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-questions', default=5,
                        help='number of evaluation_questions')
    parser.add_argument('-q', '--question-type', default='all',
                        help='Type of question.')
    
    return parser.parse_args()


def run():
    """Entrypoint for running these methods using the CLI
    """
    args = cli()
    main(args.n_questions, args.question_type)


if __name__ == '__main__':
    run()