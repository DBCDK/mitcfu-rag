import json
import pandas as pd
import os
from datetime import datetime
import argparse
import requests
import random
import pprint
import re
import logging
from openai import APIConnectionError, OpenAI, RateLimitError
from mitcfu_rag.config import DEFAULT_MODEL
from mitcfu_rag.tools.llm_formatting import gen_wrapper
from mitcfu_rag.evaluation_tools.prompt_template import (
    IN_CONTEXT_EXAMPLES,
    INSTRUCTIONS,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def sample_evaluation_questions(
    limit: int = 30, question_type: str = "all"
) -> list[dict]:
    df = pd.read_csv("src/mitcfu_rag/evaluation_tools/testset/query_answer.csv")
    if question_type == "all":
        df_finale = df

    else:
        df_finale = df[df["question_type"] == question_type]

    if limit < 30:
        list_x = random.sample(range(0, len(df_finale)), limit)
        df_finale = df_finale.loc[list_x]

    return df_finale.to_dict("records")


def load_json_file(file_path: str):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def attempt_api_call(
    client, model_name: str, messages: list[dict[str:str]], max_retries=10
):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_responses(dict_evaluation: dict):
    """Save the response from the API to a file."""
    time_now = datetime.now().strftime("evaluation-%d-%m-%Y-%H-%M-%S")
    output_directory = "api_responses/" + time_now
    os.makedirs(output_directory, exist_ok=True)
    for key, value in dict_evaluation.items():
        if key == "list_hallucination":
            file_name = datetime.now().strftime("hallucination-%d-%m-%Y-%H-%M-%S.json")
        elif key == "list_correct":
            file_name = datetime.now().strftime("correct-%d-%m-%Y-%H-%M-%S.json")
        elif key == "q_type_dict":
            file_name = datetime.now().strftime("question-type-%d-%m-%Y-%H-%M-%S.json")

        file_path = os.path.join(output_directory, file_name)
        with open(file_path, "w") as f:
            json.dump(value, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1


def generate_predictions(participant_model, evaluation_file):
    predictions = []
    for line in tqdm(evaluation_file, desc="Generating Predictions"):
        query = [line["query"]]
        list_links = [line["link_to_answer_1"], line["link_to_answer_2"]]


        response_stream = requests.post(
            participant_model,
            json={"messages": [{"role": "user", "content": query}],
                  "type": "evaluate_rag"},
            stream=True,
        )
        raw_response = [
            r for r in gen_wrapper(response_stream, DEFAULT_MODEL)
        ]
        response = "".join(raw_response)

        if "**Kilder**" in response:
            prediction, raw_reference = response.split("**Kilder**", 1)
            print(prediction)
            references = re.findall(r'\[.*?\]\((.*?)\)', raw_reference)
            print(references)
        else:
            prediction = ""
            references = []
        predictions.append(
            {
                "query": query,
                "ground_truth": str(line["answer"]).strip().lower(),
                "prediction": str(prediction).strip().lower(),
                "question_type": str(line["question_type"]).strip().lower(),
                "links": [x for x in list_links if str(x) != "nan"],
                "references": references,
            }
        )

    return predictions


def evaluate_link(links: list, prediction):
    for link in links:
        if link not in prediction:
            return False

    return True


def evaluate_responses(
    predictions: list,
    evaluation_model_name: str,
    openai_client,
    list_x: list,
    save_log: bool = True,
):
    n_correct = 0
    system_message = get_system_message()
    n_correct_q_type = {}
    n_hallucination_q_type = {}
    dict_to_print = {}
    list_hallucination = []
    list_correct = []
    n = 0
    n_links_in_prediction = 0

    for prediction_dict in tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    ):
        query, ground_truth, prediction, question_type, links = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"],
            prediction_dict["question_type"],
            prediction_dict["links"],
        )

        links_in_prediction = evaluate_link(links, prediction)
        if links_in_prediction:
            n_links_in_prediction += 1

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        # måske skal vi finde en anden måde, at evaluere når sprogmodellen svarer, at den ikke kender svaret
        # if prediction == "jeg ved det ikke" or prediction == "jeg ved det ikke.":
        #     n_miss += 1
        #     n_miss_q_type[question_type] = n_miss_q_type.get(question_type,0)+1

        #     if n in list_x:
        #         print("prediction dict (prediction = I don't know):")
        #         print(prediction_dict)
        #     n+=1
        #     miss = {'query': query, 'prediction': prediction,'ground_truth': ground_truth}
        #     list_miss.append(miss)
        #     continue

        # kan vi tjekke direkte, om ground truth er i svaret, uden at man behøver sende til gpt
        # if prediction == ground_truth:
        #     n_correct_exact += 1
        #     n_correct_exact_q_type[question_type] = n_correct_exact_q_type.get(question_type,0)+1

        #     n_correct += 1
        #     n_correct_q_type[question_type] = n_correct_q_type.get(question_type, 0)+1

        #     if n in list_x:
        #         print("prediction dict (prediction = ground truth):")
        #         print(prediction_dict)
        #     n+=1

        #     correct = {'query': query, 'prediction': prediction,'ground_truth': ground_truth}
        #     list_correct.append(correct)

        #     continue

        response = attempt_api_call(openai_client, evaluation_model_name, messages)
        if response:
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1
                n_correct_q_type[question_type] = (
                    n_correct_q_type.get(question_type, 0) + 1
                )
                correct = {
                    "query": query,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "response": response,
                    "links": links,
                    "links_in_prediction": links_in_prediction,
                }
                list_correct.append(correct)

            else:
                n_hallucination_q_type[question_type] = (
                    n_hallucination_q_type.get(question_type, 0) + 1
                )
                h = {
                    "query": query,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "response": response,
                    "links": links,
                    "links_in_prediction": links_in_prediction,
                }
                list_hallucination.append(h)

        if n in list_x and response:
            prediction_dict["evalution_response"] = response
            dict_to_print[n] = prediction_dict

            # pprint.pprint(dict_to_print, compact=True)
            # pprint.pprint(prediction_dict, compact=True)

        # elif n in list_x:
        #     print(prediction_dict)

        n += 1

    q_type_list = [
        "aggregation",
        "comparison",
        "false_premise",
        "multi-hop",
        "post_processing",
        "set",
        "simple",
    ]

    q_type_dict = {}
    for type in q_type_list:
        q_type_dict[type] = {
            "n hallucination": n_hallucination_q_type.get(type, 0),
            "n correct:": n_correct_q_type.get(type, 0),
        }

    dict_evaluation = {
        "list_hallucination": list_hallucination,
        "list_correct": list_correct,
        "q_type_dict": q_type_dict,
    }

    if save_log:
        log_responses(dict_evaluation)

    n = len(predictions)
    results = {
        # Vi skal finde en score, vi gerne vil bruge
        # "score": (2 * n_correct + n_miss) / n - 1,
        # "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct) / n,
        # "missing": n_miss / n,
        # "n_miss": n_miss,
        "n_correct": n_correct,
        # "n_correct_exact": n_correct_exact,
        "n_hallucination": n - n_correct,
        "n_links_in_predictio:": n_links_in_prediction,
        "total": n,
    }

    df = pd.DataFrame.from_dict(results, orient="index")
    # print(json.dumps(q_type_dict, indent=4, sort_keys=True))
    df_question_type = pd.DataFrame.from_dict(q_type_dict, orient="index")

    return results, df, df_question_type, dict_to_print


def evaluate_references():
    pass


def main(model:str, limit: int, question_type: str, compare_model: bool, save_log: bool):

    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4o")

    # Generate predictions
    evaluation_file = sample_evaluation_questions(int(limit), question_type)
    predictions = generate_predictions(model, evaluation_file)
    list_x = random.sample(range(0, len(predictions)), 2)

    # Evaluate Predictions
    openai_client = OpenAI()

    evaluation_results, df_results, df_question_type, dict_to_print = (
        evaluate_responses(
            predictions, EVALUATION_MODEL_NAME, openai_client, list_x, save_log
        )
    )
    df_results.rename(columns={0: "model"}, inplace=True)

    if compare_model:
        predictions_compare = generate_predictions(compare_model, evaluation_file)

        (
            evaluation_results_compare,
            df_results_compare,
            df_question_type_compare,
            dict_to_print_compare,
        ) = evaluate_responses(
            predictions_compare, EVALUATION_MODEL_NAME, openai_client, list_x, save_log
        )
        df_results_compare.rename(columns={0: "model_compare"}, inplace=True)

        df_results = pd.concat([df_results, df_results_compare], axis=1)

        for i in list_x:
            print("")
            print("Examples of RAG")
            pprint.pprint(dict_to_print[i], compact=True)
            print("")
            print("Examples of ComparisonRAG")
            pprint.pprint(dict_to_print_compare[i], compact=True)

    else:
        for i in list_x:
            print("")
            print("")
            print("Examples of RAG")
            pprint.pprint(dict_to_print[i], compact=True)

    print("")
    print("Resultater fordelt på type af spørgsmål:")
    print(df_question_type)

    print("")
    print("")
    print("Totalopgørelse for antal korrekte og forkerte:")
    print(df_results)


def cli():
    """Command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="URL to the model to evaluate")
    parser.add_argument(
        "-n", "--n-questions", default=30, help="number of evaluation_questions"
    )
    parser.add_argument(
        "-q", "--question-type", default="all", help="Type of question."
    )
    parser.add_argument(
        "-c",
        "--compare-model",
        help="Path to another model to compare output with",
    )
    parser.add_argument(
        "-s",
        "--save-log",
        default=True,
        action="store_false",
        help="set to False if logs are not to be saved",
    )

    return parser.parse_args()


def run():
    """Entrypoint for running these methods using the CLI"""
    args = cli()
    main(args.model, args.n_questions, args.question_type, args.compare_model, args.save_log)


if __name__ == "__main__":
    run()
