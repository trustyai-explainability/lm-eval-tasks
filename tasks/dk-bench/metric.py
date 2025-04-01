import json
import logging
import os

import pandas as pd
import sys
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry

sys.path.append(os.path.dirname(__file__))
from utils import validate_env_args

logger = logging.getLogger("dk-bench-metric")

''' DK-BENCH IMPLEMENTATION FOR LM-EVAL-HARNESS 
Based on the InstructLab eval framework (https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L30)
and Ragas' RubricsScore https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/general_purpose/#rubrics-based-scoring
'''


# === DEFINE TASK CONSTANTS ============================================================================================
SCORE_FEEDBACK = """
{"properties": {"feedback": {"description": "The feedback for the response", "title": "Feedback", "type": "string"}, \
"score": {"description": "The score given to the response", "title": "Score", "type": "integer"}}, \
"required": ["feedback", "score"], "title": "ScoreFeedback", "type": "object"}"""

SCORING_RUBRICS = {
    "score1_description": "The response is entirely incorrect, irrelevant, or does not "
                          "align with the reference in any meaningful way.",
    "score2_description": "The response partially matches the reference but contains major errors,"
                          " significant omissions, or irrelevant information.",
    "score3_description": "The response aligns with the reference overall but lacks sufficient detail,"
                          " clarity, or contains minor inaccuracies.",
    "score4_description": "The response is mostly accurate, aligns closely with the reference,"
                          " and contains only minor issues or omissions.",
    "score5_description": "The response is fully accurate, completely aligns with the reference,"
                          " and is clear, thorough, and detailed.",
}
MAX_GENERATION_SIZE = len(SCORE_FEEDBACK) + max(len(k) for k in SCORING_RUBRICS.values())
INSTRUCTION = ("Your task is to assign an appropriate score and provide feedback"
               " to the inputs based solely on the scoring criteria.")

SCORE_REPORTING_FORMAT = "score_{}_count"
evaluation_counter = None


# === JUDGE PROMPT FORMATTERS ==========================================================================================
def generate_output_signature() -> str:
    """Define the desired JSON schema that we ask the model to comply with"""

    return (
        f"Please return the output in a JSON format that complies with the "
        f"following schema as specified in JSON Schema:\n"
        f"{SCORE_FEEDBACK}"
        "Do not use single quotes in your response but double quotes,"
        "properly escaped with a backslash."
    )


def get_input_string(user_input, response, reference):
    """Format the (user_input, response, reference) tuple in the judge prompt"""

    return ("{\n"
            f"\t\"user_input\": \"{user_input}\"\n"
            f"\t\"response\": \"{response}\"\n"
            f"\t\"reference\": \"{reference}\"\n" 
            "}")


def get_judge_prompt(user_input, response, reference):
    """Build the judge prompt"""

    rubrics_text = "\n".join(f"{key}: {value}" for key, value in SCORING_RUBRICS.items())
    instruction = f"{INSTRUCTION}\n\nScoring Rubrics:\n{rubrics_text}\n"
    return (
            f"{instruction}\n"
            + generate_output_signature()
            + "\n"
            + "\n-----------------------------\n"
            + "\nNow perform the same with the following input\n"
            + (
                "input: " + get_input_string(user_input, response, reference) + "\n"
            )
            + "Output: "
    )


# === JUDGE REQUEST CONTENT ============================================================================================
def get_inference_payload(model_name, prompt):
    """Get the payload for the /chat/completions request"""
    return {
        "messages": [{"role": "user", "content": prompt}],
        "model": model_name,
        "seed": 42,
        "n": 1,
        "max_tokens": MAX_GENERATION_SIZE,
        "temperature": 0,
    }


# === CUSTOM AGGREGATION FUNCTIONS =====================================================================================
def count(x):
    """For whatever reason, lm-eval-harness does not provide a sum aggregation, so define one here"""
    return sum(x)


# === MAIN EVALUATION FUNCTION =========================================================================================
def evaluate(doc, predictions, total_retries=5):
    """Evaluate a single response from the student mode.

    * doc:          A single row from the original dataset. For dk-bench, this contains
                    `user_input` and `reference` fields.
    * predictions:  The output of the student model over this specific document.
    """

    # Read arguments from environment variables
    dk_bench_data_path, judge_model_url, judge_model_name, judge_api_key, requests_per_second = validate_env_args()

    # Progress logging logic
    global evaluation_counter
    if evaluation_counter is None:
        num_rows = len(pd.read_json(dk_bench_data_path, orient='records', lines=True))
        evaluation_counter = tqdm(total=num_rows, desc=f"Evaluating responses via {judge_model_name}")
    evaluation_counter.update()
    evaluation_counter.refresh()

    # Set up payload for judge model inference request
    headers = {
        "Authorization": f"Bearer {judge_api_key}",
        "content-type": "application/json",
    }
    judge_prompt = get_judge_prompt(
        user_input=doc['user_input'],
        response=predictions[0],
        reference=doc['reference']
    )
    logging.debug("\n" + judge_prompt)
    inference_payload = get_inference_payload(judge_model_name, judge_prompt)

    # Set up request session, request rate limiting, retrys
    s = requests.Session()
    sleeping_time = 1 / requests_per_second
    backoff_factor = sleeping_time / (2 ** (total_retries - 1))
    retries = Retry(total=total_retries, backoff_factor=backoff_factor)
    s.mount("http://", HTTPAdapter(max_retries=retries))

    # Post inference payload to judge model
    req_response = s.post(judge_model_url, json=inference_payload, headers=headers)

    # set up results container dictionary. This is a dictionary containing the results from evaluating this single row,
    # which is then aggregated later by lm-eval-harness
    result = {SCORE_REPORTING_FORMAT.format(i): 0 for i in range(1, 6)}
    result["invalid_score_count"] = 0  # for counting the number of parse errors we get
    result["mean_score"] = 0  # for tracking the average score over the whole dataset

    try:
        # parse the
        feedback = json.loads(req_response.json()["choices"][0]["message"]["content"])
        logging.debug(f"Judge Response: {feedback}")
    except json.JSONDecodeError:
        result['invalid_score_count'] = 1
    else:
        score = feedback['score']

        # validate that the produced score exists in the rubric, and if so, log it
        if f"score{score}_description" in SCORING_RUBRICS.keys():
            result[SCORE_REPORTING_FORMAT.format(score)] = 1
            result["mean_score"] = score
        else:
            result['invalid_score_count'] = 1

    return result