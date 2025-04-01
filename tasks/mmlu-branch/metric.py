import logging
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(__file__))
from utils import validate_env_args

logger = logging.getLogger("mmlu-branch-metric")


# === GLOBAL VARIABLES =============================================================================
comparison_df = None

def argmax(array, idx=None):
    """Implement argmax to avoid importing numpy"""
    max = None
    max_idx = -1
    for i, tup in enumerate(array):
        val = tup[idx] if idx is not None else tup
        if max is None or val > max:
            max = val
            max_idx = i
    return max_idx


def count(array):
    if any(x is not None for x in array):
        return sum(array)
    else:
        return "N/A"


def evaluate(doc, loglikelihoods):
    result = {"acc": 0, "improvements": None, "no_changes": None, "regressions": None}

    # Get answer accuracy
    model_answer = argmax(loglikelihoods, 0)
    if doc['answer'] == model_answer:
        result['acc'] = 1
    else:
        result['acc'] = 0

    # see if a historical comparison has been requested
    _, comparison_results_path = validate_env_args()
    if comparison_results_path is not None:
        # update results container if we want to perform historical comparison
        result["improvements"] = 0
        result["no_changes"] = 0
        result["regressions"] = 0

        # grab previous result file
        global comparison_df
        if comparison_df is None:  # only read the comparison df once and then store in global var
            comparison_df = pd.read_json(comparison_results_path, orient='records', lines=True)

        # find matching document in previos result
        matching_row = comparison_df[comparison_df['doc'] == doc]

        # if we have a matching doc row in the comparison data, compare the result of this evaluation
        # against the historic one
        if len(matching_row):
            matching_row = matching_row.iloc[0]
            if matching_row['acc'] > result['acc']:
                result['regressions'] = 1
            elif matching_row['acc'] < result['acc']:
                result['improvements'] = 1
            else:
                result['no_changes'] = 1
        else:
            logger.warning("Could not find a matching document in comparison results - make"
                           " sure they correspond to the same evaluation dataset.")

    return result

