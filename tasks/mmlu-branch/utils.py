import logging
import os
logger = logging.getLogger("mmlu-branch-utils")


# Data loading utilities for the dk-bench task =========================================================================
def get_env(variable, variable_desc):
    """Read an environment variable, throw an exception if it does not exist."""
    var = os.environ.get(variable)
    if var is None:
        msg = (f"{variable} is not set. If you are running the `mmlu-branch` task, please set this "
               f"environment variable to {variable_desc}.")
        logger.error(msg)
        raise ValueError(msg)
    return var


def validate_env_args():
    """Read all necessary environment arguments, and throw exceptions if any are missing"""
    dataset_path = get_env("MMLU_BRANCH_DATASET_PATH",
                           "the location of the mmlu-branch .jsonl file to use in the evaluation")
    comparison_results_file = os.environ.get("COMPARISON_RESULTS_PATH", None)
    return dataset_path, comparison_results_file


'''
Here, we validate env variables immediately upon task invocation. This catches misconfiguration immediately,
rather than waiting for the response evaluation stage.

We provide the mmlu_branch_data_locationas a global variable, which lets us access it directly in the task yaml. This
lets us dynamically specify the location of mmlu-branch file via an environment variable- it would otherwise
have to be hardcoded into the task yaml. 
'''
mmlu_branch_data_location = validate_env_args()[0]

