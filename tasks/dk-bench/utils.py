import logging
import os
logger = logging.getLogger("dk-bench-utils")


# Data loading utilities for the dk-bench task =========================================================================
def get_env(variable, variable_desc):
    """Read an environment variable, throw an exception if it does not exist."""
    var = os.environ.get(variable)
    if var is None:
        msg = (f"{variable} is not set. If you are running the `dk-bench` task, please set this "
               f"environment variable to {variable_desc}.")
        logger.error(msg)
        raise ValueError(msg)
    return var


def validate_env_args():
    """Read all necessary environment arguments, and throw exceptions if any are missing"""
    dataset_path = get_env("DK_BENCH_DATASET_PATH",
                           "the location of the dk-bench .jsonl file to use in the evaluation.")
    judge_model_url = get_env("JUDGE_MODEL_URL", "the URL of your judge model's /chat/completions/ endpoint")
    judge_model_name = get_env("JUDGE_MODEL_NAME", "the desired judge model name.")
    judge_api_key = get_env("JUDGE_API_KEY", "the API key for the judge model.")
    requests_per_second = os.environ.get("REQUESTS_PER_SECOND", 1)
    return dataset_path, judge_model_url, judge_model_name, judge_api_key, requests_per_second


'''
Here, we validate env variables immediately upon task invocation. This catches misconfiguration immediately,
rather than waiting for the response evaluation stage.

We provide the dk_bench_data_location as a global variable, which lets us access it directly in the task yaml. This
lets us dynamically specify the location of dk-bench file via an environment variable- it would otherwise
have to be hardcoded into the task yaml. 
'''
dk_bench_data_location = validate_env_args()[0]

