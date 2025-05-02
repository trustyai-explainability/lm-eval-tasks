import datasets
import logging
import numpy as np
import os

from transformers import AutoTokenizer

logger = logging.getLogger("granite-risk-utils")


def get_env(variable, variable_desc):
    """Read an environment variable, throw an exception if it does not exist."""
    var = os.environ.get(variable)
    if var is None:
        msg = (f"{variable} is not set. If you are running the `granite_risk_eval` task, please set this "
               f"environment variable to {variable_desc}.")
        logger.error(msg)
        raise ValueError(msg)
    return var


def validate_env_args():
    guard_url = get_env("GUARD_URL", "the full /v1/chat/completions endpoint of your granite-guard model")
    if "v1/completions" not in guard_url:
        msg = f"GUARD_URL must be a /v1/completions endpoint, received {guard_url}."
        logger.error(msg)
        raise ValueError(msg)

    guard_model = get_env("GUARD_MODEL", "match the name of your guard model in vLLM.")
    guard_model_api_key = os.environ.get("GUARD_MODEL_API_KEY")
    guard_repo = get_env("GUARD_REPO", "the Hugging Face repo of your guard model")
    qps = int(os.environ.get("API_QPS", 1))
    sample_size = int(os.environ.get("DATASET_SIZE", 8520))

    return guard_url, guard_model, guard_model_api_key, guard_repo, qps, sample_size


def dataset_subset(full_dataset: datasets.Dataset):
    # throw an early error if the metric env args are not valid
    _, _, _, _, _, sample_size = validate_env_args()

    # get data subset
    np.random.seed(0)
    return full_dataset.take(sample_size)

