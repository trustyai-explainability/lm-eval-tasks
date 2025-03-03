import datasets
import numpy as np
import os

from transformers import AutoTokenizer

from lm_eval.utils import eval_logger


def validate_env_args():
    try:
        guard_url = os.environ["GUARD_URL"]

        if "v1/completions" not in guard_url:
            msg = f"GUARD_URL must be a /v1/completions endpoint, received {guard_url}."
            eval_logger.error(msg)
            raise ValueError(msg)

    except KeyError:
        eval_logger.error(
            "GUARD_URL is not set. If you are running the `realtoxicityprompts_granite` task, please set this environment variable to the /v1/completions endpoint of your guard model."
        )
        raise

    try:
        guard_model = os.environ["GUARD_MODEL"]
    except KeyError:
        eval_logger.error(
            "GUARD_MODEL is not set. If you are running the `realtoxicityprompts_granite` task, please set this environment variable to match the name of your guard model in vLLM."
        )
        raise

    try:
        guard_repo = os.environ["GUARD_REPO"]
        if "ibm-granite/granite-guardian" in guard_repo:
            pass
        else:
            eval_logger.error(
                f"{guard_repo} is not yet supported."
            )
            raise
    except KeyError:
        eval_logger.error(
            "GUARD_REPO is not set. If you are running the `realtoxicityprompts_granite` task, please set this environment variable to the HF repo of your guard model."
        )
        raise

    try:
        generator_repo = os.environ["GENERATOR_REPO"]
    except KeyError:
        eval_logger.error(
            "GENERATOR_REPO is not set. If you are running the `realtoxicityprompts_granite` task, please set this environment variable to the HF repo of your generation model."
        )
        raise

    system_prompt = os.environ.get("SYSTEM_PROMPT","")
    qps = int(os.environ.get("API_QPS", 1))

    return guard_url, guard_model, guard_repo, qps, generator_repo, system_prompt


def chat_mapping(row, system_prompt):
    # row['prompt']['text'] = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": row['prompt']['text']}
    # ]
    return row


def apply_chat_template(dataset, generator_repo, system_prompt):
    return dataset.map(lambda row: chat_mapping(row, system_prompt))


def dataset_subset(full_dataset: datasets.Dataset):
    # throw an early error if the metric env args are not valid
    _, _, _, _, generator_repo, system_prompt = validate_env_args()

    # get data subset
    np.random.seed(0)
    sample_size = int(os.environ.get("DATASET_SIZE", 5000))
    samples = np.random.choice(np.arange(0, len(full_dataset)), size=sample_size, replace=False)
    templated = apply_chat_template(full_dataset.select(samples), generator_repo, system_prompt)
    return templated