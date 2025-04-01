# MMLU-BRANCH

Implementation of InstructLab's [mmlu-bench evaluation](https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L30) for the lm-evaluation-harness framework. 


## Necessary Environment Arguments
* `MMLU_BRANCH_DATASET_PATH` The path to the mmlu-branch `.jsonl` file.

## Optional Environment Arguments
* `COMPARISON_RESULTS_PATH` Path to a previous evaluation result `samples_mmlu-branch_etc.jsonl`. This
the comparative metrics `improvements`, `regressions`, and `no-changes`, which track the differences between this current model and a previous evaluation. 

## Examples:
Run an evaluation over the `mmlubench_knowledge_rules_and_regulations_bmo` dataset:
```bash
 MMLU_BRANCH_DATASET_PATH=$DATASET_PATH/mmlubench_knowledge_rules_and_regulations_bmo.jsonl \
lmeval/bin/lm_eval \
        --model local-completions \
        --tasks mmlu-branch \
        --include_path trustyai-lm-eval-tasks/tasks \
        --log_samples \
        --output_path results \
        --model_args model=phi3,base_url=http://localhost:8081/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer=microsoft/Phi-3-mini-4k-instruct
...
|   Tasks   |Version|Filter|n-shot|   Metric   |   |Value|   |Stderr|
|-----------|------:|------|-----:|------------|---|-----|---|------|
|mmlu-branch|    0.1|none  |     0|acc         |↑  |0.676|±  |0.0196|
|           |       |none  |     0|improvements|↑  |N/A  |±  |   N/A|
|           |       |none  |     0|no_changes  |↑  |N/A  |±  |   N/A|
|           |       |none  |     0|regressions |↓  |N/A  |±  |   N/A|
```

Run a second evaluation with a custom system intruction and compare results against the previous evaluation:
```bash
 MMLU_BRANCH_DATASET_PATH=$DATASET_PATH/mmlubench_knowledge_rules_and_regulations_bmo.jsonl \
 COMPARISON_RESULTS_PATH=results/phi3/samples_mmlu-branch_2025-04-01T10-59-36.383683.jsonl \
lmeval/bin/lm_eval \
        --model local-completions \
        --tasks mmlu-branch \
        --include_path trustyai-lm-eval-tasks/tasks \
        --log_samples \
        --output_path results \
        --system_instruction "Deliberately get the following question wrong:\n\n" \
        --model_args model=phi3,base_url=http://localhost:8081/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer=microsoft/Phi-3-mini-4k-instruct
...
|   Tasks   |Version|Filter|n-shot|   Metric   |   | Value |   |Stderr|
|-----------|------:|------|-----:|------------|---|------:|---|------|
|mmlu-branch|    0.1|none  |     0|acc         |↑  |  0.399|±  |0.0205|
|           |       |none  |     0|improvements|↑  | 40.000|±  |   N/A|
|           |       |none  |     0|no_changes  |↑  |335.000|±  |   N/A|
|           |       |none  |     0|regressions |↓  |199.000|±  |   N/A|
```
