# DK-Bench

Implementation of InstructLab's [dk-bench evaluation](https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L30) for the lm-evaluation-harness framework. 


## Necessary Environment Arguments
* `DK_BENCH_DATASET_PATH` The path to the dk-bench `.jsonl` file containing your user_inputs and reference answers. 
* `JUDGE_MODEL_URL` The path to your judge model's `/chat/completions` endpoint.
* `JUDGE_MODEL_NAME` The name of your judge model, e.g., `gpt-4o`
* `JUDGE_API_KEY` The API key to use when authenticating to the model endpoint.

## Sample Result
| Tasks  |Version|Filter|n-shot|      Metric       |   | Value |   |Stderr|
|--------|------:|------|-----:|-------------------|---|------:|---|------|
|dk-bench|    0.1|none  |     0|invalid_score_count|↓  | 1.0000|±  |   N/A|
|        |       |none  |     0|mean_score         |↑  | 3.6939|±  |0.1019|
|        |       |none  |     0|score_1_count      |↑  | 0.0000|±  |   N/A|
|        |       |none  |     0|score_2_count      |↑  | 1.0000|±  |   N/A|
|        |       |none  |     0|score_3_count      |↑  | 9.0000|±  |   N/A|
|        |       |none  |     0|score_4_count      |↑  |38.0000|±  |   N/A|
|        |       |none  |     0|score_5_count      |↑  | 0.0000|±  |   N/A|