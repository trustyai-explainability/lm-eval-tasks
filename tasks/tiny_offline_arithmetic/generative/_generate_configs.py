import os, sys
from os.path import dirname, abspath
sys.path.append(os.path.join(dirname(dirname(abspath(__file__))), "default"))
from utils import OPERATIONS
import yaml

if __name__ == "__main__":
    # generate subtask configs
    with open("_tiny_offline_arithmetic_generative_template_yaml", "r") as f:
        template = f.read()

    for operation in OPERATIONS:
        with open(f"tiny_offline_arithmetic_{operation.name}_generative.yaml", "w") as f:
            f.write(template.replace("OPERATION", operation.name))

    task_names = [f"tiny_offline_arithmetic_{operation.name}_generative" for operation in OPERATIONS]

    # generate parent config
    parent_yaml_dict = {
        "group": "tiny_offline_arithmetic_generative",
        "task": task_names,
        "aggregate_metric_list": [
            {
                "metric": "exact_match",
                "weight_by_size": True,
                "filter_list": "get_response"
            }
        ],
        "metadata": {
            "version": 1
        }
    }
    with open('_tiny_offline_arithmetic_generative.yaml', 'w') as outfile:
        yaml.dump(parent_yaml_dict, outfile, default_flow_style=False)