from utils import OPERATIONS
import yaml

if __name__ == "__main__":
    # generate subtask configs
    with open("_tiny_offline_arithmetic_template_yaml", "r") as f:
        template = f.read()

    for operation in OPERATIONS:
        with open(f"tiny_offline_arithmetic_{operation.name}.yaml", "w") as f:
            f.write(template.replace("OPERATION", operation.name))

    task_names = [f"tiny_offline_arithmetic_{operation.name}" for operation in OPERATIONS]

    # generate parent config
    parent_yaml_dict = {
        "group": "tiny_offline_arithmetic",
        "task": task_names,
        "aggregate_metric_list": [
            {
                "metric": "acc",
                "weight_by_size": True
            }
        ],
        "metadata": {
            "version": 1
        }
    }
    with open('_tiny_offline_arithmetic.yaml', 'w') as outfile:
        yaml.dump(parent_yaml_dict, outfile, default_flow_style=False)