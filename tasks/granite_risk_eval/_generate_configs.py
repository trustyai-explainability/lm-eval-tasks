from utils import RISKS
import yaml

if __name__ == "__main__":
    # generate subtask configs
    with open("_granite_risk_template_yaml", "r") as f:
        template = f.read()

    task_names = []
    for risk in RISKS:
        for prob in [True, False]:
            name = f"granite_risk_{risk}" if not prob else f"granite_risk_{risk}_probability"
            with open(f"{'probability' if prob else 'binary'}/{name}.yaml", "w") as f:
                proc_template = template.replace("NAME", name)
                proc_template = proc_template.replace("RISK", risk)
                if prob:

                    proc_template = proc_template.replace("metric.guard_api", "metric.guard_api_probability")
                f.write(proc_template)
            task_names.append(name)

    # generate parent config
    parent_yaml_dict = {
        "group": "granite_risk",
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
    with open('_granite_risk.yaml', 'w') as outfile:
        yaml.dump(parent_yaml_dict, outfile, default_flow_style=False)