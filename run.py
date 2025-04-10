import sys
import os
import pandas as pd
from configs import configs
from functions.experiment import experiment
from params import kwargs


def create_results_df(params):
    return pd.DataFrame(
        columns=[
            "rep_id",
            "sample_id",
            *params,
            "Prey",
            "Predator",
            "Apex",
            "Super",
            "step",
        ]
    )


def run_experiment(cfg):
    print(f"\n{cfg['name']}: {cfg.get('description', '')}")

    for i, model_cfg in enumerate(cfg["models_config"]):
        kwargs.update(model_cfg["params"])
        results = create_results_df(kwargs["params"])
        reps = kwargs.get("reps", 10)
        exp = experiment(**kwargs)

        run = exp.parallel(v=cfg.get("vars", None), rep=reps, **kwargs)
        results = pd.concat([results, run])

        prefix = cfg.get(
            "model_prefix", f"model-{i + 1}" if len(cfg["models_config"]) > 1 else ""
        )
        suffix = f"_{prefix}_results.csv" if prefix else "_results.csv"
        path = f"output/experiments/results/{cfg['name']}{suffix}"

        if cfg.get("append", False) and os.path.exists(path):
            saved = pd.read_csv(path, index_col=0)
            results = pd.concat([saved, results])

        results.to_csv(path)

        # free memory
        del results
        del exp
        del run


def main():
    os.makedirs("output/experiments/results", exist_ok=True)

    args = sys.argv[1:]
    if not args:
        print("No experiment specified. Running all.")
        for cfg in configs:
            run_experiment(cfg)
    else:
        experiment_name = args[0]
        matches = [cfg for cfg in configs if cfg["name"] == experiment_name]
        if not matches:
            print(
                f"No config found with name '{experiment_name}'. Available names are:"
            )
            for cfg in configs:
                print(f"  - {cfg['name']}")
        else:
            run_experiment(matches[0])


if __name__ == "__main__":
    main()
