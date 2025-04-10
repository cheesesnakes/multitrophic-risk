import sys
import os
from configs import configs
from params import kwargs


def main():
    args = sys.argv[1:]

    if not args:
        print(
            "Please specify a program to run. Avialable options are: Debug, Examples, Strategies, and  Experiments"
        )
        return

    if args[0] == "Debug":
        print("Running model debugging script...")

        from functions.debug import run_debug, plot_debug

        run_debug(kwargs)
        plot_debug(kwargs)

        return

    elif args[0] == "Examples":
        print("Running model examples script...")
        from functions.example import run_example, plot_example

        if len(args) == 1:
            print("No example specified. Running all.")
            models = ["lv", "apex", "super"]
            for model in models:
                run_example(kwargs, model=model)
        else:
            model_name = args[1]
            if model_name not in ["lv", "apex", "super"]:
                print(
                    f"Unknown model '{model_name}'. Available models are: lv, apex, and super"
                )
                return
            run_example(kwargs, model=model_name)
            plot_example(kwargs, model=model_name)

        return
    elif args[0] == "Strategies":
        print("Running model strategies examples script...")

        from functions.strategies import run_strategies, plot_strategy

        run_strategies(kwargs)
        plot_strategy(kwargs)
        return
    elif args[0] == "Experiments":
        print("Running model experiments script...")
        from functions.experiment import run_experiment

        os.makedirs("output/experiments/results", exist_ok=True)
        os.makedirs("output/experiments/plots", exist_ok=True)

        if len(args) == 1:
            print("No experiment specified. Running all.")
            for cfg in configs:
                run_experiment(cfg, kwargs)
        else:
            experiment_name = "Experiment-" + str(args[1])
            if experiment_name not in cfg.keys():
                print(
                    f"No config found with name '{experiment_name}'. Available names are:"
                )
                for cfg in configs.keys():
                    print(f"  - {cfg}")
            else:
                run_experiment(cfg[experiment_name], kwargs)
    else:
        print(
            f"Unknown argument '{args[0]}'. Available options are: Debug, Examples, Strategies, and Experiments"
        )
        return

    print("Script completed successfully!")


if __name__ == "__main__":
    main()
