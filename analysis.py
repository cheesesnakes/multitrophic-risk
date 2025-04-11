# import libraries

from functions.funcs import analysis, analyse_experiment_9
from configs import configs

# constants

reps = 25
steps = 1000
parameter_depth = 50

# main function


def main():
    """
    Main function to run the analysis for all experiments.
    """

    for e in configs.keys():
        # get the experiment config
        config = configs[e]

        # skip if experiment is not complete

        if config["status"] != "complete":
            print(f"Experiment {e} is not complete, skipping...")
            continue

        # skip experiment 9
        if e == "Experiment-9":
            print(f"Skipping experiment {e}...")
            continue

        print(f"{e}: {config['description']}")
        print("==================================")

        analysis(
            experiment=f"{e}",
            data_path=config["data_path"],
            multiple=not config.get("append", True),
            reps=config.get("reps", reps),
            steps=config.get("steps", steps),
            parameter_depth=config.get("parameter_depth", parameter_depth),
            n_models=config["n_models"],
            populations=config["populations"],
            variables=config["variables"],
            models=config.get("models", []),
            n_params=config.get("n_params", None),
        )

    # run experiment 9 analysis
    analyse_experiment_9()


# run the analysis for all experiments
if __name__ == "__main__":
    main()
