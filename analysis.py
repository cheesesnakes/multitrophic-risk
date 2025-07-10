# import libraries

from functions.summary import summary
from functions.compare import compare_scenarios
from configs import configs
import sys

# constants

reps = 25
steps = 1000
parameter_depth = 50

# main function


def main():
    """
    Main function to run the analysis for all experiments.
    """

    def run_summary(e):
        # get the experiment config
        config = configs[e]

        # skip if experiment is not complete

        if config["status"] != "complete":
            print(f"Experiment {e} is not complete, skipping...")
            return

        print(f"{e}: {config['description']}")
        print("==================================")

        summary(
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

    args = sys.argv[1:]

    if not args:
        print("Running summary and comparison for all experiments...")

        for e in configs.keys():
            run_summary(e)
            continue

        comparisons = ["Apex - Super", "Superpredators"]

        for comparison in comparisons:
            print(f"Comparing scenarios for {comparison}...")
            compare_scenarios(comparison)
    else:
        if args[0] == "Summary":
            if not len(args) == 2:
                print("Running all summaries...")
                for e in configs.keys():
                    run_summary(e)
                    continue
                return 0

            e = args[1]

            print(f"Running analysis for {e}...")

            if e not in configs.keys():
                print(f"Experiment {e} not found in configs.")
                return

            run_summary(e)
        elif args[0] == "Compare":
            if not len(args) == 2:
                comparisons = ["Apex - Super", "Superpredators"]
                print("Running comparisons for all scenarios...")
                for comparison in comparisons:
                    print(f"Comparing scenarios for {comparison}...")
                    compare_scenarios(comparison)
                return

            comparison = args[1]
            print(f"Comparing scenarios for {comparison}...")
            compare_scenarios(comparison)


# run the analysis for all experiments
if __name__ == "__main__":
    main()
