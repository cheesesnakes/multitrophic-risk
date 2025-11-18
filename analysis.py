# import libraries

from functions.summary import summary
from functions.compare import compare_scenarios
from functions.figures import make_figure
from functions.summary_plots import set_style
from configs import configs, scenario_meta
import sys
import pandas as pd

# constants

reps = 25
steps = 1000
parameter_depth = 50

# main function


def main():
    """
    Main function to run the analysis for all experiments.
    """
    set_style()
    
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
            name=config.get("name", []),
            models=config.get("models", []),
            n_params=config.get("n_params", None),
        )

    args = sys.argv[1:]

    # make a table with labels and descriptions from scenario_meta

    scenario_table = []
    for key, meta in scenario_meta.items():
        scenario_table.append({"Scenario": meta["label"], "Description": meta["description"]})
    
    # save table as csv
    
    df = pd.DataFrame(scenario_table)
    df.to_csv("output/experiments/scenario_descriptions.csv", index=False)
    

    if not args:
        print("Running summary and comparison for all experiments...")

        for e in configs.keys():
            run_summary(e)
            continue

        compare_scenarios("All")

        # Figure 3

        make_figure([0, 1, 6], ["phase_probability"], "figure3a")
        make_figure([0, 1, 6], ["timeseries"], "figure3b")

        # Figure 4
        make_figure(range(2, 8), ["phase_probability"], "figure4a", rows=2, cols=3)
        make_figure(range(2, 8), ["timeseries"], "figure4b", rows=2, cols=3)

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
            print("Comparing scenarios...")
            
            compare_scenarios("All")
        elif args[0] == "Figures":
            # make all figures

            print("Creating all figures...")
            make_figure([0, 1, 6], ["phase_probability"], "figure3a")
            make_figure([0, 1, 6], ["timeseries"], "figure3b")
            make_figure(range(2, 8), ["phase_probability"], "figure4a", rows=2, cols=3)
            make_figure(range(2, 8), ["timeseries"], "figure4b", rows=2, cols=3)

            make_figure([0, 1, 6], ["power_spectrum"], "figureA3_1", rows=2, cols=2)
            make_figure(range(2, 8), ["power_spectrum"], "figureA3_2", rows=3, cols=2)
        else:
            print("Invalid argument. Use 'Summary', 'Compare', or 'Figures'.")


# run the analysis for all experiments
if __name__ == "__main__":
    main()
