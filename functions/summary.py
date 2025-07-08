import numpy as np
import polars as pl
import os
from functions.summary_plots import (
    plot_attractor,
    plot_bifurcation,
    plot_max_prey,
    plot_phase_probability,
    plot_time_series,
)
from matplotlib import pyplot as plt
import seaborn as sns
from functions.summary_funcs import (
    classify_model_phase,
    phase_summary,
    set_model_order,
    summaries,
    load_data,
    verify_data,
)


# Summary for experiment 6


def summary_experiment_6(data):
    if os.path.exists("output/experiments/plots/Experiment-6_max-prey.png"):
        print("Max prey plot already exists, skipping...")
        return
    # L2 as a variable based on model

    data = data.with_columns(
        pl.when(pl.col("model") == "$L^{2}$ = 10")
        .then(10)
        .when(pl.col("model") == "$L^{2}$ = 20")
        .then(20)
        .when(pl.col("model") == "$L^{2}$ = 50")
        .then(50)
        .when(pl.col("model") == "$L^{2}$ = 100")
        .then(100)
        .alias("L2")
    )

    print(data.head())

    plot_max_prey(data)

    plt.savefig("output/experiments/plots/Experiment-6_max-prey.png")

    plt.close()

    return 0


# analysis for experiment 9


def summary_experiment_9(data):
    """
    Function to analyze experiment 9 data.
    """

    return 0


# main analysis function


def summary(
    experiment="Experiment-1",
    data_path="output/experiments/results/Experiment-1_results.csv",
    multiple=False,
    reps=25,
    steps=1000,
    parameter_depth=50,
    n_models=2,
    n_params=None,
    populations=["Prey", "Predator", "Apex"],
    variables=["s_breed", "f_breed"],
    models=["Apex", "Super"],
):
    print("Begin analysis...")
    print("\n")

    # Set plot style
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"font.size": 14, "figure.figsize": (10, 6)})

    # Check folders

    if not os.path.exists("output/experiments/outcomes/"):
        os.makedirs("output/experiments/outcomes/")

    if not os.path.exists("output/experiments/plots/"):
        os.makedirs("output/experiments/plots/")

    # load data
    print("Loading data...")

    data = load_data(
        data_path,
        experiment=experiment,
        multiple=multiple,
        n_models=n_models,
    )

    # verify data

    data = verify_data(
        data,
        n_params=n_params,
        parameter_depth=parameter_depth,
        models=models,
        reps=reps,
        steps=steps,
        n_models=n_models,
    )

    # Set order for models in experiment 2
    if experiment == "Experiment-2":
        data = set_model_order(data)

    # summaries
    print("Generating summaries...")
    summaries(data)

    # plot for experiment 6

    if experiment == "Experiment-6":
        summary_experiment_6(data)
        return

    # summary for experiment 9

    if experiment == "Experiment-9":
        summary_experiment_9(data)
        return

    # classify outcomes
    if not os.path.exists(f"output/experiments/outcomes/{experiment}_phase.csv"):
        print("Classifying outcomes...")
        phase = classify_model_phase(
            data,
            variables=variables,
        )

        phase.write_csv(
            f"output/experiments/outcomes/{experiment}_phase.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # phase summary

    if not os.path.exists(
        f"output/experiments/outcomes/{experiment}_phase_summary.csv"
    ):
        print("Generating phase summary...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")

        summary_phases = phase_summary(
            phase,
            variables=variables,
            model=True,
        )

        summary_phases.write_csv(
            f"output/experiments/outcomes/{experiment}_phase_summary.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # plot phase probability
    if not os.path.exists(
        f"output/experiments/plots/{experiment}_phase_probability.png"
    ):
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")

        print("Plotting phase probability...")
        plot_phase_probability(
            phase,
            variables=variables,
        )
        plt.savefig(f"output/experiments/plots/{experiment}_phase_probability.png")
        plt.close()
        print("Phase probability plot saved.")

    # plot attractor
    if not os.path.exists(f"output/experiments/plots/{experiment}_attractor.png"):
        print("Plotting attractor...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")
        plot_attractor(data, variables=variables)
        plt.savefig(f"output/experiments/plots/{experiment}_attractor.png")
        plt.close()
        print("Attractor plot saved.")

    # bifurcation plot
    if not os.path.exists(
        f"output/experiments/plots/{experiment}_bifurcation_{variables[-1]}_prey.png"
    ):
        for population in populations:
            if population == "Super" or population == "Apex":
                continue
            for variable in variables:
                plot_bifurcation(data, population=population, variable=variable)
                plt.savefig(
                    f"output/experiments/plots/{experiment}_bifurcation_{variable}_{population.lower()}.png"
                )
                plt.close()
                print(f"Saved {variable} bifurcation plot for {population}.")

    # plot timeseries
    if not os.path.exists(f"output/experiments/plots/{experiment}_timeseries.png"):
        print("Plotting timeseries plots...")

        plot_time_series(data=data, populations=populations, variables=variables)

        plt.savefig(f"output/experiments/plots/{experiment}_timeseries.png")
        plt.close()

    print(f"Analysis for {experiment} completed.\n")

    return 0
