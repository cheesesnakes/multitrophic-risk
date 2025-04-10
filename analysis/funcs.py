import numpy as np
import polars as pl
from plots import (
    plot_attractor,
    plot_bifurcation,
    plot_phase_probability,
    plot_time_series,
)
from matplotlib import pyplot as plt

# summaries


def summaries(data):
    """
    Function to print summaries of the data.
    Args:
        data (pd.DataFrame): The data to summarize.
    """
    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")


## create parameter space


def create_space(parameter_depth=50):
    s_breed = np.array(np.linspace(0.1, 1, parameter_depth))
    f_breed = np.array(np.linspace(0.1, 1, parameter_depth))

    vars = np.array(np.meshgrid(s_breed, f_breed))

    return vars.reshape(2, -1).T


# classify model phase


def classify_model_phase(data, variables=["s_breed", "f_breed"]):
    """
    Function to classify model outcome based on simulated data.

    Possible outcomes:

        Coexistence: Both predators and prey are present at the end of the simulation.
        Prey Only: Only prey are present at the end of the simulation.
        Extinction: Either only predators or no agents at the end of the simulation.

    Args:
        data (pl.DataFrame): The data to classify.
    """

    data = data.filter(pl.col("step") == 999)

    data = data.with_columns(
        pl.when(pl.col("Prey") > 0)
        .then(
            pl.when(pl.col("Predator") > 0)
            .then(pl.lit("Coexistence"))
            .otherwise(pl.lit("Prey Only"))
        )
        .otherwise(pl.lit("Extinction"))
        .alias("phase"),
    )

    data = data.group_by(["model", "sample_id", *variables, "phase"]).agg(pl.len())

    data = data.with_columns((pl.col("len") / 25).alias("prob"))

    return data


# analysis for experiment 9


def analyse_experiment_9():
    """
    Function to analyze experiment 9 data.
    """

    print("Experiment 9: Effect of starting density on model dynamics")
    print("==================================")

    # data

    data = pl.scan_csv("output/experiments/results/Experiment-9_results.csv")

    data = data.collect()

    # create parameter space

    prey = np.array([100, 500, 1000, 2000, 5000])
    predator = np.array([100, 500, 1000, 2000, 5000])
    apex = np.array([0, 100, 500, 1000, 2000])
    super = np.array([0, 100, 500, 1000, 2000])

    reps = 10
    steps = 1000

    vars = np.array(np.meshgrid(prey, predator, apex, super)).reshape(4, -1).T

    runs = reps * vars.shape[0]

    if runs * steps == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete")
        print(f"Expected: {runs * 1000}")
        print(f"Actual: {data.shape[0]}")
        return

    summaries(data)


# main analysis function


def analysis(
    experiment="Experiment-1",
    data_path="output/experiments/results/Experiment-1_results.csv",
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

    # load data
    print("Loading data...")
    data = pl.scan_csv(data_path)

    # calculate number of runs
    print("Calculating number of runs...")

    if n_params is None:
        param_space = create_space(parameter_depth)
        n_params = param_space.shape[0]

    runs = reps * steps * n_params * n_models
    print("Number of runs = ", runs)

    # add model names as columns
    print("Adding model names...")
    model = [m for m in models for _ in range(runs // n_models)]
    data = data.with_columns(model=pl.Series(model))
    data = data.collect()

    # summaries
    print("Generating summaries...")
    summaries(data)

    # check if data is complete
    print("Checking if data is complete...")
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0] / runs, "%")
        return

    # plot attractor
    print("Plotting attractor...")
    plot_attractor(data, variables=variables, grid_size=3)
    plt.savefig(f"output/experiments/plots/{experiment}_attractor.png")
    print("Attractor plot saved.")

    # classify outcomes
    print("Classifying outcomes...")
    phase = classify_model_phase(
        data,
        variables=variables,
    )

    # plot phase probability
    print("Plotting phase probability...")
    plot_phase_probability(
        phase,
        variables=variables,
    )
    plt.savefig(f"output/experiments/plots/{experiment}_phase_probability.png")
    print("Phase probability plot saved.")

    # bifurcation plot
    print("Plotting bifurcation plots...")

    for population in populations:
        for variable in variables:
            plot_bifurcation(data, population=population, variable=variable)
            plt.savefig(
                f"output/experiments/plots/{experiment}_{variable}_bifurcation_{population.lower()}.png"
            )
            print(f"Saved {variable} bifurcation plot for {population}.")

    # plot timeseries

    for population in populations:
        for model in models:
            if population == "Apex" and model == "Super":
                continue  # Skip plotting apex population for Super model
            print(f"Plotting timeseries for {population.lower()} in {model} model...")
            plot_time_series(
                data=data, population=population, model=model, variables=variables
            )
            plt.savefig(
                f"output/experiments/plots/{experiment}_{population.lower()}_timeseries_{model.lower()}.png"
            )
            print(f"Timeseries plot for {population.lower()} in {model} model saved.")

    print(f"Analysis for {experiment} completed.\n")
