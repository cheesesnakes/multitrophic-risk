import numpy as np
import polars as pl
from plots import (
    plot_attractor,
    plot_bifurcation,
    plot_phase_probability,
    plot_time_series,
)
from matplotlib import pyplot as plt
import seaborn as sns

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


def classify_model_phase(data, variables=["s_breed", "f_breed"], model=True, reps=25):
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

    if model:
        data = data.group_by(["model", "sample_id", *variables, "phase"]).agg(pl.len())
    else:
        data = data.group_by(["sample_id", *variables, "phase"]).agg(pl.len())

    data = data.with_columns((pl.col("len") / reps).alias("prob"))

    return data


# analysis for experiment 9


def analyse_experiment_9():
    """
    Function to analyze experiment 9 data.
    """

    print("Experiment 9: Effect of starting density on model dynamics")
    print("==================================")

    # create parameter space

    prey = np.array([100, 500, 1000, 2000, 5000])
    predator = np.array([100, 500, 1000, 2000, 5000])
    apex = np.array([0, 100, 500, 1000, 2000])
    super = np.array([0, 100, 500, 1000, 2000])

    reps = 10
    steps = 1000

    vars = np.array(np.meshgrid(prey, predator, apex, super)).reshape(4, -1).T

    runs = reps * vars.shape[0]

    # data

    data = pl.scan_csv("output/experiments/results/Experiment-9_results.csv")

    # update sample id

    sample_ids = np.array(
        [[[i] * reps * steps for i in range(0, vars.shape[0])]]
    ).flatten()

    data = data.drop("sample_id")
    data = data.with_columns(pl.Series(sample_ids).alias("sample_id"))
    data = data.collect()

    if runs * steps == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete")
        print(f"Expected: {runs * 1000}")
        print(f"Actual: {data.shape[0]}")
        return

    # create df of vars

    vars_df = pl.DataFrame(
        {
            "i_prey": vars[:, 0],
            "i_predator": vars[:, 1],
            "i_apex": vars[:, 2],
            "i_super": vars[:, 3],
        }
    ).lazy()

    # add model names as columns

    vars_df = vars_df.with_columns(
        pl.when((pl.col("i_apex") > 0) & (pl.col("i_super") == 0))
        .then(pl.lit("Apex"))
        .otherwise(
            pl.when((pl.col("i_super") > 0) & (pl.col("i_apex") == 0))
            .then(pl.lit("Super"))
            .otherwise(pl.lit("Mixed"))
        )
        .alias("model"),
    )

    # add sample id

    vars_df = vars_df.with_columns(pl.arange(0, vars.shape[0]).alias("sample_id"))

    summaries(data)

    # classify model phase
    print("Classifying model phase...")
    phase = classify_model_phase(data, variables=[], model=False)

    # add vars to phase
    phase = phase.join(
        vars_df.collect(),
        on="sample_id",
        how="left",
    )

    # plot phase probability

    print("Plotting phase probability...")

    # Apex model
    phase_apex = phase.filter(pl.col("model") == "Apex")

    phase_apex = phase_apex.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    plot = sns.relplot(
        data=phase_apex,
        x="i_prey",
        y="i_predator",
        hue="prob",
        col="phase",
        row="i_apex",
        palette="crest",
        edgecolor=".5",
        # hue_norm=plt.Normalize(0, 1),
        height=6,
        aspect=1.3,
        s=500,
    )

    plot.set_axis_labels(r"$N_{{0}}$", r"$P_{{0}}$")
    plot.set_titles(
        col_template="Phase: {col_name}", row_template=r"$A_{{0}}$ = " + "{row_name}"
    )

    plt.savefig("output/experiments/plots/experiment_9_phase_probability_apex.png")

    # Super model
    phase_super = phase.filter(pl.col("model") == "Super")
    phase_super = phase_super.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    plot = sns.relplot(
        data=phase_super,
        x="i_prey",
        y="i_predator",
        hue="prob",
        col="phase",
        row="i_super",
        palette="crest",
        edgecolor=".5",
        # hue_norm=plt.Normalize(0, 1),
        height=6,
        aspect=1.3,
        s=500,
    )

    plot.set_axis_labels(r"$N_{{0}}$", r"$P_{{0}}$")
    plot.set_titles(
        col_template="Phase: {col_name}", row_template=r"$S_{{0}}$ = " + "{row_name}"
    )
    plt.savefig("output/experiments/plots/experiment_9_phase_probability_super.png")

    plt.close()
    print("Analysis for experiment 9 completed.")
    print("\n")


# main analysis function


def analysis(
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

    # load data
    print("Loading data...")

    if multiple:
        data_paths = [
            f"output/experiments/results/{experiment}_model-{i}_results.csv"
            for i in range(1, n_models + 1)
        ]

        data = pl.concat([pl.scan_csv(path) for path in data_paths])
    else:
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
