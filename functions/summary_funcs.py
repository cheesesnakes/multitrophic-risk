import numpy as np
import polars as pl

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


def phase_summary(phase_data, variables=["s_breed", "f_breed"], model=True):
    """
    Function to summarize the transitions between different phases in the model.

    Args:
        phase_data (pl.DataFrame): The data to classify
        variables (list): The variables to classify
        model (bool): Whether to include model in classification
    """
    # Start with count
    agg_exprs = [pl.sum("len").alias("count")]

    # Add min, max, mean, std for each variable
    for var in variables:
        agg_exprs.extend(
            [
                pl.min(var).alias(f"{var}_min"),
                pl.max(var).alias(f"{var}_max"),
            ]
        )

    # Add hard-coded Coexistence outcomes
    agg_exprs.extend(
        [
            pl.mean("prob").alias("prob_mean"),
            pl.std("prob").alias("prob_std"),
        ]
    )

    # Perform groupby and aggregation

    if model:
        summary = phase_data.group_by(["model", "phase"]).agg(*agg_exprs)
    else:
        summary = phase_data.group_by("phase").agg(*agg_exprs)

    return summary


# set model order for experiment 2


def set_model_order(data):
    return data.with_columns(
        pl.col("model")
        .cast(pl.Categorical)
        .cast(
            pl.Enum(
                [
                    "Lethal -> Prey",
                    "Lethal -> Predator",
                    "Lethal -> Both",
                    "Non-lethal -> Prey",
                    "Non-lethal -> Predator",
                    "Non-lethal -> Both",
                ]
            )
        )
    )


# Load data


def load_data(data_path, experiment, multiple=False, n_models=1):
    if multiple:
        data_paths = [
            f"output/experiments/results/{experiment}_model-{i}_results.csv"
            for i in range(1, n_models + 1)
        ]

        data = pl.concat([pl.scan_csv(path) for path in data_paths])
    else:
        data = pl.scan_csv(data_path)

    data = data.collect()

    return data


# verify data


def verify_data(data, n_params, parameter_depth, models, reps, steps, n_models=1):
    print("Calculating number of runs...")

    if n_params is None:
        if vars is None:
            prey = np.array([100, 500, 1000, 2000, 5000])
            predator = np.array([100, 500, 1000, 2000, 5000])
            apex = np.array([0, 100, 500, 1000, 2000])
            super = np.array([0, 100, 500, 1000, 2000])

            param_space = (
                np.array(np.meshgrid(prey, predator, apex, super)).reshape(4, -1).T
            )
        else:
            param_space = create_space(parameter_depth)

        n_params = param_space.shape[0]

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    # add model names as columns
    print("Adding model names...")

    if models is not None:
        model = [m for m in models for _ in range(runs // n_models)]
    else:
        model = [None for _ in range(runs // n_models)]

    # check if data is complete
    print("Checking if data is complete...")
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0] * 100 / runs, "%")
        model = model[: data.shape[0]]

    data = data.with_columns(model=pl.Series(model))

    return data
