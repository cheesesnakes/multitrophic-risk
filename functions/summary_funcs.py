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
