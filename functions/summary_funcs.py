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


def identify_transition(phase_data, variables=["s_breed", "f_breed"], model=True):
    """
    Function to identify transitions between different phases in the model.
    Args:
        phase_data (pl.DataFrame): The data to classify
        variables (list): The variables to classify
        model (bool): Whether to include model in classification
    """

    states = ["Prey Only", "Coexistence", "Extinction"]

    if model:
        phase_data = phase_data.pivot(
            index=["model", "sample_id", *variables],
            columns="phase",
            values="prob",
        )
    else:
        phase_data = phase_data.pivot(
            index=["sample_id", *variables],
            columns="phase",
            values="prob",
        )

    # complete states

    for state in states:
        if state not in phase_data.columns:
            phase_data = phase_data.with_columns(pl.lit(0).alias(state))

    phase_data = phase_data.fill_null(0)

    phase_data = phase_data.with_columns(
        pl.when((pl.col("Prey Only") > 0) & (pl.col("Coexistence") > 0))
        .then(pl.lit("P - C Bistable"))
        .when((pl.col("Prey Only") > 0) & (pl.col("Extinction") > 0))
        .then(pl.lit("P - E Bistable"))
        .when((pl.col("Coexistence") > 0) & (pl.col("Extinction") > 0))
        .then(pl.lit("C - E Bistable"))
        .otherwise(
            pl.when(pl.col("Prey Only") > 0)
            .then(pl.lit("Prey Only"))
            .when(pl.col("Coexistence") > 0)
            .then(pl.lit("Coexistence"))
            .otherwise(pl.lit("Extinction"))
        )
        .alias("phase"),
    )

    return phase_data


def transition_summary(phase_data, variables=["s_breed", "f_breed"], model=True):
    """
    Function to summarize the transitions between different phases in the model.

    Args:
        phase_data (pl.DataFrame): The data to classify
        variables (list): The variables to classify
        model (bool): Whether to include model in classification
    """
    # Start with count
    agg_exprs = [pl.count().alias("count")]

    # Add min, max, mean, std for each variable
    for var in variables:
        agg_exprs.extend(
            [
                pl.min(var).alias(f"{var}_min"),
                pl.max(var).alias(f"{var}_max"),
                pl.mean(var).alias(f"{var}_mean"),
                pl.std(var).alias(f"{var}_std"),
            ]
        )

    # Add hard-coded Coexistence outcomes
    outcome_vars = ["Coexistence", "Prey Only", "Extinction"]
    for var in outcome_vars:
        agg_exprs.extend(
            [
                pl.mean(var).alias(f"prob_{var.lower().replace(' ', '_')}_mean"),
                pl.std(var).alias(f"prob_{var.lower().replace(' ', '_')}_std"),
            ]
        )

    # Perform groupby and aggregation

    if model:
        summary = phase_data.group_by(["model", "phase"]).agg(*agg_exprs)
    else:
        summary = phase_data.group_by("phase").agg(*agg_exprs)

    return summary
