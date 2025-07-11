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

    data = data.sort(["sample_id", "phase"])

    return data


def phase_summary(phase_data: pl.DataFrame) -> pl.DataFrame:
    """
    Function to summarize the transitions between different phases in the model using Bayesian bootstrap.

    Args:
        phase_data (pl.DataFrame): A DataFrame with columns: 'sample_id', 'phase', 'len' representing
                                   counts of each phase per sample.

    Returns:
        pl.DataFrame: Summarized probabilities with Bayesian credible intervals for each phase.
    """
    phases = ["Prey Only", "Coexistence", "Extinction"]
    n_boot = 1000

    # Complete cases
    sample_ids = phase_data.select("sample_id").unique()
    phase_df = pl.DataFrame({"phase": phases})

    df = (
        sample_ids.join(phase_df, how="cross")
        .join(phase_data, on=["sample_id", "phase"], how="left")
        .fill_null(0)
    )

    # Ensure proper ordering
    df = df.sort(["sample_id", "phase"])

    # Convert 'len' column into a numpy array with shape (n_samples, 3)
    alpha = (
        df.select("len").to_numpy().reshape(-1, 3)  # assumes 3 phases per sample
    ) + 1  # Add 1 for Dirichlet prior

    # Draw Dirichlet weights (shape: n_samples x 3 x n_boot)
    weights = np.stack(
        [np.random.dirichlet(a, n_boot).T for a in alpha]
    )  # shape: (n_samples, 3, n_boot)

    # Average over samples (axis=0), result shape: (3, n_boot)
    P = weights.mean(axis=0).T  # shape: (n_boot, 3)

    # Calculate summary stats per phase
    mean = np.mean(P, axis=0)
    lower = np.percentile(P, 5, axis=0)
    upper = np.percentile(P, 95, axis=0)

    # Build results dataframe
    results = pl.DataFrame(
        {"phase": phases, "mean": mean, "lower": lower, "upper": upper}
    )

    return results


# Find and summarise necessary conditions for each state


def parameter_summary(phase_data: pl.DataFrame, variables: list) -> pl.DataFrame:
    """
    Summarize the necessary conditions (min/max ranges) for each phase and parameter
    using Bayesian bootstrap.
    """
    phases = ["Prey Only", "Coexistence", "Extinction"]

    # Complete cases
    sample_ids = phase_data.select("sample_id").unique()
    phase_df = pl.DataFrame({"phase": phases})

    df = (
        sample_ids.join(phase_df, how="cross")
        .join(phase_data, on=["sample_id", "phase"], how="left")
        .fill_null(0)
    )

    # Values of the variables

    var_values = phase_data.select(["sample_id"] + variables).unique()

    # Ensure proper ordering
    df = df.sort(["sample_id", "phase"])

    def bootstrap_ranges(data_phase, n_boot=1000):
        # Use number times each phase occurs in each sample as Dirichlet prior
        alpha = (
            data_phase.select("len").to_numpy().flatten().astype(float) + 0.1
        ) / 25  # Divide by reps to get average counts

        sample_ids = data_phase.select("sample_id").to_numpy().flatten()

        # Draw Dirichlet weights
        weights = np.random.dirichlet(alpha, n_boot)

        # sampling
        sampled_ids = np.array(
            [np.random.choice(sample_ids, p=weights[i]) for i in range(n_boot)]
        )

        # Calculate min/max for each variable
        boots = []
        for var in variables:
            values = var_values.select("sample_id", var)
            for boot in range(n_boot):
                sampled_data = (
                    values.filter(pl.col("sample_id").is_in(sampled_ids[boot]))
                    .select(var)
                    .to_numpy()
                    .flatten()
                )
                if len(sampled_data) > 0:
                    boots.append(
                        {
                            "boot": boot,
                            "phase": data_phase.select("phase").to_numpy()[0][0],
                            "variable": var,
                            "value": sampled_data,
                        }
                    )
        boots_df = pl.DataFrame(boots)

        # Calculate mean, lower, and upper bounds
        summary = boots_df.group_by(["phase", "variable"]).agg(
            pl.mean("value").alias("mean"),
            pl.quantile("value", 0.05).alias("lower"),
            pl.quantile("value", 0.95).alias("upper"),
        )

        return summary

    # Group by phase and apply bootstrap ranges
    summaries = []
    for phase in phases:
        data_phase = df.filter(pl.col("phase") == phase)
        if not data_phase.is_empty():
            summary = bootstrap_ranges(data_phase)
            summaries.append(summary)

    if summaries:
        return pl.concat(summaries).sort(["phase", "variable"])
    else:
        raise ValueError("No data available for summarization.")


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


def verify_data(
    data, n_params, parameter_depth, models, reps, steps, n_models=1, variables=None
):
    print("Calculating number of runs...")

    if n_params is None:
        if len(variables) == 0:
            # Experiment 9

            initial_densities = {
                "Prey": [100, 500, 1000, 2000, 5000],
                "Predator": [100, 500, 1000, 2000, 5000],
                "Apex": [0, 100, 500, 1000, 2000],
                "Super": [0, 100, 500, 1000, 2000],
            }

            # Combinations

            param_space = (
                np.array(
                    np.meshgrid(
                        initial_densities["Prey"],
                        initial_densities["Predator"],
                        initial_densities["Apex"],
                        initial_densities["Super"],
                    )
                )
                .reshape(4, -1)
                .T
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
        model = [None for _ in range(runs)]

    # check if data is complete
    print("Checking if data is complete...")
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0] * 100 / runs, "%")
        model = model[: data.shape[0]]

    data = data.with_columns(model=pl.Series(model))

    return data
