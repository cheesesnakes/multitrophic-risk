import os
import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# calculate periodicity


def calculate_periodicity(data):
    """
    Calculate the periodicity of specified columns in a time series dataset.
    Periodicity is defined as the average distance between zero crossings.
    """

    def periods(pop):
        """
        Identify zero crossings in a population time series and calculate the average period.
        """
        pop = pop.to_numpy()  # Convert to numpy array for processing
        # Find zero crossings
        crossing = (pop[:-1] * pop[1:] < 0).nonzero()[0] + 1
        # Calculate periods between crossings
        if len(crossing) > 1:
            periods = crossing[1:] - crossing[:-1]
            return float(periods.mean())  # Return the mean period
        else:
            return 0.0  # Return 0 if there are no periods to calculate

    # List of columns to calculate periodicity for
    populations = ["Predator", "Prey", "Apex", "Super"]

    # Calculate periods for each group and column
    avg_period = data.group_by(["model", "rep_id", "sample_id"]).agg(
        pl.col(pop)
        .map_batches(periods, return_dtype=pl.Float64)
        .first()
        .alias(f"{pop}_Periodicity")
        for pop in populations
    )

    return avg_period


# Calculate amplitude


def calculate_amplitude(data):
    """
    Calculate the amplitude of a time series column.
    """

    def peaks_troughs(pop):
        """
        Identify peaks and troughs in a population time series.
        Peaks are local maxima, and troughs are local minima.
        """
        pop = pop.to_numpy()
        # identify successive peaks
        peaks = (pop[1:-1] > pop[:-2]) & (pop[1:-1] > pop[2:])
        peak_indices = peaks.nonzero()[0] + 1

        # identify successive troughs
        troughs = (pop[1:-1] < pop[:-2]) & (pop[1:-1] < pop[2:])
        trough_indices = troughs.nonzero()[0] + 1

        return peak_indices, trough_indices

    def amplitude(pop):
        """
        Calculate the amplitude of a population time series.
        Amplitude is defined as the average distance between peaks and troughs.
        """
        peak_indices, trough_indices = peaks_troughs(pop)

        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return 0.0

        # Calculate amplitudes
        amplitudes = []
        for peak in peak_indices:
            # Find the closest trough before and after the peak
            before_trough = (
                trough_indices[trough_indices < peak].max()
                if (trough_indices < peak).any()
                else None
            )
            after_trough = (
                trough_indices[trough_indices > peak].min()
                if (trough_indices > peak).any()
                else None
            )

            if before_trough is not None and after_trough is not None:
                peak = int(peak)
                before_trough = int(before_trough)
                after_trough = int(after_trough)

                amplitude = (
                    pop[peak] - pop[before_trough] + pop[peak] - pop[after_trough]
                ) / 2
                amplitudes.append(amplitude)

        return pl.Series(amplitudes).mean()

    # Calculate amplitude for each group and column
    populations = ["Predator", "Prey", "Apex"]

    avg_amplitude = data.group_by(["model", "rep_id", "sample_id"]).agg(
        pl.col(pop)
        .map_batches(amplitude, return_dtype=pl.Float64)
        .first()
        .alias(f"{pop}_Amplitude")
        for pop in populations
    )

    return avg_amplitude


# Calculate correlation


def calculate_phase_shift(data):
    """
    Calculate the correlation between prey and predator populations.
    """

    def lag(struct):
        """
        Calculate the lag between prey and predator populations.
        Returns the lag with the maximum correlation.
        """

        prey = struct.struct.field("Prey").to_numpy()
        predator = struct.struct.field("Predator").to_numpy()

        # Use numpy's correlate function to find the correlation
        correlation = np.correlate(prey, predator, mode="full")

        # Find the lag with the maximum correlation
        max_lag = np.argmax(correlation) - (len(prey) - 1)

        return 1 - abs(max_lag / 600)

    phase_shift = (
        data.group_by(["model", "rep_id", "sample_id"])
        .agg(
            pl.struct(pl.col("Prey"), pl.col("Predator"))
            .map_batches(
                lambda x: lag(x),
                return_dtype=pl.Float64,
            )
            .alias("Phase_Shift")
        )
        .explode("Phase_Shift")
    )

    return phase_shift


# make outcome dataframe


def make_outcomes(data, variables=None, Experiment="Experiment-1"):
    # filter out first 400 steps
    data = data.filter(pl.col("step") > 400)

    # Normalize the Prey and Predator columns

    data = data.with_columns(
        Prey=(pl.col("Prey") - pl.col("Prey").mean()) / pl.col("Prey").std(),
        Predator=(pl.col("Predator") - pl.col("Predator").mean())
        / pl.col("Predator").std(),
        Apex=(pl.col("Apex") - pl.col("Apex").mean()) / pl.col("Apex").std(),
        Super=(pl.col("Super") - pl.col("Super").mean()) / pl.col("Super").std(),
    )

    # Group by rep_id and sample_id

    vars = data.group_by(["model", "sample_id"]).agg(
        pl.col(var).unique().first() for var in variables
    )

    # Calculate periodicity

    avg_period = calculate_periodicity(data)

    # caluclate amplitude

    avg_amplitude = calculate_amplitude(data)

    # Calculate phase shift

    phase_shift = calculate_phase_shift(data)

    # Join outcomes

    outcomes = avg_period.join(
        avg_amplitude.select(
            [
                "model",
                "rep_id",
                "sample_id",
                "Predator_Amplitude",
                "Prey_Amplitude",
                "Apex_Amplitude",
            ]
        ),
        on=["model", "rep_id", "sample_id"],
        how="inner",
    ).join(
        phase_shift.select(["model", "rep_id", "sample_id", "Phase_Shift"]),
        on=["model", "rep_id", "sample_id"],
        how="inner",
    )

    # Join variables

    outcomes = outcomes.join(vars, on="sample_id", how="inner")

    # save outcomes to csv
    if not os.path.exists("output/experiments/outcomes"):
        os.makedirs("output/experiments/outcomes")

    outcomes.write_csv(f"output/experiments/outcomes/{Experiment}_outcomes.csv")

    return outcomes


# plot outcomes


def plot_signals(data, vars, Experiment="Experiment-1"):
    """
    Plot the phase shift of predator and prey populations against s_lethality.
    """

    def marginal_plot(data, outcome):
        pop = ["Predator", "Prey"]

        for var in vars:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 6))
            if outcome != "Phase_Shift":
                var_data = (
                    data.unpivot(
                        index=["model", "rep_id", "sample_id", var],
                        on=[f"{p}_{outcome}" for p in pop],
                        variable_name="Population",
                        value_name=outcome,
                    )
                    .group_by(["model", "rep_id", "Population", var])
                    .agg(pl.col(outcome).mean().alias(outcome))
                )

                var_data = var_data.with_columns(
                    pl.col("Population")
                    .str.replace(f"_{outcome}", "", literal=True)
                    .alias("Population")
                )

                p = sns.relplot(
                    data=var_data,
                    x=var,
                    y=f"{outcome}",
                    kind="line",
                    col="model",
                    hue="Population",
                )

            else:
                var_data = data.group_by(["model", "rep_id", var]).agg(
                    pl.col(f"{outcome}").mean().alias(outcome)
                )

                p = sns.relplot(
                    data=var_data, x=var, y=outcome, kind="line", col="model"
                )

            p.set_titles(col_template="Model: {col_name}")
            label = r"$b_{predator}$" if var == "s_breed" else r"$b_{prey}$"
            p.set_axis_labels(label, f"{outcome}")
            plt.savefig(f"output/experiments/plots/{Experiment}_{outcome}_{var}.png")
            plt.close()

    def joint_plot(data, outcome):
        if vars is None or len(vars) == 0 or len(vars) > 2:
            raise ValueError("vars must be a list of one or two variables.")
        pop = ["Predator", "Prey"]

        sns.set_theme(style="whitegrid")
        if outcome != "Phase_Shift":
            data = data.group_by(["model"] + vars).agg(
                pl.col(f"{p}_{outcome}").mean().alias(f"{p}_{outcome}") for p in pop
            )
            data = data.unpivot(
                index=["model"] + vars,
                on=[f"{p}_{outcome}" for p in pop],
                variable_name="Population",
                value_name=outcome,
            )
            data = data.with_columns(
                pl.col("Population")
                .str.replace(f"_{outcome}", "", literal=True)
                .alias("Population")
            )

            plt.figure(figsize=(12, 12))
        else:
            data = data.group_by(["model"] + vars).agg(
                pl.col(f"{outcome}").mean().alias(outcome)
            )

            plt.figure(figsize=(12, 6))

        scale = {
            "Periodicity": (-1, 50),
            "Amplitude": (0, 0.15),
            "Phase_Shift": (0, 1),
        }

        p = sns.relplot(
            data=data,
            x=vars[0],
            y=vars[1],
            row="Population" if outcome != "Phase_Shift" else None,
            col="model",
            kind="scatter",
            hue=outcome,
            size=outcome,
            sizes=(5, 50),
            size_norm=scale[outcome],
            edgecolor=".7",
            hue_norm=scale[outcome],
            palette="vlag",
        )
        p.set_titles(
            col_template="Model: {col_name}", row_template="Population: {row_name}"
        )

        labels = [
            r"$b_{predator}$" if var == "s_breed" else r"$b_{prey}$" for var in vars
        ]
        p.set_axis_labels(labels[0], labels[1])
        plt.savefig(f"output/experiments/plots/{Experiment}_{outcome}_joint.png")
        plt.close()

    outcomes = ["Periodicity", "Amplitude", "Phase_Shift"]

    for outcome in outcomes:
        if vars is None or len(vars) == 0 or len(vars) > 2:
            raise ValueError("vars must be a list of one or two variables.")
        marginal_plot(data, outcome)
        if len(vars) == 2:
            joint_plot(data, outcome)


if __name__ == "__main__":
    print(sys.path)

    from configs import configs

    Experiment = "Experiment-7"
    variables = configs[Experiment]["variables"]
    filename = configs[Experiment]["data_path"]
    data = pl.read_csv(filename)
    outcomes = make_outcomes(data=data, variables=variables, Experiment=Experiment)

    # Print the first few rows of the outcomes DataFrame
    print(outcomes.head())

    # Plot the outcomes
    plot_signals(outcomes, variables, Experiment=Experiment)
