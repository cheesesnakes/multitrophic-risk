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
    avg_period = data.group_by(["rep_id", "sample_id"]).agg(
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
    populations = ["Predator", "Prey", "Apex", "Super"]

    avg_amplitude = data.group_by(["rep_id", "sample_id"]).agg(
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
        data.group_by(["rep_id", "sample_id"])
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

    vars = data.group_by(["sample_id"]).agg(
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
                "rep_id",
                "sample_id",
                "Predator_Amplitude",
                "Prey_Amplitude",
                "Apex_Amplitude",
                "Super_Amplitude",
            ]
        ),
        on=["rep_id", "sample_id"],
        how="inner",
    ).join(
        phase_shift.select(["rep_id", "sample_id", "Phase_Shift"]),
        on=["rep_id", "sample_id"],
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

    def plot(outcome):
        if len(vars) == 0:
            print("No variables to plot.")
            return
        elif len(vars) == 1:
            var = vars[0]
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 6))

            if outcome != "Phase_Shift":
                for pop in ["Predator", "Prey", "Apex", "Super"]:
                    sns.lineplot(
                        data=data,
                        x=var,
                        y=f"{pop}_{outcome}",
                        label=pop,
                    )
                plt.legend()
            else:
                sns.lineplot(
                    data=data,
                    x=var,
                    y=outcome,
                )
            plt.xlabel(f"{var}")
            plt.ylabel(f"{outcome}")
        elif len(vars) > 1:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 6))
            for var in vars:
                if outcome != "Phase_Shift":
                    for pop in ["Predator", "Prey", "Apex", "Super"]:
                        sns.lineplot(
                            data=data,
                            x=var,
                            y=f"{pop}_{outcome}",
                            label=pop,
                        )
                else:
                    sns.lineplot(
                        data=data,
                        x=var,
                        y=outcome,
                    )
            plt.xlabel(f"{var}")
            plt.ylabel(f"{outcome}")

    outcomes = ["Periodicity", "Amplitude", "Phase_Shift"]

    for outcome in outcomes:
        plot(outcome)
        plt.savefig(f"output/experiments/plots/{Experiment}_{outcome}.png")
        plt.close()


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
