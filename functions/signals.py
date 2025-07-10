import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# calculate periodicity


def calculate_periodicity(data, populations):
    """
    Calculate the dominant period of population time series using FFT.
    """
    # Filter out early steps
    data = data.filter(pl.col("step") > 400)

    results = []

    # Group by replicate and sample
    for (rep_id, sample_id), group in data.group_by(
        ["rep_id", "sample_id"], maintain_order=True
    ):
        # Ensure data is ordered by step
        group = group.sort("step")
        row = {"rep_id": rep_id, "sample_id": sample_id}

        for pop in populations:
            if pop == "Super":
                continue

            y = group[pop].to_numpy()
            n = len(y)

            if n <= 1 or np.all(y == y[0]):
                row[f"period_{pop}"] = 0
                continue

            # Remove mean to ignore DC component
            y_detrended = y - np.mean(y)
            fft_vals = np.fft.fft(y_detrended)
            power = np.abs(fft_vals[: n // 2]) ** 2
            freqs = np.fft.fftfreq(n)[: n // 2]

            power[0] = 0  # remove zero frequency
            dom_idx = np.argmax(power)
            freq = freqs[dom_idx]
            period = 1 / freq if freq != 0 else 0

            row[f"period_{pop}"] = period

        results.append(row)

    # Create DataFrame from results
    if not results:
        raise ValueError("No valid data found for periodicity calculation.")
    else:
        return pl.DataFrame(results)


# Summary of periodicity


def summary_periodicity(periodicity, phase):
    """
    Summarize the periodicity of populations.
    """
    # Determine sample_id where predator and prey coexist
    coexist_sample_ids = phase.filter(
        (pl.col("phase") == "Coexistence") & (pl.col("prob") > 0)
    )["sample_id"].unique()

    periodicity = periodicity.filter(pl.col("sample_id").is_in(coexist_sample_ids))

    periodicity = periodicity.unpivot(
        index=["rep_id", "sample_id"],
        variable_name="Population",
        value_name="Period",
    )

    periodicity = periodicity.with_columns(
        pl.col("Population").str.replace("period_", "")
    )

    summary = (
        periodicity.group_by("Population")
        .agg(
            [
                pl.mean("Period").alias("Mean Period"),
                pl.std("Period").alias("Std Dev Period"),
                pl.min("Period").alias("Min Period"),
                pl.max("Period").alias("Max Period"),
            ]
        )
        .sort("Population")
    )

    return summary


# plot period


def plot_periodicity(periodicity, populations):
    """
    Plot the periodicity of populations.
    """

    # remove super predator
    populations = [pop for pop in populations if pop != "Super"]

    # Create a DataFrame for plotting
    plot_data = periodicity.melt(
        id_vars=["rep_id", "sample_id"],
        value_vars=[f"period_{pop}" for pop in populations],
        variable_name="Population",
        value_name="Period",
    )

    # remove period_ prefix from Population names
    plot_data = plot_data.with_columns(pl.col("Population").str.replace("period_", ""))

    # Rename populations

    pops = {
        "Prey": "Prey",
        "Predator": "Mesopredator",
        "Apex": "Apex Predator",
        "Super": "Superpredator",
    }

    plot_data = plot_data.with_columns(pl.col("Population").replace(pops))

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_data, x="Population", y="Period", palette="Set2", width=0.5)
    plt.xlabel("Population")
    plt.ylabel("Period (steps)")
    plt.tight_layout()

    return 0


if __name__ == "__main__":
    print(sys.path)

    from configs import configs

    Experiment = "Scenario-0"  # Change this to the desired experiment name
    variables = configs[Experiment]["variables"]
    filename = configs[Experiment]["data_path"]
    populations = configs[Experiment]["populations"]
    data = pl.read_csv(filename)

    # Calulate periodicity
    periodicity = calculate_periodicity(data, populations)

    # Plot periodicity
    plot_periodicity(periodicity, populations)

    plt.savefig(f"periodicity_{Experiment}.png", dpi=300)
    plt.close()

    print(periodicity.head())
