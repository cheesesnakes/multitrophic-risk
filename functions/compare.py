import polars as pl
from configs import configs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions.summary_plots import set_style
import os

# Helper functions to load data


def load_phase_data(scenario):
    """
    Load the phase data for a given scenario.
    """
    return pl.read_csv(f"output/experiments/outcomes/Scenario-{scenario}_phase.csv")


def load_period_data(scenario):
    """
    Load the period data for a given scenario.
    """
    return pl.read_csv(
        f"output/experiments/outcomes/Scenario-{scenario}_periodicity.csv"
    )


def extract_parameters(df):
    """
    Extract parameters from the DataFrame.
    """
    return df.select(
        [
            pl.col("sample_id"),
            pl.col("s_breed"),
            pl.col("f_breed"),
        ]
    ).unique()


# compare probability of each state


def effect_states(scenario):
    """
    Compare the marginal probabilities of each state across scenarios over all parameters
    """

    cfg = configs[f"Scenario-{scenario}"]

    model = cfg["models"][0]

    # Load phase data for both scenarios
    phase1 = load_phase_data(0)
    phase2 = load_phase_data(scenario)

    # Concatenate the phase data for both scenarios
    phase_data = pl.concat([phase1, phase2], how="vertical")

    del phase1, phase2

    # Pivot probabilities of each state on model

    phase_data = phase_data.pivot(
        index=["sample_id", "phase"],
        columns="model",
        values="prob",
    )

    # Calculate effect of model on state probabilities

    phase_data = phase_data.with_columns(
        (pl.col("Baseline") - pl.col(model)).alias("effect")
    )

    phase_data = phase_data.select(
        [
            pl.col("sample_id"),
            pl.col("phase"),
            pl.col("effect"),
        ]
    ).rename({"effect": model})

    return phase_data


# compare coexistence cycles


def effect_periods(scenario):
    """
    Compare the period of cycles when predators and prey coexist across scenarios over all parameters
    """

    cfg = configs[f"Scenario-{scenario}"]

    model = cfg["models"][0]

    # Load periodicity data for both scenarios
    period1 = load_period_data(0)
    period2 = load_period_data(scenario)

    # Set types
    period1 = period1.with_columns(
        pl.col("period_Prey").cast(pl.Float64),
        pl.col("period_Predator").cast(pl.Float64),
        pl.col("rep_id").cast(pl.Int64),
        pl.col("sample_id").cast(pl.Int64),
    )

    period2 = period2.with_columns(
        pl.col("period_Prey").cast(pl.Float64),
        pl.col("period_Predator").cast(pl.Float64),
        pl.col("rep_id").cast(pl.Int64),
        pl.col("sample_id").cast(pl.Int64),
    )

    # Determine samples where prey and predator coexist using state data
    coexist_samples = (
        pl.read_csv("output/experiments/outcomes/Scenario-0_phase.csv")
        .filter(pl.col("phase") == "Coexistence")
        .select(["sample_id"])
    )

    # Filter periodicity data to only include coexistence samples
    period1 = period1.join(coexist_samples, on=["sample_id"], how="inner")
    period2 = period2.join(coexist_samples, on=["sample_id"], how="inner")

    # Select relevant columns and rename for clarity

    period1 = period1.with_columns(
        (pl.Series(np.repeat("Baseline", len(period1)))).alias("model")
    )

    period2 = period2.select(
        [
            pl.col("rep_id"),
            pl.col("sample_id"),
            pl.col("period_Prey"),
            pl.col("period_Predator"),
        ]
    ).with_columns((pl.Series(np.repeat(model, len(period2)))).alias("model"))

    # Concatenate the periodicity data for both scenarios
    period_data = pl.concat([period1, period2], how="vertical")

    del period1, period2

    # Unpivot period by agent

    period_data = period_data.unpivot(
        on=["period_Prey", "period_Predator"],
        variable_name="agent",
        value_name="period",
        index=["rep_id", "sample_id", "model"],
    )

    # Remove period_ prefix from agent names

    period_data = period_data.with_columns(pl.col("agent").str.replace("period_", ""))

    # Pivot period on model

    period_data = period_data.pivot(
        index=["rep_id", "sample_id", "agent"],
        on="model",
        values="period",
    )

    # Calculate effect of model on period

    period_data = period_data.with_columns(
        (pl.col("Baseline") - pl.col(model)).alias("effect")
    )

    # Select relevant columns

    period_data = period_data.select(
        [
            pl.col("rep_id"),
            pl.col("sample_id"),
            pl.col("agent"),
            pl.col("effect"),
        ]
    ).rename({"effect": model})

    return period_data


# Plot state effect


def plot_state_effect(state_comparisons):
    """
    Plot the effect of different scenarios on state probabilities.
    """
    set_style()

    # Unpivot the DataFrame for easier plotting
    melted = (
        state_comparisons.to_pandas()
        .melt(
            id_vars=["sample_id", "phase"],
            var_name="model",
            value_name="effect",
        )
        .sort_values(by=["sample_id", "phase", "model"])
        .reset_index(drop=True)
    )

    # Set col_wrap
    if len(melted["model"].unique()) > 3:
        col_wrap = 3
        plt.figure(figsize=(8, 8))
    else:
        col_wrap = len(melted["model"].unique())

        plt.figure(figsize=(8, 6))
    plot = sns.FacetGrid(
        data=melted,
        col="model",
        height=6,
        aspect=2,
        col_wrap=col_wrap,
    )
    plot.map_dataframe(
        sns.boxplot,
        x="phase",
        y="effect",
        legend=False,
        width=0.2,
        palette="Set2",
        order=["Prey Only", "Coexistence", "Extinction"],
        linewidth=3,
        fliersize=0,
    )

    # add line at 0
    plot.map(plt.axhline, y=0, color="black", linestyle="--", linewidth=2)

    plot.set_titles("{col_name}")
    plot.set_axis_labels("Phase", r"$\Delta$ " + "Probability")
    plt.tight_layout()

    return 0


# Plot periodicity effect
def plot_period_effect(period_comparisons):
    """
    Plot the effect of different scenarios on periodicity.
    """
    set_style()

    # Unpivot the DataFrame for easier plotting
    melted = (
        period_comparisons.to_pandas()
        .melt(
            id_vars=["rep_id", "sample_id", "agent"],
            var_name="model",
            value_name="effect",
        )
        .sort_values(by=["rep_id", "sample_id", "agent", "model"])
        .reset_index(drop=True)
    )

    # Set col_wrap
    if len(melted["model"].unique()) > 3:
        col_wrap = 3

        plt.figure(figsize=(8, 8))
    else:
        col_wrap = len(melted["model"].unique())
        plt.figure(figsize=(8, 6))

    plot = sns.FacetGrid(
        data=melted,
        col="model",
        height=6,
        aspect=2,
        col_wrap=col_wrap,
    )
    plot.map_dataframe(
        sns.boxplot,
        x="agent",
        y="effect",
        legend=False,
        width=0.3,
        palette="Set2",
        order=["Prey", "Predator"],
        linewidth=3,
        fliersize=0,
    )

    # add line at 0
    plot.map(plt.axhline, y=0, color="black", linestyle="--", linewidth=2)

    plot.set_titles("{col_name}")
    plot.set_axis_labels("Agent", r"$\Delta$ " + "Period")
    plt.tight_layout()

    return 0


# Summarise effects


def summarise_effects(effect_df):
    """
    Summarise the effects of different scenarios on state probabilities or periodicity.
    """
    # Melt

    id_vars = (
        ["sample_id", "phase"]
        if "phase" in effect_df.columns
        else ["rep_id", "sample_id", "agent"]
    )

    effect_df = effect_df.unpivot(
        index=id_vars,
        variable_name="model",
        value_name="effect",
    )

    var = "phase" if "phase" in effect_df.columns else "agent"

    # prob > 0

    def P(x):
        x = x.to_numpy() if isinstance(x, pl.Series) else x
        x = x > 0
        return x.mean()

    # Calculate mean and standard deviation

    summary = effect_df.group_by(["model", var]).agg(
        pl.mean("effect").alias("mean_effect"),
        pl.quantile("effect", 0.25).alias("q25_effect"),
        pl.quantile("effect", 0.75).alias("q75_effect"),
        pl.quantile("effect", 0.05).alias("q05_effect"),
        pl.quantile("effect", 0.95).alias("q95_effect"),
        (P(pl.col("effect"))).alias("prob"),
    )

    return summary


# main comparison function


def compare_scenarios(comparison="Apex - Super"):
    """
    Compare two scenarios based on phase and periodicity data.
    """
    os.makedirs("output/experiments/comparison", exist_ok=True)

    # Set scenarios

    if comparison == "Apex - Super":
        scenarios = [1, 6]
    else:
        scenarios = range(2, 8)

    # Compare states
    compare_states = [effect_states(scenario) for scenario in scenarios]

    # Merge all states on sample_id and phase
    state_comparison = compare_states[0]
    for df in compare_states[1:]:
        state_comparison = state_comparison.join(
            df, on=["sample_id", "phase"], how="inner"
        )

    del compare_states

    # Plot the state effect
    plot_state_effect(state_comparison)
    plt.savefig(f"output/experiments/comparison/state_{comparison}.png")
    plt.close()

    # Summarise the state effects
    state_summary = summarise_effects(state_comparison)
    state_summary.write_csv(
        f"output/experiments/comparison/state_summary_{comparison}.csv"
    )

    # Compare periods
    compare_periods = [effect_periods(scenario) for scenario in scenarios]

    # Merge all periods on rep_id, sample_id, and agent
    timeseries_comparison = compare_periods[0]
    for df in compare_periods[1:]:
        timeseries_comparison = timeseries_comparison.join(
            df, on=["rep_id", "sample_id", "agent"], how="inner"
        )
    del compare_periods

    # Summarise the periodicity effects
    period_summary = summarise_effects(timeseries_comparison)
    period_summary.write_csv(
        f"output/experiments/comparison/period_summary_{comparison}.csv"
    )

    # Plot the periodicity effect
    plot_period_effect(timeseries_comparison)
    plt.savefig(f"output/experiments/comparison/period_{comparison}.png")
    plt.close()

    return 0


if __name__ == "__main__":
    # Example usage
    compare_scenarios()
