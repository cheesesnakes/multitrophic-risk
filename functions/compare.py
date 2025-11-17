from turtle import heading
import polars as pl
from configs import configs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions.summary_plots import set_style
import os
from configs import pop_meta, scenario_meta

# Helper functions to load data


def load_period_data(scenario):
    """
    Load the period data for a given scenario.
    """
    return pl.read_csv(
        f"output/experiments/outcomes/Scenario-{scenario}_periodicity.csv"
    )


def load_phase_data(scenario):
    """
    Load the phase data for a given scenario.
    """
    phase_data = pl.read_csv(
        f"output/experiments/outcomes/Scenario-{scenario}_phase.csv"
    )

    phases = ["Prey Only", "Coexistence", "Extinction"]

    # Complete cases
    sample_ids = phase_data.select("sample_id").unique()
    models = phase_data.select("model").unique()
    phase_df = pl.DataFrame({"phase": phases})

    df = (
        sample_ids.join(phase_df, how="cross")
        .join(models, how="cross")
        .join(phase_data, on=["sample_id", "phase"], how="left")
        .fill_null(0)
    )

    # Ensure proper ordering
    df = df.sort(["sample_id", "phase"])

    return df


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

    model = "Scenario-" + str(scenario)

    # Load phase data for both scenarios
    phase1 = load_phase_data(0)
    phase2 = load_phase_data(scenario)

    # Concatenate the phase data for both scenarios
    phase_data = pl.concat([phase1, phase2], how="vertical")
    # Columns: ['sample_id', "model", 'phase', 'len']

    del phase1, phase2

    # Define boot strap for state probabilities

    def bootstrap_phase(df, reps=2000):
        """
        Bootstrap the phase probabilities.
        Expects columns: 'sample_id', 'phase', 'len'
        Returns dataframe with bootstrap posterior probabilities for each phase.
        """

        # Sort values for consistent reshaping
        df = df.sort(["sample_id", "phase"])
        phases = np.sort(df["phase"].unique())
        n_samples = df["sample_id"].n_unique()
        model = df["model"].unique()[0]

        # Reshape into array: samples x phases
        arr_len = df.select("len").to_numpy().flatten()

        try:
            alpha = arr_len.reshape((n_samples, len(phases)))
        except Exception as e:
            raise ValueError(f"Reshape failed, check input DataFrame structure: {e}")

        alpha = alpha + 1  # Dirichlet prior

        # Dirichlet draws for each model/sample: reps x samples x models x phases
        # We'll average over samples as usual

        dirichlet_draws = np.array(
            [np.random.dirichlet(alpha[i], reps) for i in range(n_samples)]
        )  # samples, reps, phases

        dirichlet_draws = dirichlet_draws.reshape(n_samples, reps, len(phases))

        # Probability per reps and phase (average over samples)
        prob_mean = dirichlet_draws.mean(axis=0)  # reps, phases

        # Prepare result DataFrame
        result = pl.DataFrame(
            {
                "boot": np.repeat(np.arange(reps), len(phases)),
                "phase": np.tile(phases, reps),
                "prob": prob_mean.flatten(),
                "model": np.repeat(model, len(phases) * reps),
            }
        )

        return result

    # Run bootstrap

    effect = phase_data.group_by("model").map_groups(lambda df: bootstrap_phase(df))

    # pivot the data to have models as columns
    effect = effect.pivot(index=["boot", "phase"], columns=["model"], values="prob")

    # Calculate the effect of the scenario on state probabilities
    effect = effect.with_columns((pl.col(model) - pl.col("Scenario-0")).alias("effect"))

    # Rename effect column to model name
    effect = effect.select("boot", "phase", "effect").rename({"effect": model})

    return effect


# compare coexistence cycles


def effect_periods(scenario):
    """
    Compare the period of cycles when predators and prey coexist across scenarios over all parameters
    """

    model = "Scenario-" + str(scenario)

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
    # columns: # ['rep_id', 'sample_id', 'model', 'agent', 'period']

    # Calculate the mean period for each rep_id across samples

    period_data = period_data.group_by(["rep_id", "model", "agent"]).agg(
        pl.mean("period").alias("period")
    )

    # Bootstrap the periodicity effect
    boots = 2000  # Number of bootstrap samples

    def bootstrap_period(x, boots=boots):
        """
        Bootstrap the periodicity effect.
        """

        # Alpha parameters for Dirichlet distribution
        alpha = np.ones(len(x))

        # Dirichlet draws for each rep nested in models and agents
        dirichlet_draws = np.random.dirichlet(alpha, boots)

        # Get periods

        periods = np.tile(x, boots).reshape(len(x), boots).T  # shape: (boots, len(x))

        # Calculate the posterior distribution of periods

        posterior = dirichlet_draws * periods

        # Calculate mean across reps

        mean_period = posterior.sum(axis=1)  # models, agents, boots

        return mean_period

    # Run bootstrap
    boot_period_data = (
        period_data.group_by(["model", "agent"])
        .agg(
            pl.col("period")
            .map_batches(lambda s: bootstrap_period(s.to_numpy()))
            .alias("mean_period")
        )
        .explode("mean_period")
    )

    # Add boot column

    boot_period_data = boot_period_data.with_columns(
        pl.Series(np.tile(np.arange(boots), 4)).alias("boot")
    ).sort(["boot", "model", "agent"])

    # Pivot data

    boot_period_data = boot_period_data.pivot(
        index=["boot", "agent"],
        columns="model",
        values="mean_period",
    )

    # Calculate the effect of the scenario on periodicity
    boot_period_data = boot_period_data.with_columns(
        (pl.col(model) - pl.col("Baseline")).alias("effect")
    )

    # Rename effect to model name
    boot_period_data = boot_period_data.select("boot", "agent", "effect").rename(
        {"effect": model}
    )

    return boot_period_data


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
            id_vars=["boot", "phase"],
            var_name="model",
            value_name="effect",
        )
        .sort_values(by=["boot", "phase", "model"])
        .reset_index(drop=True)
    )

    # set model names
    
    model_mapping = {name: meta["label"] for name, meta in scenario_meta.items()}

    melted["model"] = melted["model"].replace(model_mapping)

    # plot
    plot = sns.barplot(
        x="model",
        y="effect",
        hue ="phase",
        data=melted,
        width=0.5,
        palette="Set2",
        order=range(1, len(model_mapping)),
        errorbar=None,
        linewidth=1,
        edgecolor="black",
    )

    # add line at 0
    plot.axhline(y=0, color="black", linestyle="--", linewidth=2)

    plt.legend(title="State", fontsize=12, title_fontsize=14)
    plt.xlabel("Scenario")
    plt.ylabel(r"$\Delta$ " + "Probability")
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
            id_vars=["boot", "agent"],
            var_name="model",
            value_name="effect",
        )
        .sort_values(by=["boot", "agent", "model"])
        .reset_index(drop=True)
    )

    # rename models

    models = {name: meta["label"] for name, meta in scenario_meta.items()}
    
    melted["model"] = melted["model"].replace(models)

    # rename agents
    
    agents = {"Prey": "Prey", "Predator": "Mesopredator"}

    melted["agent"] = melted["agent"].replace(agents)
    
    # plot
    plot = sns.boxplot(
        x="model",
        y="effect",
        data=melted,
        hue="agent",
        width=0.5,
        palette="Set2",
        order=range(1, len(models)),
        linewidth=1,
        fliersize=0,
    )

    # add line at 0
    plot.axhline(y=0, color="black", linestyle="--", linewidth=2)
    plt.legend(title="Agent", fontsize=12, title_fontsize=14)
    plt.xlabel("Scenario")
    plt.ylabel(r"$\Delta$ " + "Cycle Length")
    plt.tight_layout()

    return 0


# Summarise effects


def summarise_effects(effect_df):
    """
    Summarise the effects of different scenarios on state probabilities or periodicity.
    """
    # Melt

    id_vars = ["boot", "phase"] if "phase" in effect_df.columns else ["boot", "agent"]

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


def compare_scenarios(comparison="All"):
    """
    Compare two scenarios based on phase and periodicity data.
    """
    os.makedirs("output/experiments/comparison", exist_ok=True)

    # Set scenarios

    scenarios = range(1, 8)
    
    # Compare states
    compare_states = [effect_states(scenario) for scenario in scenarios]

    # Merge all states on sample_id and phase
    state_comparison = compare_states[0]
    for df in compare_states[1:]:
        state_comparison = state_comparison.join(df, on=["boot", "phase"], how="inner")

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
            df, on=["boot", "agent"], how="inner"
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
