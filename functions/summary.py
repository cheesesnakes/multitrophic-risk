import numpy as np
import polars as pl
import os
from functions.summary_plots import (
    plot_attractor,
    plot_bifurcation,
    plot_max_prey,
    plot_phase_probability,
    plot_time_series,
)
from matplotlib import pyplot as plt
import seaborn as sns
from functions.summary_funcs import (
    classify_model_phase,
    phase_summary,
    summaries,
    create_space,
)

# analysis for experiment 9


def summary_experiment_9():
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
    phase = classify_model_phase(data, variables=[], model=False, reps=10)

    # add vars to phase
    phase = phase.join(
        vars_df.collect(),
        on="sample_id",
        how="left",
    )

    # plot phase probability

    print("Plotting phase probability...")

    def plot_9(data, cat="i_apex"):
        plot = sns.relplot(
            data=data,
            x="i_prey",
            y="i_predator",
            hue="prob",
            col="phase",
            row=cat,
            edgecolor=".5",
            size="prob",
            sizes=(100, 500),
            size_norm=(0, 1),
            hue_norm=(0, 1),
            palette=sns.color_palette("viridis", as_cmap=True),
            height=6,
            aspect=1.3,
            s=500,
        )

        plot.set_axis_labels(r"$N_{{0}}$", r"$P_{{0}}$")

        return plot

    # Apex model
    phase_apex = phase.filter(pl.col("model") == "Apex")

    phase_apex = phase_apex.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    plot = plot_9(phase_apex)

    plot.set_titles(
        col_template="Phase: {col_name}",
        row_template=r"$A_{{0}}$ = " + "{row_name}",
    )

    plt.savefig("output/experiments/plots/Experiment-9_phase_probability_apex.png")

    # Super model
    phase_super = phase.filter(pl.col("model") == "Super")
    phase_super = phase_super.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    plot = plot_9(phase_super, cat="i_super")
    plot.set_titles(
        col_template="Phase: {col_name}", row_template=r"$S_{{0}}$ = " + "{row_name}"
    )
    plt.savefig("output/experiments/plots/Experiment-9_phase_probability_super.png")

    plt.close()
    print("Analysis for experiment 9 completed.")
    print("\n")


# main analysis function


def summary(
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

    # Set plot style
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"font.size": 14, "figure.figsize": (10, 6)})

    # Check folders

    if not os.path.exists("output/experiments/outcomes/"):
        os.makedirs("output/experiments/outcomes/")

    if not os.path.exists("output/experiments/plots/"):
        os.makedirs("output/experiments/plots/")

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

    data = data.collect()

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

    # check if data is complete
    print("Checking if data is complete...")
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0] * 100 / runs, "%")
        model = model[: data.shape[0]]

    data = data.with_columns(model=pl.Series(model))

    # Set order for models in experiment 2
    if experiment == "Experiment-2":
        data = data.with_columns(
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

    # summaries
    print("Generating summaries...")
    summaries(data)

    # plot for experiment 6

    if experiment == "Experiment-6":
        if os.path.exists("output/experiments/plots/Experiment-6_max-prey.png"):
            print("Max prey plot already exists, skipping...")
            return
        # L2 as a variable based on model

        data = data.with_columns(
            pl.when(pl.col("model") == "$L^{2}$ = 10")
            .then(10)
            .when(pl.col("model") == "$L^{2}$ = 20")
            .then(20)
            .when(pl.col("model") == "$L^{2}$ = 50")
            .then(50)
            .when(pl.col("model") == "$L^{2}$ = 100")
            .then(100)
            .alias("L2")
        )

        print(data.head())

        plot_max_prey(data)

        plt.savefig("output/experiments/plots/Experiment-6_max-prey.png")

        plt.close()
        return

    # classify outcomes
    if not os.path.exists(f"output/experiments/outcomes/{experiment}_phase.csv"):
        print("Classifying outcomes...")
        phase = classify_model_phase(
            data,
            variables=variables,
        )

        phase.write_csv(
            f"output/experiments/outcomes/{experiment}_phase.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # phase summary

    if not os.path.exists(f"output/experiments/outcome/{experiment}_phase_summary.csv"):
        print("Generating phase summary...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")

        summary_phases = phase_summary(
            phase,
            variables=variables,
            model=True,
        )

        summary_phases.write_csv(
            f"output/experiments/outcomes/{experiment}_phase_summary.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # plot phase probability
    if not os.path.exists(
        f"output/experiments/plots/{experiment}_phase_probability.png"
    ):
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")

        print("Plotting phase probability...")
        plot_phase_probability(
            phase,
            variables=variables,
        )
        plt.savefig(f"output/experiments/plots/{experiment}_phase_probability.png")
        plt.close()
        print("Phase probability plot saved.")

    # plot attractor
    if not os.path.exists(f"output/experiments/plots/{experiment}_attractor.png"):
        print("Plotting attractor...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")
        plot_attractor(data, variables=variables)
        plt.savefig(f"output/experiments/plots/{experiment}_attractor.png")
        plt.close()
        print("Attractor plot saved.")

    # bifurcation plot
    if not os.path.exists(
        f"output/experiments/plots/{experiment}_bifurcation_{variables[-1]}_{populations[-1].lower()}.png"
    ):
        for population in populations:
            if population == "Super" or population == "Apex":
                continue
            for variable in variables:
                plot_bifurcation(data, population=population, variable=variable)
                plt.savefig(
                    f"output/experiments/plots/{experiment}_bifurcation_{variable}_{population.lower()}.png"
                )
                plt.close()
                print(f"Saved {variable} bifurcation plot for {population}.")

    # plot timeseries
    if not os.path.exists(
        f"output/experiments/plots/{experiment}_prey_timeseries_{models[-1].lower()}.png"
    ):
        print("Plotting timeseries plots...")

        plot_time_series(data=data, populations=populations, variables=variables)

        plt.savefig(f"output/experiments/plots/{experiment}_timeseries.png")
        plt.close()

    print(f"Analysis for {experiment} completed.\n")
