import numpy as np
import polars as pl
import os
from functions.summary_plots import (
    plot_attractor,
    plot_bifurcation,
    plot_max_prey,
    plot_phase_probability,
    plot_time_series,
    plot_power_spectrum,
)
from matplotlib import pyplot as plt
import seaborn as sns
from functions.summary_funcs import (
    classify_model_phase,
    phase_summary,
    parameter_summary,
    set_model_order,
    summaries,
    load_data,
    verify_data,
)
from functions.signals import (
    calculate_periodicity,
    plot_periodicity,
    summary_periodicity,
)

# Summary for test 4


def summary_test_4(data):
    if os.path.exists("output/experiments/plots/Test-4_max-prey.png"):
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

    plt.savefig("output/experiments/plots/Test-4_max-prey.png")

    plt.close()

    return 0


# analysis for test 7


def summary_test_7(data, reps=10, steps=1000):
    """
    Varying initial density of agents
    """

    initial_densities = {
        "Prey": [100, 500, 1000, 2000, 5000],
        "Predator": [100, 500, 1000, 2000, 5000],
        "Apex": [0, 100, 500, 1000, 2000],
        "Super": [0, 100, 500, 1000, 2000],
    }

    # Combinations

    N0 = (
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

    # Repeat each row reps * steps times
    N0 = np.repeat(N0, reps * steps, axis=0)

    # Fixed initial densities for plotting

    N0_fixed = {"Prey": 2000, "Predator": 500, "Apex": 500, "Super": 100}

    # Add N0 as a column to the data
    data = data.with_columns(
        pl.lit(N0[:, 0]).alias("N0_prey"),
        pl.lit(N0[:, 1]).alias("N0_predator"),
        pl.lit(N0[:, 2]).alias("N0_apex"),
        pl.lit(N0[:, 3]).alias("N0_super"),
    )

    # Make dataframes to hold phase and summary

    phase_full = pl.DataFrame()

    # Plot

    for agent in initial_densities.keys():
        # Use fixed initial densities of other agents
        plot_data = (
            data.filter(
                (
                    pl.col(f"N0_{a.lower()}") == N0_fixed[a]
                    for a in initial_densities.keys()
                    if a != agent
                )
            )
            .drop([f"N0_{a.lower()}" for a in initial_densities.keys() if a != agent])
            .with_columns(
                (
                    r"$N_0$" + f"({agent})=" + pl.col(f"N0_{agent.lower()}").cast(str)
                ).alias("model")
            )
            .rename(
                {
                    f"N0_{agent.lower()}": "N0",
                }
            )
        )

        if plot_data.is_empty():
            print(f"No data available for {agent} with fixed initial densities.")
            continue

        if not os.path.exists(
            f"output/experiments/plots/Test-7_timeseries_{agent.lower()}.png"
        ):
            # Plot timeseries
            plot_time_series(
                data=plot_data,
                populations=["Prey", "Predator", "Apex", "Super"],
                variables=None,
            )

            plt.savefig(
                f"output/experiments/plots/Test-7_timeseries_{agent.lower()}.png"
            )
            plt.close()

        # Classify model phase

        if not os.path.exists("output/experiments/outcomes/Test-7_phase.csv"):
            phase = classify_model_phase(
                plot_data,
                variables=["N0"],
                model=True,
                reps=10,
            )

            phase_full = pl.concat([phase_full, phase])

    # Save phase data

    if not phase_full.is_empty():
        phase_full.write_csv(
            "output/experiments/outcomes/Test-7_phase.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # Plot phase probability

    if not os.path.exists("output/experiments/plots/Test-7_phase_probability.png"):
        if phase_full.is_empty():
            print("No phase data available for plotting.")
            return

        phase_full = pl.read_csv("output/experiments/outcomes/Test-7_phase.csv")

        phase_full = phase_full.with_columns(
            (pl.col("model").str.replace_all(r"=\d+", "")).alias("model")
        )

        plot_phase_probability(phase_full, variables=["N0"])
        plt.savefig("output/experiments/plots/Test-7_phase_probability.png")
        plt.close()

    print("Experiment 9 analysis completed.")

    return 0


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

    data = load_data(
        data_path,
        experiment=experiment,
        multiple=multiple,
        n_models=n_models,
    )

    # verify data

    data = verify_data(
        data,
        n_params=n_params,
        parameter_depth=parameter_depth,
        models=models,
        reps=reps,
        steps=steps,
        n_models=n_models,
        variables=variables,
    )

    # Set order for models in experiment 2
    if experiment == "Experiment-2":
        data = set_model_order(data)

    # summaries
    print("Generating summaries...")
    summaries(data)

    # plot for experiment 6

    if experiment == "Test-4":
        summary_test_4(data)
        return 0

    # summary for experiment 9

    if experiment == "Test-7":
        summary_test_7(data)
        return 0

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

    if not os.path.exists(
        f"output/experiments/outcomes/{experiment}_phase_summary.csv"
    ):
        print("Generating phase summary...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")

        summary_phases = phase_summary(
            phase,
        )

        summary_phases.write_csv(
            f"output/experiments/outcomes/{experiment}_phase_summary.csv",
            separator=",",
            include_header=True,
            quote_style="necessary",
        )

    # Parameter summary

    if not os.path.exists(
        f"output/experiments/outcomes/{experiment}_param_summary.csv"
    ):
        print("Generating parameter summary...")
        phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")
        param_summary = parameter_summary(
            phase,
            variables=variables,
        )

        param_summary.write_csv(
            f"output/experiments/outcomes/{experiment}_param_summary.csv",
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
        f"output/experiments/plots/{experiment}_bifurcation_{variables[-1]}_prey.png"
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
    if not os.path.exists(f"output/experiments/plots/{experiment}_timeseries.png"):
        print("Plotting timeseries plots...")

        plot_time_series(data=data, populations=populations, variables=variables)

        plt.savefig(f"output/experiments/plots/{experiment}_timeseries.png")
        plt.close()

    # Plot power spectrum
    if not os.path.exists(f"output/experiments/plots/{experiment}_power_spectrum.png"):
        print("Plotting power spectrum...")
        plot_power_spectrum(data, populations)
        plt.savefig(f"output/experiments/plots/{experiment}_power_spectrum.png")
        plt.close()
        print("Power spectrum plot saved.")

    # Calculate periodicity
    if "Scenario" in experiment:
        if not os.path.exists(
            f"output/experiments/outcomes/{experiment}_periodicity.csv"
        ):
            print("Calculating periodicity...")
            # Calculate periodicity
            periodicity = calculate_periodicity(data, populations=populations)
            periodicity.write_csv(
                f"output/experiments/outcomes/{experiment}_periodicity.csv",
                separator=",",
                include_header=True,
                quote_style="necessary",
            )

            # Plot periodicity
            plot_periodicity(periodicity, populations=populations)
            plt.savefig(f"output/experiments/plots/{experiment}_periodicity.png")
            plt.close()
            print("Periodicity plot saved.")

        if not os.path.exists(
            f"output/experiments/outcomes/{experiment}_periodicity_summary.csv"
        ):
            print("Summarizing periodicity...")
            periodicity = pl.read_csv(
                f"output/experiments/outcomes/{experiment}_periodicity.csv"
            )
            phase = pl.read_csv(f"output/experiments/outcomes/{experiment}_phase.csv")
            periodicity_summary = summary_periodicity(periodicity, phase)
            periodicity_summary.write_csv(
                f"output/experiments/outcomes/{experiment}_periodicity_summary.csv",
                separator=",",
                include_header=True,
                quote_style="necessary",
            )
            print("Periodicity summary saved.")

    print(f"Analysis for {experiment} completed.\n")

    return 0
