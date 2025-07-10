import polars as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# set style


def set_style():
    sns.set_theme(style="whitegrid", font_scale=2)
    plt.rcParams.update({"font.size": 30, "figure.figsize": (10, 6), "figure.dpi": 96})
    sns.color_palette()


# Helper functions for setting plot titles and axis labels


def set_plot_titles(plot, variables):
    if variables[0] == "s_breed":
        plot.set_titles(
            row_template=r"$b_{{predator}}$" + "= {row_name}",
            col_template=r"$b_{{prey}}$" + " = {col_name}",
        )
    elif variables[0] == "s_max":
        plot.set_titles(
            col_template=r"$S_{{predator}}$" + " = {col_name}",
        )
    elif variables[0] == "a_max":
        plot.set_titles(
            col_template=r"$S_{{apex}}$" + " = {col_name}",
        )
    elif variables[0] == "a_breed":
        plot.set_titles(
            col_template=r"$b_{{apex}}$" + " = {col_name}",
        )
    elif variables[0] == "f_max":
        plot.set_titles(
            col_template=r"$K$" + " = {col_name}",
        )
    elif variables[0] == "s_lethality":
        plot.set_titles(
            col_template=r"$Lethality_{{predator}}$" + " = {col_name}",
        )
    elif variables[0] == "a_lethality":
        plot.set_titles(
            col_template=r"$Lethality_{{apex}}$" + " = {col_name}",
        )

    return plot


def set_plot_axis_labels(plot, variables):
    if variables[0] == "s_breed":
        plot.set_axis_labels(x_var=r"$b_{{predator}}$", y_var=r"$b_{{prey}}$")
    elif variables[0] == "s_max":
        plot.set_axis_labels(x_var=r"$S_{{predator}}$", y_var="Probability")
    elif variables[0] == "a_max":
        plot.set_axis_labels(x_var=r"$S_{{apex}}$", y_var="Probability")
    elif variables[0] == "a_breed":
        plot.set_axis_labels(x_var=r"$b_{{apex}}$", y_var="Probability")
    elif variables[0] == "f_max":
        plot.set_axis_labels(x_var=r"$K$", y_var="Probability")
    elif variables[0] == "s_lethality":
        plot.set_axis_labels(x_var=r"$Lethality_{{predator}}$", y_var="Probability")
    elif variables[0] == "a_lethality":
        plot.set_axis_labels(x_var=r"$Lethality_{{apex}}$", y_var="Probability")

    return plot


# attractor plot


def plot_attractor(data, variables=["s_breed", "f_breed"]):
    set_style()

    # reduce number of rep_ids
    ids = np.array([0, 5, 10, 20, 25])
    data = data.filter(pl.col("rep_id").is_in(ids))

    s_breed = data.select(variables[0]).unique().sort(by=variables[0], descending=False)
    s_breed = s_breed.to_numpy().T.flatten()

    if variables[0] == "s_breed":
        data = data.filter(pl.col("s_breed") == s_breed[30])
        data = data.filter(pl.col("f_breed") == s_breed[20])
    else:
        s = s_breed.shape[0] // 2
        sample_space = s_breed[s]

        # filter data

        for v in variables:
            data = data.filter(pl.col(v).is_in(sample_space))

    # Remove first 400 steps

    data = data.filter(pl.col("step") > 400)

    # round s_breed and f_breed to 2 decimal places

    for v in variables:
        data = data.with_columns(
            [
                pl.col(v).round(2),
            ]
        )

    # set col_wrap if unique models are more than 2
    if data.select(pl.col("model").n_unique()).to_numpy()[0][0] > 2:
        col_wrap = 3
    elif data.select(pl.col("model").n_unique()).to_numpy()[0][0] == 2:
        col_wrap = 2
    else:
        col_wrap = 1

    plot = sns.relplot(
        data=data.select(["Predator", "Prey", "rep_id", "model", "step"]).sort(
            ["model", "rep_id", "step"]
        ),
        x="Predator",
        y="Prey",
        hue="rep_id",
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.5,
        col="model",
        col_wrap=col_wrap,
        legend=False,
        kind="scatter",
        marker="o",
    )

    # set titles
    plot.set_titles(col_template="Model: {col_name}")

    plt.tight_layout()

    return plot


# plot phase probability


def plot_phase_probability(phase_data, variables=["s_breed", "f_breed"]):
    """
    Function to plot phase probability as a function of breeding rates.
    """

    # set phase as ordered categorical

    phase_data = phase_data.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    # Set probability as float

    phase_data = phase_data.with_columns(pl.col("prob").cast(pl.Float64))

    # Set number of columns

    if phase_data.select(pl.col("model").n_unique()).to_numpy()[0][0] > 2:
        col_wrap = 3
    elif phase_data.select(pl.col("model").n_unique()).to_numpy()[0][0] == 1:
        col_wrap = 1
    else:
        col_wrap = 2

    if len(variables) > 1:
        plot = sns.relplot(
            data=phase_data,
            x="s_breed",
            y="f_breed",
            hue="phase",
            alpha=0.5,
            size="prob",
            col="model",
            palette="pastel",
            hue_norm=(-1, 1),
            edgecolor=".7",
            height=6,
            aspect=1,
            sizes=(1, 50),
            size_norm=(-0.2, 0.8),
            legend=False,
            col_wrap=col_wrap,
        )

        plot.set_titles(
            col_template="Model: {col_name}",
        )

    else:
        # Check if model column exists
        if "model" in phase_data.columns:
            cols = "model"
        else:
            cols = None
        plot = sns.relplot(
            data=phase_data,
            x=variables[0],
            y="prob",
            hue="phase",
            palette="Set1",
            col=cols,
            hue_norm=(-1, 1),
            height=6,
            aspect=1.3,
            kind="line",
        )

        # set limits
        plot.set(ylim=(0, 1.1))

    plot = set_plot_axis_labels(plot, variables)

    return plot


# bifurcation plot


def plot_bifurcation(data, grid_size=3, population="Prey", variable="s_breed"):
    """
    Plot change in state variables as a function of key system drivers.

    State variables:

        Predator: Number of predators
        Prey: Number of Prey
        Apex: Number of apex predators

    Args:
        data (pl.DataFrame): The data to plot

    Returns:
        plot_prey: The plot of Prey
        plot_predator: The plot of predators
        plot_apex: The plot of apex predators

    """

    # filter DataFrame

    data = data.filter(pl.col("step") > 600)
    data = data.filter(pl.col(population) < 10000)
    data = data.filter(pl.col("rep_id").is_in([0, 5, 10, 20, 25]))

    if variable == "s_breed":
        cat = "f_breed"
        # subset values

        f_breed = data.select("f_breed").unique()
        f_breed = f_breed.to_numpy().T.flatten()
        n = f_breed.shape[0] // grid_size
        samples = range(0, f_breed.shape[0], n)

        sample_space = f_breed[samples]

        # filter DataFrame

        data = data.filter(pl.col("f_breed").is_in(sample_space))

        # round f_breed to 2 decimal places

        data = data.with_columns(
            [
                pl.col("f_breed").round(2),
            ]
        )

        row_label = r"$b_{{prey}}$"
        x_label = r"$b_{{predator}}$"

    elif variable == "f_breed":
        cat = "s_breed"

        # subset values

        s_breed = data.select("s_breed").unique().sort(by="s_breed", descending=False)
        s_breed = s_breed.to_numpy().T.flatten()
        n = s_breed.shape[0] // grid_size
        samples = range(0, s_breed.shape[0], n)

        sample_space = s_breed[samples]

        # filter DataFrame

        data = data.filter(pl.col("s_breed").is_in(sample_space))

        # round s_breed to 2 decimal places

        data = data.with_columns(
            [
                pl.col("s_breed").round(2),
            ]
        )

        row_label = r"$b_{{predator}}$"
        x_label = r"$b_{{prey}}$"
    elif variable == "s_max":
        cat = None
        x_label = r"$S_{{predator}}$"
    elif variable == "a_max":
        cat = None
        x_label = r"$S_{{apex}}$"
    elif variable == "a_breed":
        cat = None
        x_label = r"$b_{{apex}}$"
    elif variable == "f_max":
        cat = None
        x_label = r"$K$"
    elif variable == "s_lethality":
        cat = None
        x_label = r"$Lethality_{{predator}}$"
    elif variable == "a_lethality":
        cat = None
        x_label = r"$Lethality_{{apex}}$"

    else:
        raise ValueError("Variable must be one of: s_breed, f_breed, s_max, a_max")

    # plot prey
    plot = sns.relplot(
        data=data,
        x=variable,
        y=population,
        col="model",
        hue=cat,
        row=cat,
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.5,
        edgecolor="w",
        legend=False,
    )

    if cat is not None:
        plot.set_titles(
            row_template=row_label + " = {row_name}",
            col_template="Model: {col_name}",
        )
    else:
        plot.set_titles(
            col_template="Model: {col_name}",
        )

    if population == "Prey":
        plot.set_axis_labels(x_var=x_label, y_var="Prey")

    elif population == "Predator":
        plot.set_axis_labels(x_var=x_label, y_var="Predator")

    elif population == "Apex":
        plot.set_axis_labels(x_var=x_label, y_var="Apex")

    else:
        raise ValueError("Population must be one of: Predator, Prey, Apex")

    return plot


# time series plotting


def plot_time_series(
    data,
    populations="Prey",
    variables=["s_breed", "f_breed"],
):
    """
    Function to plot time series of population dynamics.

    Args:
        data (pl.DataFrame): The data to plot
        population (str): The population to plot

    Returns:
        plot: The plot of the population dynamics
    """
    set_style()

    # filter reps

    ids = np.array([0, 5, 10, 20, 25])
    data = data.filter(pl.col("rep_id").is_in(ids))

    # determine sample space

    if variables is not None:
        s_breed = (
            data.select(variables[0]).unique().sort(by=variables[0], descending=False)
        )
        s_breed = s_breed.to_numpy().T.flatten()

        if variables[0] == "s_breed":
            data = data.filter(pl.col("s_breed") == s_breed[30])
            data = data.filter(pl.col("f_breed") == s_breed[20])
        else:
            s = s_breed.shape[0] // 2
            sample_space = s_breed[s]
            # filter DataFrame

            for v in variables:
                data = data.filter(pl.col(v).is_in(sample_space))

        # round s_breed and f_breed to 2 decimal places

        for v in variables:
            data = data.with_columns(
                [
                    pl.col(v).round(2),
                ]
            )

    # set col_wrap if unique models are more than 2
    if data.select(pl.col("model").n_unique()).to_numpy()[0][0] > 2:
        col_wrap = 3
    elif data.select(pl.col("model").n_unique()).to_numpy()[0][0] == 2:
        col_wrap = 2
    else:
        col_wrap = 1

    # unpivot populations

    data = data.unpivot(
        on=populations,
        variable_name="population",
        value_name="N",
        index=["step", "rep_id", "model"],
    )

    # Rename population column
    mapping = {
        "Predator": "Mesopredator",
        "Apex": "Apex predator",
        "Super": "Superpredator",
        "Prey": "Prey",
    }

    data = data.with_columns(pl.col("population").replace(mapping).alias("population"))

    plot = sns.relplot(
        kind="line",
        data=data,
        x="step",
        y="N",
        hue="population",
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.25,
        col="model",
        legend=True,
        units="rep_id",
        estimator=None,
        col_wrap=col_wrap,
    )

    # Set legend title
    plot._legend.set_title("Population")
    plot._legend.set_bbox_to_anchor((0.95, 0.8))

    # set titles

    plot.set_titles(
        col_template="Model: {col_name}",
    )

    plt.tight_layout()

    # legend
    return plot


# plot max prey based on latice size


def plot_max_prey(data):
    data = data.filter(pl.col("step") > 400)

    plot = sns.relplot(
        data,
        x="L2",
        y="Prey",
        hue="f_max",
        legend=True,
        kind="line",
        linewidth=5,
        linestyle="-",
        markers="o",
        height=6,
        aspect=1,
    )

    plot.set_axis_labels(x_var=r"$L^{2}$")

    plot._legend.set_title("S")

    return plot
