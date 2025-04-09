import polars as pl
import seaborn as sns
import numpy as np

# set style

sns.set_theme(style="whitegrid", font_scale=1.5)
sns.color_palette()

# example data


def example_data():
    data = pl.DataFrame(
        {
            "Predator": [1, 2, 3, 4, 5],
            "Prey": [1, 2, 3, 4, 5],
            "s_breed": np.linspace(0, 1, 5),
            "f_breed": np.linspace(0, 1, 5),
            "model": ["Apex"] * 5,
        }
    )

    return data


data = example_data()

# attractor plot


def plot_attractor(data, grid_size=3):
    s_breed = data.select("s_breed").unique().sort(by="s_breed", descending=False)
    s_breed = s_breed.to_numpy().T.flatten()

    # subset values
    n = s_breed.shape[0] // grid_size

    samples = range(0, s_breed.shape[0], n)

    sample_space = s_breed[samples]

    # filter data
    data = data.filter(
        pl.col("s_breed").is_in(sample_space) & pl.col("f_breed").is_in(sample_space)
    )

    # round s_breed and f_breed to 2 decimal places
    data = data.with_columns(
        [
            pl.col("s_breed").round(2),
            pl.col("f_breed").round(2),
        ]
    )

    # plot

    plot = sns.relplot(
        data=data,
        x="Predator",
        y="Prey",
        hue="model",
        style="model",
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.5,
        edgecolor="w",
        col="s_breed",
        row="f_breed",
    )

    sns.move_legend(
        plot,
        "upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
        fontsize=12,
        markerscale=2,
    )

    plot.set_titles(
        row_template=r"$b_{{predator}}$" + "= {row_name}",
        col_template=r"$b_{{prey}}$" + " = {col_name}",
    )

    return plot


# plot phase probability


def plot_phase_probability(phase_data):
    """
    Function to plot phase probability as a function of breeding rates.
    """

    # set phase as ordered categorical

    phase_data = phase_data.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(pl.Enum(["Prey Only", "Coexistence", "Extinction"]))
    )

    plot = sns.relplot(
        data=phase_data,
        x="s_breed",
        y="f_breed",
        hue="prob",
        size="prob",
        col="phase",
        row="model",
        palette="vlag",
        hue_norm=(-1, 1),
        edgecolor=".7",
        height=6,
        aspect=1,
        sizes=(1, 20),
        size_norm=(-0.2, 0.8),
        legend=False,
    )

    plot.set_titles(
        row_template="Model: {row_name}",
        col_template="Phase: {col_name}",
    )

    plot.set_axis_labels(x_var=r"$b_{{predator}}$", y_var=r"$b_{{prey}}$")

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

    else:
        raise ValueError("Variable must be one of: s_breed, f_breed")

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

    if population == "Prey":
        plot.set_titles(
            row_template=row_label + " = {row_name}",
            col_template="Model: {col_name}",
        )

        plot.set_axis_labels(x_var=x_label, y_var="Prey")

    elif population == "Predator":
        plot.set_titles(
            row_template=row_label + " = {row_name}",
            col_template="Model: {col_name}",
        )

        plot.set_axis_labels(x_var=x_label, y_var="Predator")

    elif population == "Apex":
        plot.set_titles(
            row_template=row_label + " = {row_name}",
            col_template="Model: {col_name}",
        )

        plot.set_axis_labels(x_var=x_label, y_var="Apex")

    else:
        raise ValueError("Population must be one of: Predator, Prey, Apex")

    return plot


# time series plotting


def plot_time_series(data, population="Prey", model="Apex", grid_size=3):
    """
    Function to plot time series of population dynamics.

    Args:
        data (pl.DataFrame): The data to plot
        population (str): The population to plot

    Returns:
        plot: The plot of the population dynamics
    """
    # filter model

    data = data.filter((pl.col("model").eq(model)))

    # select variables

    s_breed = data.select("s_breed").unique().sort(by="s_breed", descending=False)
    s_breed = s_breed.to_numpy().T.flatten()
    n = s_breed.shape[0] // grid_size
    samples = range(0, s_breed.shape[0], n)
    sample_space = s_breed[samples]

    # sample id

    ids = np.array([0, 5, 10, 20, 25])

    # filter DataFrame

    data = data.filter(
        pl.col("s_breed").is_in(sample_space) & pl.col("f_breed").is_in(sample_space)
    )

    data = data.filter(pl.col("rep_id").is_in(ids))

    # round s_breed and f_breed to 2 decimal places

    data = data.with_columns(
        [
            pl.col("s_breed").round(2),
            pl.col("f_breed").round(2),
        ]
    )

    # plot main line

    plot = sns.relplot(
        kind="line",
        data=data,
        x="step",
        y=population,
        hue="s_breed",
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.25,
        col="s_breed",
        row="f_breed",
        legend=False,
        units="rep_id",
        estimator=None,
    )

    # add a bold line

    for ax in plot.axes.flat:
        # filter data

        line = data.filter(
            pl.col("s_breed").is_in(sample_space)
            & pl.col("f_breed").is_in(sample_space)
        )

        line = line.filter(pl.col("rep_id") == 0)

        # plot line

        sns.lineplot(
            data=line,
            x="step",
            y=population,
            color="black",
            linewidth=1,
            ax=ax,
        )

    # set titles

    plot.set_titles(
        row_template=r"$b_{{predator}}$" + " = {row_name}",
        col_template=r"$b_{{prey}}$" + " = {col_name}",
    )

    return plot


# test plotting functions

plot_attractor(data)
