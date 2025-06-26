import polars as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
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


def plot_attractor(data, grid_size=3, variables=["s_breed", "f_breed"]):
    s_breed = data.select(variables[0]).unique().sort(by=variables[0], descending=False)
    s_breed = s_breed.to_numpy().T.flatten()

    # subset values
    n = s_breed.shape[0] // grid_size

    samples = range(0, s_breed.shape[0], n)

    sample_space = s_breed[samples]

    # filter data

    for v in variables:
        data = data.filter(pl.col(v).is_in(sample_space))

    # round s_breed and f_breed to 2 decimal places

    for v in variables:
        data = data.with_columns(
            [
                pl.col(v).round(2),
            ]
        )

    # plot

    col = variables[0]

    if len(variables) > 1:
        row = variables[1]
    else:
        row = None

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
        col=col,
        row=row,
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

    # set titles
    plot = set_plot_titles(plot, variables)

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
            col_wrap=3,
        )

        plot.set_titles(
            col_template="Phase: {col_name}",
        )

    else:
        plot = sns.relplot(
            data=phase_data,
            x=variables[0],
            y="prob",
            hue="phase",
            palette="Set1",
            col="model",
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
    data, population="Prey", model="Apex", variables=["s_breed", "f_breed"], grid_size=3
):
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

    s_breed = data.select(variables[0]).unique().sort(by=variables[0], descending=False)
    s_breed = s_breed.to_numpy().T.flatten()
    n = s_breed.shape[0] // grid_size
    samples = range(0, s_breed.shape[0], n)
    sample_space = s_breed[samples]

    # sample id

    ids = np.array([0, 5, 10, 20, 25])

    # filter DataFrame

    for v in variables:
        data = data.filter(pl.col(v).is_in(sample_space))

    data = data.filter(pl.col("rep_id").is_in(ids))

    # round s_breed and f_breed to 2 decimal places

    for v in variables:
        data = data.with_columns(
            [
                pl.col(v).round(2),
            ]
        )

    # plot main line

    if len(variables) > 1:
        row = variables[1]
    else:
        row = None

    plot = sns.relplot(
        kind="line",
        data=data,
        x="step",
        y=population,
        hue=variables[0],
        palette="Set1",
        height=6,
        aspect=1.3,
        alpha=0.25,
        col=variables[0],
        row=row,
        legend=False,
        units="rep_id",
        estimator=None,
    )

    # add a bold line

    for ax in plot.axes.flat:
        # filter data

        for v in variables:
            line = data.filter(pl.col(v).is_in(sample_space))

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

    plot = set_plot_titles(plot, variables)

    return plot


# Plot transition between phases


def plot_phase_transition(
    transition_data, variables=["s_breed", "f_breed"], model=True
):
    """
    Function to plot phase transition between phases as a function of state varible/s.
    """

    transition_data = transition_data.with_columns(
        pl.col("phase")
        .cast(pl.Categorical)
        .cast(
            pl.Enum(
                [
                    "Prey Only",
                    "P - C Bistable",
                    "Coexistence",
                    "C - E Bistable",
                    "Extinction",
                    "P - E Bistable",
                ]
            )
        )
    )
    if model:
        # filter data
        col = "model"
    else:
        col = None

    # set col_wrap if unique models are more than 2

    if (
        model
        and transition_data.select(pl.col("model").n_unique()).to_numpy()[0][0] > 2
    ):
        col_wrap = 3
    else:
        col_wrap = 2

    plot = sns.relplot(
        data=transition_data,
        x=variables[0],
        y=variables[1],
        hue="phase",
        palette="Set1",
        height=6,
        aspect=1,
        alpha=0.5,
        edgecolor="w",
        legend=True,
        col=col,
        s=25,
        marker="o",
        col_wrap=col_wrap,
    )

    if model:
        plot.set_titles(
            col_template="Model: {col_name}",
        )

    # plt.add_legend(title="Phase")

    # set titles

    plot = set_plot_axis_labels(plot, variables)

    return plot


# plot max prey based on latice size


def plot_max_prey(data):
    data = data.filter(pl.col("step") > 600)

    plot = sns.relplot(
        data,
        x="L2",
        y="Prey",
        hue="f_max",
        alpha=0.5,
        edgecolor="w",
        legend=True,
        s=50,
        height=6,
        aspect=1,
    )

    plot.set_axis_labels(x_var=r"$L^{2}$")

    plt.legend(
        title=r"$K$",
    )
    return plot
