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
    s_breed = data.select("s_breed").unique()
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


plot_attractor(data)
