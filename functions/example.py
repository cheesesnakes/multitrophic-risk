from functions.runner import model_run
from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


model_params = {
    "apex": {"model": "apex", "predator": 500, "prey": 1000, "apex": 250, "super": 0},
    "super": {"model": "super", "predator": 500, "prey": 1000, "super": 250, "apex": 0},
    "lv": {"model": "lv", "predator": 500, "prey": 1000, "apex": 0, "super": 0},
}


def run_example(kwargs, model="lv"):
    # create output directories
    os.makedirs("output/examples/results", exist_ok=True)
    os.makedirs("output/examples/plots", exist_ok=True)

    # set model parameters

    kwargs.update(model_params[model])

    # set logging

    kwargs["progress"] = True

    # Set spatial logging

    kwargs["collect_agents"] = True

    # run the model

    print(f"Running {model} model")

    m = model_run(**kwargs)

    # save data

    model_data = m.count.get_model_vars_dataframe()
    # set name for index column
    model_data.index.name = "Step"
    model_data.to_csv(f"output/examples/results/data_model_{kwargs['model']}.csv")
    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f"output/examples/results/data_agents_{kwargs['model']}.csv")


def plot_example(kwargs, model="lv"):
    sns.set_theme(style="whitegrid", font_scale=1.5)
    plt.rcParams.update({"font.size": 14, "figure.figsize": (10, 6)})
    sns.color_palette()

    kwargs.update(model_params[model])

    # load data

    model_data = pd.read_csv(
        f"output/examples/results/data_model_{kwargs['model']}.csv"
    )
    agent_data = pd.read_csv(
        f"output/examples/results/data_agents_{kwargs['model']}.csv"
    )

    # plot the number of agents over time

    plot_pop(
        model_data=model_data,
        params=kwargs,
        file=f"output/examples/plots/pop_{kwargs['model']}.png",
        steps=kwargs.get("steps", 1000),
    )

    # plot density of agents

    plot_density(
        spatial_data=agent_data,
        file=f"output/examples/plots/density_{kwargs['model']}.gif",
        steps=kwargs.get("steps", 1000),
    )

    # plot spatial distribution of agents

    plot_space(
        agent_data=agent_data,
        file=f"output/examples/plots/space_{kwargs['model']}.gif",
        steps=kwargs.get("steps", 1000),
    )

    # plot spatial distribution of agents with population size

    plot_space_pop(
        agent_data=agent_data,
        model_data=model_data,
        params=kwargs,
        file=f"output/examples/plots/space_pop_{kwargs['model']}.gif",
        steps=kwargs.get("steps", 1000),
    )


if __name__ == "__main__":
    # plot the results
    from params import kwargs

    for model in ["apex", "super", "lv"]:
        plot_example(kwargs, model=model)
