from functions.runner import model_run
from functions.runner import plot_pop, plot_space
import pandas as pd
import os

scenarios = {
    "debug-prey-only": {"predator": 0, "apex": 0, "prey": 500, "super": 0},
    "debug-predator-only": {"predator": 500, "apex": 0, "prey": 0, "super": 0},
    "debug-apex-only": {"predator": 0, "apex": 500, "prey": 0, "super": 0},
    "debug-super-only": {"predator": 0, "apex": 0, "prey": 0, "super": 500},
}

# run the model


def run_debug(kwargs):
    # check if output directory exists

    os.makedirs("output/debug/results", exist_ok=True)
    os.makedirs("output/debug/plots", exist_ok=True)

    # set loggin

    kwargs["progress"] = True

    for model, params in scenarios.items():
        kwargs["model"] = model
        kwargs.update(params)

        # run model

        m = model_run(**kwargs)

        # save data

        model_data = m.count.get_model_vars_dataframe()
        # set name for index column
        model_data.index.name = "Step"
        model_data.to_csv(f"output/debug/results/data_model_{kwargs['model']}.csv")

        agent_data = m.spatial.get_agent_vars_dataframe()
        agent_data.to_csv(f"output/debug/results/data_agents_{kwargs['model']}.csv")


def plot_debug(**kwargs):
    for model, params in scenarios.items():
        kwargs["model"] = model
        kwargs.update(params)

        # load data

        model_data = pd.read_csv(
            f"output/debug/results/data_model_{kwargs['model']}.csv"
        )
        agent_data = pd.read_csv(
            f"output/debug/results/data_agents_{kwargs['model']}.csv"
        )

        # plot population

        plot_pop(
            model_data=model_data,
            params=kwargs,
            file=f"output/debug/plots/pop_{kwargs['model']}.png",
            steps=kwargs["steps"],
        )

        # plot space

        plot_space(
            agent_data=agent_data,
            file=f"output/debug/plots/space_{kwargs['model']}.gif",
            steps=kwargs["steps"],
        )
