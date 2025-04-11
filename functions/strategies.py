from functions.runner import model_run
from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop
import pandas as pd
import os

strategy_info = {
    "Both Naive": {"prey_info": False, "predator_info": False},
    "Predator Informed": {"prey_info": False, "predator_info": True},
    "Prey Informed": {"prey_info": True, "predator_info": False},
    "Both Informed": {"prey_info": True, "predator_info": True},
}


def run_strategies(kwargs):
    os.makedirs("output/strategies/results", exist_ok=True)
    os.makedirs("output/strategies/plots", exist_ok=True)

    for strategy, info in strategy_info.items():
        # logging

        print(f"Running strategy: {strategy}")
        kwargs["progress"] = True

        # set info parameters
        kwargs.update(info)

        # run the model
        m = model_run(**kwargs)

        # save data

        model_data = m.count.get_model_vars_dataframe()
        # set name for index column
        model_data.index.name = "Step"
        model_data.to_csv(
            f"output/strategies/results/data_model_{kwargs['model']}_{strategy}.csv"
        )
        agent_data = m.spatial.get_agent_vars_dataframe()
        agent_data.to_csv(
            f"output/strategies/results/data_agents_{kwargs['model']}_{strategy}.csv"
        )


def plot_strategy(strategy_info, kwargs):
    for strategy, info in strategy_info.items():
        # set info parameters
        kwargs.update(info)

        # load data

        model_data = pd.read_csv(
            f"output/strategies/results/data_model_{kwargs['model']}_{strategy}.csv"
        )
        agent_data = pd.read_csv(
            f"output/strategies/results/data_agents_{kwargs['model']}_{strategy}.csv"
        )

        # plot the number of agents over time

        plot_pop(
            model_data=model_data,
            params=kwargs,
            file=f"output/strategies/plots/pop_{kwargs['model']}_{strategy}.png",
            steps=kwargs.get("steps", 1000),
        )

        # plot density of agents

        plot_density(
            spatial_data=agent_data,
            file=f"output/strategies/plots/density_{kwargs['model']}_{strategy}.gif",
            steps=kwargs.get("steps", 1000),
        )

        # plot spatial distribution of agents

        plot_space(
            agent_data=agent_data,
            file=f"output/strategies/plots/space_{kwargs['model']}_{strategy}.gif",
            steps=kwargs.get("steps", 1000),
        )

        # plot spatial distribution of agents with population size

        plot_space_pop(
            agent_data=agent_data,
            model_data=model_data,
            params=kwargs,
            file=f"output/strategies/plots/space_pop_{kwargs['model']}_{strategy}.gif",
            steps=kwargs.get("steps", 1000),
        )
