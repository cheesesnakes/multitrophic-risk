from functions.runner import model_run
from functions.runner import plot_pop, plot_space
import pandas as pd
from params import kwargs
import os

steps = 1000

# check if output directory exists

if not os.path.exists("output/debug"):
    os.makedirs("output/debug")
if not os.path.exists("output/debug/results"):
    os.makedirs("output/debug/results")
if not os.path.exists("output/debug/plots"):
    os.makedirs("output/debug/plots")

# run the model


def run_debug(**kwargs):
    # run model

    m = model_run(**kwargs, steps=steps)

    # save data

    model_data = m.count.get_model_vars_dataframe()
    # set name for index column
    model_data.index.name = "Step"
    model_data.to_csv(f"output/debug/results/data_model_{kwargs['model']}.csv")

    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f"output/debug/results/data_agents_{kwargs['model']}.csv")

    # plot population

    plot_pop(
        model_data=model_data,
        params=kwargs,
        file=f"output/debug/plots/pop_{kwargs['model']}.png",
        steps=steps,
    )

    # plot space

    agent_data = pd.read_csv(f"output/debug/results/data_agents_{kwargs['model']}.csv")

    plot_space(
        agent_data=agent_data,
        file=f"output/debug/plots/space_{kwargs['model']}.gif",
        steps=steps,
    )


if __name__ == "__main__":
    # run model with prey only

    kwargs["model"] = "debug-prey-only"
    kwargs["predator"] = 0
    kwargs["apex"] = 0
    kwargs["prey"] = 500
    kwargs["super"] = 0

    run_debug(**kwargs)

    # run model with predator only

    kwargs["model"] = "debug-predator-only"
    kwargs["predator"] = 500
    kwargs["apex"] = 0
    kwargs["prey"] = 0
    kwargs["super"] = 0

    run_debug(**kwargs)

    # run model with apex only

    kwargs["model"] = "debug-apex-only"
    kwargs["predator"] = 0
    kwargs["apex"] = 500
    kwargs["prey"] = 0
    kwargs["super"] = 0

    run_debug(**kwargs)

    # run model with super only

    kwargs["model"] = "debug-super-only"
    kwargs["predator"] = 0
    kwargs["apex"] = 0
    kwargs["prey"] = 0
    kwargs["super"] = 500

    run_debug(**kwargs)
