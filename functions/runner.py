# import classes and dependencies

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import warnings
from functions.model import model

warnings.filterwarnings("ignore")


def model_run(steps=50, **kwargs):
    # Run model

    print("Running model...")

    m = model(**{k: v for k, v in kwargs.items() if k != "model"})

    m.run_model(
        steps,
        progress=kwargs.get("progress", False),
        info=kwargs.get("info", False),
        limit=kwargs.get("limit", 10000),
        stop=kwargs.get("stop", False),
    )

    print("Model run complete.")

    return m


## visualize the model


def plot_pop(model_data=None, params={}, file="model_pop.png", steps=50):
    print("Creating population plot...")

    # Check if model_data is None or not a DataFrame
    if model_data is None or not hasattr(model_data, "columns"):
        raise ValueError(
            "model_data must be a pandas DataFrame with a 'columns' attribute."
        )

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

    # create plot

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the number of agents over time

    for i in list(model_data.columns):
        if i == "Step":
            continue

        ax.plot(model_data[i], label=i)

    ax.set_xlabel("Time")
    ax.set_ylabel("Number of agents")
    ax.set_title("Number of agents over time")
    ax.legend()
    ax.set_xlim(0, steps)

    # print parameters below the plot
    if isinstance(params, dict):  # Added check for dict type
        text = [
            f"{key} = {params[key]}"
            for key in params
            if key != "model" and key != "progress" and key != "info"
        ]
        text = ", ".join(text)
        fig.text(
            0.5,
            0.05,
            text,
            ha="center",
            fontsize=10,
            wrap=True,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
        )

    fig.subplots_adjust(bottom=0.25)

    # save the plot

    plt.savefig(file)

    print("Population plot created.")


## create a function to animate the spatial distribution of agents over time


def plot_space(agent_data=None, steps=100, file="space.gif"):
    print("Creating spatial plot...")

    ## get the spatial data

    spatial_data = agent_data

    ## convert index to columns

    if steps > spatial_data.Step.max():
        steps = spatial_data.Step.max()

    # define plot parameters

    marker = ["o", "^", "s"]
    agent_types = spatial_data.AgentType.unique()

    # create plot

    fig, ax = plt.subplots(figsize=(8, 8))

    # create list of images

    spaces = [
        ax.scatter([], [], label=agent, alpha=0.6, s=50, cmap="tab10", marker=marker[i])
        for i, agent in enumerate(agent_types)
    ]

    ax.set_xlim(0, spatial_data.x.max())
    ax.set_ylim(0, spatial_data.y.max())
    # remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title("Spatial distribution of agents")
    # set legend above the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

    def update(frame):
        for i, space in enumerate(spaces):
            space_data = spatial_data[
                (spatial_data.Step == frame)
                & (spatial_data.AgentType == agent_types[i])
            ]

            space.set_offsets(space_data[["x", "y"]])

        # set title
        ax.set_title(f"Step {frame}")
        return spaces

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, steps - 1), blit=False, interval=42
    )

    # save the animation

    ani.save(file, writer="imagemagick", fps=24)

    print("Spatial plot created.")

    return ani


## create a function to animate the density of agents over time


def plot_density(spatial_data=None, steps=100, file="density.gif"):
    print("Creating density plot...")

    ## convert index to columns

    spatial_data.reset_index(inplace=True)

    if steps > spatial_data.Step.max():
        steps = spatial_data.Step.max()

    # loop over types of agents

    agent_types = spatial_data.AgentType.unique()

    # create plot

    fig, axs = plt.subplots(1, len(agent_types), figsize=(len(agent_types) * 5, 5))

    # create list of images

    ims = []

    for i in range(1, steps, 1):
        spatial_data_time = spatial_data[spatial_data.Step == i]

        # temporarily store images

        step_images = []

        for j, agent in enumerate(agent_types):
            grid = np.zeros((spatial_data.x.max() + 1, spatial_data.y.max() + 1))

            spatial_data_agent = spatial_data_time[spatial_data_time.AgentType == agent]

            for index, row in spatial_data_agent.iterrows():
                x = row["x"]
                y = row["y"]

                grid[x][y] += 1

            im = fig.axes[j].imshow(grid, interpolation="nearest")

            text = fig.axes[j].text(
                1, -2, f"{agent} at Step {i}", fontsize=12, color="black"
            )

            step_images.append(im)
            step_images.append(text)

        ims.append(step_images)

    ani = animation.ArtistAnimation(
        fig, ims, interval=42, blit=False, repeat_delay=1000
    )

    # save the animation

    ani.save(file, writer="imagemagick", fps=24)

    print("Density plot created.")

    return ani


# animate space and number of agents over time together


def plot_space_pop(
    model_data=None, agent_data=None, params=None, steps=100, file="space_pop.png"
):
    print("Creating spatial and population plot...")

    # Check if model_data is None or not a DataFrame
    if model_data is None or not hasattr(model_data, "columns"):
        raise ValueError(
            "model_data must be a pandas DataFrame with a 'columns' attribute."
        )

    # Check if agent_data is None or not a DataFrame
    if agent_data is None or not hasattr(agent_data, "columns"):
        raise ValueError(
            "agent_data must be a pandas DataFrame with a 'columns' attribute."
        )

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

    ## get the spatial data

    spatial_data = agent_data

    ## convert index to columns

    if steps > spatial_data.Step.max():
        steps = spatial_data.Step.max()

    # define plot parameters

    marker = ["o", "^", "s"]
    agent_types = spatial_data.AgentType.unique()

    # create plot

    fig = plt.figure(figsize=(12, 6))

    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    # create list of images

    spaces = [
        axs[1].scatter(
            [], [], label=agent, alpha=0.6, s=25, cmap="tab10", marker=marker[i]
        )
        for i, agent in enumerate(agent_types)
    ]

    axs[1].set_xlim(0, spatial_data.x.max())
    axs[1].set_ylim(0, spatial_data.y.max())
    # set size
    axs[1].set_aspect("equal")
    # remove ticks and labels
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel(None)
    axs[1].set_ylabel(None)
    axs[1].set_title("Spatial distribution of agents")
    # set legend above the plot
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

    # plot the number of agents over time

    agent_types = model_data.columns
    # remove Step
    agent_types = agent_types[1:]

    lines = [
        axs[0].plot([], [], label=i)[0] for i in list(model_data.columns) if i != "Step"
    ]
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Number of agents")
    axs[0].set_title("Number of agents over time")
    axs[0].legend()
    axs[0].set_xlim(0, steps)
    axs[0].set_ylim(0, model_data[agent_types].max().max())

    # print parameters below the plot

    text = [
        f"{key} = {params[key]}"
        for key in params
        if key != "model" and key != "progress" and key != "info"
    ]
    text = ", ".join(text)

    fig.text(
        0.5,
        0.05,
        text,
        ha="center",
        fontsize=10,
        wrap=True,
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
    )

    fig.subplots_adjust(bottom=0.25)

    def update(frame):
        for i, space in enumerate(spaces):
            space_data = spatial_data[
                (spatial_data.Step == frame)
                & (spatial_data.AgentType == agent_types[i])
            ]

            space.set_offsets(space_data[["x", "y"]])

        model_data_time = model_data[model_data.Step < frame]

        for i, line in enumerate(lines):
            line.set_data(range(frame), model_data_time[agent_types[i]])

        # set title

        axs[0].set_title(f"Step {frame}")
        axs[1].set_title(f"Step {frame}")

        return spaces + lines

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, steps - 1), blit=False, interval=42
    )

    # save the animation

    ani.save(file, writer="imagemagick", fps=24)

    print("Spatial and population plot created.")

    return ani
