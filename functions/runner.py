# import classes and dependencies

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import warnings
from functions.model import model
import seaborn as sns

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

    if model_data is None or "columns" not in dir(model_data):
        raise ValueError("model_data must be a pandas DataFrame.")

    sns.set_theme(style="whitegrid")
    melted = model_data.melt(id_vars="Step", var_name="AgentType", value_name="Count")

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=melted, x="Step", y="Count", hue="AgentType", linewidth=2.0)

    ax.set_xlim(0, steps)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of agents")
    ax.set_title("Number of agents over time")

    plt.savefig(file)
    plt.close()
    print("Population plot created.")


## create a function to animate the spatial distribution of agents over time


def plot_space(agent_data=None, steps=100, file="space.gif"):
    print("Creating spatial plot...")

    if agent_data is None or "columns" not in dir(agent_data):
        raise ValueError("agent_data must be a pandas DataFrame.")

    if steps > agent_data.Step.max():
        steps = agent_data.Step.max()

    agent_types = agent_data.AgentType.unique()
    marker = ["o", "^", "s", "D", "v", "*"]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatters = [
        ax.scatter([], [], alpha=0.6, s=50, label=agent, marker=marker[i % len(marker)])
        for i, agent in enumerate(agent_types)
    ]

    ax.set_xlim(0, agent_data.x.max())
    ax.set_ylim(0, agent_data.y.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_title("")

    def update(frame):
        current_data = agent_data[agent_data.Step == frame]
        for i, agent in enumerate(agent_types):
            subset = current_data[current_data.AgentType == agent]
            scatters[i].set_offsets(subset[["x", "y"]].values)
        ax.set_title(f"Step {frame}")
        return scatters

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, steps), interval=42, blit=False
    )
    ani.save(file, writer="imagemagick", fps=24)
    plt.close()
    print("Spatial plot created.")
    return ani


## create a function to animate the density of agents over time


def plot_density(spatial_data=None, steps=100, file="density.gif"):
    print("Creating density plot...")

    if spatial_data is None or "columns" not in dir(spatial_data):
        raise ValueError("spatial_data must be a pandas DataFrame.")

    spatial_data = spatial_data.reset_index(drop=True)
    if steps > spatial_data.Step.max():
        steps = spatial_data.Step.max()

    agent_types = spatial_data.AgentType.unique()
    grid_shape = (spatial_data.x.max() + 1, spatial_data.y.max() + 1)

    fig, axs = plt.subplots(1, len(agent_types), figsize=(len(agent_types) * 5, 5))
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    ims = []

    for i in range(1, steps):
        step_images = []
        current = spatial_data[spatial_data.Step == i]

        for j, agent in enumerate(agent_types):
            agent_data = current[current.AgentType == agent]
            grid = np.zeros(grid_shape, dtype=int)
            np.add.at(grid, (agent_data.x.values, agent_data.y.values), 1)

            im = axs[j].imshow(grid.T, interpolation="nearest", origin="lower")
            text = axs[j].text(
                1, -2, f"{agent} at Step {i}", fontsize=12, color="black"
            )
            step_images.extend([im, text])
        ims.append(step_images)

    ani = animation.ArtistAnimation(fig, ims, interval=42, blit=False)
    ani.save(file, writer="imagemagick", fps=24)
    plt.close()
    print("Density plot created.")
    return ani


# animate space and number of agents over time together


def plot_space_pop(
    model_data=None, agent_data=None, params=None, steps=100, file="space_pop.gif"
):
    print("Creating spatial and population plot...")

    if model_data is None or "columns" not in dir(model_data):
        raise ValueError("model_data must be a pandas DataFrame.")
    if agent_data is None or "columns" not in dir(agent_data):
        raise ValueError("agent_data must be a pandas DataFrame.")

    if steps > agent_data.Step.max():
        steps = agent_data.Step.max()

    agent_types = agent_data.AgentType.unique()
    marker = ["o", "^", "s", "D", "v", "*"]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_pop = fig.add_subplot(gs[0, 0])
    ax_space = fig.add_subplot(gs[0, 1])

    # Static: Population plot
    lines = [
        ax_pop.plot([], [], label=agent_type)[0]
        for agent_type in model_data.columns
        if agent_type != "Step"
    ]
    ax_pop.set_xlim(0, steps)
    ax_pop.set_ylim(0, model_data.iloc[:, 1:].max().max())
    ax_pop.set_title("Number of agents over time")
    ax_pop.set_xlabel("Time")
    ax_pop.set_ylabel("Number of agents")
    ax_pop.legend()

    # Static: Space plot
    scatters = [
        ax_space.scatter(
            [], [], alpha=0.6, s=25, label=agent, marker=marker[i % len(marker)]
        )
        for i, agent in enumerate(agent_types)
    ]
    ax_space.set_xlim(0, agent_data.x.max())
    ax_space.set_ylim(0, agent_data.y.max())
    ax_space.set_aspect("equal")
    ax_space.set_xticks([])
    ax_space.set_yticks([])
    ax_space.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Parameters text
    text = ", ".join(
        f"{key}={val}"
        for key, val in (params or {}).items()
        if key not in {"model", "progress", "info"}
    )
    fig.text(
        0.5,
        0.05,
        text,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5, pad=10),
    )
    fig.subplots_adjust(bottom=0.25)

    def update(frame):
        current_space = agent_data[agent_data.Step == frame]
        current_model = model_data[model_data.Step < frame]

        for i, agent in enumerate(agent_types):
            data = current_space[current_space.AgentType == agent]
            scatters[i].set_offsets(data[["x", "y"]].values)

        for i, line in enumerate(lines):
            line.set_data(
                current_model.Step.values, current_model.iloc[:, i + 1].values
            )

        ax_pop.set_title(f"Step {frame}")
        ax_space.set_title(f"Step {frame}")
        return scatters + lines

    ani = animation.FuncAnimation(fig, update, frames=range(1, steps), interval=42)
    ani.save(file, writer="imagemagick", fps=24)
    plt.close()
    print("Spatial and population plot created.")
    return ani
