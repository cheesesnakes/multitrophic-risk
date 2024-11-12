# import classes and dependencies

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import warnings
from functions.model import model

warnings.filterwarnings("ignore")

def model_run(steps = 50, **kwargs):

    # Run model

    print('Running model...')
    
    m = model(**{k: v for k, v in kwargs.items() if k != 'model'})

    m.run_model(steps, progress=kwargs.get('progress', False), info=kwargs.get('info', False), limit = kwargs.get('limit', 10000), stop = kwargs.get('stop', False))

    print('Model run complete.')
    
    return m

## visualize the model

def plot_pop(model_data = None, params = {}, file = 'model_pop.png', steps = 50):

    print('Creating population plot...')
    
    # Check if model_data is None or not a DataFrame
    if model_data is None or not hasattr(model_data, 'columns'):
        raise ValueError("model_data must be a pandas DataFrame with a 'columns' attribute.")
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
    # create plot
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # plot the number of agents over time
    
    for i in list(model_data.columns):
        
        if i == 'Step':
            continue
        
        ax.plot(model_data[i], label=i)
        
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of agents')
    ax.set_title('Number of agents over time')
    ax.legend()
    ax.set_xlim(0, steps)
    
    # print parameters below the plot
    if isinstance(params, dict):  # Added check for dict type
        text = [f'{key} = {params[key]}' for key in params if key != 'model' and key != 'progress' and key != 'info']
        text = ', '.join(text)
        fig.text(0.5, 0.05, text, ha="center", fontsize=10, wrap=True,
                 bbox={"facecolor": "white", "alpha": 0.5, "pad": 10})
        
    fig.subplots_adjust(bottom=0.25)
    
    # save the plot
    
    plt.savefig(file)

    print('Population plot created.')

## create a function to animate the spatial distribution of agents over time

def plot_space(agent_data = None, steps = 100, file = 'space.gif'):
    
    print('Creating spatial plot...')
    
    ## get the spatial data

    spatial_data = agent_data

    ## convert index to columns
    
    if steps > spatial_data.Step.max():
        
        steps = spatial_data.Step.max()
        
    # define plot parameters
    
    colors = ['blue', 'red', 'green']
    marker = ['o', '^', 's']
    agent_types = spatial_data.AgentType.unique()
    
    # create plot
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # create list of images

    spaces = [ax.scatter([], [], label=agent, alpha=0.6, s=50,
                         cmap="tab10", marker=marker[i]) for i, agent in enumerate(agent_types)]
    
    ax.set_xlim(0, spatial_data.x.max())
    ax.set_ylim(0, spatial_data.y.max())
    # remove ticks and labels
    ax.set_xticks([])   
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title('Spatial distribution of agents')
    # set legend above the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

    def update(frame):
        
        for i, space in enumerate(spaces):
            
            space_data = spatial_data[(spatial_data.Step == frame) & (spatial_data.AgentType == agent_types[i])]
            
            space.set_offsets(space_data[['x', 'y']])
        
        # set title
        ax.set_title(f'Step {frame}')
        return spaces

    ani = animation.FuncAnimation(fig, update, frames=range(1, steps-1), blit=False, interval=42)
    
    # save the animation
    
    ani.save(file, writer='imagemagick', fps=24)
    
    print('Spatial plot created.')
    
    return ani    
    
## create a function to animate the density of agents over time

def plot_density(spatial_data = None, steps = 100, file = 'density.gif'):
    
    print('Creating density plot...')
    
    ## convert index to columns

    spatial_data.reset_index(inplace=True)
    
    if steps > spatial_data.Step.max():
        
        steps = spatial_data.Step.max()
        
    # loop over types of agents
    
    agent_types = spatial_data.AgentType.unique()
    
    # create plot
    
    fig, axs = plt.subplots(1, len(agent_types), figsize=(len(agent_types)*5, 5))
    
    # create list of images
    
    ims = []
    
    for i in range(1, steps, 1):
        
        spatial_data_time = spatial_data[spatial_data.Step == i]
        
        # temporarily store images
        
        step_images = []
        
        for j, agent in enumerate(agent_types):
            
            grid = np.zeros((spatial_data.x.max()+1, spatial_data.y.max()+1))
            
            spatial_data_agent = spatial_data_time[spatial_data_time.AgentType == agent]
            
            for index, row in spatial_data_agent.iterrows():
                
                x = row['x']
                y = row['y']
                
                grid[x][y] += 1
                
            im = fig.axes[j].imshow(grid, interpolation='nearest')
            
            text = fig.axes[j].text(1, -2, f'{agent} at Step {i}', fontsize=12, color='black')
            
            step_images.append(im)
            step_images.append(text)
            
        ims.append(step_images)
        
    ani = animation.ArtistAnimation(fig, ims, interval=42, blit=False, repeat_delay=1000)
    
    # save the animation
    
    ani.save(file, writer='imagemagick', fps=24)
    
    print('Density plot created.')
    
    return ani

# animate space and number of agents over time together

def plot_space_pop(model_data=None, agent_data=None, params = None, steps=100, file='space_pop.png'):
    
    print('Creating spatial and population plot...')
    
    # Check if model_data is None or not a DataFrame
    if model_data is None or not hasattr(model_data, 'columns'):
        raise ValueError("model_data must be a pandas DataFrame with a 'columns' attribute.")
    
    # Check if agent_data is None or not a DataFrame
    if agent_data is None or not hasattr(agent_data, 'columns'):
        raise ValueError("agent_data must be a pandas DataFrame with a 'columns' attribute.")
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
    ## get the spatial data

    spatial_data = agent_data

    ## convert index to columns
    
    if steps > spatial_data.Step.max():
        
        steps = spatial_data.Step.max()
        
    # define plot parameters
    
    marker = ['o', '^', 's']
    agent_types = spatial_data.AgentType.unique()
    
    # create plot
    
    fig = plt.figure(figsize=(12,6))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    # create list of images

    spaces = [axs[1].scatter([], [], label=agent, alpha=0.6, s=25,
                         cmap="tab10", marker=marker[i]) for i, agent in enumerate(agent_types)]
    
    axs[1].set_xlim(0, spatial_data.x.max())
    axs[1].set_ylim(0, spatial_data.y.max())
    #set size
    axs[1].set_aspect('equal')
    # remove ticks and labels
    axs[1].set_xticks([])   
    axs[1].set_yticks([])
    axs[1].set_xlabel(None)
    axs[1].set_ylabel(None)
    axs[1].set_title('Spatial distribution of agents')
    # set legend above the plot
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
    
    # plot the number of agents over time
    
    agent_types = model_data.columns
    #remove Step
    agent_types = agent_types[1:]
    
    lines = [axs[0].plot([], [], label=i)[0] for i in list(model_data.columns) if i != 'Step']
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Number of agents')
    axs[0].set_title('Number of agents over time')
    axs[0].legend()
    axs[0].set_xlim(0, steps)
    axs[0].set_ylim(0, model_data[agent_types].max().max())
    
    # print parameters below the plot
    
    text = [f'{key} = {params[key]}' for key in params if key != 'model' and key != 'progress' and key != 'info']
    text = ', '.join(text)
    
    fig.text(0.5, 0.05, text, ha="center", fontsize=10, wrap=True,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 10})
    
    fig.subplots_adjust(bottom=0.25)
    
    def update(frame):
        
        for i, space in enumerate(spaces):
            
            space_data = spatial_data[(spatial_data.Step == frame) & (spatial_data.AgentType == agent_types[i])]
            
            space.set_offsets(space_data[['x', 'y']])
        
        model_data_time = model_data[model_data.Step < frame]
        
        for i, line in enumerate(lines):
            
            line.set_data(range(frame), model_data_time[agent_types[i]])
            
        # set title
        
        axs[0].set_title(f'Step {frame}')
        axs[1].set_title(f'Step {frame}')
        
        return spaces + lines
            
        
    ani = animation.FuncAnimation(fig, update, frames=range(1, steps-1), blit=False, interval=42)
    
    # save the animation
    
    ani.save(file, writer='imagemagick', fps=24)
    
    print('Spatial and population plot created.')
    
    return ani

# plot mean age for each type of agent over time

def plot_age(agent_data=None, steps=100, file='mean_age.png'):
    
    print('Creating age plot...')
    
    age_data = agent_data[['Step', 'AgentType', 'Age']]
    
    # calculate mean age for each type of agent
    
    mean_age = age_data.groupby(['Step', 'AgentType']).mean().reset_index()
    
    # get sd of age for each type of agent
    
    sd_age = age_data.groupby(['Step', 'AgentType']).std().reset_index()
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
    # create plot
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # plot the mean age of agents over time
    
    for i in mean_age.AgentType.unique():
        
        if i == 'Super':
            continue
        
        agent_data = mean_age[mean_age.AgentType == i]
        sd = sd_age[sd_age.AgentType == i]
        
        ax.plot(agent_data.Step, agent_data.Age, label=i)
        ax.fill_between(sd.Step, agent_data.Age - sd.Age, agent_data.Age + sd.Age, alpha=0.2)
        
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean age')
    ax.set_title('Mean age of agents over time')
    ax.legend()
    ax.set_xlim(0, steps)
    
    # save the plot
    
    plt.savefig(file)
    
    print('Age plot created.')
    
    return fig

# plot mean energy for predator and/or apex predator over time

def plot_energy(agent_data=None, steps=100, file='mean_energy.png'):
    
    print('Creating energy plot...')
    
    energy_data = agent_data[['Step', 'AgentType', 'Energy']]
    
    # calculate mean energy for each type of agent
    
    mean_energy = energy_data.groupby(['Step', 'AgentType']).mean().reset_index()
    sd_energy = energy_data.groupby(['Step', 'AgentType']).std().reset_index()
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
    # create plot
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # plot the mean energy of agents over time
    
    for i in mean_energy.AgentType.unique():
        
        agent_data = mean_energy[mean_energy.AgentType == i]
        sd = sd_energy[sd_energy.AgentType == i]
        
        ax.plot(agent_data.Step, agent_data.Energy, label=i)
        ax.fill_between(sd.Step, agent_data.Energy - sd.Energy, agent_data.Energy + sd.Energy, alpha=0.2)
        
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean energy')
    ax.set_title('Mean energy of agents over time')
    ax.legend()
    ax.set_xlim(0, steps)
    
    # save the plot
    
    plt.savefig(file)
    
    print('Energy plot created.')
    
    return

# plot average distance between agents over time

from scipy.spatial.distance import pdist, cdist

def plot_nnd(agent_data=None, steps=100, file='mean_nnd.png'):
    print('Creating nearest neighbor distance plot...')
    
    nnd = []
    sd = []
    
    # caluclate nearest neighbor distance for the following agent type pairs
    
    pairs = [('Prey', 'Prey'), ('Predator', 'Predator'), ('Prey', 'Predator')]
    
    nnd = np.zeros((len(pairs), steps))
    sd = np.zeros((len(pairs), steps))
    
    for j, pair in enumerate(pairs):
        
        # filter data for the pair of agent types
        
        agent_data_pair = agent_data[(agent_data.AgentType == pair[0]) | (agent_data.AgentType == pair[1])]
        
        for i in range(1, steps + 1):  # Correcting range to include 'steps'
            
            if agent_data_pair[agent_data_pair.Step == i].empty:
                continue
            
            print(f'Calculating nearest neighbor distance for {pair} at step {i}')
            
            agent_data_i = agent_data_pair[agent_data_pair.Step == i]  # Filter data at each time step
            
            if pair[0] != pair[1]:
                
                cords_0 = agent_data_i[agent_data_i.AgentType == pair[0]][['x', 'y']].values
                cords_1 = agent_data_i[agent_data_i.AgentType == pair[1]][['x', 'y']].values
                
                # Calculate all pairwise distances using pdist from scipy
                
                dist_matrix = cdist(cords_0, cords_1)  # Pairwise distances as a 1D array
                
            else:     
            
                # Extract x and y coordinates of all agents at this time step
                coords = agent_data_i[['x', 'y']].values  # shape (num_agents, 2)
                
                # Calculate all pairwise distances using pdist from scipy
                dist_matrix = pdist(coords)  # Pairwise distances as a 1D array
            
            # Calculate mean and standard deviation of pairwise distances
            mean_nnd = np.mean(dist_matrix)
            sd_nnd = np.std(dist_matrix)
        
            nnd[j, i-1] = mean_nnd
            sd[j, i-1] = sd_nnd
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(range(1, steps + 1), nnd[0], label='Prey-Prey')
    ax.fill_between(range(1, steps + 1), nnd[0] - sd[0], nnd[0] + sd[0], alpha=0.2, label='±1 SD')
    ax.plot(range(1, steps + 1), nnd[1], label='Predator-Predator')
    ax.fill_between(range(1, steps + 1), nnd[1] - sd[1], nnd[1] + sd[1], alpha=0.2, label='±1 SD')
    ax.plot(range(1, steps + 1), nnd[2], label='Prey-Predator')
    ax.fill_between(range(1, steps + 1), nnd[2] - sd[2], nnd[2] + sd[2], alpha=0.2, label='±1 SD')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean distance between agents')
    ax.set_xlim(1, steps)
    ax.legend()
    
    # Save the plot
    plt.savefig(file)
    
    print('Nearest neighbor distance plot created.')
    
    return fig

# plot distane traveled by agents over time

def plot_distance(agent_data=None, steps=100, file='distance.png'):
    
    print('Creating distance plot...')
    
    # calculate mean distance for each type of agent
    
    distance_data = agent_data.groupby(['UniqueID']).sum().reset_index()
    
    print(distance_data)
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
    # create plot
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # plot the mean distance of agents over time
    
    # boxplot of agent types and distance
    
    ax.boxplot([distance_data[distance_data.AgentType == 'Prey']['DistanceTravelled'],
                distance_data[distance_data.AgentType == 'Predator']['DistanceTravelled']],
               labels=['Prey', 'Predator'])
    
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Distance travelled')
    
    # save the plot
    
    plt.savefig(file)
    
    print('DistanceTravelled plot created.')
    
    return
