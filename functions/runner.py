# import classes and dependencies

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.pyplot import colorbar
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

def model_run(model = "lv", steps = 50, **kwargs):

    if model == "lv":
        
        from functions.models.lv_model import model_1
        
    elif model == "super":

        from functions.models.super import model_1

    elif model == "apex":

        from functions.models.apex import model_1

    else:

        raise ValueError("Invalid model. Available models are lv, super and apex.")
        

    # Run model

    print('Running model...')
    
    m = model_1(**{k: v for k, v in kwargs.items() if k != 'model'})

    m.run_model(steps, progress=kwargs.get('progress', False), info=kwargs.get('info', False))

    print('Model run complete.')
    
    return m

## visualize the model

def plot_pop(model_data = None, params = {}, file = 'model_pop.png', steps = 50):

    # Check if model_data is None or not a DataFrame
    if model_data is None or not hasattr(model_data, 'columns'):
        raise ValueError("model_data must be a pandas DataFrame with a 'columns' attribute.")
    
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


## create a function to animate the spatial distribution of agents over time

def plot_space(m = None, steps = 100, duration = 60, file = 'space.gif'):
    
    ## get the spatial data

    spatial_data = m.spatial.get_agent_vars_dataframe()

    ## convert index to columns

    spatial_data.reset_index(inplace=True)
    
    if steps < spatial_data.Step.max():
        
        steps = spatial_data.Step.max()
        
    # create plot
    
    fig, ax = plt.subplots()
    
    ims = []
    
    for i in range(1, steps, 1):
        
        spatial_data_time = spatial_data[spatial_data.Step == i]
        
        grid = np.zeros((m.width, m.height))
        
        for index, row in spatial_data_time.iterrows():
            
            x = row['x']
            y = row['y']
                        
            if row['AgentType'] == 'Apex' or row['AgentType'] == 'Super':
                grid[x][y] = 30
            elif row['AgentType'] == 'Prey':
                grid[x][y] = 10
            elif row['AgentType'] == 'Predator':
                grid[x][y] = 20
            else:
                grid[x][y] = 0
            
        im = ax.imshow(grid, interpolation='nearest')       
        
        text = plt.text(1, -2, f'Step {i}', fontsize=12, color='black')
        
        ims.append([im, text])
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    
    # save the animation
    
    ani.save(file, writer='imagemagick', fps=steps/duration)
    
    return ani    
    
## create a function to animate the density of agents over time

def plot_density(m = None, duration = 60, steps = 100, file = 'density.gif'):
    
    ## get the spatial data

    spatial_data = m.spatial.get_agent_vars_dataframe()

    ## convert index to columns

    spatial_data.reset_index(inplace=True)
    
    if steps < spatial_data.Step.max():
        
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
            
            grid = np.zeros((m.width, m.height))
            
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
        
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False, repeat_delay=1000)
    
    # save the animation
    
    ani.save(file, writer='imagemagick', fps=steps/duration)
    
    return ani

# animate space and number of agents over time together

def plot_space_pop(m=None, steps=100, duration=60, file='space_pop.png'):
    
    # Retrieve data for population and space
    pop_data = m.count.get_model_vars_dataframe()
    space_data = m.spatial.get_agent_vars_dataframe()
    space_data.reset_index(inplace=True)
    
    if steps < len(pop_data):
        
        steps = len(pop_data)
        
    params = m.kwargs

    # Clean population data
    pop_data = pop_data[[col for col in pop_data.columns if '_' not in col]]

    # Prepare the grid for the two plots
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])  # Two-thirds for population, one-third for space
    
    # Population plot
    ax_pop = fig.add_subplot(gs[0, 0])
    colors = ['blue', 'red', 'green']
    lines = [ax_pop.plot([], [], label=label, color=colors[i])[0] for i, label in enumerate(pop_data.columns)]
    
    ax_pop.set_xlim(0, steps)
    ax_pop.set_ylim(0, max(pop_data.max()) * 1.1)  # Set y-limit based on max values in the population data
    ax_pop.set_xlabel('Time')
    ax_pop.set_ylabel('Number of agents')
    ax_pop.set_title('Number of agents over time')
    ax_pop.legend()

    # Space plot
    ax_space = fig.add_subplot(gs[0, 1])
    
    def init():
        # Initialize population plot
        for line in lines:
            line.set_data([], [])
        # Initialize space plot
        ax_space.clear()
        return lines

    # Create function to update both the population plot and space plot
    def update(frame):
        # Update population plot
        x = np.arange(frame)
        for i, line in enumerate(lines):
            line.set_data(x, pop_data.iloc[:frame, i])

        # Update space plot
        space_data_time = space_data[space_data.Step == frame]
        grid = np.zeros((m.width, m.height))
        
        for index, row in space_data_time.iterrows():
            x = row['x']
            y = row['y']
            
            if row['AgentType'] == 'Apex' or row['AgentType'] == 'Super':
                grid[x][y] = 30
            elif row['AgentType'] == 'Prey':
                grid[x][y] = 10
            elif row['AgentType'] == 'Predator':
                grid[x][y] = 20
            else:
                grid[x][y] = 0

        ax_space.clear()
        ax_space.imshow(grid, interpolation='nearest')
        ax_space.set_title(f'Step {frame}')
        
        return lines

    # Print parameters below the population plot
    text = [f'{key} = {params[key]}' for key in params if key != 'model' and key != 'progress' and key != 'info']
    text = ', '.join(text)
    fig.text(0.5, 0.05, text, ha="center", fontsize=10, wrap=True,
             bbox={"facecolor": "white", "alpha": 0.5, "pad": 10})

    fig.subplots_adjust(bottom=0.25)
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(1, steps-1), init_func=init, blit=False, interval=50)
    
    # Save the animation
    ani.save(file, writer='imagemagick', fps=steps/duration)
    
    return ani
