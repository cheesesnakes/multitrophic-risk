# import classes and dependencies

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.pyplot import colorbar
import numpy as np

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

def plot_pop(m = None, file = 'model_pop.png'):

    data = m.count.get_model_vars_dataframe()
    
    params = m.kwargs
    
    ## remove columns whose names contain _
    
    data = data[[col for col in data.columns if '_' not in col]]
    
    ## plot the number of prey, predator and resource agents over time

    print('Plotting number of agents over time...')

    fig = plt.figure(figsize=(10,6))
    
    colors = ['blue', 'red','green']
    
    c = 0

    for i in data:
        
        d = data[i]

        plt.plot(d, label=i, color = colors[c])

        c += 1

    plt.xlabel('Time')
    plt.ylabel('Number of agents')  
    plt.title('Number of agents over time')
    
    # print praameters below the plot
    text = [f'{key} = {params[key]}' for key in params if key != 'model' and key != 'progress' and key != 'info']
    
    # make the text a single string
    
    text = ', '.join(text)
    
    plt.figtext(0.5, 0.1, text, ha="center", fontsize=10, wrap=True,
                bbox={"facecolor":"white", "alpha":0.5, "pad":10,})

    fig.subplots_adjust(bottom=0.25)
    
    plt.legend()

    # save the plot

    plt.savefig(file)


## create a function to animate the spatial distribution of agents over time

def plot_space(m = None, steps = 100, duration = 60, file = 'space.gif'):
    
    ## get the spatial data

    spatial_data = m.spatial.get_agent_vars_dataframe()

    ## convert index to columns

    spatial_data.reset_index(inplace=True)
    
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