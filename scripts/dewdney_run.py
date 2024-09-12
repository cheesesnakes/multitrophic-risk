# import classes and dependencies

from dewdney import model_1
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# parameters

# grid size
width = 50 
height = 50

# number of agents
predator = 20
prey = 20

# duration of simulation
steps = 1000

# risk
prey_info = True
predator_info = True

# agent parameters
s_breed = 20 # shark breeding age
s_energy = 50 # shark starting energy
f_breed = 5 # fish breeding age
f_energy = 1 # fish energy value

# Run model

m = model_1(predator = predator, prey=prey, prey_info=prey_info, predator_info=predator_info, width=width, height=height, s_breed=s_breed, s_energy=s_energy, f_breed=f_breed, f_energy=f_energy)

m.run_model(steps)

## visualize the model

## get number of prey, predator and resource agents over time

data = m.count.get_model_vars_dataframe()

prey = data['Prey']
predator = data['Predator']

## plot the number of prey, predator and resource agents over time

fig = plt.figure(figsize=(10, 5))

plt.plot(prey, label='Prey', color='green')
plt.plot(predator, label='Predator', color='red')

plt.xlabel('Time')
plt.ylabel('Number of agents')
plt.title('Number of agents over time')

plt.legend()

# save the plot

plt.savefig('dewdney_pop.png')

## plot gif of spatial distribution of agents over time

from matplotlib import animation
from matplotlib.pyplot import colorbar


## get the spatial data

spatial_data = m.spatial.get_agent_vars_dataframe()

## convert index to columns

spatial_data.reset_index(inplace=True)

## create a function to animate the spatial distribution of agents over time

def animate(m = m, spatial_data = spatial_data, steps = steps):
    
    fig, ax = plt.subplots()
    
    ims = []
    
    for i in range(1, steps, 1):
        
        spatial_data_time = spatial_data[spatial_data.Step == i]
        
        grid = np.zeros((m.width, m.height))
        
        for index, row in spatial_data_time.iterrows():
            
            x = row['x']
            y = row['y']
                        
            if row['AgentType'] == 'Prey':
                grid[x][y] = 10
            elif row['AgentType'] == 'Predator':
                grid[x][y] = 20
            else:
                grid[x][y] = 0
            
        im = ax.imshow(grid, interpolation='nearest')       
        
        text = plt.text(1, -2, f'Step {i}', fontsize=12, color='black')
        
        ims.append([im, text])
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    
    return ani    
    


    
images = []

# filter spatial data for agent type

ani = animate(m = m, spatial_data = spatial_data, steps = steps)

ani.save('dewdney_space.gif', writer='imagemagick', fps=5) 
        
        
        
    
    
    
