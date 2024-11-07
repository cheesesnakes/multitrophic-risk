from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy, plot_nnd, plot_distance
import pandas as pd

# load data

model_data = pd.read_csv('data_model_lv.csv')
agent_data = pd.read_csv('data_agents_lv.csv')

params = {
    
    # model to run
    'model': 'lv',
    'progress': True,
    'info' : False,    
    'limit' : 10000,
    
    # model parameters
    'width': 100,
    'height': 100,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
}

steps = 1000

# plot the number of agents over time

plot_pop(model_data=model_data, params = params, file = 'pop_lv.png', steps=steps)

# plot the age distribution of agents

plot_age(agent_data=agent_data, file = 'age_lv.png', steps=steps)

# plot the energy distribution of agents

plot_energy(agent_data=agent_data, file = 'energy_lv.png', steps=steps)

# plot the distance distribution of agents

#plot_dist(agent_data=agent_data, file = 'dist_lv.png')

# plot the nearest neighbour distance distribution of agents

plot_nnd(agent_data=agent_data, file = 'nnd_lv.png', steps=steps)

# plot density of agents

plot_density(spatial_data=agent_data, file = 'density_lv.gif', steps=steps)

# plot spatial distribution of agents

plot_space(agent_data=agent_data,file = 'space_lv.gif', steps=steps)

# plot spatial distribution of agents with population size

plot_space_pop(agent_data=agent_data, model_data=model_data, params=params,
             file = 'space_pop_lv.gif', steps=steps)