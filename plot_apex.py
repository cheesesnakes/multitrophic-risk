from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy, plot_nnd, plot_distance
import pandas as pd

# load data

model_data = pd.read_csv('data_model_apex.csv')
agent_data = pd.read_csv('data_agents_apex.csv')

params = {
    
    # model to run
    'model': 'apex',
    'progress': True,
    'info' : False,  
    'limit' : 10000,
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    'apex' : 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.5, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    's_apex_risk': True,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_lethality': 0.15
}

steps = 1000

# plot the number of agents over time

plot_pop(model_data=model_data, params = params, file = 'pop_apex.png', steps=steps)

# plot the age distribution of agents

plot_age(agent_data=agent_data, file = 'age_apex.png', steps=steps)

# plot the energy distribution of agents

plot_energy(agent_data=agent_data, file = 'energy_apex.png', steps=steps)

# plot the distance distribution of agents

#plot_dist(agent_data=agent_data, file = 'dist_apex.png', steps=steps)

# plot the nearest neighbour distance distribution of agents

plot_nnd(agent_data=agent_data, file = 'nnd_apex.png', steps=steps)

# plot density of agents

plot_density(spatial_data=agent_data, file = 'density_apex.gif', steps=steps)

# plot spatial distribution of agents

plot_space(agent_data=agent_data,file = 'space_apex.gif', steps=steps)

# plot spatial distribution of agents with population size

plot_space_pop(agent_data=agent_data, model_data=model_data, params=params,
             file = 'space_pop_apex.gif', steps=steps)