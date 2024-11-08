from matplotlib.pylab import plot
from functions.runner import model_run
from functions.runner import plot_pop, plot_space
import pandas as pd

# using default parameters built into the model

# run model with only prey

kwargs = {
    
    # model to run
    'model': 'apex',
    'progress': True,
    'info' : False,  
    'limit' : 10000,
    # model parameters
    'width': 100,
    'height': 100,
    
    # number of agents to start with
    'predator': 0,
    'prey': 500,
    'apex' : 0,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 5,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 10,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 20,
}

steps = 1000

# run the model

m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
# set name for index column
model_data.index.name = 'Step'
model_data.to_csv(f'output/debug/results/data_model_{kwargs["model"]}_debug-prey-only.csv')

agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-prey-only.csv')

plot_pop(model_data=model_data, params = kwargs, file = 'output/debug/plots/pop_debug-prey-only.png', steps=steps)
agent_data = pd.read_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-prey-only.csv')
plot_space(agent_data=agent_data, file = 'output/debug/plots/space_debug-prey-only.gif', steps=steps)

# run model with only predators

kwargs = {
    
    # model to run
    'model': 'apex',
    'progress': True,
    'info' : False,  
    'limit' : 10000,
    # model parameters
    'width': 100,
    'height': 100,
    
    # number of agents to start with
    'predator': 500,
    'prey': 0,
    'apex' : 0,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 5,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 10,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 20,
}

steps = 1000

# run the model

m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
# set name for index column
model_data.index.name = 'Step'
model_data.to_csv(f'output/debug/results/data_model_{kwargs["model"]}_debug-predator-only.csv')

agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-predator-only.csv')

plot_pop(model_data=model_data, params = kwargs, file = 'output/debug/plots/pop_debug-predator-only.png', steps=steps)
agent_data = pd.read_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-predator-only.csv')
plot_space(agent_data=agent_data, file = 'output/debug/plots/space_debug-predator-only.gif', steps=steps)

# run model with only apex predators

kwargs = {
    
    # model to run
    'model': 'apex',
    'progress': True,
    'info' : False,  
    'limit' : 10000,
    # model parameters
    'width': 100,
    'height': 100,
    
    # number of agents to start with
    'predator': 0,
    'prey': 0,
    'apex' : 500,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 5,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 10,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 20,
}

steps = 1000

# run the model

m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
# set name for index column
model_data.index.name = 'Step'
model_data.to_csv(f'output/debug/results/data_model_{kwargs["model"]}_debug-apex-only.csv')

agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-apex-only.csv')

plot_pop(model_data=model_data, params = kwargs, file = 'output/debug/plots/pop_debug-apex-only.png', steps=steps)
agent_data = pd.read_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-apex-only.csv')
plot_space(agent_data=agent_data, file = 'output/debug/plots/space_debug-apex-only.gif', steps=steps)

# run model with super only

kwargs = {
    
    # model to run
    'model': 'super',
    'progress': True,
    'info' : False,  
    'limit' : 10000,
    # model parameters
    'width': 100,
    'height': 100,
    
    # number of agents to start with
    'predator': 500,
    'prey': 0,
    'apex' : 0,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 5,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 10,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 20,
}

steps = 1000

# run the model

m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
# set name for index column
model_data.index.name = 'Step'
model_data.to_csv(f'output/debug/results/data_model_{kwargs["model"]}_debug-super-only.csv')

agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-super-only.csv')

plot_pop(model_data=model_data, params = kwargs, file = 'output/debug/plots/pop_debug-super-only.png', steps=steps)
agent_data = pd.read_csv(f'output/debug/results/data_agents_{kwargs["model"]}_debug-super-only.csv')
plot_space(agent_data=agent_data, file = 'output/debug/plots/space_debug-super-only.gif', steps=steps)