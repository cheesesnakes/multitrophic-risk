from functions.runner import model_run

# set model parameters

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
    'prey': 500,
    'apex' : 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0,
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

steps = 100

# run the model
m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
# set name for index column
model_data.index.name = 'Step'
model_data.to_csv(f'data_model_{kwargs['model']}.csv')
agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'data_agents_{kwargs['model']}.csv')