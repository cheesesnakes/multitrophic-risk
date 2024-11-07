from functions.runner import model_run

# set model parameters

kwargs = {
    
    'model': 'lv',
    # model to run
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
    'f_steps': 5, # number of steps to move
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_steps': 10, # number of steps to move,
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