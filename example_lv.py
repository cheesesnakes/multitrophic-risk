from functions.runner import model_run

# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
    'progress': True,
    'info' : False,    
    'limit' : 50*50*4,
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    
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
}

steps = 1000

# run the model
m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
model_data.to_csv(f'data_model_{kwargs['model']}.csv')
agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'data_agents_{kwargs['model']}.csv')