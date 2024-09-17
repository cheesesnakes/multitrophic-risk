from functions.runner import model_run

# set model parameters

kwargs = {
    
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

# run the model
m = model_run(**kwargs, steps=steps)

# save data

model_data = m.count.get_model_vars_dataframe()
model_data.to_csv(f'data_model_{kwargs['model']}.csv')
agent_data = m.spatial.get_agent_vars_dataframe()
agent_data.to_csv(f'data_agents_{kwargs['model']}.csv')