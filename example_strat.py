from functions.runner import model_run

# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
    'progress': True,
    'info' : False,    
    'limit' : 50*50*4,
    'stop' : True,
    # model parameters
    'width': 100,
    'height': 100,
    
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

strategies = ['Both Naive', 'Predator Informed', 'Prey Informed', 'Both Informed']

for strategy in strategies:
    
    # set info parameters
    
    if strategy == 'Both Naive':
        
        kwargs['prey_info'] = False
        kwargs['predator_info'] = False
        
    elif strategy == 'Predator Informed':
        
        kwargs['prey_info'] = False
        kwargs['predator_info'] = True
    elif strategy == 'Prey Informed':
        
        kwargs['prey_info'] = True
        kwargs['predator_info'] = False
        
    elif strategy == 'Both Informed':
            
            kwargs['prey_info'] = True
            kwargs['predator_info'] = True
            
    else:
        
        print('Invalid Strategy')
        
        break

    # run the model
    m = model_run(**kwargs, steps=steps)

    # save data

    model_data = m.count.get_model_vars_dataframe()
    # set name for index column
    model_data.index.name = 'Step'
    model_data.to_csv(f'data_model_{kwargs['model']}_{strategy}.csv')
    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f'data_agents_{kwargs['model']}_{strategy}.csv')