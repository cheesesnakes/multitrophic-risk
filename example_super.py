from functions.runner import model_run
# set model parameters

kwargs = {
    
    # model to run
    'model': 'super',
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
    'super' : 250,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0,
    'f_super_risk': False,
    'f_steps': 5,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_die': 0.01,
    's_lethality': 0.5,
    's_super_risk': True,
    's_steps': 10,
    
    ## apex predator traits
    
    'super_target': 2,
    'super_lethality': 1,
    'super_steps': 20
}

steps = 1000

targets = ['prey', 'predator', 'both']
lethality = ['lethal', 'non-lethal']

for target in targets:
    
    if target == 'prey':
        kwargs['super_target'] = 1
        kwargs['f_super_risk'] = True
        kwargs['s_super_risk'] = False
    elif target == 'predator':
        kwargs['super_target'] = 2
        kwargs['f_super_risk'] = False
        kwargs['s_super_risk'] = True
    else:
        kwargs['super_target'] = 12
        kwargs['f_super_risk'] = True
        kwargs['s_super_risk'] = True
        
    for leth in lethality:
        
        if leth == 'lethal':
            
            kwargs['super_lethality'] = 1
        
        else:
            
            kwargs['super_lethality'] = 0

        print(f"Running model with {target} as target and {leth} super predator")
        
        # run the model
        m = model_run(**kwargs, steps=steps)

        # save data

        model_data = m.count.get_model_vars_dataframe()
        
        # set name for index column
        model_data.index.name = 'Step'
        model_data.to_csv(f'data_model_{kwargs["model"]}_{target}_{leth}.csv')
        
        agent_data = m.spatial.get_agent_vars_dataframe()
        agent_data.to_csv(f'data_agents_{kwargs["model"]}_{target}_{leth}.csv')