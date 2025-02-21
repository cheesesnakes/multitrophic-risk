from functions.runner import model_run
from functions.runner import plot_pop, plot_space
import pandas as pd

# using default parameters built into the model

kwargs = {
    
    # model to run
    'progress': True,
    'info' : False,  
    'limit' : 100000,
    
    # model parameters
    'width': 100,
    'height': 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1,
    'f_max': 10,
    'risk_cost': 0.01,
    'f_steps': 1,
    
    ## predator traits
    'predator_info': True,
    's_max': 5,
    's_breed': 0.15,
    's_die': 0.1,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 1,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_max': 10,
    'a_breed': 0.25,
    'a_die': 0.01,
    'a_lethality': 0.15,
    'a_steps': 1,
}

steps = 1000

# run the model

def run_debug(**kwargs):
    
    # run model 
    
    m = model_run(**kwargs, steps=steps)
    
    # save data
    
    model_data = m.count.get_model_vars_dataframe()
    # set name for index column
    model_data.index.name = 'Step'
    model_data.to_csv(f'output/debug/results/data_model_{kwargs["model"]}.csv')
    
    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f'output/debug/results/data_agents_{kwargs["model"]}.csv')
    
    #plot population
    
    plot_pop(model_data=model_data, params = kwargs, file = f'output/debug/plots/pop_{kwargs["model"]}.png', steps=steps)
    
    #plot space
    
    agent_data = pd.read_csv(f'output/debug/results/data_agents_{kwargs["model"]}.csv')
    
    plot_space(agent_data=agent_data, file = f'output/debug/plots/space_{kwargs["model"]}.gif', steps=steps)

if __name__ == '__main__':
    
    # run model with prey only
    
    kwargs['model'] = 'debug-prey-only'
    kwargs['predator'] = 0
    kwargs['apex'] = 0
    kwargs['prey'] = 500
    kwargs['super'] = 0 
    
    run_debug(**kwargs)
    
    # run model with predator only
    
    kwargs['model'] = 'debug-predator-only'
    kwargs['predator'] = 500
    kwargs['apex'] = 0
    kwargs['prey'] = 0
    kwargs['super'] = 0
    
    run_debug(**kwargs)
    
    # run model with apex only
    
    kwargs['model'] = 'debug-apex-only'
    kwargs['predator'] = 0
    kwargs['apex'] = 500
    kwargs['prey'] = 0  
    kwargs['super'] = 0
    
    run_debug(**kwargs)
    
    # run model with super only
    
    kwargs['model'] = 'debug-super-only'
    kwargs['predator'] = 0
    kwargs['apex'] = 0
    kwargs['prey'] = 0
    kwargs['super'] = 500
    
    run_debug(**kwargs)
    