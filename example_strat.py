from functions.runner import model_run
from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy
import pandas as pd
# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
    'progress': True,
    'info' : False,    
    'limit' : 10000,
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.6, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 10,
    'risk_cost': 0.01,
    'f_steps': 1,

    ## predator traits
    'predator_info': True,
    's_max': 5,
    's_breed': 0.15, # constant
    's_die': 0.1,
    's_lethality': 0.5,
    's_steps': 1,
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
    model_data.to_csv(f'output/strategies/results/data_model_{kwargs["model"]}_{strategy}.csv')
    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f'output/strategies/results/data_agents_{kwargs["model"]}_{strategy}.csv')
        
        # load data

    model_data = pd.read_csv(f'output/strategies/results/data_model_{kwargs["model"]}_{strategy}.csv')
    agent_data = pd.read_csv(f'output/strategies/results/data_agents_{kwargs["model"]}_{strategy}.csv')

    # plot the number of agents over time

    plot_pop(model_data=model_data, params = kwargs, file = f'output/strategies/plots/pop_{kwargs["model"]}_{strategy}.png', steps=steps)

    # plot density of agents

    plot_density(spatial_data=agent_data, file = f'output/strategies/plots/density_{kwargs["model"]}_{strategy}.gif', steps=steps)

    # plot spatial distribution of agents

    plot_space(agent_data=agent_data,file = f'output/strategies/plots/space_{kwargs["model"]}_{strategy}.gif', steps=steps)

    # plot spatial distribution of agents with population size

    plot_space_pop(agent_data=agent_data, model_data=model_data, params=kwargs,
                file = f'output/strategies/plots/space_pop_{kwargs["model"]}_{strategy}.gif', steps=steps)
