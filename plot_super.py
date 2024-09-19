from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy, plot_nnd
import pandas as pd

# load data
params = {
    
    # model to run
    'model': 'super',
    'progress': True,
    'info' : False,    
    'limit' : 50*50*4,
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    'super' : 250,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.5, # max birth rate
    'f_die': 0.1,
    'f_max': 2500,
    'risk_cost': 0,
    'f_super_risk': False,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    's_super_risk': True,
    
    ## apex predator traits
    
    'super_target': 2,
    'super_lethality': 1
}

steps = 1000

targets = ['prey', 'predator', 'both']
lethality = ['lethal', 'non-lethal']

for target in targets:
    
    if target == 'prey':
        params['super_target'] = 1
        params['f_super_risk'] = True
        params['s_super_risk'] = False
    elif target == 'predator':
        params['super_target'] = 2
        params['f_super_risk'] = False
        params['s_super_risk'] = True
    else:
        params['super_target'] = 12
        params['f_super_risk'] = True
        params['s_super_risk'] = True
        
    for leth in lethality:
        
        if leth == 'lethal':
            
            params['super_lethality'] = 1
        
        else:
            
            params['super_lethality'] = 0

        # load data
        
        model_data = pd.read_csv(f'data_model_{params["model"]}_{target}_{leth}.csv')
        agent_data = pd.read_csv(f'data_agents_{params["model"]}_{target}_{leth}.csv')
        
        print(f"Plotting model with {target} as target and {leth} super predator")
        
        # plot the number of agents over time

        plot_pop(model_data=model_data, params=params, file=f'pop_super_{target}_{leth}.png', steps=steps)

        # plot the age distribution of agents

        plot_age(agent_data=agent_data, file=f'age_super_{target}_{leth}.png', steps=steps)

        # plot the energy distribution of agents

        plot_energy(agent_data=agent_data, file=f'energy_super_{target}_{leth}.png', steps=steps)

        # plot the distance distribution of agents

        #plot_dist(agent_data=agent_data, file=f'dist_super_{target}_{leth}.png', steps=steps)

        # plot the nearest neighbour distance distribution of agents

        plot_nnd(agent_data=agent_data, file=f'nnd_super_{target}_{leth}.png', steps=steps)

        # plot density of agents

        plot_density(spatial_data=agent_data, file=f'density_super_{target}_{leth}.gif', steps=steps)

        # plot spatial distribution of agents

        plot_space(agent_data=agent_data, file=f'space_super_{target}_{leth}.gif', steps=steps)

        # plot spatial distribution of agents with population size

        plot_space_pop(agent_data=agent_data, model_data=model_data, params=params,
                   file=f'space_pop_super_{target}_{leth}.gif', steps=steps)