from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy, plot_nnd
import pandas as pd

params = {
    
    # model to run
    'model': 'lv',
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

steps = 1000

strategies = ['Both Naive', 'Predator Informed', 'Prey Informed', 'Both Informed']

for strategy in strategies:
    
    # set info parameters
    
    if strategy == 'Both Naive':
        file = 'data_agents_lv_Both Naive.csv'
        file2 = 'data_model_lv_Both Naive.csv'
        params['prey_info'] = False
        params['predator_info'] = False
    elif strategy == 'Predator Informed':
        file = 'data_agents_lv_Predator Informed.csv'
        file2 = 'data_model_lv_Predator Informed.csv'
        params['prey_info'] = False
        params['predator_info'] = True
    elif strategy == 'Prey Informed':
        file = 'data_agents_lv_Prey Informed.csv'
        file2 = 'data_model_lv_Prey Informed.csv'
        params['prey_info'] = True
        params['predator_info'] = False
    elif strategy == 'Both Informed':
        file = 'data_agents_lv_Both Informed.csv'
        file2 = 'data_model_lv_Both Informed.csv'
        params['prey_info'] = True
        params['predator_info'] = True

    # load data

    model_data = pd.read_csv(file2)
    agent_data = pd.read_csv(file)

    print(f'Plotting {strategy} Strategy....')
    
    # plot the number of agents over time

    plot_pop(model_data=model_data, params = params, file = f'pop_{strategy.replace(" ", "_")}.png', steps=steps)

    # plot the age distribution of agents

    plot_age(agent_data=agent_data, file = f'age_{strategy.replace(" ", "_")}.png', steps=steps)

    # plot the energy distribution of agents

    plot_energy(agent_data=agent_data, file = f'energy_{strategy.replace(" ", "_")}.png', steps=steps)

    # plot the distance distribution of agents

    #plot_dist(agent_data=agent_data, file = 'dist_lv.png')

    # plot the nearest neighbour distance distribution of agents

    plot_nnd(agent_data=agent_data, file = f'nnd_{strategy.replace(" ", "_")}.png', steps=steps)

    # plot density of agents

    plot_density(spatial_data=agent_data, file = f'density_{strategy.replace(" ", "_")}.gif', steps=steps)

    # plot spatial distribution of agents

    plot_space(agent_data=agent_data,file = f'space_{strategy.replace(" ", "_")}.gif', steps=steps)

    # plot spatial distribution of agents with population size

    plot_space_pop(agent_data=agent_data, model_data=model_data, params=params,
                file = f'space_pop_{strategy.replace(" ", "_")}.gif', steps=steps)
