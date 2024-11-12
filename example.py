from sys import argv
from functions.runner import model_run
from functions.runner import plot_pop, plot_density, plot_space, plot_space_pop, plot_age, plot_energy
import pandas as pd
from sys import argv

# set model parameters

kwargs = {
    
    # model to run
    'progress': True,
    'info' : False,    
    'limit' : 10000,
    'width': 100,
    'height': 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 20,

    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1, # constant
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 40,

    ## apex predator traits

    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1, # constant
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 80,

    # super predator traits

    'super_target': 2,
    'super_lethality': 1,
    'super_steps': 20,
    'super_steps': 80,
}

steps = 1000

def  set_params(kwargs = kwargs, model = 'lv'):
    
    if model == 'apex':
        
        kwargs['model'] = 'apex'
        kwargs['predator'] = 500
        kwargs['prey'] = 500
        kwargs['apex'] = 100
    
    elif model == 'super':
        
        kwargs['model'] = 'super'
        kwargs['predator'] = 500
        kwargs['prey'] = 500
        kwargs['super'] = 50
    
    else:
        
        kwargs['model'] = 'lv'
        kwargs['predator'] = 500
        kwargs['prey'] = 500
        
    return kwargs
    

def run_example(kwargs = kwargs, steps = steps, model = 'lv'):
    
    # set model parameters
    
    kwargs = set_params(kwargs = kwargs, model = model)
    
    # run the model
    
    print(f"Running {model} model")
    
    m = model_run(**kwargs, steps=steps)

    # save data

    model_data = m.count.get_model_vars_dataframe()
    # set name for index column
    model_data.index.name = 'Step'
    model_data.to_csv(f'output/examples/results/data_model_{kwargs["model"]}.csv')
    agent_data = m.spatial.get_agent_vars_dataframe()
    agent_data.to_csv(f'output/examples/results/data_agents_{kwargs["model"]}.csv')

    # load data

    model_data = pd.read_csv(f'output/examples/results/data_model_{kwargs["model"]}.csv')
    agent_data = pd.read_csv(f'output/examples/results/data_agents_{kwargs["model"]}.csv')

    # plot the number of agents over time

    plot_pop(model_data=model_data, params = kwargs, file = f'output/examples/plots/pop_{kwargs["model"]}.png', steps=steps)

    # plot the age distribution of agents

    plot_age(agent_data=agent_data, file = f'output/examples/plots/age_{kwargs["model"]}.png', steps=steps)

    # plot the energy distribution of agents

    plot_energy(agent_data=agent_data, file = f'output/examples/plots/energy_{kwargs["model"]}.png', steps=steps)

    # plot density of agents

    plot_density(spatial_data=agent_data, file = f'output/examples/plots/density_{kwargs["model"]}.gif', steps=steps)

    # plot spatial distribution of agents

    plot_space(agent_data=agent_data,file = f'output/examples/plots/space_{kwargs["model"]}.gif', steps=steps)

    # plot spatial distribution of agents with population size

    plot_space_pop(agent_data=agent_data, model_data=model_data, params=kwargs,
                file = f'output/examples/plots/space_pop_{kwargs["model"]}.gif', steps=steps)

if __name__ == '__main__':
    
    model = argv[1] if len(argv) > 0 else 'lv'
    
    run_example(kwargs = kwargs, steps = steps, model = argv[1])
    
    print("Done!")