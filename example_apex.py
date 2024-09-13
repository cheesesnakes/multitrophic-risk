from functions.runner import model_run, plot_pop, plot_space, plot_density

# set model parameters

kwargs = {
    
    # model to run
    'model': 'apex',
    'progress': True,
    'info' : False,  
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 10,
    'prey': 500,
    'apex' : 10,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.1, # max birth rate
    'f_die': 0.01,
    'f_max': 2500,
    'risk_cost': 0.01,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_lethality': 0.15,
}

# run the model
m = model_run(**kwargs, steps=100)

# plot the number of agents over time
plot_pop(m, file = f'{kwargs['model']}_pop.png')

# plot the space
plot_space(m, file = f'{kwargs['model']}_space.gif', duration=10, steps=1000)

# plot the density

plot_density(m, file = f'{kwargs['model']}_density.gif', duration = 10, steps = 1000)