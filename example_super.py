from functions.runner import model_run, plot_pop, plot_space, plot_density

# set model parameters

kwargs = {
    
    # model to run
    'model': 'super',
    'progress': True,
    'info' : False,  
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 10,
    'prey': 500,
    'super' : 50,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.1, # max birth rate
    'f_die': 0.01,
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_super_risk': True,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    's_super_risk': True,
    
    ## apex predator traits
    
    'super_target': 'predator'
}

# run the model
m = model_run(**kwargs, steps=100)

# plot the number of agents over time
plot_pop(m, file = f'{kwargs['model']}_pop.png')

# plot the space
plot_space(m, file = f'{kwargs['model']}_space.gif', duration=10, steps=1000)

# plot the density

plot_density(m, file = f'{kwargs['model']}_density.gif', duration = 10, steps = 1000)