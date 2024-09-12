from functions.runner import model_run, plot_pop, plot_space, plot_density

# set model parameters

kwargs = {
    
    # model to run
    'model': 'resource',
    'progress': True,
    
    # model parameters
    'width': 50,
    'height': 50,
    
    # number of agents to start with
    'predator': 100,
    'prey': 500,
    
    ## prey traits
    'prey_info': False,
    'f_breed': 0.1,
    'f_energy': 100,
    
    ## predator traits
    'predator_info': False,
    's_energy': 100,
    's_breed': 0.1,
    
    # resource rate
    'resource_rate': 1
}   

# run the model
m = model_run(**kwargs, steps=1000)

# plot the number of agents over time
plot_pop(m, file = f'{kwargs['model']}_pop.png')

# plot the space
plot_space(m, file = f'{kwargs['model']}_space.png', duration=10, steps=1000)

# plot the density

plot_density(m, file = f'{kwargs['model']}_density.png', duration = 10, steps = 1000)