from functions.runner import model_run, plot_pop, plot_space, plot_density

# set model parameters

kwargs = {
    
    # model to run
    'model': 'dewdney',
    'progress': True,
    'info' : True,  
    
    # model parameters
    'width': 20,
    'height': 20,
    
    # number of agents to start with
    'predator': 10,
    'prey': 100,
    
    ## prey traits
    'prey_info': False,
    'f_breed': 10,
    'f_energy': 1,
    
    ## predator traits
    'predator_info': False,
    's_energy': 10,
    's_breed': 10
}   
# run the model
m = model_run(**kwargs, steps=1000)

# plot the number of agents over time
plot_pop(m, file = f'{kwargs['model']}_pop.png')

# plot the space
plot_space(m, file = f'{kwargs['model']}_space.png', duration=10, steps=1000)

# plot the density

plot_density(m, file = f'{kwargs['model']}_density.png', duration = 10, steps = 1000)