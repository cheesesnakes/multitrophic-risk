from functions.runner import model_run, plot_pop, plot_space, plot_density, plot_space_pop

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
    'predator': 500,
    'prey': 500,
    'apex' : 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.5, # max birth rate
    'f_die': 0.01,
    'f_max': 2500,
    'risk_cost': 0,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    's_apex_risk': True,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1,
    'a_lethality': 0.15
}

steps = 1000

# run the model
m = model_run(**kwargs, steps=steps)

# save data

m.count.get_model_vars_dataframe().to_csv(f'data_{kwargs['model']}.csv')

# plot population dynamics and space

plot_space_pop(m, file = f'space_pop_{kwargs['model']}.gif', duration=10, steps=steps)

# plot the space
plot_space(m, file = f'space_{kwargs['model']}.gif', duration=10, steps=steps)

# plot the number of agents over time
plot_pop(m, file = f'pop_{kwargs['model']}.png')

# plot the density

plot_density(m, file = f'density_{kwargs['model']}.gif', duration = 10, steps = steps)