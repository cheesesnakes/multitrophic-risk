from functions.experiment import experiment
import pandas as pd
import numpy as np

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run
    'model': 'apex', 
    
    # model parameters
    'width': 50,
    'height': 50,
    'steps' : 1000,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    'apex' : 100,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.1, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 2500,
    'risk_cost': 0,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1, # constant
    's_lethality': 0.5,
    's_apex_risk': True,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1, # constant
    'a_lethality': 0.15,

    'params': ['s_energy', 'f_breed', 'a_energy']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    s_energy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    f_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    a_energy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(s_energy, f_breed, a_energy))
    
    vars = vars.T.reshape(-1, 3)
    
    # print number of experiments
    
    print('Number of experiments:', len(vars))
    
            
    # create an instance of the experiment

    exp = experiment(**kwargs)

    # run the experiment

    run = exp.parallel(v = vars, rep=100, **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])  
    
    # save results
    
    results.to_csv(f'output/{kwargs['model']}_results.csv')
            
if __name__ == '__main__':
    
    run()
    
    print('Experiment complete')          
            
            
            