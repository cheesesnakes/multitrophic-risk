from functions.experiment import experiment
import pandas as pd
import numpy as np

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
    'num_cpus': 10,
          
    # model parameters
    
    ## grid size
    'width': 50, 
    'height': 50, 
    
    ## number of agents to start with
    'predator': 10, 
    'prey': 100,
    
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
    
    ## experiment parameters
                    
    'steps': 1000, 
    'params': ['f_die', 's_energy', 's_breed', 'f_breed']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    f_die = np.array([0.1, 0.5, 0.9])
    s_energy = np.array([2, 5, 10])
    s_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    f_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(f_die, s_energy, s_breed, f_breed))
    
    vars = vars.T.reshape(-1, 4)
    
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
            
            
            