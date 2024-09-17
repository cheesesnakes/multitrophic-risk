from functions.experiment import experiment
import pandas as pd
import numpy as np

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
    'num_cpus': 10,  
    'limit' : 50*50*4,
          
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
    'f_die': 0.1, # constant
    'f_max': 2500,
    'risk_cost': 0,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1, # constant
    's_lethality': 0.5,
    
    ## experiment parameters
                    
    'steps': 1000, 
    'params': ['s_energy', 'f_breed']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    s_energy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    f_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(s_energy, f_breed))
    
    vars = vars.T.reshape(-1, 2)
    
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
            
            
            