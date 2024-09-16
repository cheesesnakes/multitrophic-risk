from functions.experiment import experiment
import pandas as pd
import numpy as np

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run
    'model': 'super', 
    'num_cpus': 10,
    
    # model parameters
    'width': 50,
    'height': 50,
    'steps' : 1000,
    
    # number of agents to start with
    'predator': 500,
    'prey': 500,
    'super' : 250,
    
    ## prey traits
    'prey_info': True,
    'f_breed': 0.1, # max birth rate
    'f_die': 0.01,
    'f_max': 2500,
    'risk_cost': 0,
    'f_super_risk': False,
    
    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1,
    's_lethality': 0.5,
    's_super_risk': True,
    
    ## apex predator traits

    'params': ['f_die', 's_energy', 'super_target', 'super_lethality', 's_breed', 'f_breed']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    f_die = np.array([0.1, 0.5, 0.9])
    s_energy = np.array([2, 5, 10])
    s_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    f_breed = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    super_target = np.array([1, 2, 12]) # 1 = prey, 2 = predator, 12 = both
    super_lethality = np.array([0, 1])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(f_die, s_energy, super_target, super_lethality, s_breed, f_breed))
    
    vars = vars.T.reshape(-1, 6)
    
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
            
            
            