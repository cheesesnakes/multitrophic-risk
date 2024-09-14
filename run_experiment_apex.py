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
    'predator': 10,
    'prey': 500,
    'apex' : 100,
    
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
    's_apex_risk': True,
    
    ## apex predator traits
    
    'apex_info': True,
    'a_energy': 50,
    'a_breed': 0.1,
    'a_lethality': 0.15,

    'params': ['s_energy', 's_breed', 'f_breed', 'f_die']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    f_die = np.array([0.1, 0.5, 0.9])
    s_energy = np.array([1, 5, 10])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(f_die, s_energy))
    
    vars = vars.T.reshape(-1, 2)
    
    # print number of experiments
    
    print('Number of experiments:', len(vars))
    
    # run experiments
    
    for e, f in vars:
            
        # print progress
        
        print(f'Running {kwargs['model']} model experiment with s_energy =', e, 'f_die =', f)
        
        # update parameters
        
        kwargs['s_energy'] = e*10
        kwargs['f_die'] = f
        
        # create an instance of the experiment

        exp = experiment(**kwargs)

        # run the experiment

        run = exp.parallel(vary=['s_breed', 'f_breed'], params = kwargs['params'], rep=5, n=20)
        
        # append results to data frame
        
        results = pd.concat([results, run])  
        
        # save results
        
        results.to_csv(f'results_{kwargs['model']}.csv')
            
if __name__ == '__main__':
    
    run()
    
    print('Experiment complete')          
            
            
            