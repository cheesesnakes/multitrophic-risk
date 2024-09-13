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
    's_super_risk': True,
    
    ## apex predator traits

    'params': ['s_energy', 's_breed', 'f_breed', 'f_die']}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
        
    # create parameter space
    
    f_die = np.array([0.1, 0.5, 0.9])
    s_energy = np.array([1, 5, 10])
    super_targets = np.array(['prey', 'predator', 'both'])
    super_lethality = np.array([0, 1])
    
    # create a meshgrid
    
    vars = np.array(np.meshgrid(f_die, s_energy, super_targets, super_lethality))
    
    vars = vars.T.reshape(-1, 4)
    
    # print number of experiments
    
    print('Number of experiments:', len(vars))
    
    for i in range(len(vars)):
            
        # print progress
        
        print(f'Running {kwargs['model']} model experiment with s_energy = {vars[1, i]}, f_die = {vars[0, i]}, super_target = {vars[2, i]}, super_lethality = {vars[3, i]}')
        
        # update parameters
        
        kwargs['s_energy'] = vars[1, i]
        kwargs['f_die'] = vars[0, i]
        kwargs['super_target'] = vars[2, i]
        kwargs['super_lethality'] = vars[3, i]
        
        # create an instance of the experiment

        exp = experiment(**kwargs)

        # run the experiment

        run = exp.parallel(vary=['s_breed', 'f_breed'], params = kwargs['params'], rep=5, n=20)
        
        # append results to data frame
        
        results = pd.concat([results, run])  
        
        # save results
        
        results.to_csv(f'{kwargs['model']}_results.csv')
    
if __name__ == '__main__':
    
    run()
    
    print('Experiment complete')          
            
            
            