from functions.experiment import experiment
import pandas as pd

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run
    'model': 'lv',
          
    # model parameters
    
    ## grid size
    'width': 20, 
    'height': 20, 
    
    ## number of agents to start with
    'predator': 10, 
    'prey': 100,
    
    ## prey traits
     
    'prey_info': False, 
    'f_breed': 0.01, 
    'f_die': 0.1,
    
    ## predator traits
    'predator_info': False, 
    's_energy': 10, 
    's_breed': 0.01,
    
    ## experiment parameters
                    
    'steps': 50, 
    'sample_id': 1, 
    'rep_id': 1,                 
    'df': pd.DataFrame(columns = ['rep_id', 'sample_id', 
                             'Prey', 'Predator', 
                             'step', 's_energy', 's_breed', 
                             'f_breed', 'f_die', 
                             'predator_info', 'prey_info'])}

def run():
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
    
    # create parameter space
    
    a = [0.1, 0.2, 0.5, 0.9, 1]
    
    # print number of experiments
    
    print('Number of experiments:', len(a) ** 2)
    
    for e in a:
        
        for f in a:
            
            # print progress
            
            print(f'Running {kwargs['model']} model experiment with s_energy =', e*10, 'f_die =', f)
            
            # update parameters
            
            kwargs['s_energy'] = e*10
            kwargs['f_die'] = f
            
            # create an instance of the experiment
    
            exp = experiment(**kwargs)

            # run the experiment

            run = exp.parallel(vary=['s_breed', 'f_breed', 'predator_info', 'prey_info'], rep=5, n=20)
            
            # append results to data frame
            
            results = pd.concat([results, run])  
            
            # save results
            
            results.to_csv(f'{kwargs['model']}_results.csv')
            
if __name__ == '__main__':
    
    run()
    
    print('Experiment complete')          
            
            
            