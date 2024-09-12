from model_experiment import experiment
import pandas as pd

# Run the experiment

# set model parameters

# parameters

# grid size
width = 20 
height = 20
f_max = width * height

# number of agents
predator = 20
prey = 20

# duration of simulation
steps = 200

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
            
            print('Running experiment with s_energy =', e*10, 'f_die =', f)
            
            # create an instance of the experiment
    
            exp = experiment(
                
                # model parameters
                predator = predator, prey=prey, 
                width=width, height=height, steps=steps,
                
                # variable parameters
                
                predator_energy = e*10, prey_die = f)

            # run the experiment

            run = exp.parallel(vary=['s_breed', 'f_breed', 'predator_info', 'prey_info'], rep=5, n=20)
            
            # append results to data frame
            
            results = pd.concat([results, run])  
            
            # save results
            
            results.to_csv('results.csv')
            
if __name__ == '__main__':
    
    run()
    
    print('Experiment complete')          
            
            
            