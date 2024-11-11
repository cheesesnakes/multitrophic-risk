from functions.experiment import experiment
import pandas as pd
import numpy as np
from sys import argv

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run   
    'limit' : 100000,
    'num_cpus': 40,
    'reps': 10,
    
    # model parameters
    'width': 50,
    'height': 50,
    'steps' : 10000,
    'prey' : 200,
    'predator': 200,
    'stop': True,

    ## prey traits
    'prey_info': True,
    'f_breed': 0.2, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 2500,
    'risk_cost': 0.01,
    'f_steps': 20,

    ## predator traits
    'predator_info': True,
    's_energy': 10,
    's_breed': 0.1, # constant
    's_die': 0.01,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 40,

    ## apex predator traits

    'apex_info': True,
    'a_energy': 10,
    'a_breed': 0.1, # constant
    'a_die': 0.001,
    'a_lethality': 0.15,
    'a_steps': 80,

    # super predator traits

    'super_target': 2,
    'super_lethality': 1,
    'super_steps': 20,
    'super_steps': 80,

    # parameters to vary

    'params': ['s_breed', 'f_breed']}

# create parameter space

def create_space():

    s_breed = np.array(np.linspace(0, 1, 100))
    f_breed = np.array(np.linspace(0, 1, 100))

    vars = np.array(np.meshgrid(s_breed, f_breed))

    return vars.reshape(2, -1).T

# experiment 1: replacing the apex predator with super predator

def experiment_1():
    
    E = "Experiment-1"
    
    print("Running experiment 1: replacing the apex predator with super predator")
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
    
    # create parameter space
    
    vars = create_space()
    
    # run model with apex predator
    
    print("Running model with apex predator")
    
    kwargs['model'] = 'apex'
    kwargs['apex'] = 100
    kwargs['super'] = 0

    # create an instance of the experiment
    
    print("Number of runs:", len(vars))
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])

    # save results

    results.to_csv(f'output/experiments/results/{E}_results.csv')
   
    # run model with super predator
    
    print("Running model with super predator")
    
    kwargs['model'] = 'super'
    kwargs['apex'] = 0
    kwargs['super'] = 100
    
    # create an instance of the experiment
    
    print("Number of runs:", len(vars))
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')
    
    print("Experiment 1 complete")
   
# Experiment 2, 3: Varying the target and lethality of superpredators
    
def experiment_2():
    
    E = "Experiment-2"

    print("Running experiment 2: varying the target and lethality of superpredators")
    
    # data frame to store results

    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator','step'])

    # create parameter space

    vars = create_space()

    # define levels of lethality

    lethalities = [0, 1]

    # define tagets

    targets = ["1","2","both"]

    print("Number of runs", len(vars)*2*3)

    for lethality in lethalities:

        for target in targets:

            print(f"Running model with superpredator target {target} and lethality {lethality}")

            kwargs['model'] = 'super'
            kwargs['super'] = 100
            kwargs['super_target'] = target
            kwargs['super_lethality'] = lethality

            # create an instance of the experiment
    
            print("Number of runs:", len(vars))
    
            exp = experiment(**kwargs)
    
            # run the experiment
    
            run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
            # append results to data frame
    
            results = pd.concat([results, run])

            # save results

            results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 4: Determine effects of predator and prey information

def experiment_3():
    
    E = "Experiment-3"
    
    print("Running experiment 3: determining the effects of predator and prey information")
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
    
    # create parameter space
    
    vars = create_space()
    
    prey_info = [True, False]
    predator_info = [True, False]
    
    print("Number of runs", len(vars)*4)
    
    for p_info in prey_info:
        
        for pred_info in predator_info:
            
            print(f"Running model with prey_info {p_info} and predator_info {pred_info}")
            
            kwargs['prey_info'] = p_info
            kwargs['predator_info'] = pred_info
            
            # create an instance of the experiment
    
            print("Number of runs:", len(vars))
    
            exp = experiment(**kwargs)
    
            # run the experiment
    
            run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
            # append results to data frame
    
            results = pd.concat([results, run])
    
            # save results
    
            results.to_csv(f'output/experiments/results/{E}_results.csv')
            
# Experiment 5: Determining the tradeoff between predator lethality and cost of anti-predator behavior


def create_space_2():
    
    risk_costs = np.array(np.linspace(0, 0.1, 100))
    
    s_breed = np.array(np.linspace(0, 1, 100))
    
    f_breed = np.array(np.linspace(0, 1, 100))
    
    vars = np.array(np.meshgrid(risk_costs, s_breed, f_breed))
    
    return vars.reshape(3, -1).T

def experiment_4():
    
    E = "Experiment-4"
    
    print("Running experiment 4: determining the tradeoff between predator lethality and cost of anti-predator behavior")
    
    # data frame to store results
    
    kwargs['params'] = ['risk_cost', 's_breed', 'f_breed']
    kwargs['model'] = 'lv'
    kwargs['predator'] = 200
    kwargs['prey'] = 200
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'step'])
    
    # create parameter space
    
    vars = create_space_2()
    
    print("Number of runs", len(vars))
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')
    
    
# running experiments

def run(exp = "All"):
    
    if exp == "1":
            
        experiment_1()
            
    elif exp == "2":
        
        experiment_2()
        
    elif exp == "3":
        
        experiment_3()
 
    elif exp == "4":
        
        experiment_4()
            
    else:
    
        experiment_1()
        experiment_2()
        experiment_3()
        experiment_4()
    
if __name__ == '__main__':
    
    exp = argv[1] if len(argv) > 0 else "All"
    
    run(exp = exp)
    
    print("Done!")
