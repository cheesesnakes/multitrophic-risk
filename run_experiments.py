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
    'steps' : 2000,
    'prey' : 2000,
    'predator': 500,
    'stop': True,

    ## prey traits
    'prey_info': True,
    'f_breed': 0.6, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 10,
    'f_steps': 1,

    ## predator traits
    'predator_info': True,
    's_max': 5,
    's_breed': 0.15, # max birth rate
    's_die': 0.1,
    's_lethality': 0.5,
    's_apex_risk': True,
    's_steps': 1,

    ## apex predator traits

    'apex_info': True,
    'a_max': 10,
    'a_breed': 0.25, # max birth rate
    'a_die': 0.01,
    'a_lethality': 0.15,
    'a_steps': 1,

    # super predator traits

    'super_target': 2,
    'super_lethality': 1,
    'super_max': 20,
    'super_steps': 1,

    # parameters to vary

    'params': ['s_breed', 'f_breed']}

# create parameter space

def create_space():

    s_breed = np.array(np.linspace(0.1, 1, 20))
    f_breed = np.array(np.linspace(0.1, 1, 20))

    vars = np.array(np.meshgrid(s_breed, f_breed))

    return vars.reshape(2, -1).T

# experiment 1: replacing the apex predator with super predator

def experiment_1():
    
    E = "Experiment-1"
    
    print("Running experiment 1: replacing the apex predator with super predator")
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'step'])
    
    # create parameter space
    
    vars = create_space()
    
    # run model with apex predator
    
    print("Running model with apex predator")
    
    kwargs['model'] = 'apex'
    kwargs['apex'] = 500
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
    kwargs['super'] = 500
    
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

    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex','step'])

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
            kwargs['super'] = 500
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
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'step'])
    
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

# Experiment 5: Effect of handling limits on apex predator and mesopredator

def experiment_4():
    
    E = "Experiment-4"
    
    print("Running experiment 4: effect of handling limit on apex predator and mesopredator")
    
    # create parameter space
   
    a_max = np.array(np.linspace(0, 100, 20))
    s_max = np.array(np.linspace(0, 100, 20))
    
    kwargs['params'] = ['a_max', 's_max']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'step'])
    
    # run experiment for lv model
    
    kwargs['model'] = 'lv'
    kwargs['apex'] = 0
    kwargs['super'] = 0
    kwargs['predator'] = 100
    kwargs['prey'] = 100
    
    vars = np.array(np.meshgrid(0, s_max)).reshape(2, -1).T
    
    print("Number of runs", len(vars)*2)
     
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')
    
    # run experiment for apex predator
    
    kwargs['model'] = 'apex'
    kwargs['apex'] = 250
    kwargs['super'] = 0
    kwargs['predator'] = 100
    kwargs['prey'] = 100
    
    vars = np.array(np.meshgrid(a_max, s_max)).reshape(2, -1).T
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 6: Varying birth rates of apex predator and mesopredator

def experiment_5():
    
    E = "Experiment-5"
    
    print("Running experiment 5: varying birth rates of apex predator and mesopredator")
    
    # create parameter space
    
    a_breed = np.array(np.linspace(0, 1, 20))
    s_breed = np.array(np.linspace(0, 1, 20))
    
    kwargs['params'] = ['a_breed', 's_breed']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'step'])
    
    # run experiment for apex predator
    
    kwargs['model'] = 'apex'
    
    vars = np.array(np.meshgrid(a_breed, s_breed)).reshape(2, -1).T
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 7: Varying local saturation of prey for lv model

def experiment_6():
    
    E = "Experiment-6"
    
    print("Running experiment 6: varying local saturation of prey for lv model")
    
    # create parameter space
    
    f_max = np.array(np.linspace(0, 100, 20))
    
    kwargs['params'] = ['f_max']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'step'])
    
    # run experiment for lv model
    
    kwargs['model'] = 'lv'
    
    vars = "f_max"
    
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
        experiment_5()
        experiment_6()
        
    
if __name__ == '__main__':
    
    exp = argv[1] if len(argv) > 0 else "All"
    
    run(exp = exp)
    
    print("Done!")
