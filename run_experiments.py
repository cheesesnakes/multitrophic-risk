from functions.experiment import experiment
import pandas as pd
import numpy as np
from sys import argv

# Run the experiment

# set model parameters

kwargs = {
    
    # model to run   
    'limit' : 100000,
    'num_cpus': 50,
    'reps': 25,
    
    # model parameters
    'width': 100,
    'height': 100,
    'steps' : 1000,
    'prey' : 500,
    'predator': 500,
    'stop': False,

    ## prey traits
    'prey_info': True,
    'f_breed': 0.6, # max birth rate
    'f_die': 0.1, # constant
    'f_max': 5,
    'f_steps': 1,

    ## predator traits
    'predator_info': True,
    's_max': 5,
    's_breed': 0.15, # max birth rate
    's_die': 0.1,
    's_lethality': 0.6,
    's_apex_risk': True,
    's_steps': 1,

    ## apex predator traits

    'apex_info': True,
    'a_max': 10,
    'a_breed': 0.25, # max birth rate
    'a_die': 0.1,
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

depth = 50

def create_space():

    s_breed = np.array(np.linspace(0.1, 1, depth))
    f_breed = np.array(np.linspace(0.1, 1, depth))

    vars = np.array(np.meshgrid(s_breed, f_breed))

    return vars.reshape(2, -1).T

# experiment 1: replacing the apex predator with super predator

def experiment_1():
    
    E = "Experiment-1"
    
    print("Running experiment 1: replacing the apex predator with super predator")
    
    # data frame to store results
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
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

    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])

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
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
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
   
    a_max = np.array(np.linspace(0, 100, depth))
    s_max = np.array(np.linspace(0, 100, depth))
    
    kwargs['params'] = ['s_max']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    # run experiment for lv model
    
    kwargs['model'] = 'lv'
    kwargs['apex'] = 0
    kwargs['super'] = 0
    kwargs['predator'] = 500
    kwargs['prey'] = 500
    
    vars = s_max
    
    print("Number of runs", len(vars)*2)
     
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_lv_results.csv')
    
    # run experiment for apex predator
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
   
    kwargs['model'] = 'apex'
    kwargs['apex'] = 500
    kwargs['super'] = 0
    kwargs['predator'] = 500
    kwargs['prey'] = 500
    
    kwargs['params'] = ['a_max']
    
    vars = a_max
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_apex_results.csv')

# Experiment 6: Varying birth rates of apex predator and mesopredator

def experiment_5():
    
    E = "Experiment-5"
    
    print("Running experiment 5: varying birth rates of apex predator")
    
    # create parameter space
    
    a_breed = np.array(np.linspace(0.1, 1, depth))
    
    kwargs['params'] = ['a_breed']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    # run experiment for apex predator
    
    kwargs['model'] = 'apex'
    kwargs['apex'] = 500
    kwargs['super'] = 0
    kwargs['predator'] = 500
    kwargs['prey'] = 500
    
    vars = a_breed
    
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
    
    print("Running experiment 6: varying lattice size and local saturation of prey for lv model")
    
    # create parameter space
    
    L2 = [10, 20, 50, 100]
    f_max = [0, 1, 2, 5, 10, 20, 25, 50]
    
    kwargs['params'] = ['f_max']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    for l in L2:
        
        for f in f_max:
            
            print(f"Running model with lattice size {l} and local saturation of prey {f}")
            
            kwargs['width'] = l
            kwargs['height'] = l
            kwargs['f_max'] = f
            
            reps = 10
            
            # run experiment for lv model
            
            kwargs['model'] = 'lv'
            kwargs['apex'] = 0
            kwargs['super'] = 0
            kwargs['predator'] = 0
            kwargs['prey'] = 2000
            
            # create an instance of the experiment
            
            exp = experiment(**kwargs)
            
            # run the experiment
            
            run = exp.parallel(v = None, rep=reps, **kwargs)
            
            # append results to data frame
            
            results = pd.concat([results, run])
            
            # save results
        
            results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 8: Varying lethality of mesopredator

def experiment_7():
    
    E = "Experiment-7"
    
    print("Running experiment 7: varying lethality of mesopredator")
    
    # create parameter space
    
    s_lethality = np.array(np.linspace(0, 1, depth))
    
    kwargs['params'] = ['s_lethality']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    # run experiment for mesopredator
    
    kwargs['model'] = 'lv'
    kwargs['apex'] = 0
    kwargs['super'] = 0
    kwargs['predator'] = 500
    kwargs['prey'] = 2000
    
    vars = s_lethality
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 9: Varying lethality of apex predator

def experiment_8():
    
    E = "Experiment-8"
    
    print("Running experiment 8: varying lethality of apex predator")
    
    # create parameter space
    
    a_lethality = np.array(np.linspace(0, 1, 20))
    
    kwargs['params'] = ['a_lethality']
    
    # data frame to store results
        
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    # run experiment for apex predator
    
    kwargs['model'] = 'apex'
    kwargs['apex'] = 500
    kwargs['super'] = 0
    kwargs['predator'] = 500
    kwargs['prey'] = 500
    
    vars = a_lethality
    
    # create an instance of the experiment
    
    exp = experiment(**kwargs)
    
    # run the experiment
    
    run = exp.parallel(v = vars, rep=kwargs.get('reps', 10), **kwargs)
    
    # append results to data frame
    
    results = pd.concat([results, run])
    
    # save results
    
    results.to_csv(f'output/experiments/results/{E}_results.csv')

# Experiment 10: varying starting density of agents

def experiment_9():
    
    E = "Experiment-9"
    
    print("Running experiment 9: varying starting density of agents")
    
    # create parameter space
    
    prey = np.array([100, 500, 1000, 2000, 5000])
    predator = np.array([100, 500, 1000, 2000, 5000])
    apex = np.array([0, 100, 500, 1000, 2000])
    super = np.array([0, 100, 500, 1000, 2000])
    
    reps = 10
    
    vars = np.array(np.meshgrid(prey, predator, apex, super)).reshape(4, -1).T
    
    results = pd.DataFrame(columns = ['rep_id', 'sample_id', *kwargs['params'], 'Prey', 'Predator', 'Apex', 'Super', 'step'])
    
    for i in range(vars.shape[0]):
        
        kwargs['prey'], kwargs['predator'], kwargs['apex'], kwargs['super'] = vars[i]
        
        print(f"Running model with prey {kwargs['prey']}, predator {kwargs['predator']}, apex {kwargs['apex']}, super {kwargs['super']}")
        
        # create an instance of the experiment
        
        exp = experiment(**kwargs)
        
        # run the experiment
        
        run = exp.parallel(v = None, rep=reps, **kwargs)
        
        # save results
        
        results = pd.concat([results, run])
        
        results.to_csv(f'output/experiments/results/{E}_results.csv')     
        
        
# diagnostic: run only apex, super, and predator agents

def diagnostic():
    
    rep = 10
    
    # create parameter space
    
    agents = ["apex", "super", "predator"]
    
    for agent in agents:
        
        kwargs['model'] = agent
        
        # create an instance of the experiment
        
        exp = experiment(**kwargs)
        
        # run the experiment
        
        run = exp.parallel(v = None, rep=rep, **kwargs)
        
        # save results
        
        run.to_csv(f'output/experiments/results/debug_{agent}_results.csv')
        
    
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
    
    elif exp == "5":
        
        experiment_5()
    
    elif exp == "6":
        
        experiment_6()
        
    elif exp == "7":
        
        experiment_7()
    
    elif exp == "8":
        
        experiment_8()
        
    elif exp == "9":
        
        experiment_9()
        
    elif exp == "debug":
        
        experiment_4()
        experiment_5()
        experiment_7()
        experiment_8()
        experiment_9()
        diagnostic()
        
        
    else:
    
        experiment_1()
        experiment_2()
        experiment_3()
        experiment_4()
        experiment_5()
        experiment_6()
        experiment_7()
        experiment_8()
        experiment_9()
        diagnostic()
        
    
if __name__ == '__main__':
    
    exp = argv[1] if len(argv) > 0 else "All"
    
    run(exp = exp)
    
    print("Done!")
