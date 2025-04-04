# import libraries

from matplotlib.pyplot import step
import pandas as pd
import numpy as np
import seaborn as sns

# constants

reps = 25
steps = 1000
parameter_depth = 50

## create parameter space

def create_space():

    s_breed = np.array(np.linspace(0.1, 1, parameter_depth))
    f_breed = np.array(np.linspace(0.1, 1, parameter_depth))

    vars = np.array(np.meshgrid(s_breed, f_breed))

    return vars.reshape(2, -1).T

# analysis for experiment 1


def analysis_experiment_1():
    """
    Analysis for experiment 1
    """
    print("Analysis for experiment 1")
    print("Replacing the apex predator with super predator")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-1_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    param_space = create_space()
    n_params = param_space.shape[0]
    n_models = 2

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 2

def analysis_experiment_2():

    """
    Analysis for experiment 2
    """
    print("Analysis for experiment 2")
    print("Varying the target and lethality of superpredators")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-2_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    param_space = create_space()
    n_params = param_space.shape[0]
    n_models = 6

    runs = reps * steps * n_params * n_models
    print("Number of runs = ", runs)
    
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 3

def analysis_experiment_3():

    """
    Analysis for experiment 3
    """
    print("Analysis for experiment 3")
    print("Determining the effects of predator and prey information")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-3_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    param_space = create_space()
    n_params = param_space.shape[0]
    n_models = 4

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")


# Experiment 4

def analysis_experiment_4():

    """
    Analysis for experiment 4
    """
    print("Analysis for experiment 4")
    print("Effect of handling limit on apex predator and mesopredator")
    print("==================================")
    print("\n")

    # load data

    data_lv = pd.read_csv("output/experiments/results/Experiment-4_lv_results.csv")
    data_apex = pd.read_csv("output/experiments/results/Experiment-4_apex_results.csv")

    # combine data

    data = pd.concat([data_lv, data_apex], axis=0)
    data.reset_index(drop=True, inplace=True)

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    n_params = 50
    n_models = 2

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 5

def analysis_experiment_5():
    """
    Analysis for experiment 5
    """
    print("Analysis for experiment 5")
    print("Varying birth rates of apex predator")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-5_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    n_params = 50
    n_models = 1

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 6

def analysis_experiment_6():
    """
    Analysis for experiment 6
    """
    print("Analysis for experiment 6")
    print("Varying lattice size and local saturation of prey for lv model")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-6_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete

    n_params = 4*8
    n_models = 1
    reps = 10
    steps = 2000
    
    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]*100/runs, "%")

# Experiment 7
def analysis_experiment_7():
    """
    Analysis for experiment 7
    """
    print("Analysis for experiment 7")
    print("Varying lethality of mesopredator")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-7_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete
    n_params = 50
    n_models = 1

    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 8

def analysis_experiment_8():

    """
    Analysis for experiment 8
    """
    print("Analysis for experiment 8")
    print("Varying lethality of apex predator")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-8_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete
    n_params = 20
    n_models = 1
    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)
    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# Experiment 9

def analysis_experiment_9():
    """
    Analysis for experiment 9
    """
    print("Analysis for experiment 9")
    print("Varying starting density of agents")
    print("==================================")
    print("\n")

    # load data

    data = pd.read_csv("output/experiments/results/Experiment-9_results.csv")

    # summaries

    print("First 5 rows of data")
    print(data.head())
    print("\n")

    print("Last 5 rows of data")
    print(data.tail())
    print("\n")

    print("Shape of data")
    print(data.shape)
    print("\n")

    print("Columns of data")
    print(data.columns)
    print("\n")

    print("Summary of data")
    print(data.describe())
    print("\n")

    # check if data is complete
    # create parameter space
    
    prey = np.array([100, 500, 1000, 2000, 5000])
    predator = np.array([100, 500, 1000, 2000, 5000])
    apex = np.array([0, 100, 500, 1000, 2000])
    super = np.array([0, 100, 500, 1000, 2000])
    
    reps = 10
    
    vars = np.array(np.meshgrid(prey, predator, apex, super)).reshape(4, -1).T

    n_params = vars.shape[0]
    n_models = 1
    runs = reps * steps * n_params * n_models

    print("Number of runs = ", runs)

    if runs == data.shape[0]:
        print("Data is complete")
    else:
        print("Data is not complete, ", data.shape[0]/runs, "%")

# run the analysis for all experiments
if __name__ == "__main__":

    analysis_experiment_1()
    analysis_experiment_2()
    analysis_experiment_3()
    analysis_experiment_4()
    analysis_experiment_5()
    analysis_experiment_6()
    analysis_experiment_7()
    analysis_experiment_8()
    analysis_experiment_9()