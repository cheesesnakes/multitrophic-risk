# import libraries

import pandas as pd
import numpy as np
import seaborn as sns

# constants

reps = 25
steps = 1000


# analysis for experiment 1


def analysis_experiment_1():
    """
    Analysis for experiment 1
    """

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


analysis_experiment_1()
