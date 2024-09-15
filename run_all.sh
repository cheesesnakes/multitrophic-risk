#!/bin/bash

# load venv

source ./abm_env/bin/activate

echo "Chapter 2: Running all scripts"

# Run all the scripts in the correct order

# run the experiments
python run_experiment_super.py
python run_experiment_apex.py
python run_experiment_lv.py

# run analysis script

cat analysis.R | r

# run the examples
python example_super.py
python example_apex.py
python example_lv.py

echo "Chapter 2: Done running all scripts"

