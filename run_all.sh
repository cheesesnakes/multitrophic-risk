#!/bin/bash

# load venv

source ./abm_env/bin/activate

echo "Chapter 2: Running all scripts"

# run all scripts

python debug.py

python example.py 'lv'
python example.py 'apex'
python example.py 'super'
python example_strat.py

python run_experiments.py

echo "Chapter 2: Done running all scripts"

