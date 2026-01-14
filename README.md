# Multi-trophic risk from a superpredator alters the outcome of predator - prey behavioural response races.

**Authors:** Shawn Dsouza, Maria Thaker, Vishwesha Guttal, Kartik Shanker

Center for Ecological Sciences, Indian Institute of Science, Bangalore, India.

## Overview

This repository contains the code and resources for analyzing the impact of multitrophic risk introduced by human "superpredators" on tritrophic ecosystems. Using an agent-based modeling approach implemented in Python's Mesa framework, the project investigates how superpredators influence population dynamics, coexistence equilibria, and ecosystem stability.

The study explores three models:
1. **Base Model**: A predator-prey system.

   ![[Base Model video (mp4)]](https://github.com/cheesesnakes/multitrophic-risk/raw/refs/heads/main/examples/space_lv.mp4)


2. **Tritrophic Model**: A system including apex predators, mesopredators, and prey.

   ![Tritrophic video (mp4)](https://github.com/cheesesnakes/multitrophic-risk/raw/refs/heads/main/examples/space_apex.mp4)

3. **Superpredator Model**: A system where apex predators are replaced by superpredators with unique characteristics (e.g., 100% lethality, no reproduction, no mortality).

   ![Superpredator video (mp4)](https://github.com/cheesesnakes/multitrophic-risk/raw/refs/heads/main/examples/space_super.mp4)

Simulations were conducted across various parameter combinations to assess the effects of superpredator targets, lethality, and behavioral adaptations.

---

## Abstract

Overexploitation has led to the trophic downgrading of many of earths ecosystems. Humans, as predators, are extremely efficient and deadly, yet they may also interact with wildlife in benign ways. This study explores how interactions with lethal and non-lethal human “superpredators” alter predator-prey dynamics using an agent-based modelling approach. Our model incorporates both the consumptive (lethal) and non-consumptive (behavioural) effects of predators on prey. We aimed to (1) understand how the extirpation and replacement of apex predators by humans may affect the dynamics of mesopredators and prey, (2) compare the outcomes of scenarios where lethal superpredators target different trophic levels (mesopredators and prey), separately and simultaneously, and (3) assess the effects of superpredators when they have lethal versus non-lethal intent. We found that superpredators have a greater effect on model outcomes than apex predators. When superpredators consume mesopredators alone or with prey, the probability of mesopredator-prey coexistence increases to a greater extent than when apex predators consume mesopredators. In contrast, superpredators consuming prey slightly increases overall extinction risks and reduces coexistence. Non-lethal superpredators, despite eliciting anti-predator responses in mesopredators and prey, had a negligible effect on population dynamics. Our findings demonstrate that human superpredators may functionally replace apex predators when they are lethal.  However, benign interactions with humans may not be as ecologically significant as lethal interactions, even when humans induce antipredator responses. Future research should integrate habitat heterogeneity, resource distribution, and behavioural adaptations to predict and mitigate the impacts of human activities on ecosystems.

**Keywords:** human disturbance, agent-based modeling, predator-prey interactions, behavioral adaptations

## Repository Structure

```
superpredator/
├── functions/              # helper functions
│   ├── compare.py          # comparison utilities
│   ├── debug.py            # debugging functions
│   ├── example.py          # example simulations
│   ├── experiment.py       # experiment logic
│   ├── figures.py          # plotting figures
│   ├── model.py            # ABM model definitions
│   ├── runner.py           # model runner and plotting functions
│   ├── signals.py          # signal processing
│   ├── summary_funcs.py    # summary statistics functions
│   ├── summary_plots.py    # summary plotting functions
│   ├── summary.py          # summary utilities
│   └── __pycache__/        # python cache
├── literature/             # literature survey data and analysis
│   ├── lit-data.R          # R script for literature data processing
│   ├── articles.csv        # database of articles
│   └── articles_full-text.csv # articles chosen for full-text screening
├── analysis.py             # main analysis script
├── configs.py              # configuration file for experiments
├── params.py               # global parameter definitions for model
├── run.py                  # main script to run the model
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Python dependencies
├── sample.csv              # sample data
├── tables.R                # R script for tables
├── LICENSE                 # license file
├── README.md               # this file
└── __pycache__/            # python cache
```


## Installation

### Using `uv` (Recommended)

To install the project using `uv`, follow these steps:

1. Install `uv`:
   ```bash
   pip install uv
   ```
2. Clone the repository
3. Create a new environment:
   ```bash
   uv sync
   ```   
4. Run the desired program e.g.:
   ```bash
   uv run run.py
   ```

### Using `pip`

**Prerequisites**:
- Python 3.8+
- R (for literature analysis)
- Required Python libraries: `mesa`, `pandas`, `matplotlib`, `seaborn`, `polars`, `numpy`.

**Installation Steps**:

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install R dependencies:
   ```r
   install.packages(c("dplyr", "tidyr", "stringr", "purrr"))
   ```

## Usage

The `run.py` script serves as the main entry point for running various components of the superpredator analysis framework. It provides options to execute debugging, example simulations, strategy visualizations, and experiments. 

To run the script, use the following command in the terminal:

```bash
uv run run.py [<program>] [<option>]
```

### Available Programs:

*Debug*: Runs debugging scripts for the model.

*Examples*: Executes predefined example simulations.

*Strategies*: Runs and visualizes strategy examples.

*Experiments*: Executes model experiments based on configurations.
Options in Detail

#### Debug

Runs debugging scripts to test the model and visualize debugging outputs.

Command:

```bash
uv run run.py Debug
```

What it does:

- Executes the run_debug function to test the model.
- Generates debugging plots using plot_debug.
 
#### Examples

Runs predefined example simulations for different models (lv, apex, super).

Command:

```bash
uv run run.py Examples [model_name]
```

Arguments:

- model_name (optional): Specifies the model to run. 

Available models:
- lv: Lotka-Volterra (base predator-prey model).
- apex: Tritrophic model with apex predators.
- super: Superpredator model.

If no model_name is provided, all models (lv, apex, super) are executed.
If an invalid model_name is provided, the script will display an error message.

#### Strategies

Runs and visualizes strategy examples for the model.

Command:

```bash
uv run run.py Strategies
```

What it does:

- Executes the `run_strategies` function to simulate various predator and prey strategies.
- Generates strategy plots using `plot_strategy`.

#### Experiments

Runs model experiments based on predefined configurations in `configs.py`.

Command:

```bash
uv run run.py Experiments [experiment_number]
```

Arguments:

- `experiment_number` (optional): Specifies the experiment to run (e.g., `2` for Experiment-2).

Behavior:

- If no `experiment_number` is provided, all experiments in `configs.py` are executed.
- If an invalid `experiment_number` is provided, the script will display available experiment names.

Examples:

Run all experiments:

```bash
uv run run.py Experiments
```

Run a specific experiment (e.g., Experiment-2):

```bash
uv run run.py Experiments 2
```

## Configuration

The configuration for experiments is defined in the `configs.py` file. Global model parameters are set in `params.py`. 

## Citation

This project is currently in development. If you use this code or data in your research, please contact the authors for citation information.

## License

This project is licensed under the MIT License. See `LICENSE` for details.