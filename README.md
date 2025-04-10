# README: Superpredator Analysis Framework

## Overview

This repository contains the code and resources for analyzing the impact of multitrophic risk introduced by human "superpredators" on tritrophic ecosystems. Using an agent-based modeling approach implemented in Python's Mesa framework, the project investigates how superpredators influence population dynamics, coexistence equilibria, and ecosystem stability.

The study explores three models:
1. **Base Model**: A predator-prey system.
2. **Tritrophic Model**: A system including apex predators, mesopredators, and prey.
3. **Superpredator Model**: A system where apex predators are replaced by superpredators with unique characteristics (e.g., 100% lethality, no reproduction, no mortality).

Simulations were conducted across various parameter combinations to assess the effects of superpredator targets, lethality, and behavioral adaptations.

---

## Abstract

Predator-prey interactions involve continuous behavioral adaptations that influence population dynamics and ecosystem stability. While traditional models have examined these dynamics at single trophic levels, the impact of multitrophic risk particularly from human "superpredators" targeting multiple levels remains underexplored. We investigated how pervasive multitrophic risk introduced by a superpredator affects a tritrophic system of apex predators, mesopredators, and primary consumers using an agent-based model in Python’s Mesa framework. We explored the effects of replacing apex predators with superpredators, varying superpredator targets, and altering superpredator lethality. Three models were constructed: (1) a base model with predators and prey, (2) a tritrophic model including an apex predator, and (3) a superpredator model with a superpredator that does not die or reproduce and has 100% lethality. Agents were programmed to exhibit behaviors such as reproduction, movement, and predator avoidance. Simulations were conducted across a range of breeding rates with 100 runs for each parameter combination. Our results demonstrate that superpredators significantly destabilize ecosystems. Replacing apex predators with superpredators eliminated coexistence equilibria seen in the tritrophic model, leading to prey-only equilibria or extinction. Superpredators targeting both mesopredators and prey or only mesopredators resulted in extinction trends, while coexistence occurred when superpredators targeted only prey, within a limited range. Importantly, non-lethal superpredators—inducing antipredator responses without direct mortality—enabled coexistence across scenarios, underscoring the role of behavioral adaptations in stability. These findings highlight the significant impact of human superpredators on multitrophic interactions and underscore the importance of integrating human activities into ecological models and conservation strategies to mitigate unintended ecological consequences.

**Keywords:** human disturbance, agent-based modeling, predator-prey interactions, behavioral adaptations

## Repository Structure

```
superpredator/
├── functions/          # helper functions
|  ├── models.py        # ABM models definitions
|  ├── runner.py        # Model runner and plotting functions
|  ├── debug.py         # Debugging functions
|  ├── example.py       # example simulations
|  ├── strategies.py    # example simulations varying predator and prey strategies
|  ├── funcs.py         # analysis functions
|  ├── plots.py         # anlysis plotting functions 
├── literature/         # literature survey data and analysis
|  ├── lit-data.R       # R script for literature data processing
|  ├── articles.csv     # Database of articles
|  ├── articles_full-text.csv # Articles chosen for full-text screeening
├── analysis.py        # main analysis script   
├── configs.py          # configuration file for experiments
├── params.py           # globalparameter definitions for model
├── run.py              # main script to run the model
├── pyproject.toml      # Python project configuration
├── requirements.txt    # Python dependencies
├── uv.lock             # Python dependency lock file
├── README.md           # This file
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