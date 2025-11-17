import numpy as np

depth = 50
space = np.linspace(0.1, 1, depth)
space_100 = np.linspace(0, 100, depth)
space_lethality = np.linspace(0, 1, depth)


def create_space():
    x = space
    return np.array(np.meshgrid(x, x)).reshape(2, -1).T


# models configs

configs = {
    "Scenario-0": {
        "name": "Scenario-0",
        "description": "Baseline model with mesopredator and prey",
        "data_path": "output/experiments/results/scenario-0_results.csv",
        "status": "complete",
        "append": True,
        "n_models": 1,
        "populations": ["Prey", "Predator"],
        "variables": ["s_breed", "f_breed"],
        "models": ["Baseline"],
        "vars": create_space(),
        "models_config": [
            {
                "description": "lv",
                "params": {"model": "lv", "apex": 0, "super": 0},
            }
        ],
    },
    "Scenario-1": {
        "name": "Scenario-1",
        "description": "Apex predator targets mesopredator",
        "data_path": "output/experiments/results/scenario-1_results.csv",
        "status": "complete",
        "append": True,
        "n_models": 1,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["s_breed", "f_breed"],
        "models": ["Apex predator"],
        "vars": create_space(),
        "models_config": [
            {
                "description": "apex",
                "params": {"model": "apex", "apex": 500, "super": 0},
            },
        ],
    },
    "Test-1": {
        "name": "Test-1",
        "description": "Effect of handling limit on mesopredator",
        "data_path": "output/experiments/results/test-1_results.csv",
        "status": "complete",
        "n_models": 1,
        "populations": ["Prey", "Predator"],
        "variables": ["s_max"],
        "n_params": 50,
        "models": ["LV"],
        "vars": space_100,
        "models_config": [
            {
                "params": {
                    "model": "lv",
                    "apex": 0,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["s_max"],
                }
            }
        ],
        "model_prefix": "lv",
    },
    "Test-2": {
        "name": "Test-2",
        "description": "Effect of handling limit on apex predator",
        "data_path": "output/experiments/results/test-2_results.csv",
        "status": "complete",
        "n_models": 1,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["a_max"],
        "n_params": 50,
        "models": ["Apex"],
        "vars": space_100,
        "models_config": [
            {
                "params": {
                    "model": "apex",
                    "migrate": False,
                    "apex": 500,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["a_max"],
                }
            }
        ],
        "model_prefix": "apex",
    },
    "Test-3": {
        "name": "Test-3",
        "description": "Varying birth rates of apex predator",
        "data_path": "output/experiments/results/test-3_results.csv",
        "status": "complete",
        "n_models": 1,
        "n_params": 50,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["a_breed"],
        "models": ["Apex"],
        "vars": space,
        "models_config": [
            {
                "params": {
                    "model": "apex",
                    "migrate": False,
                    "apex": 500,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["a_breed"],
                }
            }
        ],
    },
    "Test-4": {
        "name": "Test-4",
        "description": "Varying lattice size and local saturation of prey for LV model",
        "data_path": "output/experiments/results/test-4_results.csv",
        "status": "complete",
        "n_models": 4,
        "n_params": 4,
        "populations": ["Prey"],
        "variables": ["f_max"],
        "models": [r"$L^{2}$ = 10", r"$L^{2}$ = 20", r"$L^{2}$ = 50", r"$L^{2}$ = 100"],
        "reps": 5,
        "steps": 500,
        "vars": None,
        "append": True,
        "models_config": [
            {
                "params": {
                    "width": l,
                    "height": l,
                    "f_max": f,
                    "model": "lv",
                    "apex": 0,
                    "super": 0,
                    "predator": 0,
                    "prey": 2000,
                    "params": ["f_max"],
                }
            }
            for l in [10, 20, 50, 100]  # noqa: E741
            for f in [1, 2, 5, 10]
        ],
    },
    "Test-5": {
        "name": "Test-5",
        "description": "Varying lethality of mesopredator",
        "data_path": "output/experiments/results/test-5_results.csv",
        "status": "complete",
        "n_models": 1,
        "n_params": 50,
        "populations": ["Prey", "Predator"],
        "variables": ["s_lethality"],
        "models": ["LV"],
        "vars": space_lethality,
        "models_config": [
            {
                "params": {
                    "model": "lv",
                    "apex": 0,
                    "super": 0,
                    "predator": 500,
                    "prey": 2000,
                    "params": ["s_lethality"],
                }
            }
        ],
    },
    "Test-6": {
        "name": "Test-6",
        "description": "Apex predator lethality",
        "data_path": "output/experiments/results/test-6_results.csv",
        "status": "complete",
        "n_models": 1,
        "n_params": 20,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["a_lethality"],
        "models": ["Apex"],
        "append": True,
        "vars": np.linspace(0, 1, 20),
        "models_config": [
            {
                "params": {
                    "model": "apex",
                    "migrate": False,
                    "apex": 500,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["a_lethality"],
                }
            }
        ],
    },
    "Test-7": {
        "name": "Test-7",
        "description": "Effect of starting density on model dynamics",
        "data_path": "output/experiments/results/test-7_results.csv",
        "status": "complete",
        "reps": 10,
        "steps": 1000,
        "n_models": 1,
        "n_params": None,
        "populations": ["Prey", "Predator", "Apex", "Super"],
        "variables": [],
        "models": None,
        "vars": None,
        "append": True,
        "models_config": [
            {
                "params": {
                    "prey": p,
                    "predator": pr,
                    "apex": a,
                    "super": s,
                    "model": "lv",
                    "migrate": False,
                }
            }
            for p in [100, 500, 1000, 2000, 5000]
            for pr in [100, 500, 1000, 2000, 5000]
            for a in [0, 100, 500, 1000, 2000]
            for s in [0, 100, 500, 1000, 2000]
        ],
    },
}

configs_b = {
    f"Scenario-{i + 1}": {
        "name": f"Scenario-{i + 1}",
        "description": "Varying the target and lethality of superpredators",
        "data_path": f"output/experiments/results/scenario-{i + 1}_results.csv",
        "status": "complete",
        "append": True,
        "n_models": 1,
        "populations": ["Prey", "Predator", "Super"],
        "variables": ["s_breed", "f_breed"],
        "models": [f"{lethality} -> {target}"],
        "vars": create_space(),
        "models_config": [
            {
                "params": {
                    "model": "super",
                    "super": 100,
                    "super_target": t,
                    "super_lethality": l,
                }
            }
        ],
    }
    for i, (lethality, l, target, t) in enumerate(  # noqa: E741
        (
            (lethality, l, target, t)
            for lethality, l in {"Non-lethal": 0, "Lethal": 1}.items()  # noqa: E741
            for target, t in {"Prey": "1", "Predator": "2", "Both": "Both"}.items()
        ),
        start=1,
    )
}

configs.update(configs_b)

# combined population metadata (names + colors) with backward-compatible mappings
pop_meta = {
    "Prey": {"label": "Prey", "color": "#2CF561"},
    "Predator": {"label": "Mesopredator", "color": "#2C41F5"},
    "Apex": {"label": "Apex Predator", "color": "#F5C82C"},
    "Super": {"label": "Superpredator", "color": "#F52D34"},
}

# set scenario info (label + description)
scenario_meta = {
    "Scenario-0": {"label": "0", "description": "Mesopredator consumes prey"},
    "Scenario-1": {"label": "1", "description": "Apex predator consumes mesopredator"},
    "Scenario-2": {"label": "5", "description": "Prey respond to non-lethal superpredator"},
    "Scenario-3": {"label": "6", "description": "Mesopredator respond to non-lethal superpredator"},
    "Scenario-4": {"label": "7", "description": "Both prey and mesopredator respond to non-lethal superpredator"},
    "Scenario-5": {"label": "2", "description": "Superpredator consumes prey"},
    "Scenario-6": {"label": "3", "description": "Superpredator consumes mesopredator"},
    "Scenario-7": {"label": "4", "description": "Superpredator consumes both prey and mesopredator"},
}
