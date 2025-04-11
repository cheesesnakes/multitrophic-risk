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
    "Experiment-1": {
        "name": "Experiment-1",
        "description": "Replacing the apex predator with super predator",
        "data_path": "output/experiments/results/Experiment-1_results.csv",
        "status": "SKIP",
        "append": True,
        "n_models": 2,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["s_breed", "f_breed"],
        "models": ["Apex", "Super"],
        "vars": create_space(),
        "models_config": [
            {
                "description": "apex",
                "params": {"model": "apex", "apex": 500, "super": 0},
            },
            {
                "description": "super",
                "params": {"model": "super", "apex": 0, "super": 100},
            },
        ],
    },
    "Experiment-2": {
        "name": "Experiment-2",
        "description": "Varying the target and lethality of superpredators",
        "data_path": "output/experiments/results/Experiment-2_results.csv",
        "status": "pending",
        "append": False,
        "n_models": 6,
        "populations": ["Prey", "Predator"],
        "variables": ["s_breed", "f_breed"],
        "models": [
            f"{leth} -> {t}"
            for leth in ["Non-lethal", "Lethal"]
            for t in ["Prey", "Predator", "Both"]
        ],
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
            for l in [0, 1]  # noqa: E741
            for t in ["1", "2", "Both"]
        ],
    },
    "Experiment-3": {
        "name": "Experiment-3",
        "description": "Determining the effects of predator and prey information",
        "data_path": "output/experiments/results/Experiment-3_results.csv",
        "status": "pending",
        "append": True,
        "n_models": 4,
        "populations": ["Prey", "Predator"],
        "variables": ["s_breed", "f_breed"],
        "models": [
            "Both Informed",
            "Prey Informed",
            "Predator Informed",
            "Both Naive",
        ],
        "vars": create_space(),
        "models_config": [
            {
                "params": {
                    "prey_info": p,
                    "predator_info": d,
                    "model": "lv",
                    "apex": 0,
                    "super": 0,
                }
            }
            for p in [True, False]
            for d in [True, False]
        ],
    },
    "Experiment-4a": {
        "name": "Experiment-4a",
        "description": "Effect of handling limit on mesopredator",
        "data_path": "output/experiments/results/Experiment-4_lv_results.csv",
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
    "Experiment-4b": {
        "name": "Experiment-4b",
        "description": "Effect of handling limit on apex predator",
        "data_path": "output/experiments/results/Experiment-4_apex_results.csv",
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
    "Experiment-5": {
        "name": "Experiment-5",
        "description": "Varying birth rates of apex predator",
        "data_path": "output/experiments/results/Experiment-5_results.csv",
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
                    "apex": 500,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["a_breed"],
                }
            }
        ],
    },
    "Experiment-6": {
        "name": "Experiment-6",
        "description": "Varying lattice size and local saturation of prey for LV model",
        "data_path": "output/experiments/results/Experiment-6_results.csv",
        "status": "pending",
        "n_models": 4,
        "n_params": 8,
        "populations": ["Prey"],
        "variables": ["f_max"],
        "models": [r"L^{2} = 10", r"L^{2} = 20", r"L^{2} = 50", r"L^{2} = 100"],
        "reps": 10,
        "steps": 2000,
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
            for f in [0, 1, 2, 5, 10, 20, 25, 50]
        ],
    },
    "Experiment-7": {
        "name": "Experiment-7",
        "description": "Varying lethality of mesopredator",
        "data_path": "output/experiments/results/Experiment-7_results.csv",
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
    "Experiment-8": {
        "name": "Experiment-8",
        "description": "Apex predator lethality",
        "data_path": "output/experiments/results/Experiment-8_results.csv",
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
                    "apex": 500,
                    "super": 0,
                    "predator": 500,
                    "prey": 500,
                    "params": ["a_lethality"],
                }
            }
        ],
    },
    "Experiment-9": {
        "name": "Experiment-9",
        "description": "Varying lethality of apex predator",
        "data_path": "output/experiments/results/Experiment-8_results.csv",
        "status": "complete",
        "n_models": 1,
        "n_params": 20,
        "populations": ["Prey", "Predator", "Apex"],
        "variables": ["a_lethality"],
        "models": ["Apex"],
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
                }
            }
            for p in [100, 500, 1000, 2000, 5000]
            for pr in [100, 500, 1000, 2000, 5000]
            for a in [0, 100, 500, 1000, 2000]
            for s in [0, 100, 500, 1000, 2000]
        ],
    },
}
