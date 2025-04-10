import numpy as np

depth = 50
space = np.linspace(0.1, 1, depth)
space_100 = np.linspace(0, 100, depth)
space_lethality = np.linspace(0, 1, depth)


def create_space():
    x = space
    return np.array(np.meshgrid(x, x)).reshape(2, -1).T


configs = [
    {
        "name": "Experiment-1",
        "description": "Replacing the apex predator with super predator",
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
    {
        "name": "Experiment-2",
        "description": "Varying super predator target & lethality",
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
    {
        "name": "Experiment-3",
        "description": "Predator/prey info toggles",
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
    {
        "name": "Experiment-4a",
        "description": "Handling limits (LV)",
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
    {
        "name": "Experiment-4b",
        "description": "Handling limits (Apex)",
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
    {
        "name": "Experiment-5",
        "description": "Apex birth rate",
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
    {
        "name": "Experiment-6",
        "description": "Lattice size & local saturation",
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
    {
        "name": "Experiment-7",
        "description": "Mesopredator lethality",
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
    {
        "name": "Experiment-8",
        "description": "Apex predator lethality",
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
    {
        "name": "Experiment-9",
        "description": "Varying initial densities",
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
]
