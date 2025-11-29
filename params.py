# set model parameters

kwargs = {
    # model to run
    "limit": 100000,
    "num_cpus": 40,
    "cluster": True,
    "reps": 25,
    "debug": False,
    # model parameters
    "width": 100,
    "height": 100,
    "steps": 1000,
    "prey": 500,
    "predator": 500,
    "stop": False,
    ## prey traits
    "prey_info": True,
    "f_breed": 0.6,  # max birth rate
    "f_die": 0.1,  # constant
    "f_max": 5,
    "f_steps": 1,
    ## predator traits
    "predator_info": True,
    "s_max": 5,
    "s_breed": 0.3,  # max birth rate
    "s_die": 0.1,
    "s_lethality": 0.6,
    "s_apex_risk": True,
    "s_steps": 1,
    ## apex predator traits
    "apex_info": True,
    "a_max": 10,
    "a_breed": 0.15,  # max birth rate
    "a_die": 0.1,
    "a_lethality": 0.35,
    "a_target": "both",
    "a_steps": 1,
    "migrate": True,
    # super predator traits
    "super_target": 2,
    "super_lethality": 1,
    "super_max": 20,
    "super_steps": 1,
    # parameters to vary
    "params": ["s_breed", "f_breed"],
}
