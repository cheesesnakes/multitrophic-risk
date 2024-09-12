# binary search for appropriate parameter space of the model

from dewdney import model_1
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# parameters

# grid size
width = 50 
height = 50

## number of agents
predator = 20
prey = 20

# duration of simulation
steps = 1000

# risk
prey_info = True
predator_info = True

# agent parameters
s_energy = 50 # shark starting energy
f_breed = 10 # fish breeding age
f_energy = 1 # fish energy value

## define parameter search function

def search(start = 1, end = 50, s_energy = s_energy, f_breed = f_breed, f_energy = f_energy, predator = predator, prey = prey, prey_info = prey_info, predator_info = predator_info, width = width, height = height, steps = steps):
    
    if end > start:
        
        s = (start + end) // 2

        ## run model with the random value of shark breeding age

        m = model_1(predator = predator, prey=prey, prey_info=prey_info, predator_info=predator_info, width=width, height=height, s_breed=s, s_energy=s_energy, f_breed=f_breed, f_energy=f_energy)

        m.run_model(steps)

        ## get number of prey, predator and resource agents over time

        data = m.count.get_model_vars_dataframe()

        prey = data['Prey']
        predator = data['Predator']

        ## get the number of prey and predator agents at the end of the simulation

        prey_end = prey.iloc[-1]
        predator_end = predator.iloc[-1]
        
        # get last step
        
        last_step = data.index[-1]
        
        print(f"Shark breeding age: {s} - Prey: {prey_end}, Predator: {predator_end} at step {last_step}")

        if prey_end > 0 and predator_end > 0:

            return print(f"Shark breeding age: {s} - Prey: {prey_end}, Predator: {predator_end}")
        
        elif prey_end == 0:
            
            return search(start = s + 1, end = end)
        
        elif predator_end == 0:
            
            return search(start = start, end = s - 1)
        
    else:
        
        return print("No suitable parameter found")

## run the search function

for i in range(1, 50):

    print(f"Fish breeding age: {i}")
    
    search(start = 1, end = 50, s_energy = s_energy, f_breed = i, f_energy = f_energy, predator = predator, prey = prey, prey_info = prey_info, predator_info = predator_info, width = width, height = height, steps = steps)
       
            
    
