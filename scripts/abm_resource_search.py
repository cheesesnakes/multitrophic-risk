# binary search for appropriate parameter space of the model

from abm_resource import model_1
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# parameters

# grid size
width = 100 
height = 100

# model

steps = 1000

# agents
predator = 50
prey = 50

# agent parameters

## predator

predator_info = True # if True, the predator will have a memory of the prey
p_energy = 10 # starting energy level

## prey
prey_info = True # if True, the prey will have a memory of the predator
prey_energy = 10 # starting energy level

## resource
resource_rate = 0.5 # rate of resource growth

# make a vector of 0 to 1 by 0.01

a = np.arange(0, 1, 0.01)

## define parameter search function

class parameter_search():
    
    def __init__(self, start = 1, end = 100, predator = predator, prey=prey,
                prey_info=prey_info, predator_info=predator_info,
                width=width, height=height,
                predator_energy=p_energy, prey_energy=prey_energy, steps=steps):
    
        self.start = start
        self.end = end
        self.predator = predator
        self.prey = prey
        self.prey_info = prey_info
        self.predator_info = predator_info
        self.width = width
        self.height = height
        self.predator_energy = predator_energy
        self.prey_energy = prey_energy
        self.steps = steps
        self.a = a
    
    def search(self, start = 1, end = 100, predator = predator, prey=prey, 
                prey_info=prey_info, predator_info=predator_info, 
                width=width, height=height,
                prey_r=0.5,
                predator_energy=p_energy, prey_energy=prey_energy, steps=steps, a=a):
        
        if end > start:
            
            s = (start + end) // 2
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]

            m = model_1(predator = predator, prey=prey, 
                prey_info=prey_info, predator_info=predator_info, 
                width=width, height=height,
                predator_r=j, prey_r=prey_r,
                predator_energy=predator_energy, prey_energy=prey_energy)
            
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
            
            print(f"Shark breeding rate: {j} - Prey: {prey_end}, Predator: {predator_end} at step {last_step}")

            self.res = prey_end < predator_end
            
            if prey_end > 0 and predator_end > 0:

                print(f"Shark breeding rate: {j} - Prey: {prey_end}, Predator: {predator_end}")
                
                return 0
            
            elif prey_end == 0:
                
                return self.search(start = start, end = s - 1)
                        
            elif predator_end == 0:
                

                return self.search(start = s + 1, end = end)
            
        else:
            
            print("No suitable parameter found")
            
            return 1

## run the search function
    
def search_fish(start = 1, end = 100, predator = predator, prey=prey, 
            prey_info=prey_info, predator_info=predator_info, 
            width=width, height=height,
            predator_energy=p_energy, prey_energy=prey_energy, steps=steps):
    
    if end > start:
        
        s = (start + end) // 2
        
        ## run model with the random value of shark breeding age
        
        # get the parameter value from a
        
        j = a[s]
        
        print(f"Fish breeding rate: {j}")
        
        test = parameter_search()
        
        test.search(start = 1, end = 100, predator = predator, prey=prey,
                        prey_info=prey_info, predator_info=predator_info, 
                        width=width, height=height,prey_r=j,
                        predator_energy=p_energy, prey_energy=prey_energy, steps=steps)
        
        if test.res:
            
            return search_fish(start = s + 1, end = end)
        
        else:
            
            return search_fish(start = start, end = s - 1)
        
search_fish(start = 1, end = 100, predator = predator, prey=prey,
            prey_info=prey_info, predator_info=predator_info, 
            width=width, height=height,
            predator_energy=p_energy, prey_energy=prey_energy, steps=steps)