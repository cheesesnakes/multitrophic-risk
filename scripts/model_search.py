# binary search for appropriate parameter space of the model

from model import model_1
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


## define parameter search function

class parameter_search():
    
    def __init__(self, predator = 50, prey=50,
                prey_info=False, predator_info=False,
                width=50, height=50,
                predator_energy=1, predator_breed = 0.1, 
                prey_breed = 0.1, prey_die = 0.1,
                steps=100,
                sample_id = 1, rep_id = 1,
                df = pd.DataFrame(columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])):

        # grid size
        self.width = width
        self.height = height
        self.f_max = width * height
        
        # number of agents
        self.predator = predator
        self.prey = prey
        
        #prey traits
        self.prey_info = prey_info
        self.f_breed = prey_breed
        self.f_die = prey_die
        
        # predator traits
        self.predator_info = predator_info
        self.s_energy = predator_energy
        self.s_breed = predator_breed
        
        # model steps
        self.steps = steps
        
        # data frame to store the results
        
        self.data = df
        
        # current sample id
        
        self.sample_id = sample_id
        
        # current rep id
        
        self.rep_id = rep_id
        
    def search_s_breed(self, start = 1, end = 100, a = None):
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]
            
            # model 
            
            m = model_1(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            s_energy = self.s_energy, s_breed = j,
                            f_breed = self.f_breed, f_die = self.f_die, f_max = self.f_max)
            
            
            m.run_model(self.steps)

            ## get number of prey, predator and resource agents at the end

            sample = m.count.get_model_vars_dataframe()

            prey = sample['Prey'].iloc[-1]
            predator = sample['Predator'].iloc[-1]
            
            # get last step
            
            last_step = sample.index[-1]
            
            #print(f"Shark breeding rate: {j} - Prey: {prey}, Predator: {predator} at step {last_step}")

            ## input for f_breed search
            
            self.res = prey < predator
            
            ## cpllect data
            
            temp = pd.DataFrame({'rep_id': self.rep_id, 'sample_id': self.sample_id, 'Prey': prey, 'Predator': predator,
                                 'step': last_step, 's_energy': self.s_energy, 's_breed': j, 'f_breed': self.f_breed,
                                 'f_die': self.f_die, 'predator_info': self.predator_info, 'prey_info': self.prey_info}, index = [0])
            
            self.data = pd.concat([self.data, temp], axis = 0)
            
            self.sample_id += 1
            
            self.data.to_csv('data.csv', index = False)

            print(f"Shark breeding rate: {j}")
            
            print(f"Replicate ID: {self.rep_id} - Sample ID: {self.sample_id}")
            
            if prey > 0 and predator > 0:
                
                if last_step < self.steps: # change the condition for a random search <-------------------------
                
                    if prey < predator:
                        
                        return self.search_s_breed(start = start, end = s - 1, a = a)
                    
                    else:
                        
                        return self.search_s_breed(start = s + 1, end = end, a = a)
                else:

                    #print(f"Shark breeding rate: {j} - Prey: {prey}, Predator: {predator}")
                
                    return 0
            
            elif prey == 0:
                
                return self.search_s_breed(start = start, end = s - 1, a=a)
                        
            elif predator == 0:
                

                return self.search_s_breed(start = s + 1, end = end, a=a)
            
        else:
            
            #print("No suitable shark breeding rate found")
            
            return 1
    
    def search_f_breed(self, start = 1, end = 100, a = None):
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]
            
            # define parameter search 
            
            print(f"Fish breeding rate: {j}")
            
            p_1 = parameter_search(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            predator_energy = self.s_energy, predator_breed = self.s_breed,
                            prey_breed = j, prey_die = self.f_die, steps = self.steps, sample_id = self.sample_id, rep_id= self.rep_id,
                            df = self.data) 
            
            search = p_1.search_s_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, p_1.data], axis = 0)
            
            self.sample_id += p_1.sample_id
            
            if search:
                
                if p_1.res:
                    
                    return p_1.search_f_breed(start = s + 1, end = end, a = a)
                
                else:
                    
                    return p_1.search_f_breed(start = start, end = s - 1, a = a)
            
            else:
            
                #print("No suitable fish breeding rate found")
                
                return 1
    
    def search_f_die(self, start = 1, end = 100, a = None):
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]
            
            # define parameter search 
            
            print(f"Fish death rate: {j}")
            
            p_1 = parameter_search(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            predator_energy = self.s_energy, predator_breed = self.s_breed,
                            prey_breed = self.f_breed, prey_die = j, steps = self.steps, sample_id = self.sample_id, rep_id = self.rep_id,
                            df = self.data) 
            
            search = p_1.search_f_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, p_1.data], axis = 0)
            
            self.sample_id
            
            if search:
                
                if p_1.res:
                    
                    return p_1.search_f_die(start = s + 1, end = end, a = a)
                
                else:
                    
                    return p_1.search_f_die(start = start, end = s - 1, a = a)
            
        else:
        
            #print("No suitable fish death rate found")
            
            return 1
    
    def vary_f_die(self, n, start = 1, end = 100, a = None):
        
        f_die = np.linspace(0, 1, n) 
        
        for i in f_die:
            
            print(f"Fish death rate: {i}")
            
            p_1 = parameter_search(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            predator_energy = self.s_energy, predator_breed = self.s_breed,
                            prey_breed = self.f_breed, prey_die = i, steps = self.steps, sample_id=self.sample_id,
                            rep_id = self.rep_id, df = self.data) 
            
            p_1.search_f_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, p_1.data], axis = 0)
            
            self.sample_id += p_1.sample_id
            
    def vary_s_energy(self, max, n, start = 1, end = 100, a = None):
        
        s_energy = np.linspace(0, max, n)
        
        for i in s_energy:
            
            print(f"Shark energy: {i}")
            
            p_1 = parameter_search(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            predator_energy = i, predator_breed = self.s_breed,
                            prey_breed = self.f_breed, prey_die = self.f_die, steps = self.steps, sample_id=self.sample_id, rep_id = self.rep_id,
                            df = self.data) 
            
            p_1.search_f_breed(start = start, end = end, a = a)
            
            self.sample_id += p_1.sample_id
            
            self.data = pd.concat([self.data, p_1.data], axis = 0)            

    # replication function
    
    def replicate(self, rep = 100, n = 5, start = 0, end = 99, a = np.linspace(0, 1, 100), max_energy = 5):
        
        for i in range(rep):
            
            print(f"Replication: {i}")
            
            p_1 = parameter_search(width = self.width, height = self.height,
                            predator = self.predator, prey = self.prey,
                            prey_info = self.prey_info, predator_info = self.predator_info,
                            predator_energy = self.s_energy, predator_breed = self.s_breed,
                            prey_breed = self.f_breed, prey_die = self.f_die, steps = self.steps, sample_id=self.sample_id, rep_id = self.rep_id,
                            df=self.data) 
            
            p_1.vary_s_energy(n = n, max = max_energy, start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, p_1.data], axis = 0)
            
            self.rep_id += 1
            

## run the search

# parameter space

a_step = 100
a = np.linspace(0, 1, a_step)
start = 0
end = a_step - 1


p = parameter_search(width=20, height=20, steps = 100)

#p.vary_s_energy(n = 5, max = 5, start = start, end = end, a = a)

p.replicate(rep = 1, n = 5, start = start, end = end, a = a, max_energy = 5)

# save the data

p.data.to_csv('data.csv', index = False)