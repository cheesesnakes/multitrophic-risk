# binary search for appropriate parameter space of the model

from functions.runner import model_run
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

## define parameter search function

class parameter_search():
    
    def __init__(self, **kwargs):
        
        # set default values
        
        defaults = {'model': 'lv',
                    'width': 20, 'height': 20, 
                    'predator': 10, 'prey': 100, 
                    'prey_info': False, 'predator_info': False, 
                    's_energy': 10, 's_breed': 0.01,
                    'f_breed': 0.01, 'f_die': 0.1,
                    'steps': 50, 'sample_id': 1, 'rep_id': 1, 
                    'df': pd.DataFrame(columns = ['rep_id', 'sample_id', 
                             'Prey', 'Predator', 
                             'step', 's_energy', 's_breed', 
                             'f_breed', 'f_die', 
                             'predator_info', 'prey_info'])}

        # if kwargs is empty, use defaults
        
        if not kwargs:
            
            kwargs = defaults
        
        # set the parameters
        
        self.model = kwargs['model']
        
        # grid size
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.f_max = self.width * self.height
        
        # number of agents
        self.predator = kwargs['predator']
        self.prey = kwargs['prey']
        
        #prey traits
        self.prey_info = kwargs['prey_info']
        self.f_breed = kwargs['prey_breed']
        self.f_die = kwargs['prey_die']
        
        # predator traits
        self.predator_info = kwargs['predator_info']
        self.s_energy = kwargs['s_energy']
        self.s_breed = kwargs['s_breed']
        
        # model steps
        self.steps = kwargs['steps']
        
        # data frame to store the results
        
        self.data = kwargs['df']
        
        # current sample id
        
        self.sample_id = kwargs['sample_id']
        
        # current rep id
        
        self.rep_id = kwargs['rep_id']
        
    def search_s_breed(self, start = 1, end = 100, a = None):
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s] 
            
            ## set params
            
            kwargs = self.kwargs
            
            ## update the shark breeding rate
            
            kwargs['s_breed'] = j
            
            # model 
            
            m = model_run(**kwargs)

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
            
            self.data.to_csv('sample.csv', index = False)

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
        
        ## set params
            
        kwargs = self.kwargs
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]
            
            # define parameter search 
            
            print(f"Fish breeding rate: {j}")
            
            ## update the fish breeding rate
            
            kwargs['f_breed'] = j
            
            # run search
            
            sample = parameter_search(kwargs=kwargs) 
            
            search = sample.search_s_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, sample.data], axis = 0)
            
            self.sample_id += sample.sample_id
            
            if search:
                
                if sample.res:
                    
                    return sample.search_f_breed(start = s + 1, end = end, a = a)
                
                else:
                    
                    return sample.search_f_breed(start = start, end = s - 1, a = a)
            
            else:
            
                #print("No suitable fish breeding rate found")
                
                return 1
    
    def search_f_die(self, start = 1, end = 100, a = None):
        
        ## set params
        
        kwargs = self.kwargs
        
        if end > start:
            
            s = np.random.choice(range(start, end))
            
            ## run model with the random value of shark breeding age
            
            # get the parameter value from a
            
            j = a[s]
            
            # define parameter search 
            
            print(f"Fish death rate: {j}")
            
            ## update the fish breeding rate
            
            kwargs['f_die'] = j
            
            sample = parameter_search(kwargs=kwargs) 
            
            search = sample.search_f_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, sample.data], axis = 0)
            
            self.sample_id += sample.sample_id
            
            if search:
                
                if sample.res:
                    
                    return sample.search_f_die(start = s + 1, end = end, a = a)
                
                else:
                    
                    return sample.search_f_die(start = start, end = s - 1, a = a)
            
        else:
        
            #print("No suitable fish death rate found")
            
            return 1
    
    def vary_f_die(self, n, start = 1, end = 100, a = None):
        
        f_die = np.linspace(0, 1, n) 
        
        ## set params
        
        kwargs = self.kwargs
        
        for i in f_die:
            
            print(f"Fish death rate: {i}")
            
            # update the fish death rate
            
            kwargs['f_die'] = i
            
            sample = parameter_search(kwargs=kwargs) 
            
            sample.search_f_breed(start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, sample.data], axis = 0)
            
            self.sample_id += sample.sample_id
            
    def vary_s_energy(self, max, n, start = 1, end = 100, a = None):
        
        s_energy = np.linspace(0, max, n)
        
        ## set params
        
        kwargs = self.kwargs
        
        for i in s_energy:
            
            print(f"Shark energy: {i}")
            
            ## update the shark energy
            
            kwargs['s_energy'] = i
            
            sample = parameter_search(kwargs=kwargs) 
            
            sample.search_f_breed(start = start, end = end, a = a)
            
            self.sample_id += sample.sample_id
            
            self.data = pd.concat([self.data, sample.data], axis = 0)            

    # replication function
    
    def replicate(self, rep = 100, n = 5, start = 0, end = 99, a = np.linspace(0, 1, 100), max_energy = 5):
        
        
        # set kwargs
        
        kwargs = self.kwargs
        
        for i in range(rep):
                
            print(f"Replication: {i}")
            
            ## update the rep id
            
            kwargs['rep_id'] = self.rep_id
            
            sample = parameter_search(kwargs=kwargs) 
            
            sample.vary_s_energy(n = n, max = max_energy, start = start, end = end, a = a)
            
            self.data = pd.concat([self.data, sample.data], axis = 0)
            
            self.rep_id += 1
        
        # save the data
        
        self.data.to_csv('search.csv', index = False)
        
        return self.data