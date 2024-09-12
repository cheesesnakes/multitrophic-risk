# binary search for appropriate parameter space of the model

from functions.runner import model_run
import numpy as np
import pandas as pd
from math import factorial
import ray

import warnings

warnings.filterwarnings("ignore")


## define parameter search function@

class experiment():
    
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
        
        self.kwargs = kwargs
        
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
        self.f_breed = kwargs['f_breed']
        self.f_die = kwargs['f_die']
        
        # predator traits
        self.predator_info = kwargs['predator_info']
        self.s_energy = kwargs['s_energy']
        self.s_breed = kwargs['s_breed']
        
        # model steps
        self.steps = kwargs['steps']
        
        # data frame to store the results
        
        self.data = kwargs['df']
        
        warnings.filterwarnings("ignore")
        
    # function: create vector of vaiables for the model
    
    def variables(self, vary = ["s_breed", "f_breed"], n = 10):
        
        # possible parameter values
        
        params = ["s_energy", "s_breed", "f_breed", "f_die", "predator_info", "prey_info"]
        
        # catch error
        
        if len(vary) > len(params):
            
            raise ValueError("vary should be a subset of s_energy, s_breed, f_breed, f_die, predator_info, prey_info")
        
        # catch wrong parameter names
        
        for v in vary:
            
            if v not in params:
                
                raise ValueError("vary should be a subset of s_energy, s_breed, f_breed, f_die, predator_info, prey_info")        

        # create a list of possible values
        
        vars = []
        
        for i in range(len(vary)):
            
            if vary[i] == "predator_info" or vary[i] == "prey_info":
                
                t = np.array([True, False])
            
            else:
            
                t = np.linspace(0.1, 1, n)
            
            vars.append(t)
        
        # create a meshgrid of all combinations
        mesh = np.meshgrid(*vars)
        
        # reshape the meshgrid to the correct shape
        vars = np.array(mesh).T.reshape(-1, len(vary))
        
        # create the vector of parameters
        
        vec = np.zeros((vars.shape[0], len(params)))
        
        # iterate over the parameters
        
        for p in params:
            
            if p in vary:
                
                i = vary.index(p)
                
                if p == "s_energy":
                    
                    vec[:, params.index(p)] = vars[:, i] * 10
                
                else:
                
                    vec[:, params.index(p)] = vars[:, i]
                
            else:
                
                vec[:, params.index(p)] = self.__dict__[p]    
            
        
        # return the vector of parameters
        
        return vec
    
    # function: run the model
    
    @ray.remote
    def run(self, v, rep_id = 1, sample_id = 1):
        
        warnings.filterwarnings("ignore")
        
        # results data frame
        
        results = pd.DataFrame(columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
        
        # sample id
        
        self.sample_id = sample_id
        
        # iterate over the parameter values
        
        # get the parameter values
                
        # if v is one dimensional
        
        # set params
        
        kwargs = self.kwargs
         
        if len(v.shape) == 1:
                    
            # update kwargs
            
            kwargs['s_energy'] = v[0] * 10
            kwargs['s_breed'] = v[1]
            kwargs['f_breed'] = v[2]
            kwargs['f_die'] = v[3]
            kwargs['predator_info'] = v[4]
            kwargs['prey_info'] = v[5]
            
            # progress
            
            print(f"Running replicate {rep_id}, sample {sample_id}")
                
            # create the model
            
            m = model_run(kwargs=kwargs)
            
            # run the model
            
            m.run_model(steps = self.steps)
            
            # get the results
            
            sample = m.count.get_model_vars_dataframe()
            
            # get the results and append to the data frame
            
            res = [rep_id, self.sample_id, sample['Prey'].iloc[-1], sample['Predator'].iloc[-1], sample.index[-1], 
                   kwargs['s_energy'], kwargs['s_breed'], kwargs['f_breed'], kwargs['f_die'], kwargs['predator_info'], kwargs['prey_info']]
            
            # create a data frame
            
            res = pd.DataFrame([res], columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
            
            # append to the results
            
            results = pd.concat([results, res], axis = 0)         
            
            results.to_csv(f"replicate.csv", index = False)               
            
            # update the sample id
            
            self.sample_id += 1
        
        else: 
            
            for i in range(v.shape[0]):
                
                # update kwargs
            
                kwargs['s_energy'] = v[i, 0] * 10
                kwargs['s_breed'] = v[i, 1]
                kwargs['f_breed'] = v[i, 2]
                kwargs['f_die'] = v[i, 3]
                kwargs['predator_info'] = v[i, 4]
                kwargs['prey_info'] = v[i, 5]
                
                # progress
                
                print(f"Running replicate {rep_id}, sample {sample_id}")
                    
                # create the model
                
                m = model_run(kwargs=kwargs)
                
                # run the model
                
                m.run_model(steps = self.steps)
                
                # get the results
                
                sample = m.count.get_model_vars_dataframe()
                
                # get the results and append to the data frame
                
                res = [rep_id, self.sample_id, sample['Prey'].iloc[-1], sample['Predator'].iloc[-1], sample.index[-1], 
                    kwargs['s_energy'], kwargs['s_breed'], kwargs['f_breed'], kwargs['f_die'], kwargs['predator_info'], kwargs['prey_info']]
                
                # create a data frame
                
                res = pd.DataFrame([res], columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
                
                # append to the results
                
                results = pd.concat([results, res], axis = 0)         
                
                results.to_csv("sample.csv", index = False)               
                
                # update the sample id
                
                self.sample_id += 1
            
        return results

    # replicate the experiment
    
    def replicate(self, v, rep = 10):
        
        # results data frame
        
        self.data = pd.DataFrame(columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
        
        rep_id = 1
        
        # iterate over the replicates
        
        for i in range(rep):
            
            # run the experiment
            
            res = self.run(v, rep_id = rep_id)
            
            # add the results to the data frame
            
            self.data = pd.concat([self.data, res])
            
            # write
            
            self.data.to_csv(f"{self.model}_results.csv")
            
            rep_id += 1
        
        return self.data

    # parallel processing

    def parallel(self, vary = ['s_breed', 'f_breed'], rep = 1, n = 5):
        
        ray.init(num_cpus = 10)
        
        v = self.variables(vary = vary, n = n)
        
        # results data frame
        
        self.data = pd.DataFrame(columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
        
        # iterate over the replicates
        
        print("Number of parameter combinations: ", v.shape[0])
        
        # create an empty future
        
        future = []
        
        # iterate over the replicates
        
        for i in range(rep):
            
            for j in range(v.shape[0]):
                
                # concatenate the futures
            
                future.append(self.run.remote(self = self, v = v[j, :], rep_id = i + 1, sample_id = j + 1))

        print("Number of runs: ", len(future))
        
        res = ray.get(future)
            
        for r in res:
            
            self.data = pd.concat([self.data, r], axis = 0)
        
        self.data.to_csv(f"{self.model}_results.csv", index = False)
    
        ray.shutdown()
        
        return self.data

test = experiment()