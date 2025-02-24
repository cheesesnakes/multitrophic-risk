# binary search for appropriate parameter space of the model

from functions.runner import model_run
import pandas as pd
import ray

import warnings

warnings.filterwarnings("ignore")


## define parameter search function@

class experiment():
    
    def __init__(self, **kwargs):

        # if kwargs is empty, use defaults
        
        self.kwargs = kwargs
        
        # set the parameters
        
        self.model = kwargs.get('model', 'lv')
        
        # grid size
        self.width = kwargs.get('width', 100)
        self.height = kwargs.get('height', 100)
        self.f_max = kwargs.get('f_max', 2500) 
        
        # number of agents
        self.predator = kwargs.get('predator', 500)
        self.prey = kwargs.get('prey', 500)
        
        # model steps
        self.steps = kwargs.get('steps', 100)
        
        warnings.filterwarnings("ignore")
        
    # function: run the model
    
    @ray.remote
    def run(self, v, rep_id = 1, sample_id = 1, params = [], **kwargs):
        
        warnings.filterwarnings("ignore")
        
        # results data frame
        
        results = pd.DataFrame(columns = ['rep_id', 'sample_id', *params, 'Prey', 'Predator', 'Apex', 'Super', 'step'])
        
        # sample id
        
        self.sample_id = sample_id
        
        # iterate over the parameter values
        
        # get the parameter values
                
        # if v is one dimensional
        
        # set params
        
        kwargs = self.kwargs
        
        if v is None:
            
            # progress
            
            print(f"Running replicate {rep_id}, sample {sample_id}")
                
        
        else:
            
            if len(v.shape) == 1:
                        
                # update kwargs.valeus
                
                for i, p in enumerate(params):
                    
                    kwargs[p] = v[i]
                
                # progress
                
                print(f"Running replicate {rep_id}, sample {sample_id}, with {params} = {v}")
                
            else:
                
                # update kwargs.valeus
                
                for p in params:
                    
                    kwargs[p] = v
                
                # progress
                
                print(f"Running replicate {rep_id}, sample {sample_id}, with {params} = {v}")
                    
        # create the model
        
        m = model_run(**kwargs)
        
        # get the results
        
        sample = m.count.get_model_vars_dataframe()
        
        sample['step'] = sample.index
        
        # get the results and append to the data frame
        
        
        sample['rep_id'] = rep_id
        sample['sample_id'] = self.sample_id
        
        ## append the parameters
        
        for p in params:
            
            sample[p] = kwargs[p]
            
        
        # append to the results
        
        results = pd.concat([results, sample], axis = 0)
        
        results.to_csv(f"sample.csv", index = False)               
        
        # update the sample id
        
        self.sample_id += 1
            
        return results

    # parallel processing

    def parallel(self, v, params = [],rep = 1, **kwargs):
        
        num_cpus = kwargs.get('num_cpus', 4)
        
        ray.init(num_cpus = num_cpus)
        
        # results data frame
        
        self.data = pd.DataFrame(columns = ['rep_id', 'sample_id', *params, 'Prey', 'Predator', 'Apex', 'Super', 'step'])
        
        # iterate over the replicates
        
        if v is None:
            
            print("Number of runs: ", rep)
        
        else:            
            print("Number of parameter combinations: ", v.shape[0])
         
            print("Number of runs: ", v.shape[0]*rep)
        
        # iterate over the replicates
        
        future = []
        
        for i in range(rep):
            
            # assign num_cpu tasks at a time
            
            if not v is None:
                
                if len(v.shape) == 1:
                    
                    for j in range(len(v)):

                        future.append(self.run.remote(self = self, v = v[j], rep_id = i+1, sample_id = j, params = params, **kwargs))
                    
                else:

                    for j in range(v.shape[0]):
                            
                        future.append(self.run.remote(self = self, v = v[j,:], rep_id = i+1, sample_id = j, params = params, **kwargs))
            
            else:
                
                future.append(self.run.remote(self = self, v = None, rep_id = i+1, sample_id = 1, params = params, **kwargs))
        
        res = ray.get(future)
            
        for r in res:
            
            self.data = pd.concat([self.data, r], axis = 0)
            
        ray.shutdown()
        
        return self.data
