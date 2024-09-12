# binary search for appropriate parameter space of the model

from model import model_1
import numpy as np
import pandas as pd
from math import factorial
import ray

import warnings

warnings.filterwarnings("ignore")


## define parameter search function@

class experiment():
    
    def __init__(self, predator = 50, prey=50,
                prey_info=False, predator_info=False,
                width=50, height=50,
                predator_energy=8, predator_breed = 0.1, 
                prey_breed = 0.1, prey_die = 0.1,
                steps=100):

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
        self.sample_id = 1
        
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
                
        if len(v.shape) == 1:
                    
            s_energy, s_breed, f_breed, f_die, predator_info, prey_info = v
            
            print(f"Running replicate {rep_id}, sample {sample_id}")
                
            # create the model
            
            m = model_1(width = self.width, height = self.height, 
                        predator = self.predator, prey = self.prey,
                        s_energy = s_energy, s_breed = s_breed,
                        f_breed = f_breed, f_die = f_die,
                        predator_info = predator_info, prey_info = prey_info)
            
            # run the model
            
            m.run_model(steps = self.steps)
            
            # get the results
            
            sample = m.count.get_model_vars_dataframe()
            
            # get the results and append to the data frame
            
            res = [rep_id, self.sample_id, sample['Prey'].iloc[-1], sample['Predator'].iloc[-1], sample.index[-1], s_energy, s_breed, f_breed, f_die, predator_info, prey_info]
            
            # create a data frame
            
            res = pd.DataFrame([res], columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
            
            # append to the results
            
            results = pd.concat([results, res], axis = 0)         
            
            results.to_csv("replicate.csv", index = False)               
            
            # update the sample id
            
            self.sample_id += 1
        
        else: 
            
            for i in range(v.shape[0]):
                
                s_energy, s_breed, f_breed, f_die, predator_info, prey_info = v[i, :]
                
                print(f"Running replicate {rep_id}, sample {sample_id}")
                
                # create the model
                
                m = model_1(width = self.width, height = self.height, 
                            predator = self.predator, prey = self.prey,
                            s_energy = s_energy, s_breed = s_breed,
                            f_breed = f_breed, f_die = f_die,
                            predator_info = predator_info, prey_info = prey_info)
                
                # run the model
                
                m.run_model(steps = self.steps)
                
                # get the results
                
                sample = m.count.get_model_vars_dataframe()
                
                # get the results and append to the data frame
                
                res = [rep_id, self.sample_id, sample['Prey'].iloc[-1], sample['Predator'].iloc[-1], sample.index[-1], s_energy, s_breed, f_breed, f_die, predator_info, prey_info]
                
                # create a data frame
                
                res = pd.DataFrame([res], columns = ['rep_id', 'sample_id', 'Prey', 'Predator', 'step', 's_energy', 's_breed', 'f_breed', 'f_die', 'predator_info', 'prey_info'])
                
                # append to the results
                
                results = pd.concat([results, res], axis = 0)         
                
                results.to_csv("replicate.csv", index = False)               
                
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
            
            self.data.to_csv("results.csv")
            
            rep_id += 1

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
        
        self.data.to_csv("results.csv", index = False)
    
        ray.shutdown()
        
        return self.data    
            
# test the function
      
def test():        
    # set model parameters

    # parameters

    # grid size
    width = 20 
    height = 20
    f_max = width * height

    # number of agents
    predator = 20
    prey = 20

    # duration of simulation
    steps = 200

    # risk
    prey_info = False
    predator_info = False

    # agent parameters
    s_breed = 0.1 # shark breeding rate
    s_energy = 5 # shark starting energy
    f_breed = 0.1 # fish breeding rate
    f_die = 0.1 # fish death rate


    exp = experiment(predator = predator, prey=prey, 
                    prey_info=prey_info, predator_info=predator_info, 
                    width=width, height=height, steps=steps, 
                    prey_die=f_die, prey_breed=f_breed,
                    predator_breed=s_breed, predator_energy=s_energy)

    # run the experiment

    exp.parallel(vary=['s_breed', 'f_breed', 'predator_info', 'prey_info'], rep=5, n=20)