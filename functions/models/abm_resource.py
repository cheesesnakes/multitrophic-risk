# Title: Multi trophic risk alters predator - prey behavioural response races
# Description: Agent Based Model (ABM) of multi-trophic predator-prey interactions 
# in a spatially explicit environment 
# as an extension of Dewdney (1998) and Mitchel and Lima (2002).

# Import libraries
 
import mesa
import numpy as np
from functions.models import behaviors

# Define agent classes

## the abm contains a predator, prey and resource agent class

class Prey(mesa.Agent):
    '''
    Prey agent class
    
    Attributes:
    - unique_id: int
    - energy: float, amount of energy the prey has
    - age: int, age of the prey, determines survival
    - prey eats resource, if resource is available
    - pos: tuple, x,y coordinates of the prey on the grid   
    '''
    
    def __init__(self, model, unique_id, pos, energy = 10, info = False, r = 0.1):
        
        self.unique_id = unique_id
        self.model = model
        self.energy = energy
        self.pos = pos
        self.amount = 1
        self.info = info
        self.r = r

    ## reproduce function
    
    def prey_reproduce_energy(self):
        
        ## get the current energy
        
        energy = self.energy
        
        ## check if the energy is enough to reproduce
        
        reproductivity = np.random.binomial(1, self.r)
        
        if energy > 1 and reproductivity == 1:
            
            ## create a new prey agent
            
            a = Prey(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos)
            
            self.model.schedule.add(a)
            
            self.model.grid.place_agent(a, self.pos)
            
            #print('Prey agent reproduced:', a.unique_id, a.pos)
        
            #energy = energy / 2
            
            #a.energy = energy
            
        ## update the energy
        
        self.energy = energy    
        
    ## step function
    
    def step(self):
        
        ## die
        
        if self.energy < 1:
            
            behaviors.die(self)
        
        else:
            
            ## reproduce
            
            self.prey_reproduce_energy()
            
            # move
            
            if self.info:
                
                behaviors.risk(self)
            
            else:
                
                behaviors.move(self)
                            
            ## eat the resource
            
            behaviors.prey_eat(self)
            
            ## update the age
            
            self.energy -= 1
        
class Predator(mesa.Agent):
    ''' 
    Predator agent class
    
    Attributes:
    - unique_id: int
    - energy: float, amount of energy the predator has
    - age: int, age of the predator, determines survival
    - location: tuple, x,y coordinates of the predator on the grid
    - predator eats prey, if prey is available
    '''
    
    def __init__(self, model, unique_id, pos, energy = 10, info = False, r = 0.1):
        
        self.unique_id = unique_id
        self.model = model
        self.energy = energy
        self.pos = pos
        self.amount = 1
        self.info = info
        self.r = r 
    
    ## reproduce function
    
    def predator_reproduce_energy(self):
        
        ## get the current energy
        
        energy = self.energy
        
        ## binomial distribution for reproductivity
        
        reproductivity = np.random.binomial(1, self.r)
        
        ## check if the energy is enough to reproduce
        
        if energy > 1 and reproductivity == 1:
            
            ## create a new predator agent
            
            a = Predator(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos)
            
            self.model.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.model.grid.place_agent(a, self.pos)
            
            ## give energy to the new agent
            
            #energy = energy / 2
            
            #a.energy = energy
            
            #print('Predator agent reproduced:', a.unique_id, a.pos)
        
        ## update the energy
        
        self.energy = energy
        
    ## step function
    
    def step(self):
        
        
        ## die
        
        if self.energy < 1:
            
            behaviors.die(self)
        
        else:   
            ## reproduce

            self.predator_reproduce_energy()
             
            ## move the agent

            if self.info:
                
                behaviors.hunt(self)
            
            else:
                
                behaviors.move(self)
                    
            ## eat the prey

            behaviors.predator_eat(self)

            ## update the age
            
            self.energy -= 0.1
            
class Resource(mesa.Agent):
    '''
    Resource agent class
    
    Attributes:
    - unique_id: int
    - amount: float, amount of resource available
    - pos: tuple, x,y coordinates of the resource on the grid
    '''
    
    def __init__(self, model, unique_id, pos, r = 0.5):
    
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.r = r
        self.amount = np.random.binomial(1, self.r)
            
    ## resource growth function
    
    def grow(self):
        
        ## get total amount of resource
        
        total_resource = sum([a.amount for a in self.model.schedule.agents if isinstance(a, Resource)])
        
        k = 1 - (total_resource / (self.model.width * self.model.height))
        
        if k < 0:
            
            productivity = 0
            
        else:
        
            productivity = np.random.binomial(1, k*self.r)
        
        if productivity == 1:
            
            ## get the current amount

            amount = self.amount
        
            ## get the new amount
        
            amount += 1
        
            ## update the amount
        
            self.amount = amount
        
    ## step function
    
    def step(self):
        
        ## grow the resource
        
        self.grow()    
        
# Define model class

class model_1(mesa.Model):
    '''
    Define model rules
    - 20x20 grid
    - Create agents
    - Move agents
    - Interact agents
    '''
    
    def __init__(self, width=20, height=20, prey=100, predator=100, prey_info=False, predator_info=False, predator_r = 0.1, prey_r = 0.1, prey_energy = 10, predator_energy = 10, patchy = 0, resource_rate = 0.5):
        
        # initialize model
        self.width = width
        self.height = height
        
        ## create grid
        ## MultiGrid allows multiple agents per cell
        
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        
        ## create schedule
        
        self.schedule = mesa.time.RandomActivationByType(self)
        
        ## create agents
        
        self.create_agents(predator=predator, prey=prey, prey_info=prey_info, predator_info=predator_info, 
                           prey_r=prey_r, predator_r=predator_r, prey_energy=prey_energy, predator_energy=predator_energy, patchy=patchy, resource_rate=0.5)
        
        ## defione data collector
        
        ## counts

        model_reporters = {'Prey': lambda m: m.data_collector(Prey), 
                           'Predator': lambda m: m.data_collector(Predator),
                           }

        
        self.count = mesa.DataCollector(
            model_reporters = {'Prey': lambda m: m.data_collector(Prey), 
                           'Predator': lambda m: m.data_collector(Predator),
                           # sum of resource amounts
                            'Resource': lambda m: sum([a.amount for a in m.schedule.agents if isinstance(a, Resource)]),
                           }
            )
        
        ## spatial data
        
        self.spatial = mesa.DataCollector(
            agent_reporters = {'x': lambda a: a.pos[0],
                               'y': lambda a: a.pos[1],
                               'AgentType': lambda a: type(a).__name__,
                                'Amount': lambda a: a.amount
                                }
        )
    ## data collector function
    
    def data_collector(self, agent_type):
        
        ## get list of agents
        
        agents = self.schedule.agents
        
        ## get number of agents
        
        agents = len([a for a in agents if isinstance(a, agent_type)])
        
        ## return the number of prey, predator and resource agents
        
        return agents
    
    ## create agents function
    
    def create_agents(self, prey = 100, predator = 100, prey_info = False, predator_info = False, 
                      prey_r = 0.1, predator_r = 0.1, prey_energy = 10, predator_energy = 10, 
                      patchy = 0, resource_rate = 0.5):    
        ## create prey agents
        
        for i in range(prey):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Prey(unique_id=i, model=self, pos=(x,y), energy=prey_energy, info=prey_info, r=prey_r)
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))
        
        for i in range(predator):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Predator(unique_id=i, model=self, pos=(x,y), energy=predator_energy, info=predator_info, r=predator_r)
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))

        ## create resource agents
        
        # place a resource agent in each cell
        
        if patchy == 0:
            
            for x in range(self.width):
                
                for y in range(self.height):
                        
                    a = Resource(unique_id=i, model=self, pos=(x,y), r = resource_rate)
                
                    self.schedule.add(a)
                
                    self.grid.place_agent(a, (x,y))
        
        else:
            
            ## create vector of positions
            
            x = np.random.choice(self.width, patchy)
            y = np.random.choice(self.height, patchy)
            
            for i in range(len(x)):
                
                a = Resource(unique_id=i, model=self, pos=(x[i],y[i]), r = resource_rate)
                
                self.schedule.add(a)
                
                self.grid.place_agent(a, (x[i],y[i]))
                
    ## step function
    
    def step(self):
        
        self.schedule.step()
        
        ## collect data
       
        self.count.collect(self)
        self.spatial.collect(self)
        
    ## run function
    
    def run_model(self, steps = 100):
        
        for i in range(steps):
            
            # stop if there are no more prey or predators
            
            if self.data_collector(Predator) == 0:
                    
                    break
            
            else:
                
                print('Step:', i)
            
                self.step()
            
