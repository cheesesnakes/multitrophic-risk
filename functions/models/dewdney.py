# Title: Multi trophic risk alters predator - prey behavioural response races
# Description: Agent Based Model (ABM) of multi-trophic predator-prey interactions 
# in a spatially explicit environment 
# as an extension of Dewdney (1998) and Mitchel and Lima (2002).

# Import libraries
 
import mesa
import numpy as np

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
    
    def __init__(self, model, unique_id, pos, **kwargs):
        
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.amount = 1
        self.info = kwargs.get('info', False)
        self.age = 0
        self.f_breed = kwargs.get('f_breed', 5)
        self.f_max = kwargs.get('f_max', self.model.width * self.model.height)
        
        self.kwargs = kwargs
        
    ## reproduce function
    
    def reproduce(self):
        
        if self.age > self.f_breed and self.model.data_collector(Prey) < self.f_max:
            
            ## create a new prey agent
            
            a = Prey(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos, **{k:v for k,v in self.kwargs.items() if k != 'model'})
            
            self.model.schedule.add(a)
            
            self.model.grid.place_agent(a, self.pos)
            
            self.age = 0
            
            #print('Prey agent reproduced:', a.unique_id, a.pos)
 
    ## movement function: brownian motion
    
    def move(self):
        
        ## get the current position
        
        x, y = self.pos
        
        ## find an empty cell
        
        neighbours = self.model.grid.get_neighborhood((x,y), moore=True, include_center=False)
        
        ## count the number of empty cells
        
        empty = [cell for cell in neighbours if self.model.grid.is_cell_empty(cell)]
        
        ## if there are empty cells, move to a random empty cell
        
        if len(empty) > 0:
            
            x, y = self.model.random.choice(empty)
            
            ## move the agent
        
            self.model.grid.move_agent(self, (x,y))
        
    ## information function 
    
    def risk(self):
        
        ## get current pos
        
        x, y = self.pos
        
        ## get neighbouring predator agents
        
        neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
        
        ## count the number of predator agents
        
        num_predators = len([a for a in neighbours if isinstance(a, Predator)])
        
        ## if there are predators, move away
        
        if num_predators > 0:
            
            # list of predator agents
            
            predators = [a for a in neighbours if isinstance(a, Predator)]
            
            ## get the predator agent
            
            predator = self.model.random.choice(predators)
            
            ## get the predator pos
            
            x_p, y_p = predator.pos
            
            ## move away from the predator
            
            self.escape(x, y, x_p, y_p)     
        
        else:
            
            ## if there are no predators, move randomly
            
            self.move()       

    def escape(self, x, y, x_p, y_p):
        
        ## move away from the predator
        
        x += np.random.choice([-1,0,1])
        y += np.random.choice([-1,0,1])
        
        if ((x,y) == (x_p, y_p) or (x,y) == (x_p+1, y_p) or (x,y) == (x_p-1, y_p) or (x,y) == (x_p, y_p+1) or (x,y) == (x_p, y_p-1)):
            
            # avoid moving to the predator's position
            
            self.escape(x, y, x_p, y_p)
        
        elif (x,y) == self.pos:
            
            # avoid staying in the same position
             
            self.escape(x, y, x_p, y_p)
        
        else:
            
            # move
            
            self.model.grid.move_agent(self, (x,y)) 
        
        
    ## step function
    
    def step(self):
        
        ## reproduce
        
        self.reproduce()
        
        ## move the agent
        
        if self.info:
            
            self.risk()
        
        else:
            
            self.move()        
        
        ## increase age
        
        self.age += 1
        
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
    
    def __init__(self, model, unique_id, pos, **kwargs):
        
        self.unique_id = unique_id
        self.model = model
        self.age = 0
        self.pos = pos
        self.info = kwargs.get('info', False)
        self.s_breed = kwargs.get('s_breed', 5)
        self.energy = kwargs.get('energy', 10)
        self.f_energy = kwargs.get('f_energy', 1)
        
        self.kwargs = kwargs
        
        
    ## movement function: brownian motion
    
    def move(self):
        
        ## get the current position
        
        x, y = self.pos
        
        ## find an empty cell
        
        neighbours = self.model.grid.get_neighborhood((x,y), moore=True, include_center=False)
        
        ## count the number of empty cells
        
        empty = [cell for cell in neighbours if self.model.grid.is_cell_empty(cell)]
        
        ## if there are empty cells, move to a random empty cell
        
        if len(empty) > 0:
            
            x, y = self.model.random.choice(empty)
        
        ## move the agent
        
        self.model.grid.move_agent(self, (x,y))
    
    ## eat function\
        
    def eat(self):
        
        ## get the current location
        
        x, y = self.pos
        
        ## get the prey at the current pos
        
        this_cell = self.model.grid.get_cell_list_contents((x,y))
        
        ## count the number of prey agents
        
        prey = [a for a in this_cell if isinstance(a, Prey)]
        
        ## choose a random prey agent
        
        if len(prey) > 0:
    
            prey = self.model.random.choice(prey)
            
            ## remove the prey agent
            
            self.model.grid.remove_agent(prey)
            
            ## remove the prey agent from the schedule
            
            self.model.schedule.remove(prey)
            
            ## increase the predator's energy
            
            self.energy += self.f_energy
            
            #print('Predator agent ate:', prey.unique_id)
            
                    
        
    ## reproduce function
    
    def reproduce(self):
        
        
        ## check if the energy is enough to reproduce
        
        if self.age > self.s_breed:
            
            ## create a new predator agent
            
            a = Predator(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos, **self.kwargs)
            
            self.model.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.model.grid.place_agent(a, self.pos)
            
            self.age = 0
            
            #print('Predator agent reproduced:', a.unique_id, a.pos)
        
    
    ## die function
    
    def die(self):
        
        ## remove the agent from the schedule
        
        self.model.schedule.remove(self)
        
        ## remove the agent from the grid
        
        self.model.grid.remove_agent(self)
        
        #print('Predator agent starved:', self.unique_id)
        
            
        
    # information function
    
    def risk(self):
        
        ## get current pos
        
        x, y = self.pos
        
        ## get neighbouring prey agents
        
        neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
        
        ## count the number of prey agents
        
        num_prey = len([a for a in neighbours if isinstance(a, Prey)])
        
        ## if there are prey, move towards
        
        if num_prey > 0:
            
            ## get the prey agent
            
            prey = self.model.random.choice(neighbours)
            
            ## get the prey pos
            
            x_p, y_p = prey.pos
            
            ## move towards the prey
            
            x += x_p
            y += y_p
            
            ## move the agent
            
            self.model.grid.move_agent(self, (x,y))
        
        else:
            
            ## if there are no prey, move randomly
            
            self.move()
        
    ## step function
    
    def step(self):
        
        if self.energy < 1:
            
            self.die()
            
        else:
        
            ## reproduce

            self.reproduce()
            
            ## move the agent

            if self.info:
                
                self.risk()
            
            else:
                
                self.move()
            
            ## eat the prey

            self.eat()
            
            ## decrease energy
            
            self.energy -= 1

            ## increase age
            
            self.age += 1

        
# Define model class

class model_1(mesa.Model):
    '''
    Define model rules
    - 20x20 grid
    - Create agents
    - Move agents
    - Interact agents
    '''
    
    def __init__(self, **kwargs):
        
        # initialize model
        self.width = kwargs.get('width', 20)
        self.height = kwargs.get('height', 20)
        
        self.kwargs = kwargs
        
        ## create grid without multiple agents
        
        self.grid = mesa.space.MultiGrid(self.width, self.height, True)
        
        ## create schedule
        
        self.schedule = mesa.time.RandomActivation(self)
        
        ## create agents
        
        self.create_agents(**self.kwargs)
        
        ## defione data collector

        
        self.count = mesa.DataCollector(
            model_reporters = {'Prey': lambda m: m.data_collector(Prey), 
                           'Predator': lambda m: m.data_collector(Predator)
                           }
            )
        
        ## spatial data
        
        self.spatial = mesa.DataCollector(
            agent_reporters = {'x': lambda a: a.pos[0],
                               'y': lambda a: a.pos[1],
                               'AgentType': lambda a: type(a).__name__
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
    
    def create_agents(self, **kwargs):
            
        ## create prey agents
        
        for i in range(kwargs.get('prey', 100)):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Prey(unique_id= i, model=self, pos=(x,y), **{k:v for k,v in kwargs.items() if k != 'model'})
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))
        
        for i in range(kwargs.get('predator', 10)):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Predator(unique_id=i, model=self, pos=(x,y), **{k:v for k,v in kwargs.items() if k != 'model'})
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))

        
    ## step function
    
    def step(self):
        
        self.schedule.step()
        
        ## collect data
       
        self.count.collect(self)
        self.spatial.collect(self)
                
        
    ## run function
    
    def run_model(self, steps = 100, **kwargs):
        
        for i in range(steps):
            
            ## end the model if there are no prey or predator agents
            
            if self.data_collector(Prey) == 0 or self.data_collector(Predator) == 0:
                
                break
            
            ## if there are more than 1000 agents, end the model
            
            elif self.data_collector(Prey) + self.data_collector(Predator) > 50000:
                
                break
    
            else:
                
                if kwargs.get('progress', False): 
                    
                    print(f'Step: {i}, Prey: {self.data_collector(Prey)}, Predator: {self.data_collector(Predator)}')
                    
                self.step()
            
