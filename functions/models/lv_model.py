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
        self.info = kwargs.get('prey_info', False)
        self.f_breed = kwargs.get('f_breed', 0.5)
        self.f_die = kwargs.get('f_die', 0.1)
        self.f_max = kwargs.get('f_max', self.model.width*self.model.height/2)
        self.risk_cost = kwargs.get('risk_cost', 0.1)
        self.amount = 1
        self.energy = 0 #dummy variable
        self.age = 0
        self.lineage = kwargs.get('lineage', self.unique_id)
        self.dist = 0
        
        self.kwargs = kwargs
        self.kwargs['lineage'] = self.lineage
        
    def prey_random_reproduce(self):
    
        # get the number of prey agents
        
        num_prey = self.model.data_collector(Prey)
        
        ## calculate the probability of reproduction
        
        k = 1 - (num_prey/self.f_max)
        
        if k > 0 and self.f_breed > 0:
            
            breed = self.f_breed*k
            
            reproduce = np.random.binomial(1, breed)   
        
        else:
            
            reproduce = 0
        
        if reproduce == 1:
            
            ## create a new prey agent
            
            a = Prey(unique_id=self.model.schedule.get_agent_count(), model=self.model, 
                     pos=self.pos, **{k: v for k, v in self.kwargs.items()})
            
            self.model.schedule.add(a)
            
            self.model.grid.place_agent(a, self.pos)
            
            #print('Prey agent reproduced:', a.unique_id, a.pos)`     
    
    ##Prey information function 

    def risk(self):
        
        ## get current pos
        
        x, y = self.pos
        
        ## get neighbouring predator agents
        
        neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
        
        ## count the number of predator agents
        
        predators = [a for a in neighbours if isinstance(a, Predator)]
        
        ## if there are predators, move away
        
        if len(predators) > 0:
            
            predator = self.model.random.choice(predators)
                        
            ## get the predator pos
            
            x_p, y_p = predator.pos
            
            ## move away from the predator
            
            ## move away from the predator without jumping over the grid
            
            self.escape(x, y, x_p, y_p)
            
            ## add cost to birth rate
            
                
            self.f_breed = self.f_breed - self.risk_cost
            
        else:
            
            self.move()
            
            if self.f_breed < self.kwargs.get('f_breed', 0.5): # max birth rate
            
                self.f_breed = self.f_breed + self.risk_cost
    
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
        
    ## move function

    def move(self):
        
        ## get the current position
        
        x, y = self.pos
        
        ## find an empty cell
        
        #empty = self.model.grid.get_neighborhood((x,y), moore=True, include_center=False)
        
        ## if there are empty cells, move to a random empty cell
        
        #if len(empty) > 0:
            
        #    x, y = self.model.random.choice(empty)
            
            ## move the agent
            
        #    self.model.grid.move_agent(self, (x,y))
            
        ## if there are no empty cells, stay in the same position
        
        #else:
            
        #    self.model.grid.move_agent(self, (x,y))
        
        ## random walk
        
        x += np.random.choice([-1,0,1])
        y += np.random.choice([-1,0,1])
        
        ## move the agent
        
        self.model.grid.move_agent(self, (x,y))       
    
    # deterministic death function

    def die(self):
        
        ## remove the agent from the schedule
        
        self.model.schedule.remove(self)
        
        ## remove the agent from the grid
        
        self.model.grid.remove_agent(self)

    # random death function
        
    def random_die(self):
        
        death = np.random.binomial(1, self.f_die)
        
        if death == 1:
            
            self.die()
               
    ## step function
    
    def step(self):
        ##
        
        self.age += 1
        
        x,y = self.pos
        
        ## reproduce
        
        self.prey_random_reproduce()
        
        ## move the agent
        
        if self.info:
            
            self.risk()
        
        else:
            
            self.move()
        
        ## calculate distance travelled
        
        self.dist += np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        
        ## die
        
        self.random_die()
                        
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
        self.info = kwargs.get('predator_info', False)
        self.s_breed = kwargs.get('s_breed', 0.1)
        self.energy = kwargs.get('s_energy', 10)
        self.lethality = kwargs.get('s_lethality', 0.5)
        self.amount = 1
        self.lineage = kwargs.get('lineage', self.unique_id)
        self.dist = 0
        
        self.kwargs = kwargs
        self.kwargs['lineage'] = self.lineage
        self.kwargs = kwargs

    
    ## reproduce function predator

    def predator_random_reproduce(self):
            
        # get probability of reproduction
        
        reproduce = np.random.binomial(1, self.s_breed)  
        
        ## check if the energy is enough to reproduce
        
        if reproduce == 1:
            
            ## create a new predator agent
            
            a = Predator(unique_id=self.model.schedule.get_agent_count(), 
                        model=self.model, pos=self.pos, 
                        **{k: v for k, v in self.kwargs.items()})
            
            self.model.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.model.grid.place_agent(a, self.pos)
            
            #print('Predator agent reproduced:', a.unique_id, a.pos)
    
    #Predator information function

    def hunt(self):
        
        ## get current pos
        
        x, y = self.pos
        
        ## get neighbouring prey agents
        
        neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
        
        ## count the number of prey agents
        
        prey = [a for a in neighbours if isinstance(a, Prey)]
        
        ## choose a random prey agent
        
        if len(prey) > 0:
            
            a = self.model.random.choice(prey)
            
            ## get the prey pos
            
            x_p, y_p = a.pos
            
            ## move towards the prey
            
            x = x_p
            y = y_p
            
            ## move the agent
            
            self.model.grid.move_agent(self, (x,y))
        
        else:
            
            ## if there are no prey, move randomly
            
            self.move()

    ## move function

    def move(self):
        
        ## get the current position
        
        x, y = self.pos
        
        ## find an empty cell
        
        #empty = self.model.grid.get_neighborhood((x,y), moore=True, include_center=False)
        
        ## if there are empty cells, move to a random empty cell
        
        #if len(empty) > 0:
            
        #    x, y = self.model.random.choice(empty)
            
            ## move the agent
            
        #    self.model.grid.move_agent(self, (x,y))
            
        ## if there are no empty cells, stay in the same position
        
        #else:
            
        #    self.model.grid.move_agent(self, (x,y))
        
        ## random walk
        
        x += self.model.random.randint(-1,1)
        y += self.model.random.randint(-1,1)
        
        ## move the agent
        
        self.model.grid.move_agent(self, (x,y))        
    
    def predator_eat(self):
    
        ## get the current location
        
        x, y = self.pos
        
        ## get the prey at the current pos
        
        this_cell = self.model.grid.get_cell_list_contents((x,y))
        
        ## count the number of prey agents
        
        prey = [a for a in this_cell if isinstance(a, Prey)]
        
        ## choose a random prey agent
        
        if len(prey) > 0:
            
            a = self.model.random.choice(prey)
            
            ## remove the prey agent
            
            gain = 1
            
            l = np.random.binomial(1, self.lethality)
            
            if l == 1:
            
                self.model.grid.remove_agent(a)
                
                self.model.schedule.remove(a)
                
                ## increase the energy
                
                self.energy += gain
                
    def die(self):
    
        ## remove the agent from the schedule
        
        self.model.schedule.remove(self)
        
        ## remove the agent from the grid
        
        self.model.grid.remove_agent(self)
                
    ## step function
    
    def step(self):
        
        self.age += 1
        
        x,y = self.pos
        
        if self.energy <= 1:
            
            self.die()            
        else:
        
            ## reproduce

            self.predator_random_reproduce()
            
            ## move the agent

            if self.info:
                
                self.hunt()
            
            else:
                
                self.move()
                
            ## eat the prey

            self.predator_eat()
            
            ## decrease energy
            
            self.energy -= 1
            
            ## calculate distance travelled
            
            self.dist += np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
            
            #print('Predator energy:', self.energy)

        
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
        
        self.create_agents(**{k: v for k, v in kwargs.items()})
        
        ## defione data collector

        
        self.count = mesa.DataCollector(
            model_reporters = {'Prey': lambda m: m.data_collector(Prey), 
                           'Predator': lambda m: m.data_collector(Predator)
                           }
            )
        
        ## spatial data
        
        self.spatial = mesa.DataCollector(
            agent_reporters= {'x': lambda a: a.pos[0],
                               'y': lambda a: a.pos[1],
                               'AgentType': lambda a: type(a).__name__,
                               'Age': lambda a: a.age,
                                'Energy': lambda a: a.energy,
                                'UniqueID': lambda a: a.unique_id,
                                'Lineage': lambda a: a.lineage,
                                'DistanceTravelled': lambda a: a.dist
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
            
            a = Prey(unique_id=i, model=self, pos=(x,y), 
                     **{k: v for k, v in kwargs.items()})
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))
        
        for i in range(kwargs.get('predator', 10)):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Predator(unique_id=i, model=self, pos=(x,y), 
                         **{k: v for k, v in kwargs.items()})
            
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
    
    def run_model(self, steps = 100, progress = False, info = False, limit = 10000, stop = False):
        
        for i in range(steps):
            
            ## end the model if there are no prey or predator agents
            
            if self.data_collector(Predator) + self.data_collector(Prey) > limit:
                
                break
            
            elif (stop and self.data_collector(Predator) == 0) or (stop and self.data_collector(Prey) == 0):
                
                break
            
            else:
                self.step()
                
                if info:
                    
                    print('Number of prey:', self.data_collector(Prey), 
                          'Number of predators:', self.data_collector(Predator))
                
                if progress:
                    
                    print ('Number of prey:', self.data_collector(Prey), 
                          'Number of predators:', self.data_collector(Predator),
                          'Step:', i)
            
