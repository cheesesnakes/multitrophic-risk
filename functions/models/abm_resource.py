# Title: Multi trophic risk alters predator - prey behavioural response races
# Description: Agent Based Model (ABM) of multi-trophic predator-prey interactions 
# in a spatially explicit environment 
# as an extension of Dewdney (1998) and Mitchel and Lima (2002).

# Import libraries
 
import mesa
import numpy as np

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
        self.energy = kwargs.get('f_energy', 10)
        self.pos = pos
        self.amount = 1
        self.info = kwargs.get('prey_info', False)
        self.r = kwargs.get('f_breed', 0.1)

        self.kwargs = kwargs
        
    ## reproduce function
    
    def prey_reproduce_energy(self):
        
        ## get the current energy
        
        energy = self.energy
        
        ## check if the energy is enough to reproduce
        
        reproductivity = np.random.binomial(1, self.r)
        
        if energy > 1 and reproductivity == 1:
            
            ## create a new prey agent
            
            traits = {k : v for k, v in self.kwargs.items()}
            
            ## update energy
            
            energy = traits['f_energy'] = energy / 2
            
            a = Prey(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos, 
                     **{k : v for k, v in traits.items()})
            
            self.model.schedule.add(a)
            
            self.model.grid.place_agent(a, self.pos)
            
            #print('Prey agent reproduced:', a.unique_id, a.pos)
        
            
        ## update the energy
        
        self.energy = energy    

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

    ##Prey information function 

    def risk(self):
        
        ## get current pos
        
        x, y = self.pos
        
        ## get neighbouring predator agents
        
        neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
        
        ## count the number of predator agents
        
        predators = [a for a in neighbours if isinstance(a, Predator)]
        
        ## if there are predators, move away from them
        
        if len(predators) > 0:
            
            predator = self.model.random.choice(predators)
            
            ## get the predator pos
            
            x_p, y_p = predator.pos
            
            ## move away from the predator
            
            ## calculate the distance between the prey and the predator
            
            distance = np.sqrt((x-x_p)**2 + (y-y_p)**2)
            
            ## move away from the predator without jumping over the grid
            
            self.escape(x, y, x_p, y_p)
            
        else:
            
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
    
    ## predator eat function
        
    def prey_eat(self):
        
        ## get the current energy
        
        energy = self.energy
        
        ## get the current pos
        
        x, y = self.pos
        
        ## get the resource at the current pos
        
        this_cell = self.model.grid.get_cell_list_contents((x,y))
        
        ## check if there is a resource in the cell
        
        for agent in this_cell:
            
            if isinstance(agent, Resource):
                
                if agent.amount > 0:
                    
                    ## if there is a resource, eat it
                    
                    # pick a random amount of resource to eat
                    
                    food = np.random.choice(list(range(0, agent.amount+1)))
                    
                    energy += food
                    
                    agent.amount -= food
                
            break
        
        ## update the energy
        
        self.energy = energy     

    # deterministic death function

    def die(self):
        
        ## remove the agent from the schedule
        
        self.model.schedule.remove(self)
        
        ## remove the agent from the grid
        
        self.model.grid.remove_agent(self)
                                
    ## step function
    
    def step(self):
        
        ## die
        
        if self.energy < 1:
            
            self.die()
        
        else:
            
            ## reproduce
            
            self.prey_reproduce_energy()
            
            # move
            
            if self.info:
                
                self.risk()
            
            else:
                
                self.move()
                                            
            ## eat the resource
            
            self.prey_eat()
            
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
    
    def __init__(self, model, unique_id, pos, **kwargs):
        
        self.unique_id = unique_id
        self.model = model
        self.energy = kwargs.get('s_energy', 10)
        self.pos = pos
        self.amount = 1
        self.info = kwargs.get('predator_info', False)
        self.r = kwargs.get('s_breed', 0.1)
        
        self.kwargs = kwargs
    
    ## reproduce function
    
    def predator_reproduce_energy(self):
        
        ## get the current energy
        
        energy = self.energy
        
        ## binomial distribution for reproductivity
        
        reproductivity = np.random.binomial(1, self.r)
        
        ## check if the energy is enough to reproduce
        
        if energy > 1 and reproductivity == 1:
            
            ## create a new predator agent
            
            ## get traits
            
            traits = {k : v for k, v in self.kwargs.items()}
            
            ## update energy
            
            energy = traits['s_energy'] = energy / 2
            
            a = Predator(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos, 
                         **{k : v for k, v in traits.items()})
            
            self.model.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.model.grid.place_agent(a, self.pos)
    
            #print('Predator agent reproduced:', a.unique_id, a.pos)
        
        ## update the energy
        
        self.energy = energy

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
            
            x += x_p
            y += y_p
            
            ## move the agent
            
            self.model.grid.move_agent(self, (x,y))
        
        else:
            
            ## if there are no prey, move randomly
            
            self.move()

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
            
            gain = a.energy
            
            ## eat the prey
            
            self.energy += gain
            
            ## remove the prey from the schedule
            
            self.model.schedule.remove(a)
            
            ## remove the prey from the grid
            
            self.model.grid.remove_agent(a)
            
    ## step function
    
    def step(self):
        
        
        ## die
        
        if self.energy < 1:
            
            self.die()
        
        else:   
            ## reproduce

            self.predator_reproduce_energy()
             
            ## move the agent

            if self.info:
                
                self.hunt()
            
            else:
                
                self.move()
                    
            ## eat the prey

            self.predator_eat()

            ## update the age
            
            self.energy -= 1
   
   # deterministic death function

    def die(self):
        
        ## remove the agent from the schedule
        
        self.model.schedule.remove(self)
        
        ## remove the agent from the grid
        
        self.model.grid.remove_agent(self)            
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
    
    def __init__(self, **kwargs):
        
        # initialize model
        self.width = kwargs.get('width', 20)
        self.height = kwargs.get('height', 20)
        
        ## create grid
        ## MultiGrid allows multiple agents per cell
        
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        
        ## create schedule
        
        self.schedule = mesa.time.RandomActivationByType(self)
        
        ## create agents
        
        self.create_agents(**kwargs)
        
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
    
    def create_agents(self, **kwargs):    
        ## create prey agents
        
        for i in range(kwargs.get('prey', 10)):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Prey(unique_id=i, model=self, pos=(x,y), **kwargs)
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))
        
        for i in range(kwargs.get('predator', 10)):
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            a = Predator(unique_id=i, model=self, pos=(x,y), **kwargs)
            
            self.schedule.add(a)
            
            ## add the agent to a random grid cell
            
            self.grid.place_agent(a, (x,y))

        ## create resource agents
        
        # place a resource agent in each cell
        
        patchy = kwargs.get('patchy', 0)
        
        if patchy == 0:
            
            for x in range(self.width):
                
                for y in range(self.height):
                        
                    a = Resource(unique_id=i, model=self, pos=(x,y), r = kwargs.get('resource_rate', 0.5))
                
                    self.schedule.add(a)
                
                    self.grid.place_agent(a, (x,y))
        
        else:
            
            ## create vector of positions
            
            x = np.random.choice(self.width, patchy)
            y = np.random.choice(self.height, patchy)
            
            for i in range(len(x)):
                
                a = Resource(unique_id=i, model=self, pos=(x[i],y[i]), r = kwargs.get('resource_rate', 0.5))
                
                self.schedule.add(a)
                
                self.grid.place_agent(a, (x[i],y[i]))
                
    ## step function
    
    def step(self):
        
        self.schedule.step()
        
        ## collect data
       
        self.count.collect(self)
        self.spatial.collect(self)
        
    ## run function
    
    def run_model(self, steps = 100, info = False, progress = False):
        
        for i in range(steps):
            
            ## end the model if there are no prey or predator agents
            
            if self.data_collector(Predator) == 0:
                
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
            
