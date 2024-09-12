from functions.models.abm_resource import Prey, Predator, Resource
import numpy as np

## reproduce function

def prey_reproduce_k(self):
    
    # get the number of prey agents
    
    num_prey = self.model.data_collector(Prey)
    
    ## calculate the probability of reproduction
    
    k = 1 - (num_prey/self.f_max)
    
    if k > 0:
        
        breed = self.f_breed*k
        
        reproduce = np.random.binomial(1, breed)   
    
    else:
        
        reproduce = 0
    
    if reproduce == 1:
        
        ## create a new prey agent
        
        a = Prey(unique_id=self.model.schedule.get_agent_count(), model=self.model, pos=self.pos)
        
        self.model.schedule.add(a)
        
        self.model.grid.place_agent(a, self.pos)
        
        #print('Prey agent reproduced:', a.unique_id, a.pos)

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
        
        p#rint('Prey agent reproduced:', a.unique_id, a.pos)
    
        #energy = energy / 2
        
        #a.energy = energy
        
    ## update the energy
    
    self.energy = energy    
                
## reproduce function predator

def predator_random_reproduce(self):
        
    # get probability of reproduction
    
    reproduce = np.random.binomial(1, self.s_breed)  
    
    ## check if the energy is enough to reproduce
    
    if reproduce == 1:
        
        ## create a new predator agent
        
        a = Predator(unique_id=self.model.schedule.get_agent_count(), 
                     model=self.model, pos=self.pos)
        
        self.model.schedule.add(a)
        
        ## add the agent to a random grid cell
        
        self.model.grid.place_agent(a, self.pos)
        
        #print('Predator agent reproduced:', a.unique_id, a.pos)

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
        
##Prey information function 

def risk(self):
    
    ## get current pos
    
    x, y = self.pos
    
    ## get neighbouring predator agents
    
    neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
    
    ## count the number of predator agents
    
    num_predators = len([a for a in neighbours if isinstance(a, Predator)])
    
    ## if there are predators, move away
    
    if num_predators > 0:
        
        ## get the predator agent
        for a in neighbours:
                
            if isinstance(a, Predator):
                
                predator = a
                
                break
        
        ## get the predator pos
        
        x_p, y_p = predator.pos
        
        ## move away from the predator
        
        x += x_p - self.model.random.randint(0,1)
        y += y_p - self.model.random.randint(0,1)
        
        ## move the agent
        
        self.model.grid.move_agent(self, (x,y))     
    
    else:
        
        ## if there are no predators, move randomly
        
         ## get the current position
    
        x, y = self.pos
        
        ## find an empty cell
        
        empty = self.model.grid.get_neighborhood((x,y), moore=True, include_center=False)
        
        ## if there are empty cells, move to a random empty cell
        
        if len(empty) > 0:
            
            x, y = self.model.random.choice(empty)
            
            ## move the agent
        
            self.model.grid.move_agent(self, (x,y))      

#Predator information function

def hunt(self):
    
    ## get current pos
    
    x, y = self.pos
    
    ## get neighbouring prey agents
    
    neighbours = self.model.grid.get_neighbors((x,y), moore=True, include_center=False)
    
    ## count the number of prey agents
    
    num_prey = len([a for a in neighbours if isinstance(a, Prey)])
    
    ## if there are prey, move towards
    
    if num_prey > 0:
        
        for a in neighbours:
                
            if isinstance(a, Prey):
                
                prey = a
                
                break
        
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
        
## death function

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
        
        die(self)

## predator eat function: predator agent eats prey agent
    
def predator_eat(self):
    
    ## get the current location
    
    x, y = self.pos
    
    ## get the prey at the current pos
    
    this_cell = self.model.grid.get_cell_list_contents((x,y))
    
    ## count the number of prey agents
    
    num_prey = len([a for a in this_cell if isinstance(a, Prey)])
    
    ## choose a random prey agent
    
    if num_prey > 0:

        for prey in this_cell:
            
            if isinstance(prey, Prey):
                
                ## remove the prey
                
                self.model.grid.remove_agent(prey)
                self.model.schedule.remove(prey)
                
                print('Predator agent ate:', prey.unique_id, prey.pos)
                
                self.energy += 20
                
                break

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
                
                food = self.model.random.randint(1, agent.amount)
                
                energy += food
                
                agent.amount -= food
            
        break
    
    ## update the energy
    
    self.energy = energy