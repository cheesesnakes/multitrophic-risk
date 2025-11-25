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
    """
    Prey agent class

    Attributes:
    - unique_id: int
    - energy: float, amount of energy the prey has
    - age: int, age of the prey, determines survival
    - prey eats resource, if resource is available
    - pos: tuple, x,y coordinates of the prey on the grid
    """

    def __init__(self, model, unique_id, pos, **kwargs):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos

        # traits
        self.info = kwargs.get("prey_info", False)
        self.f_super_risk = kwargs.get("f_super_risk", False)
        self.risk_cost = kwargs.get("risk_cost", 0.1)
        self.f_breed = kwargs.get("f_breed", 0.2)
        self.f_die = kwargs.get("f_die", 0.1)
        self.f_max = kwargs.get("f_max", 10)
        self.steps = kwargs.get("f_steps", 5)

        # state variables

        self.kwargs = kwargs

    ## move function

    def move(self):
        ## get the current position

        x, y = self.pos

        ## choose a new position

        x += np.random.choice([-1, 0, 1])
        y += np.random.choice([-1, 0, 1])

        ## move the agent

        self.model.grid.move_agent(self, (x, y))

    ## escape

    def escape(self, x, y, x_p, y_p):
        ## possible moves

        if not x_p == x:
            a = 2 * x - x_p

        else:
            a = np.random.choice([x - 1, x + 1])

        if not y_p == y:
            b = 2 * y - y_p

        else:
            b = np.random.choice([y - 1, y + 1])

        # move

        self.model.grid.move_agent(self, (a, b))

    ##Prey information function

    def risk(self):
        ## get current pos

        x, y = self.pos

        ## get neighbouring predator agents

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=False
        )

        ## count the number of predator agents

        if self.f_super_risk:
            predators = [
                a for a in neighbours if isinstance(a, Super) or isinstance(a, Predator)
            ]

        else:
            predators = [a for a in neighbours if isinstance(a, Predator)]

        ## if there are predators, move away

        if len(predators) > 0:
            predator = self.model.random.choice(predators)

            ## get the predator pos

            x_p, y_p = predator.pos

            ## move away from the predator

            ## move away from the predator without jumping over the grid

            self.escape(x, y, x_p, y_p)

            ## breeding cost

            self.f_breed = self.f_breed * (1 - self.risk_cost)

        else:
            self.move()

            self.f_breed = self.kwargs.get("f_breed", 0.2)

    # prey reproduction function
    def prey_random_reproduce(self):
        ## get neighbours

        x, y = self.pos

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=True
        )

        ## count the number of prey agents

        prey = [a for a in neighbours if isinstance(a, Prey)]

        ## calculate the probability of reproduction

        reproduce = np.random.binomial(1, self.f_breed)

        if len(prey) < self.f_max and reproduce == 1:
            ## create a new prey agent

            a = Prey(
                unique_id=self.model.schedule.get_agent_count(),
                model=self.model,
                pos=self.pos,
                **{k: v for k, v in self.kwargs.items()},
            )

            self.model.schedule.add(a)

            self.model.grid.place_agent(a, self.pos)

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
        ## move the agent

        for i in range(self.steps):
            if self.info:
                self.risk()

            else:
                self.move()

        ## reproduce

        self.prey_random_reproduce()

        ## die

        self.random_die()


class Predator(mesa.Agent):
    """
    Predator agent class

    Attributes:
    - unique_id: int
    - energy: float, amount of energy the predator has
    - age: int, age of the predator, determines survival
    - location: tuple, x,y coordinates of the predator on the grid
    - predator eats prey, if prey is available
    """

    def __init__(self, model, unique_id, pos, **kwargs):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos

        # traits
        self.info = kwargs.get("predator_info", False)
        self.lethality = kwargs.get("s_lethality", 0.5)

        self.super_risk = kwargs.get("s_super_risk", False)
        self.apex_risk = kwargs.get("s_apex_risk", False)
        self.risk_cost = kwargs.get("risk_cost", 0.01)

        self.breed = kwargs.get("s_breed", 0.1)
        self.s_die = kwargs.get("s_die", 0.01)
        self.steps = kwargs.get("s_steps", 10)

        # state variables
        self.energy = 1
        self.max_energy = kwargs.get("s_max", 1)

        self.kwargs = kwargs

    ## move function

    def move(self):
        ## get the current position

        x, y = self.pos

        ## random walk

        x += self.model.random.randint(-1, 1)
        y += self.model.random.randint(-1, 1)

        ## move the agent

        self.model.grid.move_agent(self, (x, y))

    ## escape

    def escape(self, x, y, x_p, y_p):
        ## possible moves

        if not x_p == x:
            a = 2 * x - x_p

        else:
            a = np.random.choice([x - 1, x + 1])

        if not y_p == y:
            b = 2 * y - y_p

        else:
            b = np.random.choice([y - 1, y + 1])

        # move

        self.model.grid.move_agent(self, (a, b))

    ## hunt

    def hunt(self):
        ## get current pos

        x, y = self.pos

        ## get neighbouring prey agents

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=True
        )

        ## count the number of prey agents

        prey = [a for a in neighbours if isinstance(a, Prey)]

        ## choose a random prey agent

        if len(prey) > 0:
            a = self.model.random.choice(prey)

            ## get the prey pos

            x, y = a.pos

            ## move the agent

            self.model.grid.move_agent(self, (x, y))

        else:
            ## if there are no prey, move randomly

            self.move()

    # Predator information function

    def predator_risk(self):
        ## get current pos

        x, y = self.pos

        ## get neighbouring predator agents

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=True
        )

        ## count the number of predator agents

        if self.super_risk:
            predators = [a for a in neighbours if isinstance(a, Super)]

        elif self.apex_risk:
            predators = [a for a in neighbours if isinstance(a, Apex)]

        ## if there are predators, move away

        if len(predators) > 0:
            predator = self.model.random.choice(predators)

            ## get the predator pos

            x_p, y_p = predator.pos

            ## move away from the predator

            ## move away from the predator without jumping over the grid

            self.escape(x, y, x_p, y_p)

            self.breed = self.breed * (1 - self.risk_cost)

        else:
            self.hunt()

            self.breed = self.kwargs.get("s_breed", 0.5)

    ## eat function

    def predator_eat(self):
        ## get the current location

        x, y = self.pos

        ## get the prey at the current pos

        this_cell = self.model.grid.get_cell_list_contents((x, y))

        ## count the number of prey agents

        prey = [a for a in this_cell if isinstance(a, Prey)]

        ## choose a random prey agent until satiated or no prey left

        while self.energy < self.max_energy and len(prey) > 0:
            a = self.model.random.choice(prey)

            ## pop the prey agent

            prey.pop(prey.index(a))

            ## remove the prey agent

            l = np.random.binomial(1, self.lethality)

            if l == 1:
                self.model.grid.remove_agent(a)

                self.model.schedule.remove(a)

                ## increase the energy

                self.energy += 1

    ## reproduce function predator

    def predator_random_reproduce(self):
        # get probability of reproduction

        reproduce = np.random.binomial(1, self.breed)

        ## check if the energy is enough to reproduce

        if reproduce == 1:
            ## create a new predator agent

            a = Predator(
                unique_id=self.model.schedule.get_agent_count(),
                model=self.model,
                pos=self.pos,
                **{k: v for k, v in self.kwargs.items()},
            )

            self.model.schedule.add(a)

            ## add the agent to a random grid cell

            self.model.grid.place_agent(a, self.pos)

            # print('Predator agent reproduced:', a.unique_id, a.pos)

    # deterministic death function

    def die(self):
        ## remove the agent from the schedule

        self.model.schedule.remove(self)

        ## remove the agent from the grid

        self.model.grid.remove_agent(self)

    # random death function

    def random_die(self):
        death = np.random.binomial(1, self.s_die)

        if death == 1:
            self.die()

    ## step function

    def step(self):
        ## move the agent

        for i in range(self.steps):
            if (self.info and self.super_risk) or (self.info and self.apex_risk):
                self.predator_risk()

            elif self.info:
                self.hunt()

            else:
                self.move()

        ## eat the prey

        self.predator_eat()

        ## reproduce

        while self.energy > 0:
            self.energy -= 1

            self.predator_random_reproduce()

        ## die

        self.random_die()


class Apex(mesa.Agent):
    """
    Apex Predator agent class

    Attributes:
    - unique_id: int
    - energy: float, amount of energy the predator has
    - age: int, age of the predator, determines survival
    - location: tuple, x,y coordinates of the predator on the grid
    - predator eats prey, if prey is available
    """

    def __init__(self, model, unique_id, pos, **kwargs):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos

        # traits
        self.info = kwargs.get("apex_info", False)
        self.lethality = kwargs.get("a_lethality", 0.15)
        self.target = kwargs.get("a_target", "2")

        self.steps = kwargs.get("a_steps", 20)

        self.breed = kwargs.get("a_breed", 0.1)
        self.a_die = kwargs.get("a_die", 0.001)
        self.max_energy = kwargs.get("a_max", 10)
        self.energy = 1

        self.kwargs = kwargs

        if self.target == "1":
            self.target = "prey"

        elif self.target == "2":
            self.target = "predator"

        else:
            self.target = "both"

    ## move function

    def move(self):
        ## get the current position

        x, y = self.pos

        ## random walk

        x += self.model.random.randint(-1, 1)
        y += self.model.random.randint(-1, 1)

        ## move the agent

        self.model.grid.move_agent(self, (x, y))

    ## hunt, Predator information function

    def apex_hunt(self):
        ## get current pos

        x, y = self.pos

        ## get neighbouring prey agents

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=True
        )

        if self.target == "predator":
            prey = [a for a in neighbours if isinstance(a, Predator)]

        elif self.target == "prey":
            prey = [a for a in neighbours if isinstance(a, Prey)]

        else:
            prey = [
                a for a in neighbours if isinstance(a, Predator) or isinstance(a, Prey)
            ]

        ## choose a random prey agent

        if len(prey) > 0:
            a = self.model.random.choice(prey)

            ## get the prey pos

            x_p, y_p = a.pos

            ## move towards the prey

            x = x_p
            y = y_p

            ## move the agent

            self.model.grid.move_agent(self, (x, y))

        else:
            ## if there are no prey, move randomly

            self.move()

    # eat

    def apex_eat(self):
        ## get the current location

        x, y = self.pos

        ## get the prey at the current pos

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=False
        )

        ## count the number of prey agents
        if self.target == "predator":
            prey = [a for a in neighbours if isinstance(a, Predator)]

        elif self.target == "prey":
            prey = [a for a in neighbours if isinstance(a, Prey)]

        else:
            prey = [
                a for a in neighbours if isinstance(a, Predator) or isinstance(a, Prey)
            ]

        ## choose a random prey agent

        while self.energy < self.max_energy and len(prey) > 0:
            ## choose
            a = self.model.random.choice(prey)

            ## pop the prey agent

            prey.pop(prey.index(a))

            ## remove the prey agent

            l = np.random.binomial(1, self.lethality)

            if l == 1:
                self.model.grid.remove_agent(a)

                self.model.schedule.remove(a)

                ## increase the energy

                self.energy += 1

    ## reproduce function predator

    def apex_random_reproduce(self):
        ## get probability of reproduction

        reproduce = np.random.binomial(1, self.breed)

        if reproduce == 1:
            ## create a new predator agent

            a = Apex(
                unique_id=self.model.schedule.get_agent_count(),
                model=self.model,
                pos=self.pos,
                **{k: v for k, v in self.kwargs.items()},
            )

            self.model.schedule.add(a)

            ## add the agent to a random grid cell

            self.model.grid.place_agent(a, self.pos)

            # print('Predator agent reproduced:', a.unique_id, a.pos)

    # die

    def die(self):
        ## remove the agent from the schedule

        self.model.schedule.remove(self)

        ## remove the agent from the grid

        self.model.grid.remove_agent(self)

    # random death function

    def random_die(self):
        death = np.random.binomial(1, self.a_die)

        if death == 1:
            self.die()

    ## step function

    def step(self):
        ## move the agent

        for i in range(self.steps):
            if self.info:
                self.apex_hunt()

            else:
                self.move()

        ## eat the prey

        self.apex_eat()

        ## reproduce

        while self.energy > 0:
            self.apex_random_reproduce()

            self.energy -= 1

        ## die

        self.random_die()


class Super(mesa.Agent):
    """
    Super Predator agent class

    Attributes:
    - unique_id: int
    - energy: float, amount of energy the predator has
    - age: int, age of the predator, determines survival
    - location: tuple, x,y coordinates of the predator on the grid
    - predator eats prey, if prey is available
    """

    def __init__(self, model, unique_id, pos, **kwargs):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos

        # traits
        self.info = True
        self.lethality = kwargs.get("super_lethality", 1)
        self.target = kwargs.get("super_target", "2")
        self.steps = kwargs.get("super_steps", 20)
        self.max_energy = kwargs.get("super_max", 10)
        self.energy = 0

        self.kwargs = kwargs

        if self.target == "1":
            self.target = "prey"

        elif self.target == "2":
            self.target = "predator"

        else:
            self.target = "both"

    ## move function

    def move(self):
        ## get the current position

        x, y = self.pos

        ## random walk

        x += self.model.random.randint(-1, 1)
        y += self.model.random.randint(-1, 1)

        ## move the agent

        self.model.grid.move_agent(self, (x, y))

    # Predator information function

    def super_hunt(self):
        ## get current pos

        x, y = self.pos

        ## get neighbouring prey agents

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=True
        )

        if self.target == "predator":
            prey = [a for a in neighbours if isinstance(a, Predator)]

        elif self.target == "prey":
            prey = [a for a in neighbours if isinstance(a, Prey)]

        else:
            prey = [
                a for a in neighbours if isinstance(a, Predator) or isinstance(a, Prey)
            ]

        ## choose a random prey agent

        if len(prey) > 0:
            a = self.model.random.choice(prey)

            ## get the prey pos

            x_p, y_p = a.pos

            ## move towards the prey

            x = x_p
            y = y_p

            ## move the agent

            self.model.grid.move_agent(self, (x, y))

        else:
            ## if there are no prey, move randomly

            self.move()

    ## eat function

    def super_eat(self):
        ## get the current location

        x, y = self.pos

        ## get the prey at the current pos

        neighbours = self.model.grid.get_neighbors(
            (x, y), moore=True, include_center=False
        )

        ## count the number of prey agents
        if self.target == "predator":
            prey = [a for a in neighbours if isinstance(a, Predator)]

        elif self.target == "prey":
            prey = [a for a in neighbours if isinstance(a, Prey)]

        else:
            prey = [
                a for a in neighbours if isinstance(a, Predator) or isinstance(a, Prey)
            ]

        ## choose a random prey agent

        while self.energy < self.max_energy and len(prey) > 0:
            ## choose
            a = self.model.random.choice(prey)

            ## pop the prey agent

            prey.pop(prey.index(a))

            ## remove the prey agent

            l = np.random.binomial(1, self.lethality)

            if l == 1:
                self.model.grid.remove_agent(a)

                self.model.schedule.remove(a)

                ## increase the energy

                self.energy += 1

    ## step function

    def step(self):
        ## move the agent

        for i in range(self.steps):
            self.super_hunt()

        ## eat the prey

        self.super_eat()

        ## reset energy

        self.energy = 0


# Define model class


class model(mesa.Model):
    """
    Define model rules
    - 20x20 grid
    - Create agents
    - Move agents
    - Interact agents
    """

    def __init__(self, **kwargs):
        # initialize model
        self.width = kwargs.get("width", 20)
        self.height = kwargs.get("height", 20)

        self.kwargs = kwargs

        ## create grid without multiple agents

        self.grid = mesa.space.MultiGrid(self.width, self.height, True)

        ## create schedule

        self.schedule = mesa.time.RandomActivation(self)

        ## create agents

        self.create_agents(**{k: v for k, v in kwargs.items()})

        ## defione data collector

        self.count = mesa.DataCollector(
            model_reporters={
                "Prey": lambda m: m.n_Prey,
                "Predator": lambda m: m.n_Predators,
                "Apex": lambda m: m.n_Apex,
                "Super": lambda m: m.n_Super,
            }
        )

        ## spatial data

        self.spatial = mesa.DataCollector(
            agent_reporters={
                "x": lambda a: a.pos[0],
                "y": lambda a: a.pos[1],
                "AgentType": lambda a: type(a).__name__,
                "UniqueID": lambda a: a.unique_id,
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

        for i in range(kwargs.get("prey", 0)):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)

            a = Prey(
                unique_id=i, model=self, pos=(x, y), **{k: v for k, v in kwargs.items()}
            )

            self.schedule.add(a)

            ## add the agent to a random grid cell

            self.grid.place_agent(a, (x, y))

        for i in range(kwargs.get("predator", 0)):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)

            a = Predator(
                unique_id=i, model=self, pos=(x, y), **{k: v for k, v in kwargs.items()}
            )

            self.schedule.add(a)

            ## add the agent to a random grid cell

            self.grid.place_agent(a, (x, y))

        for i in range(kwargs.get("apex", 0)):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)

            a = Apex(
                unique_id=i, model=self, pos=(x, y), **{k: v for k, v in kwargs.items()}
            )

            self.schedule.add(a)

            ## add the agent to a random grid cell

            self.grid.place_agent(a, (x, y))

        for i in range(kwargs.get("super", 0)):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)

            a = Super(
                unique_id=i, model=self, pos=(x, y), **{k: v for k, v in kwargs.items()}
            )

            self.schedule.add(a)

            ## add the agent to a random grid cell

            self.grid.place_agent(a, (x, y))

    ## Apex predator pulse

    def migrate_apex(self, N_apex, N_meso, **kwargs):

        if N_apex < 50 and N_meso > 0:

            for i in range(kwargs.get("apex", 0)):
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

                a = Apex(
                    unique_id=i, model=self, pos=(x, y), **{k: v for k, v in kwargs.items()}
                )

                self.schedule.add(a)

                ## add the agent to a random grid cell

                self.grid.place_agent(a, (x, y))
            
            return 1
        else:
            pass

        return 1

    ## step function

    def step(self):
        self.schedule.step()

        ## collect data

        self.count.collect(self)
        self.spatial.collect(self)

    ## run function

    def run_model(self, steps=100, progress=False, limit=10000, stop=False):
        for i in range(steps):
            
            ## end the model if there are no prey or predator agents

            self.n_Predators = self.data_collector(Predator)
            self.n_Prey = self.data_collector(Prey)
            self.n_Apex = self.data_collector(Apex)
            self.n_Super = self.data_collector(Super)

            # Pulse apex
            
            if self.migrate:

                self.migrate_apex(self.n_Apex, self.n_Predators, **self.kwargs)

            if self.n_Predators + self.n_Prey > limit:
                break

            elif (stop and self.n_Predators == 0) or (stop and self.n_Prey == 0):
                break

            else:
                self.step()

                if progress:
                    print(
                        "Number of prey:",
                        self.n_Prey,
                        "Number of predators:",
                        self.n_Predators,
                        "Number of apex predators:",
                        self.n_Apex,
                        "Number of super predators:",
                        self.n_Super,
                        "Step:",
                        i,
                    )
