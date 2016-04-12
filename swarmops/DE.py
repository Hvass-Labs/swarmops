########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Differential Evolution (DE).
#
# DE is a heuristic optimizer which does not use the gradient of the
# problem. The variant implemented here is known as DE/Rand/1/Bin.
#
# The DE has some control parameters that may greatly affect its
# performance. If the default parameters do not give satisfactory
# results then you may try some of the parameters listed below
# which were tuned for different optimization scenarios, otherwise
# you can try tuning the parameters using meta-optimization.
#
# References:
#
# [1] R. Storn and K. Price. Differential evolution - a simple
#     and efficient heuristic for global optimization over
#     continuous spaces. Journal of Global Optimization,
#     11:341-359, 1997.
#
# [2] M.E.H. Pedersen. Good parameters for Differential Evolution.
#     Hvass Laboratories Report HL-1002, 2010.
#     http://www.hvass-labs.org/people/magnus/publications/pedersen10good-de.pdf
#
# ########################################################################

import numpy as np
from swarmops.Optimize import SingleRun
from swarmops import tools


##################################################

class DE(SingleRun):
    """
        Perform a single optimization run using Differential Evolution (DE).

        This is the DE/Rand/1/Bin variant.

        In practice, you would typically perform multiple optimization runs using
        the MultiRun-class. The reason is that DE is a heuristic optimizer so
        there is no guarantee that an acceptable solution is found in any single
        run. It is more likely that an acceptable solution is found if you perform
        multiple optimization runs.

        Control parameters have been tuned for different optimization scenarios.
        First try and use the default parameters. If that does not give
        satisfactory results, then you may try some of the following.
        Select the parameters that most closely match your problem.
        For example, if you want to optimize a problem where the search-space
        has 15 dimensions and you can perform 30000 evaluations, then you could
        first try using parameters_20dim_40000eval. If that does not give
        satisfactory results then you could try using parameters_10dim_20000eval.
        If that does not work then you will either need to meta-optimize the
        parameters for the problem at hand, or you should try using another optimizer.
    """

    # Name of this optimizer.
    name = "DE"
    name_full = "Differential Evolution (DE/Rand/1/Bin)"

    # Number of control parameters for DE. Used by MetaFitness-class.
    num_parameters = 3

    # Lower boundaries for the control parameters of DE. Used by MetaFitness-class.
    parameters_lower_bound = [4.0, 0.0, 0.0]

    # Upper boundaries for the control parameters of DE. Used by MetaFitness-class.
    parameters_upper_bound = [200.0, 1.0, 2.0]

    @staticmethod
    def parameters_list(num_agents, crossover_probability, differential_weight):
        """
        Create a list with DE parameters in the correct order.

        :param num_agents:
            Number of agents in the population (aka. NP in the research literature).

        :param crossover_probability:
            Crossover probability (aka. CR in the research literature).

        :param differential_weight:
            Differential weight (aka. F in the research literature).

        :return:
            List with parameters in correct order.
        """

        return [num_agents, crossover_probability, differential_weight]

    @staticmethod
    def parameters_dict(parameters):
        """
        Create and return a dict from a list of DE parameters.
        This is useful for printing the named parameters.

        :param parameters: List of parameters.
        :return: Dict with parameters.
        """

        return {'num_agents': parameters[0],
                'crossover_probability': parameters[1],
                'differential_weight': parameters[2]}

    # Default parameters for the DE which will be used if no other parameters are specified.
    # These are a compromise of the tuned parameters below. Try this first and see if it works.
    parameters_default = [40.0, 0.7, 0.7]

    # Parameters tuned by hand. These are common in the older research literature on DE.
    parameters_hand_tuned = [300.0, 0.9000, 0.5000]

    # Parameters tuned for benchmark problems in 2 dimensions using 400 fitness evaluations.
    parameters_2dim_400eval_a = [13.0, 0.7450, 0.9096]

    # Parameters tuned for benchmark problems in 2 dimensions using 400 fitness evaluations.
    parameters_2dim_400eval_b = [10.0, 0.4862, 1.1922]

    # Parameters tuned for benchmark problems in 2 dimensions using 4000 fitness evaluations.
    parameters_2dim_4000eval_a = [24.0, 0.2515, 0.8905]

    # Parameters tuned for benchmark problems in 2 dimensions using 4000 fitness evaluations.
    parameters_2dim_4000eval_b = [20.0, 0.7455, 0.9362]

    # Parameters tuned for benchmark problems in 5 dimensions using 1000 fitness evaluations.
    parameters_5dim_1000eval = [17.0, 0.7122, 0.6301]

    # Parameters tuned for benchmark problems in 5 dimensions using 10000 fitness evaluations.
    parameters_5dim_10000eval = [20.0, 0.6938, 0.9314]

    # Parameters tuned for benchmark problems in 10 dimensions using 2000 fitness evaluations.
    parameters_10dim_2000eval_a = [28.0, 0.9426, 0.6607]

    # Parameters tuned for benchmark problems in 10 dimensions using 2000 fitness evaluations.
    parameters_10dim_2000eval_b = [12.0, 0.2368, 0.6702]

    # Parameters tuned for benchmark problems in 10 dimensions using 20000 fitness evaluations.
    parameters_10dim_20000eval = [18.0, 0.5026, 0.6714]

    # Parameters tuned for benchmark problems in 20 dimensions using 40000 fitness evaluations.
    parameters_20dim_40000eval = [37.0, 0.9455, 0.6497]

    # Parameters tuned for benchmark problems in 20 dimensions using 400000 fitness evaluations.
    parameters_20dim_400000eval = [35.0, 0.4147, 0.5983]

    # Parameters tuned for benchmark problems in 30 dimensions using 6000 fitness evaluations.
    parameters_30dim_6000eval = [11.0, 0.0877, 0.6419]

    # Parameters tuned for benchmark problems in 30 dimensions using 60000 fitness evaluations.
    parameters_30dim_60000eval = [19.0, 0.1220, 0.4983]

    # Parameters tuned for benchmark problems in 50 dimensions using 100000 fitness evaluations.
    parameters_50dim_100000eval = [48.0, 0.9784, 0.6876]

    # Parameters tuned for benchmark problems in 100 dimensions using 200000 fitness evaluations.
    parameters_100dim_200000eval = [46.0, 0.9565, 0.5824]

    def __init__(self, problem, parameters=parameters_default, parallel=False, *args, **kwargs):
        """
        Create object instance and perform a single optimization run using DE.

        :param problem: The problem to be optimized. Instance of Problem-class.

        :param parameters:
            Control parameters for the DE.
            These may have a significant impact on the optimization performance.
            First try and use the default parameters and if they don't give satisfactory
            results, then experiment with other parameters or meta-optimization.

        :param parallel:
            Evaluate the fitness for the agents in parallel.
            See the README.md file for a discussion on this.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is an instance of the FitnessTrace-class.
        """

        # Copy arguments to instance variables.
        self.problem = problem
        self.parallel = parallel

        # Unpack control parameters.
        self.num_agents, self.crossover_probability, self.differential_weight = parameters

        # The number of agents must be an integer.
        self.num_agents = int(self.num_agents)

        # Initialize population with random positions in the search-space.
        # The first index is for the agent number.
        # The second index is for the search-space dimension.
        # These are actually the best-known positions for each agent
        # and will only get replaced when improvements are found.
        self.population = tools.rand_population(lower=problem.lower_init,
                                                upper=problem.upper_init,
                                                num_agents=self.num_agents,
                                                dim=problem.dim)

        # Population used for generating new agents.
        self.new_population = np.copy(self.population)

        # Initialize fitness for all agents to infinity.
        self.fitness = np.repeat(np.inf, self.num_agents)

        # Initialize parent-class which also starts the optimization run.
        SingleRun.__init__(self, *args, **kwargs)

    def _optimize(self):
        """
        Perform a single optimization run.
        This function is called by the parent-class.
        """

        # Calculate fitness for the initial agent positions.
        self._update_fitness()

        # Optimization iterations.
        # The counting starts with num_agents because the fitness has
        # already been calculated once for each agent during initialization.
        for i in range(self.num_agents, self.max_evaluations, self.num_agents):
            # Calculate the new agent positions but don't update the old yet.
            self._new_agents()

            # Calculate the fitness for each new agent position and
            # update the population if the fitness is an improvement.
            self._update_fitness()

            # Call parent-class to print status etc. during optimization.
            self._iteration(i)

    def _new_agents(self):
        """
        Calculate a new population of agents using the DE method.
        The fitness is not calculated in this function.
        """

        # Convenience variables.
        population = self.population    # 2-d array with the population of agents.
        num_agents = self.num_agents    # Number of agents in population.
        dim = self.problem.dim          # Search-space dimensionality.
        differential_weight = self.differential_weight          # Control parameter for DE.
        crossover_probability = self.crossover_probability      # Control parameter for DE.

        # Create a new population of agents.
        for i in range(num_agents):
            # This loop only processes a single agent in each iteration.
            # It might be possible to make a faster implementation using Numpy
            # to process all agents simultaneously. However, this
            # implementation is simpler and easier to understand, and the
            # greatest runtime usage is typically in the fitness function.

            # Pick random and distinct indices for agents in the population.
            a, b, c, k = tools.rand_choice(num_agents, size=4, replace=False)

            # Indices a, b, c, k are all different from each other.
            # Now ensure that indices a, b, c are also different from i.
            # Simply check if an index equals i and then replace it with k,
            # which is different from i, a, b, c.
            if i == a:
                a = k
            elif i == b:
                b = k
            elif i == c:
                c = k

            assert i != a != b != c

            # Original position in the search-space for the agent to be updated.
            original = population[i, :]

            # Calculate crossover of randomly selected agents from the population.
            crossover = population[a, :] + differential_weight * (population[b, :] - population[c, :])

            # Create a new agent (i.e. position) in the search-space from the crossover.
            # The crossover probability decides whether to use the crossover or
            # the agent's original position in the search-space.
            new_agent = np.where(tools.rand_uniform(dim) < crossover_probability, crossover, original)

            # Ensure at least one element of the agent's new position is from the crossover.
            rand_index = tools.rand_int(lower=0, upper=dim)
            new_agent[rand_index] = crossover[rand_index]

            # Bound the agent's new position to the search-space boundaries.
            new_agent = tools.bound(x=new_agent,
                                    lower=self.problem.lower_bound,
                                    upper=self.problem.upper_bound)

            # Assign the agent's new position to the temporary population so the
            # fitness can be calculated in another function.
            self.new_population[i, :] = new_agent

    def _fitness(self, i):
        """
        Calculate the fitness for the i'th agent.
        """

        # Note that self.new_population is used because they are the potentially
        # new positions for the agents.
        return self.problem.fitness(self.new_population[i, :], limit=self.fitness[i])

    def _update_fitness(self):
        """
        Calculate the fitness for each agent and update the agent's and population's
        best-known fitness and position if an improvement is found.
        """

        if not self.parallel:
            # Calculate the fitness for each agent. Not parallel.
            new_fitness = [self._fitness(i) for i in range(self.num_agents)]
        else:
            import multiprocessing as mp

            # Create a pool of workers sized according to the CPU cores available.
            pool = mp.Pool()

            # Calculate the fitness for each agent in parallel.
            new_fitness = pool.map(self._fitness, range(self.num_agents))

            # Close the pool of workers and wait for them all to finish.
            pool.close()
            pool.join()

        # For each agent in the population.
        for i in range(self.num_agents):
            # If the new fitness is an improvement over the agent's best-known fitness.
            if new_fitness[i] < self.fitness[i]:
                # Update the agent's best-known fitness and position.
                self.fitness[i] = new_fitness[i]
                self.population[i, :] = self.new_population[i, :]

                # Update the entire population's best-known fitness and position
                # if an improvement.
                # The parent-class is used for this.
                self._update_best(fitness=self.fitness[i],
                                  x=self.population[i, :])


##################################################
