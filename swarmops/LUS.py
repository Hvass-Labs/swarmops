########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Local Unimodal Sampling (LUS).
#
# Performs localized sampling of the search-space with a sampling range
# that initially covers the entire search-space and is decreased
# exponentially as optimization progresses. LUS works especially well
# for optimization problems where only short runs can be performed
# and the search-space is fairly smooth.
#
# References:
#
# [1] M.E.H. Pedersen. Tuning & Simplifying Heuristical Optimization (PhD thesis).
#     University of Southampton, School of Engineering Sciences. 2010
#     http://www.hvass-labs.org/people/magnus/thesis/pedersen08thesis.pdf
#
########################################################################

from swarmops.Optimize import SingleRun
from swarmops import tools


########################################################################

class LUS(SingleRun):
    """
        Perform a single optimization run using Local Unimodal Sampling (LUS).

        In practice, you would typically perform multiple optimization runs using
        the MultiRun-class. The reason is that LUS is a heuristic optimizer so
        there is no guarantee that an acceptable solution is found in any single
        run. It is more likely that an acceptable solution is found if you perform
        multiple optimization runs.
    """

    # Name of this optimizer.
    name = "LUS"
    name_full = "Local Unimodal Sampling"

    # Number of control parameters for LUS. Used by MetaFitness-class.
    num_parameters = 1

    # Lower boundaries for the control parameters of LUS. Used by MetaFitness-class.
    parameters_lower_bound = [0.1]

    # Upper boundaries for the control parameters of LUS. Used by MetaFitness-class.
    parameters_upper_bound = [100.0]

    @staticmethod
    def parameters_dict(parameters):
        """
        Create and return a dict from a list of LUS parameters.
        This is useful for printing the named parameters.

        :param parameters: List with LUS parameters assumed to be in the correct order.
        :return: Dict with LUS parameters.
        """

        return {'gamma': parameters[0]}

    @staticmethod
    def parameters_list(gamma):
        """
        Create a list with LUS parameters in the correct order.

        :param gamma: Gamma-parameter (see paper reference for explanation).
        :return: List with LUS parameters.
        """

        return [gamma]

    # Default parameters for LUS which will be used if no other parameters are specified.
    parameters_default = [3.0]

    def __init__(self, problem, parameters=parameters_default, parallel=False, *args, **kwargs):
        """
        Create object instance and perform a single optimization run using LUS.

        :param problem:
            The problem to be optimized. Instance of Problem-class.

        :param parameters:
            Control parameters for LUS.

        :param parallel: False. LUS cannot run in parallel except through MultiRun.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is an instance of the FitnessTrace-class.
        """

        # Copy arguments to instance variables.
        self.problem = problem

        # Unpack control parameters.
        gamma = parameters[0]

        # Derived control parameter.
        self.decrease_factor = 0.5 ** (1.0 / (gamma * problem.dim))

        # Initialize parent-class which also starts the optimization run.
        SingleRun.__init__(self, *args, **kwargs)

    def _optimize(self):
        """
        Perform a single optimization run.
        This function is called by the parent-class.
        """

        # Convenience variable for fitness function.
        f = self.problem.fitness

        # Convenience variables for search-space boundaries.
        lower_init = self.problem.lower_init
        upper_init = self.problem.upper_init
        lower_bound = self.problem.lower_bound
        upper_bound = self.problem.upper_bound

        # Initialize the range-vector to full search-space.
        d = upper_bound - lower_bound

        # Search-space dimensionality.
        dim = self.problem.dim

        # Initialize x with random position in search-space.
        x = tools.rand_array(lower=lower_init, upper=upper_init)

        # Compute fitness of initial position.
        fitness = f(x)

        # Update the best-known fitness and position.
        # The parent-class is used for this.
        self._update_best(fitness=fitness, x=x)

        # Perform optimization iterations until the maximum number
        # of fitness evaluations has been performed.
        # Count starts at one because we have already calculated fitness once above.
        evaluations = 1
        while evaluations < self.max_evaluations:
            # Sample new position y from the bounded surroundings
            # of the current position x.
            y = tools.sample_bounded(x=x, d=d, lower=lower_bound, upper=upper_bound)

            # Compute new fitness.
            new_fitness = f(y, limit=fitness)

            # If improvement to fitness.
            if new_fitness < fitness:
                # Update fitness and position.
                fitness = new_fitness
                x = y

                # Update the best-known fitness and position.
                # The parent-class is used for this.
                self._update_best(fitness=fitness, x=x)
            else:
                # Otherwise decrease the search-range.
                d *= self.decrease_factor

            # Call parent-class to print status etc. during optimization.
            self._iteration(evaluations)

            # Increment counter.
            evaluations += 1

########################################################################
