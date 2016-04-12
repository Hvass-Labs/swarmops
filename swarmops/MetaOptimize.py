########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Classes used for tuning the control parameters of an optimizer.
#
# The basic idea is to have two layers of optimization, the optimizer
# whose control parameters we wish to tune (e.g. PSO or DE), and a
# meta-optimizer (typically LUS) for meta-optimizing those control
# parameters.
#
# The meta-fitness measures the performance of the optimizer on several
# optimization problems. It is basically just the sum of the best
# fitness achieved in several optimization runs on each problem.
#
# To save execution time, the calculation of the meta-fitness may be
# aborted pre-emptively if the fitness sum exceeds a limit.
# This is known as Pre-Emptive Fitness Evaluation.
#
# Meta-optimization is described in detail in:
#
# [1] M.E.H. Pedersen. Tuning & Simplifying Heuristical Optimization (PhD thesis).
#     University of Southampton, School of Engineering Sciences. 2010
#     http://www.hvass-labs.org/people/magnus/thesis/pedersen08thesis.pdf
#
########################################################################

import numpy as np
from swarmops.LogSolutions import LogSolutions
from swarmops.Optimize import MultiRun
from swarmops.Problem import Problem
from swarmops.Timer import Timer
from swarmops.LUS import LUS


########################################################################

class _ProblemRank:
    """
    Helper-class for ranking and sorting the list of problems according
    to how well an optimization method performed on each problem.

    This is used with Pre-Emptive Fitness Evaluation for more quickly
    aborting the computation of the meta-fitness measure when possible.
    """

    def __init__(self, problem, weight=1.0):
        """
        Create object instance.

        :param problem: Instance of the Problem-class.
        :param weight: Weight for the problem.
        :return: Object instance.
        """

        # Copy arguments to instance variables.
        self.problem = problem
        self.weight = weight

        # Initialize the fitness-sum.
        self.fitness_sum = 0.0

        # Initialize the best-found solution and fitness.
        self.best = None
        self.best_fitness = np.Infinity

    def update_best(self, best, best_fitness):
        """
        Update the best-found solution to the problem.

        :param best: Best-found solution from an optimization run.
        :param best_fitness: Fitness of the best-found solution.
        :return: Nothing.
        """

        # If the fitness is an improvement over the best-known.
        if best_fitness < self.best_fitness:
            # Update the best-known solution and fitness.
            self.best = best
            self.best_fitness = best_fitness


########################################################################

class _MetaFitness(Problem):
    """
    Used for measuring the performance of an optimization method on
    several problems. This is called the meta-fitness which can
    then be optimized by another overlaying optimizer which is called
    the meta-optimizer.
    """

    def __init__(self, optimizer, problems, num_runs, max_evaluations, weights=None):
        """
        Create object instance.

        :param optimizer: Optimizer-class, e.g. PSO, MOL or DE.
        :param problems: List of instances of the Problem-class.
        :param num_runs: Number of optimization runs to perform for each problem.
        :param max_evaluations: Number of fitness evaluations for each optimization run.
        :param weights: List of weights for the problems to adjust their mutual importance.
        :return: Object instance.
        """

        # Copy arguments to instance variables.
        self.optimizer = optimizer
        self.num_runs = num_runs
        self.max_evaluations = max_evaluations

        # Wrap the problems and weights. This is used for ranking the problems
        # which significantly speeds up the execution time, as explained below.
        if weights is None:
            # No weights were supplied so we just wrap the problems.
            self.problem_ranks = [_ProblemRank(problem) for problem in problems]
        else:
            # Wrap both the problems and weights.
            self.problem_ranks = [_ProblemRank(problem, weight) for problem, weight in zip(problems, weights)]

        # The MetaFitness-class is actually an optimization problem,
        # so init the parent-class.
        # The dimensionality of the search-space is the number of
        # control parameters for the optimizer, and the search-space
        # boundaries are the boundaries for the control parameters.
        Problem.__init__(self, name="MetaFitness",
                         dim=optimizer.num_parameters, fitness_min=0.0,
                         lower_bound=optimizer.parameters_lower_bound,
                         upper_bound=optimizer.parameters_upper_bound)

    def fitness(self, x, limit=np.Infinity):
        """
        Calculate the meta-fitness measure.

        :param x:
            Control parameters for the optimization method.

        :param limit:
            Abort the calculation of the meta-fitness when it
            becomes greater than this limit.

        :return:
            The meta-fitness measures how well the optimizer
            performed on the list of problems and using the given
            control parameters.
        """

        # Start a timer so we can later print the time-usage.
        timer = Timer()

        # Convenience variables.
        optimizer = self.optimizer
        max_evaluations = self.max_evaluations

        # Initialize the meta-fitness to zero.
        # The meta-fitness is just the (adjusted) sum of the
        # fitness obtained on multiple optimization runs.
        fitness_sum = 0.0

        # For each problem do the following.
        # Note that we iterate over self.problem_ranks which
        # is sorted so that we first try and optimize the problems
        # that are most likely to cause fitness_sum to exceed the
        # limit so the calculation can be aborted. This is called
        # Pre-Emptive Fitness Evaluation and greatly saves run-time.
        for problem_rank in self.problem_ranks:
            # Convenience variables.
            problem = problem_rank.problem
            weight = problem_rank.weight

            # Initialize the fitness sum for this problem.
            fitness_sum_inner = 0.0

            # Perform a number of optimization runs on the problem.
            for i in range(self.num_runs):
                # Perform one optimization run on the given problem
                # using the given control parameters.
                result = optimizer(problem=problem,
                                   max_evaluations=max_evaluations,
                                   parameters=x)

                # Keep track of the best-found solution for this problem.
                problem_rank.update_best(best=result.best,
                                         best_fitness=result.best_fitness)

                # Adjust the fitness so it is non-negative.
                fitness_adjusted = result.best_fitness - problem.fitness_min

                # Print warning if adjusted fitness is negative. Due to tiny rounding
                # errors this might occur without being an issue. But if the adjusted
                # fitness is negative and large, then problem.fitness_min must be corrected
                # in order for Pre-Emptive Fitness Evaluation to work properly.
                # It is better to print a warning than to use an assert which would
                # stop the execution.
                if fitness_adjusted < 0.0:
                    msg = "WARNING: MetaFitness.py, fitness_adjusted is negative {0:.4e} on problem {1}"
                    print(msg.format(fitness_adjusted, problem.name))

                # Accumulate the fitness sum for the inner-loop.
                fitness_sum_inner += weight * fitness_adjusted

                # Accumulate the overall fitness sum.
                fitness_sum += weight * fitness_adjusted

                # If the fitness sum exceeds the limit then break from the inner for-loop.
                if fitness_sum > limit:
                    break

            # Update the problem's ranking with the fitness-sum.
            # This is the key used in sorting below.
            problem_rank.fitness_sum = fitness_sum_inner

            # If the fitness sum exceeds the limit then break from the outer for-loop.
            if fitness_sum > limit:
                break

        # Sort the problems using the fitness_sum as the key in descending order.
        # This increases the probability that the for-loops above can be
        # aborted pre-emptively the next time the meta-fitness is calculated.
        self.problem_ranks = sorted(self.problem_ranks,
                                    key=lambda rank: rank.fitness_sum,
                                    reverse=True)

        # Stop the timer.
        timer.stop()

        # Print various results so we can follow the progress.
        print("- Parameters tried: {0}".format(x))
        msg = "- Meta-Fitness: {0:.4e}, Improvement: {1}"
        improvement = fitness_sum < limit
        print(msg.format(fitness_sum, improvement))
        print("- Time-Usage: {0}".format(timer))

        return fitness_sum


########################################################################

class MetaOptimize:
    """
    Performs meta-optimization, that is, tuning of an optimizer's
    control parameters by using another overlaying optimizer.
    """

    def __init__(self, optimizer, problems, num_runs, max_evaluations,
                 meta_num_runs=5, meta_max_evaluations=None, weights=None,
                 log_capacity=20, parallel=False):
        """
        Create object instance and perform the meta-optimization.

        :param optimizer:
            Optimizer-class, e.g. PSO, MOL or DE.

        :param problems:
            List of instances of the Problem-class.

        :param num_runs:
            Number of optimization runs to perform for each problem.

        :param max_evaluations:
            Number of fitness evaluations for each optimization run.

        :param meta_num_runs:
            Number of runs for the meta-optimizer (default 5).

        :param meta_max_evaluations:
            Number of iterations for each run of the meta-optimizer.
            If None then it is set to 20 * number of control parameters for the optimizer.

        :param weights:
            List of weights for the problems to adjust their mutual importance
            when tuning the control parameters of the optimizer.
            If weights=None then the weight is set to 1.0 for all problems.

        :param log_capacity:
            How many of the best control parameters are logged.

        :param parallel:
            Execute the meta-optimization runs in parallel. If True then only
            the best-found control parameters are available afterwards, as the
            log of best parameters will be empty. The best-found solutions
            to the optimization problems will also be empty.
            See README.md for more details.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   results.best are the best control parameters found for the optimizer.
            -   results.best_fitness is the meta-fitness of the best-found parameters.
            -   log.solutions holds several of the best control parameters.
            -   meta_fitness.problem_rank is a list which also holds the best-found
                solutions to the actual optimization problems.
        """

        # Default number of iterations for the meta-optimizer.
        if meta_max_evaluations is None:
            meta_max_evaluations = 20 * optimizer.num_parameters

        # Create a meta-fitness object using the optimizer, problems, weights, etc.
        self.meta_fitness = _MetaFitness(optimizer=optimizer,
                                         problems=problems,
                                         weights=weights,
                                         num_runs=num_runs,
                                         max_evaluations=max_evaluations)

        # Wrap the meta_fitness in a logger so we keep several of the best solutions.
        self.log = LogSolutions(problem=self.meta_fitness, capacity=log_capacity)

        # Perform multiple runs using LUS as the meta-optimizer which is
        # used because it usually requires very few iterations to find
        # near-optimal control parameters for the optimizer. However,
        # it typically needs about 5 runs because it can easily get
        # stuck with sub-optimal parameters.
        self.results = MultiRun(optimizer=LUS,
                                problem=self.log,  # Wrapper for self.meta_fitness
                                num_runs=meta_num_runs,
                                max_evaluations=meta_max_evaluations,
                                parallel=parallel,
                                display_interval=1, trace_len=100)


########################################################################
