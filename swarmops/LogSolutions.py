########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Provides logging of best solutions found for an optimization problem,
# so we get a list of good solutions rather than just a single solution.
########################################################################

import numpy as np


########################################################################

class _LogElement:
    """
    Used for storing parameters and associated fitness in the log.
    """

    def __init__(self, x=None, fitness=np.Infinity):
        """
        Create object instance.

        :param x: Position in the search-space aka. candidate solution.
        :param fitness: Associated fitness of the position in the search-space.
        :return: Object instance.
        """

        # Copy arguments to instance variables.
        self.x = x
        self.fitness = fitness


########################################################################

class LogSolutions:
    """
    Transparently wraps a Problem-object and provides logging of the
    best solutions found. This allows us to get a list of good
    solutions for a problem rather than just the single best solution.
    """

    def __init__(self, problem, capacity=20):
        """
        Create object instance.

        :param problem: Instance of the Problem-class to be wrapped.
        :param capacity: Capacity of the log, default 20.
        :return: Object instance.
        """

        # Copy all the attributes of the problem-object to self. This essentially
        # wraps the problem-object and makes e.g. self.dim the same as problem.dim, etc.
        # Note that a shallow copy is being made, otherwise self.__dict__
        # would be a reference to problem.__dict__ which would cause the
        # original problem-object to become modified e.g. with self.capacity, etc.
        self.__dict__ = problem.__dict__.copy()

        # Copy arguments to instance variables.
        self.problem = problem
        self.capacity = capacity

        # Initialize log with empty solutions.
        self.solutions = [_LogElement() for i in range(capacity)]

    def fitness(self, x, limit=np.Infinity):
        """
        Wraps the fitness-function of the actual problem, whose results
        are logged if the fitness is an improvement.
        """

        # Calculate fitness of the actual problem.
        new_fitness = self.problem.fitness(x=x, limit=limit)

        # If log is desired.
        if self.capacity > 0:
            # If the new fitness is an improvement over the worst-known fitness in the log.
            if new_fitness < self.solutions[-1].fitness:
                # Update the worst solution in the log with the new solution.
                self.solutions[-1] = _LogElement(x=x, fitness=new_fitness)

            # Sort the logged solutions.
            self.solutions = sorted(self.solutions, key=lambda solution: solution.fitness)

        return new_fitness

########################################################################
