########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Problem-class used for implementing optimization problems. This
# essentially just wraps the fitness-function, boundaries, etc.
#
# This file also implements several common benchmark problems whose
# mathematical definitions are given in this paper:
#
# [1] M.E.H. Pedersen. Good parameters for particle swarm optimization.
#     Technical Report HL-1001, Hvass Laboratories. 2010.
#     http://www.hvass-labs.org/people/magnus/publications/pedersen10good-pso.pdf
#
########################################################################

import numpy as np
from swarmops import tools


########################################################################

class Problem:
    """
    Implements optimization problems by wrapping the fitness-function,
    boundaries, etc.
    """

    def __init__(self, name, dim, fitness_min,
                 lower_bound, upper_bound,
                 lower_init=None, upper_init=None,
                 func=None):
        """
        Create the object instance for an optimization problem.

        You can implement a fitness function in either of two ways:

        1) You can sub-class this Problem-class and override the self.fitness() function.
           This is demonstrated in the benchmark problems below.

        2) You can pass the fitness function as the func-argument to __init__.
           The arguments of the fitness function must match those of self.fitness().
           Note that if you want to do parallel execution then the fitness function
           must not be declared inside "if __name__ == '__main__':" due to some
           peculiarities in how multiprocessing uses pickle.
           This is demonstrated in demo-optimize.py

        :param name: Name of the optimization problem.
        :param dim: Dimensionality of the search-space.
        :param fitness_min: Minimum possible fitness value, or a reasonable lower boundary.
        :param lower_bound: Lower boundary for the search-space.
        :param upper_bound: Upper boundary for the search-space.
        :param lower_init: Lower initialization boundary (if None then user lower_bound).
        :param upper_init: Upper initialization boundary (if None then user upper_bound).
        :param func: Wrap a fitness-function whose arguments must match self.fitness().
        """

        # Copy arguments to instance variables.
        self.name = name
        self.name_full = "{0} ({1} dim)".format(name, dim)
        self.dim = dim
        self.fitness_min = fitness_min

        # If a fitness function is supplied as argument,
        # then use it instead of the class method.
        if func is not None:
            self.fitness = func

        # Boundaries for the search-space. Ensure they are numpy float arrays.
        self.lower_bound = np.array(lower_bound, dtype='float64')
        self.upper_bound = np.array(upper_bound, dtype='float64')

        # Lower initialization boundary.
        if lower_init is not None:
            # Use the initialization boundary provided.
            self.lower_init = np.array(lower_init, dtype='float64')
        else:
            # Or use the search-space boundary.
            self.lower_init = self.lower_bound

        # Upper initialization boundary.
        if upper_init is not None:
            # Use the initialization boundary provided.
            self.upper_init = np.array(upper_init, dtype='float64')
        else:
            # Or use the search-space boundary.
            self.upper_init = self.upper_bound

        # Ensure boundaries have the correct dimensions.
        assert dim == len(self.lower_bound) == len(self.upper_bound)
        assert dim == len(self.lower_init) == len(self.upper_init)

    def fitness(self, x, limit=np.Infinity):
        """
        This is the fitness function that must be implemented for an
        optimization problem.

        It is also sometimes called the cost- or error-function.

        :param x:
            Calculate the fitness value for these parameters in the search-space.

        :param limit:
            Calculation of the fitness can be aborted if the value is greater than this limit.
            This is used for so-called Pre-Emptive Fitness Evaluation in the MetaFitness-class.
            You can ignore this value.

        :return:
            Fitness-value of x.
        """

        # Raise an exception if the child-class has not implemented this function.
        raise NotImplementedError


########################################################################

class Benchmark(Problem):
    """
    Same as the Problem-class except the boundaries are assumed to be scalar
    values used for creating numpy arrays. This is convenient for optimization
    problems whose boundaries are identical for all dimensions, such as
    the benchmark-problems below.
    """

    def __init__(self, dim, lower_bound, upper_bound, lower_init, upper_init,
                 *args, **kwargs):
        """
        Create the object instance for an optimization problem.

        :param dim: Search-space dimensionality.
        :param lower_bound: Scalar value for the lower search-space boundary.
        :param upper_bound: Scalar value for the upper search-space boundary.
        :param lower_init: Scalar value for the lower initialization boundary.
        :param upper_init: Scalar value for the upper initialization boundary.
        :param kwargs:
        :return:
        """

        Problem.__init__(self, dim=dim,
                         lower_bound=np.repeat(lower_bound, dim),
                         upper_bound=np.repeat(upper_bound, dim),
                         lower_init=np.repeat(lower_init, dim),
                         upper_init=np.repeat(upper_init, dim),
                         *args, **kwargs)


########################################################################
# Benchmark problems.

class Sphere(Benchmark):
    """
    The Sphere benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Sphere', dim=dim, fitness_min=0.0,
                           lower_bound=-100.0, upper_bound=100.0,
                           lower_init=50.0, upper_init=100.0)

    def fitness(self, x, limit=np.Infinity):
        return np.sum(x ** 2)

class Rosenbrock(Benchmark):
    """
    The Rosenbrock benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Rosenbrock', dim=dim, fitness_min=0.0,
                           lower_bound=-100.0, upper_bound=100.0,
                           lower_init=15.0, upper_init=30.0)

    def fitness(self, x, limit=np.Infinity):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


class Griewank(Benchmark):
    """
    The Griewank benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Griewank', dim=dim, fitness_min=0.0,
                           lower_bound=-600.0, upper_bound=600.0,
                           lower_init=300.0, upper_init=600.0)

        # Helper array for faster calculation of fitness.
        # This is the square root of the array index.
        self.sqrt_idx = np.sqrt(np.linspace(start=1.0, stop=dim, num=dim))

    def fitness(self, x, limit=np.Infinity):
        return 1.0 + np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / self.sqrt_idx))


class Rastrigin(Benchmark):
    """
    The Rastrigin benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Rastrigin', dim=dim, fitness_min=0.0,
                           lower_bound=-5.12, upper_bound=5.12,
                           lower_init=2.56, upper_init=5.12)

    def fitness(self, x, limit=np.Infinity):
        return np.sum(x ** 2 + 10.0 - 10.0 * np.cos(2 * np.pi * x))


class Schwefel1_2(Benchmark):
    """
    The Schwefel 1.2 benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Schwefel 1-2', dim=dim, fitness_min=0.0,
                           lower_bound=-30.0, upper_bound=30.0,
                           lower_init=15.0, upper_init=30.0)

    def fitness(self, x, limit=np.Infinity):
        return np.sum([np.sum(x[:i])**2 for i in range(self.dim)])


class Schwefel2_21(Benchmark):
    """
    The Schwefel 2.21 benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Schwefel 2-21', dim=dim, fitness_min=0.0,
                           lower_bound=-100.0, upper_bound=100.0,
                           lower_init=50.0, upper_init=100.0)

    def fitness(self, x, limit=np.Infinity):
        return np.max(np.abs(x))


class Schwefel2_22(Benchmark):
    """
    The Schwefel 2.22 benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Schwefel 2-22', dim=dim, fitness_min=0.0,
                           lower_bound=-10.0, upper_bound=10.0,
                           lower_init=5.0, upper_init=10.0)

    def fitness(self, x, limit=np.Infinity):
        absx = np.abs(x)
        fitness = np.sum(absx) + np.prod(absx)

        # For large dim, np.prod(absx) often overflows to infinity.
        # This means the optimizers cannot work properly.
        # So we limit the fitness to 1e20.
        if fitness == np.Infinity or fitness > 1e20:
            fitness = 1e20

        return fitness


class Step(Benchmark):
    """
    The Step benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Step', dim=dim, fitness_min=0.0,
                           lower_bound=-100.0, upper_bound=100.0,
                           lower_init=50.0, upper_init=100.0)

    def fitness(self, x, limit=np.Infinity):
        return np.sum(np.floor(x + 0.5) ** 2)


class QuarticNoise(Benchmark):
    """
    The Quartic-Noise benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='QuarticNoise', dim=dim, fitness_min=0.0,
                           lower_bound=-1.28, upper_bound=1.28,
                           lower_init=0.64, upper_init=1.28)

        # Helper array with the index.
        self.idx_array = np.linspace(start=1.0, stop=dim, num=dim)

    def fitness(self, x, limit=np.Infinity):
        # If using PSO, MOL or DE in parallel mode and not run through MultiRun,
        # then a new PRNG must be created for each thread because the PRNG is not
        # thread-safe. This is a bit slow so it should only be done when necessary.
        if False:
            tools.new_prng()

        # Array with random values between zero and one.
        r = tools.rand_uniform(size=self.dim)

        return np.sum(self.idx_array * (x ** 4) + r)


class Ackley(Benchmark):
    """
    The Ackley benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Ackley', dim=dim, fitness_min=0.0,
                           lower_bound=-30.0, upper_bound=30.0,
                           lower_init=15.0, upper_init=30.0)

    def fitness(self, x, limit=np.Infinity):
        value = np.e + 20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.dim)) \
                - np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dim)

        # Due to rounding errors, the value is sometimes slightly negative.
        # This gives a warning in the MetaFitness-class.
        if value < 0.0:
            value = 0.0

        return value


def _penalty(x, a, k, m):
    """
    Helper-function for the Penalized-problems below.
    """

    # Penalty for each dimension of the search-space.
    p = np.where(x < -a, k * (-x - a)**m,
                 np.where(x > a, k * (x - a)**m, 0.0))

    # Sum the penalties.
    return np.sum(p)


class Penalized1(Benchmark):
    """
    The Penalized1 benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Penalized1', dim=dim, fitness_min=0.0,
                           lower_bound=-50.0, upper_bound=50.0,
                           lower_init=5.0, upper_init=50.0)

    def fitness(self, x, limit=np.Infinity):
        # Split the calculation into smaller formulas.
        y = 1.0 + (x + 1.0) / 4.0
        a = 10.0 * np.sin(np.pi * y[0]) ** 2
        b = np.sum(((y[:-1] - 1.0) ** 2) * (1.0 + 10.0 * (np.sin(np.pi * y[1:]) ** 2)))
        c = (y[-1] - 1.0) ** 2

        # Penalty term.
        penalty = _penalty(x, 10.0, 100.0, 4.0)

        return np.pi * (a + b + c) / self.dim + penalty


class Penalized2(Benchmark):
    """
    The Penalized2 benchmark problem.
    """

    def __init__(self, dim):
        Benchmark.__init__(self, name='Penalized2', dim=dim, fitness_min=0.0,
                           lower_bound=-50.0, upper_bound=50.0,
                           lower_init=5.0, upper_init=50.0)

    def fitness(self, x, limit=np.Infinity):
        # Split the calculation into smaller formulas.
        a = np.sin(3 * np.pi * x[0]) ** 2
        b = np.sum(((x[:-1] - 1.0) ** 2) * (1.0 + np.sin(3 * np.pi * x[1:]) ** 2))
        c = (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * x[-1])**2)

        # Penalty term.
        penalty = _penalty(x, 5.0, 100.0, 4.0)

        return 0.1 * (a + b + c) / self.dim + penalty


########################################################################

# List of all of the above classes for benchmark problems.
all_benchmark_problems = [Ackley, Griewank, Penalized1, Penalized2,
                          QuarticNoise, Rastrigin, Rosenbrock,
                          Schwefel1_2, Schwefel2_21, Schwefel2_22,
                          Sphere, Step]

########################################################################
