########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Various tools for random numbers, boundaries, etc.
########################################################################

import numpy as np


########################################################################
# Pseudo-Random Number Generator (PRNG).

# The PRNG used to generate random numbers for an execution thread.
# This is not thread-safe so a new PRNG must be created for each thread
# using the new_prng() function below.
_prng = np.random.RandomState()


def new_prng(seed=None):
    """
    Create a new PRNG. Call this function when you start a new execution thread.
    This is necessary because the PRNG is not thread-safe so each thread must have
    its own PRNG object.

    :param seed: Starting seed for the PRNG.
    :return: Nothing.
    """

    # This ensures that we update the module's variable instead of
    # merely creating a new local variable inside this function.
    global _prng

    # Create a new PRNG object for the module.
    _prng = np.random.RandomState(seed=seed)


def rand_uniform(size):
    """
    Creates an array of the given size with random uniform numbers between [0, 1).

    Wraps the rand() function from numpy.
    """

    return _prng.rand(size)


def rand_array(lower, upper):
    """
    Create array with uniform random numbers between the given lower and upper bounds.
    """

    return (upper - lower) * _prng.rand(*lower.shape) + lower


def rand_population(lower, upper, num_agents, dim):
    """
    Create 2-d array with uniform random numbers between the lower and upper bounds.

    The first index is for the agent numbers.
    The second index is for the search-space dimensionality.

    :param lower: Lower boundary.
    :param upper: Upper boundary.
    :param num_agents: Number of agents in the population.
    :param dim: Dimensionality of the search-space.
    :return: 2-d array with uniform random values between bounds.
    """

    return (upper - lower) * _prng.rand(num_agents, dim) + lower


def rand_int(lower, upper):
    """
    Return a single random integer between lower (inclusive) and upper (exclusive).

    Wraps the randint() function from numpy.
    """

    return _prng.randint(low=lower, high=upper)


def rand_choice(a, size=None, replace=True, p=None):
    """
    Generate a random sample from an array.

    Wraps the choice() function from numpy so the parameters are the same.

    :param a:
        If a numpy ndarray, then a random sample is generated from its elements.
        If an int, then the random sample is generated as if a was np.arange(a)

    :param size:
        Output shape. Int or tuple of ints.

    :param replace:
        Sample with or without replacement.

    :param p:
        The probabilities associated with each entry in a.
        If not given then the sample assumes a uniform distribution over
        all entries in a.

    :return:
        Array with random sample.
    """

    return _prng.choice(a=a, size=size, replace=replace, p=p)


def sample_bounded(x, d, lower, upper):
    """
    Generate a random sample between x-d and x+d, while ensuring the range
    is bounded by lower and upper.
    """

    # Adjust sampling range so it does not exceed the search-space boundaries.
    l = np.maximum(x - d, lower)
    u = np.minimum(x + d, upper)

    # Return a random sample.
    return rand_array(lower=l, upper=u)


########################################################################

def bound(x, lower, upper):
    """
    Bound x between lower and upper, where x is a numpy array.
    """

    # Lower bound.
    y = np.where(x < lower, lower, x)

    # Upper bound.
    z = np.where(y > upper, upper, y)

    return z


def bound_scalar(x, lower, upper):
    """
    Bound x between lower and upper, where x is a scalar value.
    """

    return min(upper, max(lower, x))

########################################################################

def denormalize_trunc(x):
    """
    If x is very close to zero then it is truncated to zero.
    This is done to avoid denormalized floating point values.

    It is unclear from the Python and numpy manuals whether this is a problem, but
    in other languages (e.g. C# and C++) arithmetic operations can perform extremely slowly
    on denormalized floats. Some choices of PSO parameters cause this problem to occur.

    The truncating is very quick and only has a tiny runtime penalty.
    """

    # Truncate values to zero if they are below this limit.
    limit = 1e-30

    return np.where(np.abs(x) > limit, x, 0.0)

########################################################################
