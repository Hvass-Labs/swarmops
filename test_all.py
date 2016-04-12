########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Nose-tests.
#
# These tests take TWO HOURS or more to run on a PC from the year 2011.
#
# There are 5 generator functions for testing SingleRun, MultiRun and
# MetaOptimize. Each generator function combines different parameters
# to make up thousands of configurations that are being tested.
# The optimization results are then asserted to be of the expected
# numpy array-type and length.
#
# The main purpose of these tests is to try the optimizers with
# many different configurations to see if an exception is raised
# somewhere. The actual optimization results are not tested for
# "correctness" because the optimizers are heuristic and stochastic
# so they will give different results on each run. The random-number
# generator could be seeded with a fixed number so as to generate
# the exact same results on each optimization run, but this would
# still not guarantee the implementation is correct, and even small
# modifications to the implementation of an optimizer may give very
# different results.
#
# This means you should also manually run and inspect the output of
# demo-optimize.py and demo-meta-optimize.py to assess whether the
# implementation is correct and the performance is satisfactory.
########################################################################

import random
import numpy as np
import nose
from nose.tools import assert_equals, assert_is_instance

import swarmops.Problem as Problem
from swarmops.DE import DE
from swarmops.MetaOptimize import MetaOptimize
from swarmops.Optimize import MultiRun
from swarmops.PS import PS
from swarmops.PSO import PSO, MOL
from swarmops.Timer import Timer
from swarmops.LUS import LUS


########################################################################
# Test all optimizers with a single run for each configuration.
# Note that each optimizer inherits from the SingleRun-class.

def test_SingleRun():
    """
    Test-generator for single-runs of the optimizers using many different configurations.

    This generates test-configurations that ensures boundary-cases are tested, e.g.
    a search-space dimensionality of 1, fitness-trace-length of 0, etc.
    These boundary-cases might not be tested if the configuration was completely random.
    """

    # For each optimizer.
    for optimizer in [PSO, MOL, DE, LUS, PS]:
        # For different search-space dimensionalities.
        for dim in [1, 7, 31, 773]:
            # For different display intervals.
            for display_interval in [0, 1, 13, 269]:
                # For different number of fitness evaluations.
                for max_evaluations in [9, 997, 5521, 20011]:
                    # For different fitness-trace-lengths.
                    for trace_len in [0, 43, 101]:
                        # Take a benchmark problem at random.
                        problem_class = random.choice(Problem.all_benchmark_problems)
                        problem = problem_class(dim=dim)

                        # Non-parallel.
                        parallel = False

                        # Run the test using this configuration.
                        yield _do_test_SingleRun, optimizer, problem, dim, max_evaluations, display_interval, trace_len, parallel

        # Test Parallel execution on the optimizer-level.
        # This is quite slow so we only do it once for each optimizer.

        # Configuration.
        dim = 17
        display_interval = 127
        trace_len = 151
        max_eval = 1013
        parallel = True

        # Use the Rosenbrock problem because it doesn't use random numbers so there's
        # no issues with thread-safety.
        problem = Problem.Rosenbrock(dim=dim)

        # Run the test using this configuration.
        yield _do_test_SingleRun, optimizer, problem, dim, max_eval, display_interval, trace_len, parallel


def test_SingleRun2():
    """
    Test-generator for single-runs of the optimizers using many different configurations.

    This keeps generating random test-configurations for a desired number of minutes.
    """

    # Keep testing configurations for this many minutes.
    max_minutes = 10

    # Start a timer.
    timer = Timer()

    # For the desired number of minutes we select random configurations to test.
    while timer.less_than(minutes=max_minutes):
        # Select an optimizer at random.
        optimizer = random.choice([PSO, MOL, DE, LUS, PS])

        # Search-space dimensionality.
        dim = np.random.randint(1, 1000)

        # Display intervals.
        display_interval = np.random.randint(0, 250)

        # Max fitness evaluations.
        max_evaluations = np.random.randint(1, 20000)

        # Fitness-trace-length.
        trace_len = np.random.randint(0, 1000)

        # Take a benchmark problem at random.
        problem_class = random.choice(Problem.all_benchmark_problems)
        problem = problem_class(dim=dim)

        # Non-parallel.
        parallel = False

        # Run the test using this configuration.
        yield _do_test_SingleRun, optimizer, problem, dim, max_evaluations, display_interval, trace_len, parallel


def _do_test_SingleRun(optimizer, problem, dim, max_evaluations,
                       display_interval, trace_len, parallel):
    """
    Helper-function that performs the actual optimization and asserts the results
    are as expected.
    """

    # Perform the optimization using the given configuration.
    result = optimizer(parallel=parallel, problem=problem,
                       max_evaluations=max_evaluations,
                       display_interval=display_interval,
                       trace_len=trace_len)

    # Assert that the result of the optimization is as expected.
    # Note that the "correctness" is not tested, see discussion above.
    assert_is_instance(obj=result.best, cls=np.ndarray)
    assert_equals(len(result.best), dim)


########################################################################
# Test multiple runs of each optimizer.

def test_MultiRun():
    """
    Test-generator for multi-runs of the optimizers using many different configurations.

    This generates test-configurations that ensures boundary-cases are tested, e.g.
    a search-space dimensionality of 1, fitness-trace-length of 0, etc.
    These boundary-cases might not be tested if the configuration was completely random.
    """

    # For each optimizer.
    for optimizer in [PSO, MOL, DE, LUS, PS]:
        # For different search-space dimensionalities.
        for dim in [2, 47]:
            # For different display intervals.
            for display_interval in [0, 11, 167]:
                # For different number of fitness evaluations.
                for max_evaluations in [53, 10391]:
                    # For different fitness-trace-lengths.
                    for trace_len in [0, 101]:
                        # For different number of optimization runs.
                        for num_runs in [1, 5]:
                            # For parallel and non-parallel.
                            for parallel in [True, False]:
                                # Take a benchmark problem at random.
                                problem_class = random.choice(Problem.all_benchmark_problems)
                                problem = problem_class(dim=dim)

                                # Run the test using this configuration.
                                yield _do_test_MultiRun, optimizer, problem, dim, max_evaluations, display_interval, trace_len, parallel, num_runs


def test_MultiRun2():
    """
    Test-generator for multi-runs of the optimizers using many different configurations.

    This keeps generating random test-configurations for a desired number of minutes.
    """

    # Keep testing configurations for this many minutes.
    max_minutes = 10

    # Start a timer.
    timer = Timer()

    # For the desired number of minutes we select random configurations to test.
    while timer.less_than(minutes=max_minutes):
        # Select an optimizer at random.
        optimizer = random.choice([PSO, MOL, DE, LUS, PS])

        # Search-space dimensionality.
        dim = np.random.randint(1, 1000)

        # Display intervals.
        display_interval = np.random.randint(0, 250)

        # Max fitness evaluations.
        max_evaluations = np.random.randint(1, 2000)

        # Number of optimization runs.
        num_runs = np.random.randint(1, 10)

        # Fitness-trace-length.
        trace_len = np.random.randint(0, 1000)

        # Take a benchmark problem at random.
        problem_class = random.choice(Problem.all_benchmark_problems)
        problem = problem_class(dim=dim)

        # Either parallel or not.
        parallel = random.choice([True, False])

        # Run the test using this configuration.
        yield _do_test_MultiRun, optimizer, problem, dim, max_evaluations, display_interval, trace_len, parallel, num_runs


def _do_test_MultiRun(optimizer, problem, dim, max_evaluations, display_interval, trace_len, parallel, num_runs):
    """
    Helper-function that performs the actual optimization and asserts the results
    are as expected.
    """

    # Perform the optimization using the given configuration.
    results = MultiRun(optimizer=optimizer,
                       num_runs=num_runs, problem=problem,
                       parallel=parallel,
                       max_evaluations=max_evaluations,
                       display_interval=display_interval, trace_len=trace_len)

    # Assert that the result of the optimization is as expected.
    # Note that the "correctness" is not tested, see discussion above.
    assert_is_instance(obj=results.best, cls=np.ndarray)
    assert_equals(len(results.best), dim)


########################################################################
# Test meta-optimization.

def test_MetaOptimize():
    """
    Test-generator for meta-optimization using many different configurations.

    Meta-optimization is quite slow so fewer configurations are tested.
    """

    # For each optimizer that has tunable parameters (not PS, it has no parameters).
    for optimizer in [PSO, MOL, DE, LUS]:
        # For different search-space dimensionalities of the problems.
        for dim in [3]:
            # For different number of fitness evaluations.
            for max_evaluations in [5119]:
                # For different number of optimization runs.
                for num_runs in [1, 5]:
                    # For different number of meta-optimization runs.
                    for meta_num_runs in [1, 3]:
                        # For different number of evaluations of the meta-optimizer.
                        for meta_max_evaluations in [47]:
                            for log_capacity in [17]:
                                # For parallel and non-parallel.
                                for parallel in [False, True]:
                                    # Take several benchmark problems at random.
                                    num_problems = np.random.randint(1, 6)
                                    idx = np.random.choice(len(Problem.all_benchmark_problems), size=num_problems, replace=False)
                                    problems = [Problem.all_benchmark_problems[int(i)](dim=dim) for i in idx]

                                    # Generate random problem weights.
                                    weights = 1000.0 * np.random.rand(num_problems)

                                    # Run the test using this configuration.
                                    yield _do_test_MetaOptimize, optimizer, problems, num_runs, max_evaluations, meta_num_runs, meta_max_evaluations, weights, log_capacity, parallel


def _do_test_MetaOptimize(optimizer, problems, num_runs, max_evaluations, meta_num_runs,
                          meta_max_evaluations, weights, log_capacity, parallel):
    """
    Helper-function that performs the actual meta-optimization and asserts the results
    are as expected.
    """

    # Perform the meta-optimization using the given configuration.
    meta = MetaOptimize(optimizer=optimizer, problems=problems, weights=weights,
                        num_runs=num_runs, max_evaluations=max_evaluations,
                        meta_num_runs=meta_num_runs, meta_max_evaluations=meta_max_evaluations,
                        log_capacity=log_capacity, parallel=parallel)

    # Assert that the result of the optimization is as expected.
    # Note that the "correctness" is not tested, see discussion above.
    assert_is_instance(obj=meta.results.best, cls=np.ndarray)
    assert_equals(len(meta.results.best), optimizer.num_parameters)


########################################################################

if __name__ == "__main__":
    nose.main()

########################################################################
