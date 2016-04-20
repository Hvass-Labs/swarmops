########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Demonstration of optimizing benchmark problems.
########################################################################

import numpy as np
import swarmops.Problem as Problem
from swarmops.Optimize import MultiRun
from swarmops.PSO import PSO, MOL
from swarmops.DE import DE
from swarmops.PS import PS
from swarmops.LUS import LUS
from swarmops.Timer import Timer


########################################################################

# Fitness function used as example how to wrap a function in a Problem-object.
# Note that it is defined outside the "if __name__ == '__main__'" statement,
# in order for it to work with parallel execution.
def sphere_func(x, limit=np.Infinity):
    return np.sum(x*x)


########################################################################

if __name__ == "__main__":
    # Number of optimization runs.
    num_runs = 8

    # Execute the optimization runs in parallel.
    parallel = True

    # Search-space dimensionality.
    dim = 3

    # Max number of fitness evaluations.
    max_evaluations = dim * 5000

    # Number of fitness evaluations between printing of status messages.
    display_interval = 5000

    # Max-length of fitness trace.
    trace_len = 100

    # Create object-instance for a benchmark problem.
    #problem = Problem.Ackley(dim=dim)
    #problem = Problem.Sphere(dim=dim)
    #problem = Problem.Griewank(dim=dim)
    #problem = Problem.Rastrigin(dim=dim)
    problem = Problem.Rosenbrock(dim=dim)
    #problem = Problem.Schwefel1_2(dim=dim)
    #problem = Problem.Schwefel2_21(dim=dim)
    #problem = Problem.Schwefel2_22(dim=dim)
    #problem = Problem.Step(dim=dim)
    #problem = Problem.QuarticNoise(dim=dim)
    #problem = Problem.Penalized1(dim=dim)
    #problem = Problem.Penalized2(dim=dim)

    # Example of how to wrap a function in a problem-object.
    if False:
        problem = Problem.Benchmark(name="Sphere Wrap Func", dim=dim, fitness_min=0.0,
                                    lower_bound=-100.0, upper_bound=100.0,
                                    lower_init=50.0, upper_init=100.0,
                                    func=sphere_func)

    if True:
        # Use PSO as optimizer.
        optimizer = PSO

        # Control parameters.
        parameters = PSO.parameters_default

        # Examples of other control parameters.
        #parameters = PSO.parameters_30dim_60000eval
        #parameters = [71.0,  0.26, -1.1, 3.7]
    elif False:
        # Use MOL as optimizer.
        optimizer = MOL

        # Control parameters.
        parameters = MOL.parameters_default
        #parameters = [ 274.3153726  ,  -0.80864214  , -3.86480522]
        #parameters = [ 159.98393514 ,  -0.26711631  ,  5.51016079]
        #parameters = MOL.parameters_20dim_40000eval
    elif False:
        # Use DE as optimizer.
        optimizer = DE

        # Control parameters.
        parameters = DE.parameters_default
        #parameters = [ 115.79605113  ,  0.88891246  ,  0.50018078]
        #parameters = DE.parameters_20dim_40000eval
    elif False:
        # Use LUS as optimizer.
        optimizer = LUS

        # Control parameters.
        parameters = LUS.parameters_default

        # LUS converges fairly quickly so we allow fewer fitness evaluations.
        max_evaluations = dim * 50
        display_interval = 50
    else:
        # Use PS as optimizer.
        optimizer = PS

        # PS does not have any control parameters.
        parameters = None

        # PS converges fairly quickly so we allow fewer fitness evaluations.
        max_evaluations = dim * 50
        display_interval = 50

    # Print control parameters.
    print("Control parameters used for {0}:".format(optimizer.name))
    print(optimizer.parameters_dict(parameters))
    print()  # Newline.

########################################################################

    if False:
        # Demonstrate the optimizer's parallel execution.
        # This has significant overhead especially on Windows
        # and should generally only be used on fitness functions
        # that are themselves time-consuming to compute.

        # Start a timer.
        timer = Timer()

        # Perform a single optimization run using the optimizer
        # where the fitness is evaluated in parallel.
        result = optimizer(parallel=True, problem=problem,
                           max_evaluations=max_evaluations,
                           display_interval=display_interval)

        # Stop the timer.
        timer.stop()

        print()  # Newline.
        print("Time-Usage: {0}".format(timer))
        print()  # Newline.

        print("Best fitness from heuristic optimization: {0:0.4e}".format(result.best_fitness))
        print("Best solution:")
        print(result.best)


########################################################################

    # Start a timer.
    timer = Timer()

    # Perform multiple optimization runs.
    results = MultiRun(optimizer=optimizer, parameters=parameters,
                       num_runs=num_runs, problem=problem,
                       parallel=parallel,
                       max_evaluations=max_evaluations,
                       display_interval=display_interval, trace_len=trace_len)

    # Stop the timer.
    timer.stop()

    print()  # Newline.
    print("Time-Usage: {0}".format(timer))
    print()  # Newline.

    # Print statistics for the optimization results.
    results.print_statistics()

    print()  # Newline.

    # Print best-found solution.
    print("Best fitness from heuristic optimization: {0:0.4e}".format(results.best_fitness))
    print("Best solution:")
    print(results.best)

    # Refine the best-found solution.
    if True:
        print()  # Newline.
        print("Refining using SciPy's L-BFGS-B (this may be slow on some problems) ...")

        # Do the actual refinement using the L-BFGS-B optimizer.
        refined_fitness, refined_solution = results.refine()

        print("Best fitness from L-BFGS-B optimization: {0:0.4e}".format(refined_fitness))
        print("Best solution:")
        print(refined_solution)

    # Plot the fitness trace.
    if trace_len > 0:
        results.plot_fitness_trace()

########################################################################
