########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Demonstration of meta-optimization, that is, the tuning of the
# control parameters of an optimizer using another overlaying optimizer.
#
# Meta-optimization is described in detail in:
#
# [1] M.E.H. Pedersen. Tuning & Simplifying Heuristical Optimization (PhD thesis).
#     University of Southampton, School of Engineering Sciences. 2010
#     http://www.hvass-labs.org/people/magnus/thesis/pedersen08thesis.pdf
#
# [2] Short video on YouTube explaining meta-optimization.
#     https://www.youtube.com/watch?v=O6OQPpzVHBc
#
# [3] Short video demonstrating meta-optimization.
#     https://www.youtube.com/watch?v=6cM-e10YRdI
#
########################################################################

from swarmops.MetaOptimize import MetaOptimize
from swarmops.PSO import MOL
from swarmops.Problem import *
from swarmops.Timer import Timer

if __name__ == "__main__":

    ########################################################################
    # Settings for the optimization layer.

    # The optimizer whose control parameters must be meta-optimized / tuned.
    optimizer = MOL     # Tune the control parameters of MOL.
    #optimizer = PSO    # Tune the control parameters of PSO.
    #optimizer = DE     # Tune the control parameters of DE.
    #optimizer = LUS    # Tune the control parameters of LUS.

    # The dimensionality of the benchmark problems used in the tuning.
    # This may greatly affect the best choice of control parameters so it
    # should match the dim of the problems you ultimately want to optimize.
    dim = 3

    # Benchmark problems that the optimizer must be tuned for.
    # You can include as many as you like but it will make the
    # execution slower. So you should select a few problems
    # with different characteristics so the control parameters
    # can be expected to generalize better to new problems.
    # You can also use a single optimization problem if you like.
    problems = [Ackley(dim=dim),
                Griewank(dim=dim),
                Sphere(dim=dim),
                Penalized1(dim=dim),
                Rosenbrock(dim=dim)]

    # Weights for the benchmark problems. These can be used to change
    # the mutual importance of the problems in case an optimizer
    # cannot be made to perform well on all problems simultaneously.
    # You may have to use rather extreme weights. These are examples.
    weights = [1.0, 0.5, 0.001, 1.0, 1.0]

    # Number of fitness evaluations used in each optimization run.
    max_evaluations = dim * 5000

    # Number of optimization runs on each problem.
    # TODO: This should be 50 in a real run, but it becomes much slower.
    num_runs = 5

    ########################################################################
    # Settings for the Meta-Optimization layer.

    # Execute the meta-optimization runs in parallel.
    # This means the log of best control parameters will be empty
    # and the best-found solutions to the problems are also empty.
    # See README.md for more details.
    parallel = False

    # Number of meta-optimization runs.
    # This is typically set to 5 or more because the meta-optimizer
    # may fail to find good parameters in some of the runs.
    # TODO: Set to 5 or more.
    meta_num_runs = 4

    # Number of iterations for each meta-optimization run.
    # This can be left out when initializing the MetaOptimize-class which
    # will then select an appropriate number, typically 20 times the
    # number of control parameters to be tuned.
    meta_max_evaluations = optimizer.num_parameters * 20

    # Number of best control parameters to log and show afterwards.
    log_capacity = 20

    ########################################################################
    # Print various info before starting.

    print("Meta-optimizing the control parameters of:")
    print("{0} - {1}".format(optimizer.name, optimizer.name_full))

    msg = "Performing {0} meta-optimization runs with {1} iterations per run."
    print(msg.format(meta_num_runs, meta_max_evaluations))
    print()

    print("Number of problems: {0}".format(len(problems)))
    print("Dimensionality: {0}".format(dim))
    print("Iterations per run: {0}".format(max_evaluations))

    print("Problems:")
    for problem in problems:
        print(" - {0}".format(problem.name))

    print()  # Newline.
    print("Warning: This may be very slow!")
    print()  # Newline.

    ########################################################################
    # Perform meta-optimization.

    # Start a timer.
    timer = Timer()

    # Perform the meta-optimization.
    meta = MetaOptimize(optimizer=optimizer, problems=problems, weights=weights,
                        num_runs=num_runs, max_evaluations=max_evaluations,
                        meta_num_runs=meta_num_runs, meta_max_evaluations=meta_max_evaluations,
                        log_capacity=log_capacity, parallel=parallel)

    # Stop the timer.
    timer.stop()

    ########################################################################
    # Print results.

    # Print the time-usage.
    print("-------------------------------------------")
    print("Meta-Optimization Finished for {0}.".format(optimizer.name))
    print("Time-Usage: {0}".format(timer))
    print()

    # Print the best-found meta-fitness and the associated control parameters.
    print("Best meta-fitness: {0:.4e}".format(meta.results.best_fitness))
    print("Best control parameters:")
    print(meta.results.best)
    print(optimizer.parameters_dict(meta.results.best))
    print()  # Newline.

    # Print more control parameters from the log.
    print("Best control parameters ({0} best):".format(log_capacity))
    print("Meta-Fitn.:\tParameters:")
    for solution in meta.log.solutions:
        # Control parameters and meta-fitness.
        parameters = solution.x
        meta_fitness = solution.fitness

        # Print the meta-fitness and the associated control parameters.
        print("{0:.4e}\t{1}".format(meta_fitness, parameters))

    # Plot the fitness-trace.
    meta.results.plot_fitness_trace()

    print()  # Newline
    print("Best solutions found for the problems:")
    print()  # Newline.

    # Print the best solution found for each of the problems used
    # in the meta-optimization.
    # Note that these are sorted descendingly on problem_rank.fitness_sum
    for problem_rank in sorted(meta.meta_fitness.problem_ranks, key=lambda rank: rank.problem.name):
        # The optimization problem.
        problem = problem_rank.problem

        # The best solution found on that problem.
        best = problem_rank.best

        # The best fitness found.
        best_fitness = problem_rank.best_fitness

        # Print it.
        print("Problem: {0}, Best fitness: {1:.4e}".format(problem.name, best_fitness))
        print("Best solution:")
        print(best)
        print()  # Newline.

    if parallel:
        print("Meta-optimization was executed in parallel mode.")
        print("This means the list of best control parameters was empty,")
        print("and the best-found solutions were also empty.")
        print("If you need these, then you can run meta-optimization in")
        print("non-parallel mode and simply run several instances of")
        print("the program simultaneously.")
        print("See README.md for more details.")
        print()  # Newline.

########################################################################
