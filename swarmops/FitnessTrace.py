########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Trace the fitness progress during optimization so it can be plotted.
########################################################################


class FitnessTrace:
    """
    Trace / log the fitness at regular intervals during optimization.
    This is used for plotting the optimization progress afterwards.
    """

    def __init__(self, trace_len, max_evaluations):
        """
        Create the object instance.

        :param trace_len: Max length of fitness-trace. Zero if no fitness-trace is wanted.
        :param max_evaluations: Number of optimization iterations that will be performed.
        :return: Object instance.
        """

        # The length of the fitness trace cannot be greater than
        # the number of optimization iterations.
        self.max_len = min(trace_len, max_evaluations)

        # Initialize empty arrays for the fitness-trace and iteration counter.
        # See performance note below.
        self.iteration = []         # Plot this as the x-axis.
        self.fitness = []           # Plot this as the y-axis.

        # Initialize the iteration counter used to decide when to trace/log the fitness.
        self.next_iteration = 0

        if self.max_len > 0:
            # The iteration interval between each trace of the fitness.
            self.interval = max_evaluations // self.max_len

    def trace(self, iteration, fitness):
        """
        Trace the fitness at regular intervals.

        :param iteration:  The number of optimization iterations performed so far.
        :param fitness: The best fitness found so far.
        :return: Nothing.
        """

        # Note:
        # Appending to a list is by far the most elegant way of implementing this.
        # The performance overhead of extending the list is tiny in comparison to
        # the total runtime of the optimization. Using a pre-allocated array instead
        # requires math with special cases and makes the implementation error-prone.

        # Proceed if a trace is desired.
        if self.max_len > 0:
            # Trace the fitness at regular intervals.
            if iteration >= self.next_iteration:
                # Append the fitness to the trace-array.
                self.fitness.append(fitness)

                # Append the iteration counter to the array.
                self.iteration.append(iteration)

                # Increment the counter for the next trace.
                self.next_iteration = iteration + self.interval

########################################################################
