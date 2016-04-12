########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Parent-classes for doing optimization. These provide various logistics,
# printing of status messages, running multiple optimizations, etc.
########################################################################

import numpy as np
from swarmops.FitnessTrace import FitnessTrace
from swarmops import tools


##################################################

class SingleRun:
    """
    Parent-class for performing a single optimization run.
    The class provides various logistics that are common to all
    optimizers. If you make a new optimizer, then you should derive
    from this class.
    """

    def __init__(self, max_evaluations, run_number=0, trace_len=0, display_interval=0):
        """
        Create object instance and initialize variables for keeping track
        of the optimization progress.
        Then call self._optimize() to perform the actual optimization.

        :param max_evaluations:
            Maximum number of fitness evaluations for the problem.

        :param run_number:
            The optimization run number, if performing multiple optimization runs.

        :param trace_len:
            Approximate length of the fitness-trace.
            Default is zero which means that no fitness-trace will be created.

        :param display_interval:
            Approximate interval between printing status messages.
            Default is zero which means that no status messages will be printed.

        :return:
            Object instance.
        """

        # Copy arguments to instance variables.
        self.max_evaluations = max_evaluations
        self.run_number = run_number
        self.display_interval = display_interval

        # Initialize the counter for the next status-display.
        self.display_next = 0

        # Create an object used for tracing the fitness at regular intervals.
        self.fitness_trace = FitnessTrace(trace_len, max_evaluations)

        # Initialize best-known position and fitness.
        self.best = None
        self.best_fitness = np.inf

        # Print status at the beginning of the optimization run.
        self._start_run()

        # Perform the actual optimization iterations.
        self._optimize()

        # Print status at the end of the optimization run.
        self._end_run()

    def _optimize(self):
        """
        Perform the actual optimization.
        This function should be implemented by the child-class.
        The function does not return anything, instead it will
        call self._update_best() to update the best-known solution
        during optimization.
        """

        # Raise an exception if the child-class has not implemented this function.
        raise NotImplementedError

    def _start_run(self):
        """
        Print status at the beginning of the optimization run.
        """

        if self.display_interval > 0:
            msg = "Starting optimization run {0} using {1} ..."
            print(msg.format(self.run_number, self.name))

    def _iteration(self, i):
        """
        Print status and trace fitness during optimization.
        This function should be called regularly during optimization.
        """

        if self.display_interval > 0 and i >= self.display_next:
            # Print the status.
            msg = "Run: {0}, Iteration: {1}, Best Fitness: {2:.4e}"
            print(msg.format(self.run_number, i, self.best_fitness))

            # Increment the counter for the next status-display.
            self.display_next = i + self.display_interval

        # Trace the fitness.
        self.fitness_trace.trace(i, self.best_fitness)

    def _end_run(self):
        """
        Print status at the end of the optimization run.
        """

        if self.display_interval > 0:
            # Print the status.
            msg = "Finished optimization run {0}, Best fitness: {1:.4e}"
            print(msg.format(self.run_number, self.best_fitness))

    def _update_best(self, fitness, x):
        """
        Update the best-known solution and fitness if an improvement.

        WARNING: This function does NOT copy the array x so you must
        ensure that the optimizer does not modify the array you pass
        in as x here.

        :param fitness: New fitness.
        :param x: New solution.
        :return: Boolean whether the fitness was an improvement.
        """

        # If the fitness is an improvement over the best-known fitness.
        improvement = fitness < self.best_fitness
        if improvement:
            # Update the best-known fitness and position.
            self.best_fitness = fitness
            self.best = x

        return improvement


##################################################

class MultiRun:
    """
    Perform multiple optimization runs with an optimizer and
    calculate statistics on the results.

    This has been separated into its own class for the sake of
    modularity so it is easier to read and maintain. There is only a
    tiny overhead of instantiating the SingleRun-class for each run.
    """

    def __init__(self, optimizer, num_runs, problem, parallel=True, *args, **kwargs):
        """
        Create object instance and perform multiple optimization runs.
        To retrieve the results, access the object variables afterwards.

        The parameters are the same as for the Optimize-class,
        except for the following:

        :param optimizer:
            Class for the optimizer e.g. PSO or DE.

        :param num_runs:
            Number of optimization runs to perform.

        :param problem:
            The problem to be optimized. Instance of Problem-class.

        :param parallel:
            Perform the optimization runs in parallel (True) or serial (False).

        :param args:
            Arguments to pass along to the Optimize-class.

        :param kwargs:
            Arguments to pass along to the Optimize-class.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best_solution is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is a 2-d numpy array with the fitness trace of each run.
        """

        # Copy arguments to instance variables.
        self.problem = problem
        self.optimizer = optimizer

        # Store the args and kwargs to be passed on to the optimizer.
        self.args = args
        self.kwargs = kwargs

        if not parallel:
            # Run the optimizer multiple times on one processor.
            self.runs = [self._optimize(run_number=i) for i in range(num_runs)]
        else:
            import multiprocessing as mp

            # Create a pool of workers sized according to the CPU cores available.
            pool = mp.Pool()

            # Run the optimizer multiple times in parallel.
            # We must use a helper-function for this.
            self.runs = pool.map(self._optimize_parallel, range(num_runs))

            # Close the pool of workers and wait for them all to finish.
            pool.close()
            pool.join()

        # Put the best solutions from all the optimization runs into an array.
        self.solution = np.array([run.best for run in self.runs])

        # Put the best fitness from all the optimization runs into an array.
        self.fitness = np.array([run.best_fitness for run in self.runs])

        # Put the fitness traces from all the optimization runs into an array.
        # This only works when the fitness-traces are all the same length.
        # If you make changes to the optimizers so they can stop early, then
        # you may have to change this.
        self.fitness_trace = np.array([run.fitness_trace.fitness for run in self.runs])

        # Index for the best fitness (minimization).
        i = np.argmin(self.fitness)

        # Best optimization run. This is an instance of the SingleRun-class.
        self.best_run = self.runs[i]

        # Best fitness of all the optimization runs.
        self.best_fitness = self.fitness[i]

        # Best solution of all the optimization runs.
        self.best = self.solution[i]

    def _optimize(self, run_number):
        """
        Helper-function used for execution of an optimization run. Non-parallel.

        :param run_number: Counter for the optimization run.
        :return: Instance of the SingleRun-class.
        """

        return self.optimizer(problem=self.problem, run_number=run_number,
                              *self.args, **self.kwargs)

    def _optimize_parallel(self, run_number):
        """
        Helper-function used for parallel execution of an optimization run.

        :param run_number: Counter for the optimization run.
        :return: Instance of the SingleRun-class.
        """

        # Create a new Pseudo-Random Number Generator for the thread.
        tools.new_prng()

        # Do the optimization.
        return self._optimize(run_number=run_number)

    def refine(self):
        """
        Refine the best result from heuristic optimization using SciPy's L-BFGS-B method.
        This may significantly improve the results on some optimization problems,
        but it is sometimes very slow to execute.

        NOTE: This function imports SciPy, which should make it possible
        to use the rest of this source-code library even if SciPy is not installed.
        SciPy should first be loaded when calling this function.

        :return:
            A tuple with:
            -   The best fitness found.
            -   The best solution found.
        """

        # SciPy requires bounds in another format.
        bounds = list(zip(self.problem.lower_bound, self.problem.upper_bound))

        # Start SciPy optimization at best found solution.
        import scipy.optimize
        res = scipy.optimize.minimize(fun=self.problem.fitness,
                                      x0=self.best,
                                      method="L-BFGS-B",
                                      bounds=bounds)

        # Get best fitness and parameters.
        refined_fitness = res.fun
        refined_solution = res.x

        return refined_fitness, refined_solution

    def print_statistics(self):
        """
        Print statistics for the fitness.
        """

        # Print header.
        print("{0} - Optimized by {1}".format(self.problem.name_full, self.optimizer.name))
        print("Fitness Statistics:")

        # Print mean and standard deviation.
        print("- Mean:\t\t{0:.4e}".format(self.fitness.mean()))
        print("- Std.:\t\t{0:.4e}".format(self.fitness.std()))

        # Print quartiles.
        print("- Min:\t\t{0:.4e}".format(self.fitness.min()))
        print("- 1st Qrt.:\t{0:.4e}".format(np.percentile(self.fitness, 0.25)))
        print("- Median:\t{0:.4e}".format(np.percentile(self.fitness, 0.5)))
        print("- 3rd Qrt.:\t{0:.4e}".format(np.percentile(self.fitness, 0.75)))
        print("- Max:\t\t{0:.4e}".format(self.fitness.max()))

    def plot_fitness_trace(self, y_log_scale=True, filename=None):
        """
        Plot the fitness traces.

        NOTE: This function imports matplotlib, which should make it possible
        to use the rest of this source-code library even if it is not installed.
        matplotlib should first be loaded when calling this function.

        :param y_log_scale: Use log-scale for y-axis.
        :param filename: Output filename e.g. "foo.svg". If None then plot to screen.
        :return: Nothing.
        """

        import matplotlib.pyplot as plt

        # Setup plotting.
        plt.grid()

        # Axis labels.
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (Lower is better)")

        # Title.
        title = "{0} - Optimized by {1}".format(self.problem.name_full, self.optimizer.name)
        plt.title(title)

        # Use log-scale for Y-axis.
        if y_log_scale:
            plt.yscale("log", nonposy="clip")

        # Plot the fitness-trace for each optimization run.
        for run in self.runs:
            # Array with iteration counter for the optimization run.
            iteration = run.fitness_trace.iteration

            # Array with fitness-trace for the optimization run.
            fitness_trace = run.fitness_trace.fitness

            # Plot the fitness-trace.
            plt.plot(iteration, fitness_trace, 'r-', color='black', alpha=0.25)

        # Plot to screen or file.
        if filename is None:
            # Plot to screen.
            plt.show()
        else:
            # Plot to file.
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


##################################################
