########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Particle Swarm Optimization (PSO).
#
# PSO is a heuristic optimizer that does not use the gradient of the problem
# being optimized. A so-called global-best variant of the PSO is implemented here.
# A simple PSO variant is also implemented here.
#
# Search-space boundaries are necessary for this PSO variant to work properly.
# So if your optimization problem does not have natural boundaries, you should
# simply choose some boundaries that are reasonable.
#
# PSO starts by creating a number of random trials called particles. In each
# iteration, these particles are moved around in the search-space using a
# formula that involves the particle's best-known position as well as the
# entire swarm's best-known position. This has been found to work well for
# optimizing many difficult problems, although a satisfactory solution is
# not guaranteed to be found.
#
# The PSO was originally proposed around year 1995, see [1] and [2]. In the
# following 20 years, thousands of PSO variants have been proposed.
# One of the early and basic variants of the PSO is implemented here.
# Newer PSO variants often claim to adapt the control parameters during
# optimization, thus making the PSO adapt better to new problems. But it
# was found in [3] that the basic PSO could perform just as well if using
# proper control parameters. Control parameters tuned for different
# optimization scenarios are given in [4] and included in this file below.
#
# References:
#
# [1] J. Kennedy, R.C. Eberhart. Particle Swarm Optimization. Proceedings of
#     IEEE International Conference on Neural Networks. pp. 1942-1948. 1995.
#
# [2] Y. Shi, R.C. Eberhart. A modified particle swarm optimizer. Proceedings
#     of IEEE International Conference on Evolutionary Computation. pp. 69-73. 1998.
#
# [3] M.E.H. Pedersen. Tuning & Simplifying Heuristical Optimization (PhD thesis).
#     University of Southampton, School of Engineering Sciences. 2010
#     http://www.hvass-labs.org/people/magnus/thesis/pedersen08thesis.pdf
#
# [4] M.E.H. Pedersen. Good parameters for particle swarm optimization.
#     Technical Report HL-1001, Hvass Laboratories. 2010.
#     http://www.hvass-labs.org/people/magnus/publications/pedersen10good-pso.pdf
#
########################################################################

import numpy as np
from swarmops.Optimize import SingleRun
from swarmops import tools


##################################################

class Base(SingleRun):
    def __init__(self, problem, parallel=False, *args, **kwargs):
        """
        Create object instance and perform a single optimization run using PSO.

        :param problem: The problem to be optimized. Instance of Problem-class.

        :param parallel:
            Evaluate the fitness for the particles in parallel.
            See the README.md file for a discussion on this.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is an instance of the FitnessTrace-class.
        """

        # Copy arguments to instance variables.
        self.problem = problem
        self.parallel = parallel

        # Initialize all particles with random positions in the search-space.
        # The first index is for the particle number.
        # The second index is for the search-space dimension.
        # Note that self.num_particles must be set prior to this by the sub-class.
        self.particle = tools.rand_population(lower=problem.lower_init,
                                              upper=problem.upper_init,
                                              num_agents=self.num_particles,
                                              dim=problem.dim)

        # Initialize best-known positions for the particles to their starting positions.
        # A copy is made because the particle positions will change during optimization
        # regardless of improvement to the particle's fitness.
        self.particle_best = np.copy(self.particle)

        # Initialize fitness of best-known particle positions to infinity.
        self.particle_best_fitness = np.repeat(np.inf, self.num_particles)

        # Boundaries for the velocity. These are set to the range of the search-space.
        bound_range = np.abs(problem.upper_bound - problem.lower_bound)
        self.velocity_lower_bound = -bound_range
        self.velocity_upper_bound = bound_range

        # Initialize all velocities with random values in the allowed range.
        self.velocity = tools.rand_population(lower=self.velocity_lower_bound,
                                              upper=self.velocity_upper_bound,
                                              num_agents=self.num_particles,
                                              dim=problem.dim)

        # Initialize parent-class which also starts the optimization run.
        SingleRun.__init__(self, *args, **kwargs)

    def _optimize(self):
        """
        Perform a single optimization run.
        This function is called by the parent-class.
        """

        # Calculate fitness for the initial particle positions.
        self._update_fitness()

        # Optimization iterations.
        # The counting starts with num_particles because the fitness has
        # already been calculated once for each particle during initialization.
        for i in range(self.num_particles, self.max_evaluations, self.num_particles):
            # Update the particle velocities and positions.
            self._update_particles()

            # Update the fitness for each particle.
            self._update_fitness()

            # Call parent-class to print status etc. during optimization.
            self._iteration(i)

    def _fitness(self, i):
        """
        Calculate the fitness for the i'th particle.
        """

        return self.problem.fitness(self.particle[i, :], limit=self.particle_best_fitness[i])

    def _update_fitness(self):
        """
        Calculate and update the fitness for each particle. Also updates the particle's
        and swarm's best-known fitness and position if an improvement is found.
        """

        if not self.parallel:
            # Calculate the fitness for each particle. Not parallel.
            new_fitness = [self._fitness(i) for i in range(self.num_particles)]
        else:
            import multiprocessing as mp

            # Create a pool of workers sized according to the CPU cores available.
            pool = mp.Pool()

            # Calculate the fitness for each particle in parallel.
            new_fitness = pool.map(self._fitness, range(self.num_particles))

            # Close the pool of workers and wait for them all to finish.
            pool.close()
            pool.join()

        # For each particle.
        for i in range(self.num_particles):
            # If the fitness is an improvement over the particle's best-known fitness.
            if new_fitness[i] < self.particle_best_fitness[i]:
                # Update the particle's best-known fitness and position.
                self.particle_best_fitness[i] = new_fitness[i]
                self.particle_best[i, :] = self.particle[i, :]

                # Update the entire swarm's best-known fitness and position if an improvement.
                # The parent-class is used for this.
                self._update_best(fitness=new_fitness[i],
                                  x=self.particle_best[i, :])


##################################################

class PSO(Base):
    """
        Perform a single optimization run using Particle Swarm Optimization (PSO).

        This is a so-called global-best variant, although it may have slightly
        different features than other global-best variants in the research literature.

        In practice, you would typically perform multiple optimization runs using
        the MultiRun-class. The reason is that PSO is a heuristic optimizer so
        there is no guarantee that an acceptable solution is found in any single
        run. It is more likely that an acceptable solution is found if you perform
        multiple optimization runs.

        Control parameters have been tuned for different optimization scenarios.
        First try and use the default parameters. If that does not give
        satisfactory results, then you may try some of the following.
        Select the parameters that most closely match your problem.
        For example, if you want to optimize a problem where the search-space
        has 15 dimensions and you can perform 30000 evaluations, then you could
        first try using parameters_20dim_40000eval. If that does not give
        satisfactory results then you could try using parameters_10dim_20000eval.
        If that does not work then you will either need to meta-optimize the
        parameters for the problem at hand, or you should try using another optimizer.
    """

    # Name of this optimizer.
    name = "PSO"
    name_full = "Particle Swarm Optimization (Global-Best Variant)"

    # Number of control parameters for PSO. Used by MetaFitness-class.
    num_parameters = 4

    # Lower boundaries for the control parameters of PSO. Used by MetaFitness-class.
    parameters_lower_bound = [1.0, -2.0, -4.0, -4.0]

    # Upper boundaries for the control parameters of PSO. Used by MetaFitness-class.
    parameters_upper_bound = [300.0, 2.0, 4.0, 6.0]

    @staticmethod
    def parameters_list(num_particles, omega, phi_p, phi_g):
        """
        Create a list with PSO parameters in the correct order.

        :param num_particles: Number of particles for the PSO swarm.
        :param omega: The omega parameter (aka. inertia weight) for the PSO.
        :param phi_p: The phi_p parameter (aka. particle weight) for the PSO.
        :param phi_g: The phi_g parameter (aka. social weight) for the PSO.
        :return: List with PSO parameters.
        """

        return [num_particles, omega, phi_p, phi_g]

    @staticmethod
    def parameters_dict(parameters):
        """
        Create and return a dict from a list of PSO parameters.
        This is useful for printing the named parameters.

        :param parameters: List with PSO parameters assumed to be in the correct order.
        :return: Dict with PSO parameters.
        """

        return {'num_particles': parameters[0],
                'omega': parameters[1],
                'phi_p': parameters[2],
                'phi_g': parameters[3]}

    # Default parameters for the PSO which will be used if no other parameters are specified.
    # These are a compromise of the tuned parameters below. Try this first and see if it works.
    parameters_default = [50.0, -0.4, -0.3, 3.9]

    # Parameters tuned by hand. These are common in the older research literature on PSO
    # but perform much worse than meta-optimized parameters, especially for this PSO variant.
    parameters_hand_tuned = [50.0, 0.729, 1.49445, 1.49445]

    # Parameters tuned for benchmark problems in 2 dimensions using 400 fitness evaluations.
    parameters_2dim_400eval_a = [25.0, 0.3925, 2.5586, 1.3358]
    parameters_2dim_400eval_b = [29.0, -0.4349, -0.6504, 2.2073]

    # Parameters tuned for benchmark problems in 2 dimensions using 4000 fitness evaluations.
    parameters_2dim_4000eval_a = [156.0, 0.4091, 2.1304, 1.0575]
    parameters_2dim_4000eval_b = [237.0, -0.2887, 0.4862, 2.5067]

    # Parameters tuned for benchmark problems in 5 dimensions using 1000 fitness evaluations.
    parameters_5dim_1000eval_a = [63.0, -0.3593, -0.7238, 2.0289]
    parameters_5dim_1000eval_b = [47.0, -0.1832, 0.5287, 3.1913]

    # Parameters tuned for benchmark problems in 5 dimensions using 10000 fitness evaluations.
    parameters_5dim_10000eval_a = [223.0, -0.3699, -0.1207, 3.3657]
    parameters_5dim_10000eval_b = [203.0, 0.5069, 2.5524, 1.0056]

    # Parameters tuned for benchmark problems in 10 dimensions using 2000 fitness evaluations.
    parameters_10dim_2000eval_a = [63.0, 0.6571, 1.6319, 0.6239]
    parameters_10dim_2000eval_b = [204.0, -0.2134, -0.3344, 2.3259]

    # Parameters tuned for benchmark problems in 10 dimensions using 20000 fitness evaluations.
    parameters_10dim_20000eval = [53.0, -0.3488, -0.2746, 4.8976]

    # Parameters tuned for benchmark problems in 20 dimensions using 40000 fitness evaluations.
    parameters_20dim_40000eval = [69.0, -0.4438, -0.2699, 3.395]

    # Parameters tuned for benchmark problems in 20 dimensions using 400000 fitness evaluations.
    parameters_20dim_400000eval_a = [149.0, -0.3236, -0.1136, 3.9789]
    parameters_20dim_400000eval_b = [60.0, -0.4736, -0.97, 3.7904]
    parameters_20dim_400000eval_c = [256.0, -0.3499, -0.0513, 4.9087]

    # Parameters tuned for benchmark problems in 30 dimensions using 60000 fitness evaluations.
    parameters_30dim_60000eval = [134.0, -0.1618, 1.8903, 2.1225]

    # Parameters tuned for benchmark problems in 30 dimensions using 600000 fitness evaluations.
    parameters_30dim_600000eval = [95.0, -0.6031, -0.6485, 2.6475]

    # Parameters tuned for benchmark problems in 50 dimensions using 100000 fitness evaluations.
    parameters_50dim_100000eval = [106.0, -0.2256, -0.1564, 3.8876]

    # Parameters tuned for benchmark problems in 100 dimensions using 200000 fitness evaluations.
    parameters_100dim_200000eval = [161.0, -0.2089, -0.0787, 3.7637]

    def __init__(self, parameters=parameters_default, *args, **kwargs):
        """
        Create object instance and perform a single optimization run using PSO.

        :param parameters:
            Control parameters for the PSO.
            These may have a significant impact on the optimization performance.
            First try and use the default parameters and if they don't give satisfactory
            results, then experiment with other parameters.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is an instance of the FitnessTrace-class.
        """

        # Unpack control parameters.
        self.num_particles, self.omega, self.phi_p, self.phi_g = parameters

        # The number of particles must be an integer.
        self.num_particles = int(self.num_particles)

        # Initialize parent-class which also starts the optimization run.
        Base.__init__(self, *args, **kwargs)

    def _update_particles(self):
        """
        Update the velocities and positions for all particles.
        This does not update the fitness for each particle.
        """

        # Random values between zero and one. One random value per particle.
        rand_p = tools.rand_uniform(size=self.num_particles)
        rand_g = tools.rand_uniform(size=self.num_particles)

        # Update velocity for all particles using numpy operations.
        # For an explanation of this formula, see the research papers referenced above.
        # Note that self.best is the swarm's best-known position aka. global-best.
        self.velocity = (self.omega * self.velocity.T \
                         + self.phi_p * rand_p * (self.particle_best - self.particle).T \
                         + self.phi_g * rand_g * (self.best - self.particle).T).T

        # Fix de-normalized floating point values which can make the execution very slow.
        self.velocity = tools.denormalize_trunc(self.velocity)

        # Bound velocity.
        self.velocity = tools.bound(self.velocity, self.velocity_lower_bound, self.velocity_upper_bound)

        # Update particle positions in the search-space by adding the velocity.
        self.particle = self.particle + self.velocity

        # Bound particle position to search-space.
        self.particle = tools.bound(self.particle, self.problem.lower_bound, self.problem.upper_bound)


##################################################

class MOL(Base):
    """
        Perform a single optimization run using Many Optimizing Liaisons (MOL).

        In practice, you would typically perform multiple optimization runs using
        the MultiRun-class. The reason is that MOL is a heuristic optimizer so
        there is no guarantee that an acceptable solution is found in any single
        run. It is more likely that an acceptable solution is found if you perform
        multiple optimization runs.

        Control parameters have been tuned for different optimization scenarios.
        First try and use the default parameters. If that does not give
        satisfactory results, then you may try some of the following.
        Select the parameters that most closely match your problem.
        For example, if you want to optimize a problem where the search-space
        has 15 dimensions and you can perform 30000 evaluations, then you could
        first try using parameters_20dim_40000eval. If that does not give
        satisfactory results then you could try using parameters_10dim_20000eval.
        If that does not work then you will either need to meta-optimize the
        parameters for the problem at hand, or you should try using another optimizer.
    """

    # Name of this optimizer.
    name = "MOL"
    name_full = "Many Optimizing Liaisons (Simple Variant of PSO)"

    # Number of control parameters for MOL. Used by MetaFitness-class.
    num_parameters = 3

    # Lower boundaries for the control parameters of MOL. Used by MetaFitness-class.
    parameters_lower_bound = [1.0, -2.0, -4.0]

    # Upper boundaries for the control parameters of MOL. Used by MetaFitness-class.
    parameters_upper_bound = [300.0, 2.0, 6.0]

    @staticmethod
    def parameters_dict(parameters):
        """
        Create and return a dict from a list of MOL parameters.
        This is useful for printing the named parameters.

        :param parameters: List with MOL parameters assumed to be in the correct order.
        :return: Dict with MOL parameters.
        """

        return {'num_particles': parameters[0],
                'omega': parameters[1],
                'phi_g': parameters[2]}

    @staticmethod
    def parameters_list(num_particles, omega, phi_g):
        """
        Create a list with MOL parameters in the correct order.

        :param num_particles: Number of particles for the MOL swarm.
        :param omega: The omega parameter (aka. inertia weight) for the MOL.
        :param phi_g: The phi_g parameter (aka. social weight) for the MOL.
        :return: List with MOL parameters.
        """

        return [num_particles, omega, phi_g]

    # Default parameters for MOL which will be used if no other parameters are specified.
    # These are a compromise of the tuned parameters below. Try this first and see if it works.
    parameters_default = [100.0, -0.35, 3.0]

    # Parameters tuned for benchmark problems in 2 dimensions using 400 fitness evaluations.
    parameters_2dim_400eval_a = [23.0, -0.3328, 2.8446]
    parameters_2dim_400eval_b = [50.0,  0.2840, 1.9466]

    # Parameters tuned for benchmark problems in 2 dimensions using 4000 fitness evaluations.
    parameters_2dim_4000eval_a = [183.0, -0.2797, 3.0539]
    parameters_2dim_4000eval_b = [139.0,  0.6372, 1.0949]

    # Parameters tuned for benchmark problems in 5 dimensions using 10000 fitness evaluations.
    parameters_5dim_1000eval = [50.0, -0.3085, 2.0273]

    # Parameters tuned for benchmark problems in 5 dimensions using 10000 fitness evaluations.
    parameters_5dim_10000eval = [96.0, -0.3675, 4.1710]

    # Parameters tuned for benchmark problems in 10 dimensions using 2000 fitness evaluations.
    parameters_10dim_2000eval = [60.0, -0.2700, 2.9708]

    # Parameters tuned for benchmark problems in 10 dimensions using 20000 fitness evaluations.
    parameters_10dim_20000eval = [116.0, -0.3518, 3.8304]

    # Parameters tuned for benchmark problems in 20 dimensions using 40000 fitness evaluations.
    parameters_20dim_40000eval = [228.0, -0.3747, 4.2373]

    # Parameters tuned for benchmark problems in 20 dimensions using 400000 fitness evaluations.
    parameters_20dim_400000eval = [125.0, -0.2575, 4.6713]

    # Parameters tuned for benchmark problems in 30 dimensions using 600000 fitness evaluations.
    parameters_30dim_60000eval = [198.0, -0.2723, 3.8283]

    # Parameters tuned for benchmark problems in 50 dimensions using 100000 fitness evaluations.
    parameters_50dim_100000eval = [290.0, -0.3067, 3.6223]

    # Parameters tuned for benchmark problems in 100 dimensions using 200000 fitness evaluations.
    parameters_100dim_200000eval = [219.0, -0.1685, 3.9162]

    def __init__(self, parameters=parameters_default, *args, **kwargs):
        """
        Create object instance and perform a single optimization run using MOL.

        :param problem: The problem to be optimized. Instance of Problem-class.

        :param parameters:
            Control parameters for the MOL.
            These may have a significant impact on the optimization performance.
            First try and use the default parameters and if they don't give satisfactory
            results, then experiment with other the parameters.

        :return:
            Object instance. Get the optimization results from the object's variables.
            -   best is the best-found solution.
            -   best_fitness is the associated fitness of the best-found solution.
            -   fitness_trace is an instance of the FitnessTrace-class.
        """

        # Unpack control parameters.
        self.num_particles, self.omega, self.phi_g = parameters

        # The number of particles must be an integer.
        self.num_particles = int(self.num_particles)

        # Initialize parent-class which also starts the optimization run.
        Base.__init__(self, *args, **kwargs)

    def _update_particles(self):
        """
        Update the velocities and positions for all particles.
        This does not update the fitness for each particle.
        """

        # Random values between zero and one. One random value per particle.
        rand_g = tools.rand_uniform(size=self.num_particles)

        # Update velocity for all particles using numpy operations.
        # For an explanation of this formula, see the research papers referenced above.
        # Note that self.best is the swarm's best-known position aka. global-best.
        self.velocity = (self.omega * self.velocity.T \
                         + self.phi_g * rand_g * (self.best - self.particle).T).T

        # Fix de-normalized floating point values which can make the execution very slow.
        self.velocity = tools.denormalize_trunc(self.velocity)

        # Bound velocity.
        self.velocity = tools.bound(self.velocity, self.velocity_lower_bound, self.velocity_upper_bound)

        # Update particle positions in the search-space by adding the velocity.
        self.particle = self.particle + self.velocity

        # Bound particle position to search-space.
        self.particle = tools.bound(self.particle, self.problem.lower_bound, self.problem.upper_bound)


##################################################
