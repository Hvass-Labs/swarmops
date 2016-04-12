########################################################################
# SwarmOps - Heuristic optimization for Python.
# Copyright (C) 2003-2016 Magnus Erik Hvass Pedersen.
# See the file README.md for instructions.
# See the file LICENSE.txt for license details.
# SwarmOps on the internet: http://www.Hvass-Labs.org/
########################################################################

########################################################################
# Timer-class used for timing execution.
########################################################################

import time
from datetime import timedelta


########################################################################

class Timer:
    """
    Used for timing execution.
    """

    def __init__(self):
        """
        Start the timer.

        :return: Object instance.
        """

        # Note that time.process_time() doesn't work with multiprocessing.

        self.start_time = time.time()
        self.end_time = self.start_time

    def less_than(self, hours=0, minutes=0, seconds=0):
        """
        Return True if the time that has passed between now and the timer was
        started is less than the given hours, minutes and seconds,
        otherwise return False.

        Note that this does not stop the timer.
        """

        # The current time.
        now = time.time()

        # Time-difference between now and the timer was started.
        time_dif = now - self.start_time

        # Time-difference as a time-delta.
        delta = timedelta(seconds=time_dif)

        # Limit as a time-delta.
        limit_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Compare the two time-deltas.
        return delta < limit_delta

    def stop(self):
        """
        Stop the timer.
        """

        self.end_time = time.time()

    def __str__(self):
        """
        The difference between start and end-time is converted to a string.
        """

        time_dif = self.end_time - self.start_time
        return str(timedelta(seconds=int(round(time_dif))))

########################################################################
