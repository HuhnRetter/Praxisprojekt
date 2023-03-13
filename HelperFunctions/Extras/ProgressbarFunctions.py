import sys
import time

def progressBarWithTime(i, n_total_steps, starting_time):
    """overwrites the previous progressbar to create a dynamic progressbar with time approximation

    :param i: current step as int
    :param n_total_steps: total steps/batches of the epoch as int
    :param starting_time: starting time of the progressbar
    """
    bar_len = 60
    filled_len = int(round(bar_len * i / float(n_total_steps)))

    status = calculateRemainingTime(starting_time, time.time(), i, n_total_steps)

    percents = round(100.0 * i / float(n_total_steps), 1)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def calculateRemainingTime(starting_time, current_time, i, n_total_steps):
    """calculated the remaining time with the help of the given parameters

    :param starting_time: starting time of the progressbar
    :param current_time: current time as the calculations take place
    :param i: current step as int
    :param n_total_steps: total steps/batches of the epoch as int

    Returns: returns approximated remaining time in seconds

    """
    if i <= 0:
        return "remaining Time: NULL"

    time_taken = current_time - starting_time
    steps_taken = i / n_total_steps
    remaining_time = time_taken / steps_taken * (1 - steps_taken)
    return f"remaining Time: {remaining_time:.1f}s"