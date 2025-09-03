import numpy as np


def round_up_to_multiple(x, base):
    """Rounds a number up to the nearest multiple of a base."""
    return int(np.ceil(x / base) * base)
