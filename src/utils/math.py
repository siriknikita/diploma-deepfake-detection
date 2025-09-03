import numpy as np
from pydantic import PositiveInt


def round_up_to_multiple(number: PositiveInt, base: PositiveInt) -> int:
    """
    Rounds a number up to the nearest multiple of a base.

    Args:
        number (PositiveInt): The number to be rounded up.
        base (PositiveInt): The base to which the number should be rounded up.
    Returns:
        int: The rounded number.
    """
    return int(np.ceil(number / base) * base)
