import numpy as np

from functools import wraps


def _seed(func):
    @wraps(func)
    def function_wrapper(*args, **kwargs):
        orig_seed = np.random.get_state()
        results = func(*args, **kwargs)
        np.random.set_state(orig_seed)
        return results
    return function_wrapper
