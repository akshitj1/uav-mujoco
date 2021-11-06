import numpy as np
from gym  import spaces


def denormalize(x_normed: np.ndarray, x_space :spaces.Box):
    return rescale(x_normed, normed_space(x_normed.size), x_space)

def normalize(x: np.ndarray, x_space :spaces.Box):
    return rescale(x, x_space, normed_space(x.size))

def rescale(x: np.ndarray, x_range: spaces.Box, y_range: spaces.Box):
    return y_range.low + (y_range.high-y_range.low) * ((x-x_range.low)/(x_range.high-x_range.low))

def normed_space(ndims: int) -> spaces.Box:
    return spaces.Box(low=-1, high=1, shape=(ndims,), dtype=np.float32)

def normed_space_like(eg_space: spaces.Box) -> spaces.Box:
    return normed_space(eg_space.shape[0])