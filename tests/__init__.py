import numpy as np
from veccs.orderings2 import find_prev_nearest_neighbors


def find_nn_l2(locs, max_nn=10):
    return find_prev_nearest_neighbors(
        locs,
        np.arange(locs.shape[0]),
        max_nn,
    )
