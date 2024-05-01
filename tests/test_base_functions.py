import numpy as np
import torch
from veccs import orderings

from batram.base_functions import compute_scale


def test_base_functions_compute_scale() -> None:
    locs = np.random.uniform(size=(50, 2)).astype(np.float32)
    locs = locs[orderings.maxmin_cpp(locs)]
    nbrs = orderings.find_nns_l2(locs, 10)
    scale = compute_scale(torch.from_numpy(locs), torch.from_numpy(nbrs))
    assert scale.numel() == 50
    assert all(scale[1:] <= scale[:-1])
