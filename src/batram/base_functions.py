import numpy as np
import torch

all = [
    "compute_scale",
]


def compute_scale(
    ordered_locs: np.ndarray | torch.Tensor, NN: np.ndarray | torch.Tensor
) -> torch.Tensor:
    """Computes scaling for the data.

    Args:
    -----
    ordered_locs: Locations of the data. Shape (N, d)
        Each row is one location in a d-dimensional space.

    NN: Conditioning sets of the data. Shape (N, m)
        Each row represents one location with references to the m nearest
        neighbors. -1 indicates not to condition on more neighbors.

    Returns:
    --------
    scale: Scaling of the data. Shape (N,)
    """
    ordered_locs = torch.as_tensor(ordered_locs)
    scale = (ordered_locs[1:, :] - ordered_locs[NN[1:, 0], :]).square().sum(1).sqrt()
    scale = torch.cat((scale[0].square().div(scale[4]).unsqueeze(0), scale))
    scale = scale.div(scale[0])
    return scale
