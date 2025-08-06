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


def scaling_mf(
    locs_ord: np.ndarray | torch.Tensor,
    NN: np.ndarray | torch.Tensor,
    fs: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Computes scaling for multifidelity data.

    Args:
    -----
    all_locs: Locations of the data in all Fidelities. Shape (N, d),
    where N = N_1 + N_2 + ... + N_r

    NN: Multifidelity conditioning sets. Shape (N, 2m), where m is the
    maximum number of neighbors of each fidelity, with -1 indicating no
    neighbors

    fs: Fidelity sizes (N_1, N_2, ..., N_r)

    Returns:
    scale: Scaling of the data. Shape (N, )
    """
    scales = torch.zeros(sum(fs))
    for i in range(1, sum(fs)):
        loc = locs_ord[i]
        nn_now = NN[i]
        nn_now_f = nn_now[nn_now != -1]
        locs_nn = locs_ord[nn_now_f]
        distances = torch.norm(locs_nn - loc, dim=1)
        scales[i] = distances.min()
    scales[0] = scales[1].square().div(scales[5])
    scales = scales.div(scales[0])
    return scales
