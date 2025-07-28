from dataclasses import dataclass

import torch
import numpy as np
from batram.legmods import AugmentedData
from batram.base_functions import scaling_mf

@dataclass
class Data:
    """Data class

    Holds $n$ replicates of spatial field observed at $N$ locations. The data in
    this class has not been normalized.

    Note
    ----
    scales refers scaled distance to the nearest neighbor.

    Attributes
    ----------
    locs
        Locations of the data. shape (N, d)
    response
        Response of the data. Shape (n, N)

    augmented_response
        Augmented response of the data. Shape (n, N, m + 1). nan indicates no
        conditioning.

    conditioning_sets
        Conditioning sets of the data. Shape (N, m)
    """

    locs: torch.Tensor
    response: torch.Tensor
    augmented_response: torch.Tensor
    conditioning_sets: torch.Tensor

    @staticmethod
    def new(locs, response, conditioning_set):
        """Creates a new data object."""
        nlocs = locs.shape[0]
        ecs = torch.hstack([torch.arange(nlocs).reshape(-1, 1), conditioning_set])
        augmented_response = torch.where(ecs == -1, torch.nan, response[:, ecs])

        return Data(locs, response, augmented_response, conditioning_set)
    
@dataclass
class MultiFidelityData:
    """Data class

    Holds $n$ replicates of spatial field observed at $N$ locations. The data in
    this class has not been normalized.

    Note
    ----
    scales refers scaled distance to the nearest neighbor.

    Attributes
    ----------
    locs
        Locations of the data. shape (N, d)
    responsez
        Response of the data. Shape (n, N)

    augmented_response
        Augmented response of the data. Shape (n, N, m + 1). nan indicates no
        conditioning.

    conditioning_sets
        Conditioning sets of the data. Shape (N, m)
    """

    locs: torch.Tensor
    response: torch.Tensor
    augmented_response: torch.Tensor
    conditioning_sets: torch.Tensor
    fidelity_sizes: torch.Tensor
    max_m: int

    @staticmethod
    def new(locs, response, conditioning_set, fidelity_sizes):
        """Creates a new data object."""
        nlocs = locs.shape[0]
        ecs = torch.hstack([torch.arange(nlocs).reshape(-1, 1), conditioning_set])
        augmented_response = torch.where(ecs == -1, torch.nan, response[:, ecs])
        max_m = int((augmented_response.shape[-1] - 1)/2)

        return MultiFidelityData(locs, response, augmented_response, 
                                 conditioning_set, fidelity_sizes, max_m)
    
class AugmentDataMF(torch.nn.Module):
    """Augments data

    Right now this just adds the scales to the data and creates a batch based on
    the provided index.

    Calculating the scales could be cached.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, data: MultiFidelityData, batch_idx: None | torch.Tensor = None
    ) -> AugmentedData:
        if batch_idx is None:
            batch_idx = torch.arange(data.response.shape[1])
        # batched_data = data[batch_idx]
        scales = scaling_mf(data.locs, data.conditioning_sets, data.fidelity_sizes)

        return AugmentedData(
            data_size=data.response.shape[1],
            batch_size=batch_idx.shape[0],
            batch_idx=batch_idx,
            locs=data.locs[batch_idx, :],
            augmented_response=data.augmented_response[:, batch_idx, :],
            scales=scales[batch_idx],
            data=data,
        )