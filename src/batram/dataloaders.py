from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils import data

from .base_functions import compute_scale

__all__ = [
    "MinibatchSample",
    "Dataset",
    "DataLoader",
]


# We don't care what the type of the index is for any purpose
# except indexing into variational parameters, so there is no need
# to deal with logic pinning the memory or otherwise.
def _pin(tensor: torch.Tensor, pin_memory: bool) -> torch.Tensor:
    """A helper function to easily pin memory during data collation.

    This function is used as part of a collate function when defining
    the behavior of a `DataLoader`. It is only necessary when we are
    using a GPU and want to pin memory for faster transfer to the GPU.
    """
    return tensor.pin_memory() if pin_memory else tensor


# Our data are highly structured, and we want to reference the items
# using a `DataSample` object instead of a tuple during training. The
# `collate_fn` ensures that we get a `DataSample` object back in each
# batch from this loader, while guaranteeing the minibatches are
# randomized during training of transport maps.
def collate_function(
    batch: Sequence["MinibatchSample"],
    pin_memory: bool = False,
) -> "MinibatchSample":
    """Collate a minibatch of data into a `MinibatchSample` object.

    This function is defined to work with `SimpleTM` models. It must be
    redefined for other models and passed into a DataLoader to ensure
    correct behavior.
    """

    idx = [b.idx for b in batch]
    response = torch.stack([_pin(b.response, pin_memory) for b in batch])
    augmented_response = torch.stack(
        [_pin(b.augmented_response, pin_memory) for b in batch]
    )
    scales = torch.stack([_pin(b.scales, pin_memory) for b in batch])

    if batch[0].x is not None:
        # This code path only works if b.x has covariates. mypy cannot infer
        # the type, so ignore it in this line
        x = torch.stack([_pin(b.x, pin_memory) for b in batch])  # type: ignore
    else:
        x = None

    return MinibatchSample(idx, response, augmented_response, scales, x)


@dataclass
class MinibatchSample:
    """Subsets `Dataset` into a minibatch for training a `TransportMap`.

    This container is used to facilitate minibatching during model training.
    It is also a convenient data abstraction for capturing a subset of the
    `Dataset` any time we are indexing into that parent class.
    """

    idx: Sequence
    response: torch.Tensor
    augmented_response: torch.Tensor
    scales: torch.Tensor
    x: Optional[torch.Tensor] = None

    def to(self, device, dtype):
        """Move the data to the specified device and dtype."""
        return MinibatchSample(
            self.idx,
            self.response.to(device, dtype),
            self.augmented_response.to(device, dtype),
            self.scales.to(device, dtype),
            self.x.to(device, dtype) if self.x else None,
        )


class Dataset(data.Dataset):
    """Data container for covariate-based transport maps.

    Holsd ..math:`n` replicate spatial fields observed at ..math:`N` locations.
    No assumptions about data normalization are made. The object can be used
    for minibatching via `torch.utils.data.DataLoader`s.

    Attributes:
    -----------
    locs: Spatial locations in d-dimensional space of shape ..math:`(N, d)`.

    x: Covariate matrix of shape ..math:`(N, n, p)`.

    y: Response matrix of shape ..math:`(N, n, 1)`.

    augy: Augmented response matrix of shape ..math:`(N, n, m)`.

    scales: Scale parameters for the spatial grid of shape ..math:`(N, 1, 1)`.

    max_m: The maximum number of conditional sets for any location in the grid.

    Notes:
    ------
    - See docs https://pytorch.org/docs/stable/data.html for more info on how
      to use these.

    - If we use a `DataLoader` then we need to determine how to get the actual
      batch indices back as part of the return object. It may be as simple as
      returning the `idx` in `__getitem__`, but requires checking.
    """

    def __init__(
        self,
        locs: torch.Tensor,
        response: torch.Tensor,
        condsets: torch.Tensor,
        x: None | torch.Tensor,
    ):
        """Initialize the data container. Reindexes data used in the model.

        Args:
        -----
        locs: Spatial locations in d-dimensional space of shape
              ..math:`(N, d)`.

        x: Covariate matrix of shape ..math:`(N, n, p)`.
            If these are not in shape (N, n, p) before passing them in, then
            they must be reshaped by the user. This can be done by
            `x.permute(1, 0, 2)` before construction or by referencing the
            attribute after initializing the object.

        y: Response matrix of shape ..math:`(n, N)`.

        condsets: Conditioning sets for each spatial location of shape
        ..math:`(N, m)`.
        """
        augmented_response = torch.where(
            condsets == -1, torch.nan, response[:, condsets]
        )
        scales = compute_scale(locs, condsets)

        self.locs = locs
        self.condsets = condsets
        self.response = response.unsqueeze(-1)
        self.augmented_response = augmented_response
        self.scales = scales.reshape(-1, 1, 1)
        self.x = x if x else None
        self.permute_dims()

    @property
    def max_m(self):
        return self.augmented_response.shape[-1]

    def permute_dims(self) -> None:
        self.response = self.response.permute(1, 0, 2)
        self.augmented_response = self.augmented_response.permute(1, 0, 2)

    def __len__(self):
        return self.locs.shape[0]

    def __getitem__(self, idx) -> MinibatchSample:
        return MinibatchSample(
            # We require the index to pass into the variational parameters.
            idx,
            self.response[idx],
            self.augmented_response[idx],
            self.scales[idx],
            self.x[idx] if self.x else None,
        )

    def __repr__(self) -> str:  # pragma: no cover
        if self.x:
            x_repr = self.x.shape
        else:
            x_repr = "None"
        return (
            f"CovariatesData(\n"
            f"  locs = {self.locs.shape},\n"
            f"  condsets = {self.condsets.shape},\n"
            f"  response = {self.response.shape},\n"
            f"  augmented_response = {self.augmented_response.shape},\n"
            f"  scales = {self.scales.shape},\n"
            f"  x = {x_repr},\n"
            f"  max_m = {self.max_m} \n"
            f")"
        )


class DataLoader(data.DataLoader):
    """A customized data loader for minibatch training of `CovariateTM` models.

    This `DataLoader` is a specialized version of the default PyTorch
    `DataLoader`which assumes training with randomized minibatches. The user
    may tune the batch size and number of workers for parallelization. A
    generator may also be passed in to ensure reproducibility.

    Attributes:
    -----------
    dataset: The `CovariatesData` object to be used for training.

    batch_size: The number of samples in each minibatch.

    num_workers: The number of workers to use for parallelization.

    generator: A random number generator to use for reproducibility.

    sampler: A `Sampler` object to use for sampling minibatches.

    batch_sampler: A `BatchSampler` object to use for sampling minibatches.

    collate_fn: A function to use for collating minibatches.
        This function enforces returning a `DataSample` object in each batch.
        This behavior must be preserved under any training regime.

    Notes:
    ------
     - If modifying this class, the `collate_fn` defined here should be reused
       to ensure correct behavior in the `fit` method.
     - See https://pytorch.org/docs/stable/data.html# for more details on how
       to extend the `DataLoader` class.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        num_workers: int = 0,
        collate_fn=collate_function,
        generator=None,
        pin_memory: bool = False,
    ):
        sampler = data.RandomSampler(dataset, generator=generator)
        if batch_size is None:
            batch_size = len(dataset)
        batch_sampler = data.BatchSampler(sampler, batch_size, drop_last=False)

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            generator=generator,
        )
