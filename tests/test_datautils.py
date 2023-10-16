import numpy as np
import pytest
import torch
from veccs import orderings

from batram.datautils import DataLoader, Dataset, MinibatchSample


@pytest.fixture
def raw_data() -> dict:
    """Return a Dataset object."""
    locs = np.random.normal(size=(100, 2))
    ordering = orderings.maxmin_cpp(locs)
    ordered_locs = locs[ordering]
    condsets = orderings.find_nns_l2(ordered_locs, 10)

    ordered_locs = torch.from_numpy(locs).float()
    condsets = torch.from_numpy(condsets).long()
    return {
        "ordered_locs": ordered_locs,
        "condsets": condsets,
        "response": torch.randn(10, 100),
    }


@pytest.fixture
def data(raw_data: dict) -> Dataset:
    return Dataset(
        locs=raw_data["ordered_locs"],
        response=raw_data["response"],
        condsets=raw_data["condsets"],
    )


def test_datautils_dataset_has_attributes(data: Dataset) -> None:
    assert hasattr(data, "locs")
    assert hasattr(data, "response")
    assert hasattr(data, "condsets")
    assert hasattr(data, "scales")
    assert hasattr(data, "augmented_response")
    assert hasattr(data, "max_m")
    assert hasattr(data, "x")


def test_datautils_dataset_x_none(data: Dataset) -> None:
    assert data.x is None


def test_datautils_dataset_x_not_none(raw_data: dict) -> None:
    x = torch.randn(100, 1, 1)
    new_data = Dataset(
        locs=raw_data["ordered_locs"],
        response=raw_data["response"],
        condsets=raw_data["condsets"],
        x=x,
    )
    assert (new_data.x == x).all()


def test_datautils_dataset_subsets(data: Dataset) -> None:
    mb = data[:10]
    assert isinstance(mb, MinibatchSample)


def test_datautils_minibatch_sample_to(data: Dataset) -> None:
    mb = data[:10]
    mb = mb.to(device="cpu", dtype=torch.float32)
    attrs = ["response", "augmented_response", "scales"]
    for attr in attrs:
        assert getattr(mb, attr).device.type == "cpu"
        assert getattr(mb, attr).dtype == torch.float32


def test_datautils_dataloader_batching(data: Dataset) -> None:
    data_loader = DataLoader(data, batch_size=10)
    for mb in data_loader:
        assert isinstance(mb, MinibatchSample)
        assert mb.response.shape[0] == 10
