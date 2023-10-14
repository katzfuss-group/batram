import pathlib
import pickle

import numpy as np
import pytest
import torch
import veccs.orderings

import batram.legmods as legmods
from batram.legacy import fit_map
from batram.legmods import (
    AugmentData,
    Data,
    Nugget,
    OldTransportMapKernel,
    SimpleTM,
    TransportMapKernel,
)


@pytest.fixture
def simple_data() -> Data:
    test_folder = pathlib.Path(__file__).parent
    data_fp = test_folder.joinpath("data/simple_data.pkl")
    with open(data_fp, "rb") as fh:
        pickled_data: dict[str, np.ndarray] = pickle.load(fh)

    # data is ordered
    locs = pickled_data["locations"]
    obs = torch.as_tensor(pickled_data["observations"])

    max_size_cs = 30
    nn = veccs.orderings.find_nns_l2(locs, max_size_cs)

    data = Data.new(torch.as_tensor(locs), obs, torch.as_tensor(nn))
    return data


def test_create_data() -> None:
    response = torch.linspace(0, 0.5, 6).reshape(2, 3)  # 2 replicates, 3 locations
    locations = torch.linspace(0, 1, 6).reshape(3, 2)
    cond_set = torch.tensor([[-1, -1], [0, -1], [1, 0]])

    data = Data.new(locations, response, cond_set)

    augment_response = torch.tensor(
        [
            [
                [0.0000, torch.nan, torch.nan],
                [0.1000, 0.0000, torch.nan],
                [0.2000, 0.1000, 0.0000],
            ],
            [
                [0.3000, torch.nan, torch.nan],
                [0.4000, 0.3000, torch.nan],
                [0.5000, 0.4000, 0.3000],
            ],
        ]
    )

    assert torch.allclose(data.augmented_response, augment_response, equal_nan=True)


def test_legmods_intlik_simple_data(simple_data: Data) -> None:
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    tm = SimpleTM(simple_data, theta_init, False, smooth=1.5, nug_mult=4.0)

    with torch.no_grad():
        intlik: float = float(tm(None))
    assert intlik == pytest.approx(-128.11582946777344, rel=1e-3)


def test_legmods__intlik_mini_batch_simple_data(simple_data: Data) -> None:
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    tm = SimpleTM(simple_data, theta_init, False, smooth=1.5, nug_mult=4.0)

    with torch.no_grad():
        idx = torch.arange(simple_data.response.shape[1]).flip(0)
        intlik: float = float(tm(idx))
    assert intlik == pytest.approx(-128.11582946777344, rel=1e-3)


def test_legmods_cond_samp_bayes(simple_data: Data) -> None:
    torch.manual_seed(0)
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    tm = SimpleTM(simple_data, theta_init, False, smooth=1.5, nug_mult=4.0)

    with torch.no_grad():
        sample = tm.cond_sample()
        sample_abs_sum = sample.abs().sum()

    # we do not expect this to hold because of the RNG involved
    # however, it should be the same for the legacy method and the legmods
    assert sample_abs_sum == pytest.approx(65.3635, abs=1e-2)


def test_legmods_cond_samp_multi_samples(simple_data: Data) -> None:
    tm = SimpleTM(simple_data)
    with torch.no_grad():
        sample = tm.cond_sample(num_samples=10)
    assert sample.shape == (10, simple_data.response.shape[1])


def test_legmods_score(simple_data: Data) -> None:
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    tm = SimpleTM(simple_data, theta_init, False, smooth=1.5, nug_mult=4.0)
    with torch.no_grad():
        score = tm.score(simple_data.response[0, :])

    assert score == pytest.approx(-49.6006, abs=1e-3)


def test_legmods_nugget_mean(simple_data: Data) -> None:
    augdata = AugmentData()(simple_data)
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    nugget = legmods.Nugget(theta_init[:2])

    n_locs = augdata.scales.numel()
    legacy_nugget = fit_map.nug_fun(torch.arange(n_locs), theta_init, augdata.scales)
    new_nugget = nugget(augdata).squeeze()
    assert torch.allclose(new_nugget, legacy_nugget)


def test_kernel_equiv(simple_data: Data) -> None:
    augdata = legmods.AugmentData()(simple_data, None)

    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    theta = legmods.ParameterBox(theta_init)
    nugget = Nugget(theta_init[:2])
    kernel = OldTransportMapKernel(theta, fix_m=simple_data.conditioning_sets.shape[1])

    with torch.no_grad():
        # Currently equiv to calling the following:
        #   new = kernel(augdata, new_method = True)
        #   old = kernel(augdata, new_method = False)
        # This implementation runs ~32x faster than the legacy method
        nugget_mean = nugget(augdata)
        new = kernel.new_forward(augdata, nugget_mean)
        old = kernel.old_forward(augdata, nugget_mean)

    assert torch.allclose(new.G, old.G, rtol=1e-2)
    assert torch.allclose(new.GChol, old.GChol, rtol=1e-1)
    assert torch.allclose(new.nug_mean.squeeze(), old.nug_mean.squeeze())


def test_refactored_kernel(simple_data: Data) -> None:
    augdata = legmods.AugmentData()(simple_data, None)
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    nugget = legmods.Nugget(theta_init[:2])
    theta = legmods.ParameterBox(theta_init)
    kernel = OldTransportMapKernel(theta, fix_m=augdata.max_m)
    refactored_kernel = TransportMapKernel(theta_init[2:], fix_m=augdata.max_m)

    with torch.no_grad():
        nugget_mean = nugget(augdata)
        ker = kernel(augdata, nugget_mean)
        refac = refactored_kernel.forward(augdata, nugget_mean)

    assert torch.allclose(ker.G, refac.G, rtol=1e-2)
    assert torch.allclose(ker.GChol, refac.GChol, rtol=1e-1)


def test_optim_simple(simple_data: Data) -> None:
    # sanity check that the minibatch and non-minibatch versions
    # yield similar results

    torch.manual_seed(0)

    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    tm = SimpleTM(simple_data, theta_init.clone(), False, smooth=1.5, nug_mult=4.0)
    res = tm.fit(100, 0.3, test_data=tm.data)

    tm = SimpleTM(simple_data, theta_init.clone(), False, smooth=1.5, nug_mult=4.0)
    res2 = tm.fit(100, 0.3, batch_size=50, test_data=tm.data)
    assert res2.losses[-1] == pytest.approx(res.losses[-1], abs=1e-1)


def test_optim_no_theta_init(simple_data: Data) -> None:
    torch.manual_seed(0)

    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.3, 0.0, 0.0, 0.1, -1.0]
    )

    no_theta = SimpleTM(simple_data)
    use_theta = SimpleTM(simple_data, theta_init.clone())

    no_theta_res = no_theta.fit(100, 0.3, test_data=no_theta.data)
    use_theta_res = use_theta.fit(100, 0.3, test_data=use_theta.data)

    assert no_theta_res.losses[-1] == pytest.approx(use_theta_res.losses[-1], abs=1e-1)


def test_init(simple_data) -> None:
    theta_init = torch.tensor(
        [simple_data.response[:, 0].square().mean().log(), 0.2, 0.0, 0.0, 0.0, -1.0]
    )

    tm_no_theta = SimpleTM(simple_data)
    no_theta_params = torch.nn.ParameterList(tm_no_theta.parameters())
    tm_theta = SimpleTM(simple_data, theta_init)
    theta_params = torch.nn.ParameterList(tm_theta.parameters())

    for ntp, tp in zip(no_theta_params, theta_params):
        assert torch.allclose(ntp, tp)


def test_linear_tm_raises(simple_data) -> None:
    with pytest.raises(ValueError):
        SimpleTM(simple_data, linear=True)


def test_optim_simNR900() -> None:
    # sanity check if performs silimar to the legacy code
    # the optimizer is not yet converged, but the loss is similar
    # should be replaced with a better test

    test_folder = pathlib.Path(__file__).parent
    data_fp = test_folder.joinpath("data/simNR900.pkl")
    with open(data_fp, "rb") as fh:
        data: dict[str, np.ndarray] = pickle.load(fh)

    locs = data["locations"]
    obs = torch.as_tensor(data["observations"])

    # use only a subset of the available ensemble
    obs = obs[:10, :]

    # order data and determine conditioning sets
    ord = veccs.orderings.maxmin_cpp(locs)
    locs = locs[ord, :]
    obs = obs[:, ord]

    max_size_cs = 30
    nn = veccs.orderings.find_nns_l2(locs, max_nn=max_size_cs)

    tmdata = Data.new(torch.as_tensor(locs), obs, torch.as_tensor(nn))
    theta_init = torch.tensor(
        [tmdata.response[:, 0].square().mean().log(), 0.2, 0.0, 0.0, 0.0, -1.0]
    )

    tm = SimpleTM(tmdata, theta_init, False, smooth=1.5, nug_mult=4.0)
    opt = torch.optim.Adam(tm.parameters(), lr=0.01)

    # catch the warning that the conditioning set is not large enough
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tm.fit(100, 1.0, test_data=tm.data, optimizer=opt, silent=True)
    assert res.test_losses[-1] == pytest.approx(10055.03, rel=3e-3)  # type: ignore
