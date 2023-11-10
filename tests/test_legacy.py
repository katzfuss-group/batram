import pathlib
import pickle

import numpy as np
import pytest
import torch
import veccs.orderings

import batram.legacy.fit_map as legacy


@pytest.mark.skip("test is numerically unstable")
def test_legacy_intloglik() -> None:
    # this test fails on github.
    # See https://github.com/wiep/batrama/issues/2

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
    nn = veccs.orderings.find_nns_l2(locs)
    scal = legacy.compute_scal(locs, nn)

    tm = legacy.TransportMap(
        thetaInit=torch.tensor(
            [obs[:, 0].square().mean().log(), 0.2, -1.0, 0.0, 0.0, -1.0]
        ),
        linear=False,
        tuneParm=None,
    )
    intlik: float = float(
        tm.forward(
            obs,
            NNmax=nn,
            mode="intlik",
            m=torch.as_tensor(max_size_cs),
            inds=None,
            scal=scal,
        ).detach()
    )
    assert intlik == pytest.approx(10308.78125)


def test_legacy_simple_data() -> None:
    test_folder = pathlib.Path(__file__).parent
    data_fp = test_folder.joinpath("data/simple_data.pkl")
    with open(data_fp, "rb") as fh:
        data: dict[str, np.ndarray] = pickle.load(fh)

    # data is ordered
    locs = data["locations"]
    obs = torch.as_tensor(data["observations"])

    max_size_cs = 30
    nn = veccs.orderings.find_nns_l2(locs, max_size_cs)
    scal = legacy.compute_scal(locs, nn)

    tm = legacy.TransportMap(
        thetaInit=torch.tensor(
            [obs[:, 0].square().mean().log(), 0.3, -1.0, 0.0, 0.1, -1.0]
        ),
        linear=False,
        tuneParm=None,
    )
    with torch.no_grad():
        intlik: float = float(
            tm.forward(
                obs,
                NNmax=nn,
                mode="intlik",
                m=torch.as_tensor(max_size_cs),
                inds=None,
                scal=scal,
            )
        )
    assert intlik == pytest.approx(-128.11582946777344)


def test_legacy_cond_samp_bayes() -> None:
    torch.manual_seed(0)
    test_folder = pathlib.Path(__file__).parent
    data_fp = test_folder.joinpath("data/simple_data.pkl")
    with open(data_fp, "rb") as fh:
        data: dict[str, np.ndarray] = pickle.load(fh)

    # data is ordered
    locs = data["locations"]
    obs = torch.as_tensor(data["observations"])

    max_size_cs = 30
    nn = veccs.orderings.find_nns_l2(locs, max_size_cs)
    scal = legacy.compute_scal(locs, nn)

    tm = legacy.TransportMap(
        thetaInit=torch.tensor(
            [obs[:, 0].square().mean().log(), 0.3, -1.0, 0.0, 0.1, -1.0]
        ),
        linear=False,
        tuneParm=None,
    )
    with torch.no_grad():
        fit = tm.forward(
            obs,
            NNmax=nn,
            mode="fit",
            m=torch.as_tensor(max_size_cs),
            inds=None,
            scal=scal,
        )

        sample = legacy.cond_samp(fit, "bayes")
        sample_abs_sum = sample.abs().sum()

    # we do not expect this to hold because of the RNG involved
    # however, it should be the same for the legacy method and the legmods
    assert sample_abs_sum == pytest.approx(65.3635)


def test_legacy_score() -> None:
    torch.manual_seed(0)
    test_folder = pathlib.Path(__file__).parent
    data_fp = test_folder.joinpath("data/simple_data.pkl")
    with open(data_fp, "rb") as fh:
        data: dict[str, np.ndarray] = pickle.load(fh)

    # data is ordered
    locs = data["locations"]
    obs = torch.as_tensor(data["observations"])

    max_size_cs = 30
    nn = veccs.orderings.find_nns_l2(locs, max_size_cs)
    scal = legacy.compute_scal(locs, nn)

    tm = legacy.TransportMap(
        thetaInit=torch.tensor(
            [obs[:, 0].square().mean().log(), 0.3, -1.0, 0.0, 0.1, -1.0]
        ),
        linear=False,
        tuneParm=None,
    )
    with torch.no_grad():
        fit = tm.forward(
            obs,
            NNmax=nn,
            mode="fit",
            m=torch.as_tensor(max_size_cs),
            inds=None,
            scal=scal,
        )

        score = legacy.cond_samp(fit, "score", obs[0, :])

    # we do not expect this to hold because of the RNG involved
    # however, it should be the same for the legacy method and the legmods
    assert score == pytest.approx(-49.6006)


@pytest.mark.skip()
def test_legacy_fitmap() -> None:
    # same sanity check as for the legmod code
    # test_optim_simNR900

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
    scal = legacy.compute_scal(locs, nn)

    res, loss = legacy.fit_map_mini(
        obs, nn, scal, maxEpoch=100, lr=1e-2, batsz=obs.shape[1], track_loss=True
    )
    print(res["theta"])
    print(loss[-1])
    assert loss[-1].item() == pytest.approx(10055.03, abs=1e-2)
