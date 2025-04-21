from itertools import product

import numpy as np
import scipy.spatial.distance
import torch
from veccs.orderings import maxmin_cpp
from veccs.orderings2 import find_prev_nearest_neighbors

from .data import Data, MultVarData
from .utils import calc_u_d_b, cov_exponential


def sim_LR(
    gen: np.random.Generator,
    num_locs_dim: int = 30,
    num_reps: int = 5,
    max_num_neibours: int = 30,
    sd_noise: float = 1.0,
    lrange: float = 0.3,
    non_linear: bool = True,
) -> Data:
    """
    Simulates similar to NR from the TransportMap paper.
    """

    generator = gen
    s1 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    s2 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    locs = np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2)

    ord = maxmin_cpp(locs)

    locs = locs[ord, :]
    nn = find_nns_l2(locs, max_nn=max_num_neibours)

    dists = scipy.spatial.distance.cdist(locs, locs, metric="euclidean")
    cov_true = cov_exponential(dists, 1, lrange)
    _, d, b = calc_u_d_b(np.linalg.inv(cov_true))

    weights = np.zeros((locs.shape[0], max_num_neibours))
    for i in range(1, locs.shape[0]):
        weights[i, :] = b[nn[i, :], i]

    x_all = np.zeros((num_reps, locs.shape[0]), dtype=np.float32)
    for rep in range(num_reps):
        x = np.zeros(locs.shape[0])
        for i in range(len(x)):
            nnvals = np.where(nn[i, :] >= 0, x[nn[i, :]], 0.0)
            lins = nnvals * weights[i, :]
            if non_linear:
                nonlin = 2 * np.sin(4 * lins[:2].sum())
            else:
                nonlin = 0.0
            fx = lins.sum() + nonlin
            eps = generator.normal(scale=sd_noise * np.sqrt(d[i]))
            x[i] = fx + eps
        x_all[rep, :] = x

    data = Data.new(
        locs=locs,
        response=x_all,
        conditioning_sets=nn,
        order=ord,
    )

    return data


def sim_LR_mv(
    gen: np.random.Generator,
    process_locations: np.ndarray,
    num_locs_dim: int = 30,
    num_reps: int = 5,
    max_num_neibours: int = 30,
    sd_noise: float = 1.0,
    lrange: float = 0.3,
    non_linear: bool = True,
) -> MultVarData:
    """
    Simulates a MV sptial field similar similar `sim_LR` but in an augmented
    space.
    """

    generator = gen

    # create spatial locations on a regular grid
    s1 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    s2 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    locs_sp = np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2)

    nproc = process_locations.shape[0]
    nlocs = locs_sp.shape[0]

    # create augmented locations
    locs_augm = np.stack(
        [np.concatenate((pl, loc)) for pl, loc in product(process_locations, locs_sp)],
        0,
    )

    # create ordering
    ord = maxmin_cpp(locs_augm)

    # order the data
    locs_augm = locs_augm[ord, :]
    process_ids = np.repeat(np.arange(nproc), nlocs)[ord]
    location_ids = np.tile(np.arange(nlocs), nproc)[ord]

    # determine conditioning sets
    nn = find_prev_nearest_neighbors(locs_augm, np.arange(locs_augm.shape[0]), max_nn=max_num_neibours)

    # calculate respnses
    dists = scipy.spatial.distance.cdist(locs_augm, locs_augm, metric="euclidean")
    cov_true = cov_exponential(dists, 1, lrange)
    _, d, b = calc_u_d_b(np.linalg.inv(cov_true))

    weights = np.zeros((locs_augm.shape[0], max_num_neibours))
    # change weights according to process distance (might not work.)
    for i in range(1, locs_augm.shape[0]):
        weights[i, :] = b[nn[i, :], i]

    x_all = np.zeros((num_reps, locs_augm.shape[0]), dtype=np.float32)
    for rep in range(num_reps):
        x = np.zeros(locs_augm.shape[0])
        for i in range(len(x)):
            nnvals = np.where(nn[i, :] >= 0, x[nn[i, :]], 0.0)
            lins = nnvals * weights[i, :]
            if non_linear:
                nonlin = 2 * np.sin(4 * lins[:2].sum())
                # = g(spatial distance) + h(inter process distance)
            else:
                nonlin = 0.0
            fx = lins.sum() + nonlin
            eps = generator.normal(scale=sd_noise * np.sqrt(d[i]))
            x[i] = fx + eps
        x_all[rep, :] = x

    data_simple = Data.new(
        locs=locs_augm, response=x_all, conditioning_sets=nn, order=ord
    )

    data = MultVarData(
        response=torch.as_tensor(x_all),
        locs_sp=torch.as_tensor(locs_sp),
        locs_proc=torch.as_tensor(process_locations, dtype=torch.float32),
        process_ids=torch.as_tensor(process_ids),
        location_ids=torch.as_tensor(location_ids),
        conditioning_sets=torch.as_tensor(nn),
        ordering=ord,
        response_augmented=data_simple.augmented_response,
        order_last_var_last=False,
    )

    return data


def sim_LR_mv2(
    gen: np.random.Generator,
    process_locations: np.ndarray,
    num_locs_dim: int = 30,
    num_reps: int = 5,
    max_num_neibours: int = 30,
    sd_noise: float = 1.0,
    lrange: float = 0.3,
    non_linear: bool = True,
) -> MultVarData:
    """
    Simulates a MV sptial field similar similar `sim_LR` but in an augmented
    space.
    """

    generator = gen

    pmat = gen.normal(1, 0.3, (5, 5))
    pmat = np.diag(1 / pmat.diagonal()) @ pmat
    pmat = 4 * (pmat + pmat.T) / 2
    pmat

    # create spatial locations on a regular grid
    s1 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    s2 = np.linspace(0, 1, num_locs_dim, dtype=np.float32)
    locs_sp = np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2)

    nproc = process_locations.shape[0]
    nlocs = locs_sp.shape[0]

    # create augmented locations
    locs_augm = np.stack(
        [np.concatenate((pl, loc)) for pl, loc in product(process_locations, locs_sp)],
        0,
    )

    # create ordering
    ord = maxmin_cpp(locs_augm)

    # order the data
    locs_augm = locs_augm[ord, :]
    process_ids = np.repeat(np.arange(nproc), nlocs)[ord]
    location_ids = np.tile(np.arange(nlocs), nproc)[ord]

    # determine conditioning sets
    nn = find_nns_l2(locs_augm, max_nn=max_num_neibours)

    # calculate respnses
    dists = scipy.spatial.distance.cdist(locs_augm, locs_augm, metric="euclidean")
    cov_true = cov_exponential(dists, 1, lrange)
    _, d, b = calc_u_d_b(np.linalg.inv(cov_true))

    weights = np.zeros((locs_augm.shape[0], max_num_neibours))
    # change weights according to process distance (might not work.)
    for i in range(1, locs_augm.shape[0]):
        weights[i, :] = b[nn[i, :], i]

    x_all = np.zeros((num_reps, locs_augm.shape[0]), dtype=np.float32)
    nl_all = np.zeros((num_reps, locs_augm.shape[0]), dtype=np.float32)
    for rep in range(num_reps):
        x = np.zeros(locs_augm.shape[0])
        for i in range(len(x)):
            nnvals = np.where(nn[i, :] >= 0, x[nn[i, :]], 0.0)
            lins = nnvals * weights[i, :]
            if non_linear:
                l0, l1 = lins[0], lins[0]
                pi = process_ids[i]
                p0, p1 = process_ids[nn[i, 0]], process_ids[nn[i, 1]]
                # pmat = pmat * 0 + 4
                nonlin = 2 * np.sin(l0 * pmat[pi, p0] + l1 * pmat[pi, p1])
                # nonlin = 2 * np.sin(4 * l0 * pmat[pi, p0] + 4 * l1 * pmat[pi, p1])# * np.sqrt(d[i])
                # nonlin = l0 + l1
                # nonlin = 2 * lins[:2].sum()
                # = g(spatial distance) + h(inter process distance)
            else:
                nonlin = 0.0
            nl_all[rep, i] = nonlin
            fx = lins.sum() + nonlin
            eps = generator.normal(scale=sd_noise * np.sqrt(d[i]))
            x[i] = fx + eps
        x_all[rep, :] = x

    data_simple = Data.new(
        locs=locs_augm, response=x_all, conditioning_sets=nn, order=ord
    )

    data = (
        MultVarData(
            response=torch.as_tensor(x_all),
            locs_sp=torch.as_tensor(locs_sp),
            locs_proc=torch.as_tensor(process_locations, dtype=torch.float32),
            process_ids=torch.as_tensor(process_ids),
            location_ids=torch.as_tensor(location_ids),
            conditioning_sets=torch.as_tensor(nn),
            ordering=ord,
            response_augmented=data_simple.augmented_response,
            order_last_var_last=False,
        ),
        nl_all,
        d,
    )

    return data
