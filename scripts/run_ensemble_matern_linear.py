import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Implementation of Matern covariance and negative log-likelihood,
# not using GPyTorch, since we want to fit smoothness nu as well.
# Using L-BFGS-B.
def matern_correlation(dists: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """
    M(r) = 2^{1-ν}/Γ(ν) (r/ρ)^ν K_ν(r/ρ), with M(0) = 1.
    """
    d = dists / rho
    corr = np.zeros_like(d, dtype=float)

    zero_mask = d == 0
    corr[zero_mask] = 1.0

    pos_mask = ~zero_mask
    if np.any(pos_mask):
        d_pos = d[pos_mask]
        factor = (2.0 ** (1.0 - nu)) / gamma(nu)
        corr[pos_mask] = factor * (d_pos**nu) * kv(nu, d_pos)

    return corr


def matern_covariance(dists: np.ndarray, log_params: np.ndarray) -> np.ndarray:
    phi = np.exp(log_params[0])
    rho = np.exp(log_params[1])
    nu = np.exp(log_params[2])

    corr = matern_correlation(dists, rho=rho, nu=nu)
    return phi * corr


def matern_nloglik(log_params: np.ndarray, dat: np.ndarray, dists: np.ndarray) -> float:
    cov_mat = matern_covariance(dists, log_params)
    n_dim = dat.shape[0]

    # dat.T has shape (n_samples, n_dim), as mvtnorm expects
    mvn = multivariate_normal(mean=np.zeros(n_dim), cov=cov_mat, allow_singular=True)
    logpdf_vals = mvn.logpdf(dat.T)
    return -np.mean(logpdf_vals)


def matern_param(
    locs: np.ndarray, data_train: np.ndarray, data_test: np.ndarray
) -> np.ndarray:
    dists = cdist(locs, locs)
    init = np.log(np.array([1.0, 0.1 * np.max(dists), 1.0]))

    def obj(p):
        return matern_nloglik(p, data_train, dists)

    res = minimize(obj, init, method="L-BFGS-B", options=dict(maxiter=100, ftol=1e-3))
    # Optional: you can evaluate the holdout score as in R:
    # ls = -matern_nloglik(res.x, data_test, dists)
    return res.x


def get_logscore(n, data_pkl):
    locs_mf = data_pkl["locs_mf"]
    locs_hf = data_pkl["locs_hf"]

    obs_lf = data_pkl["obs_lf"]
    obs_mf = data_pkl["obs_mf"]
    obs_hf = data_pkl["obs_hf"]

    # Averaging matrix
    list_of_cols = []
    list_of_coords = []
    for j in range(10):
        for i in range(10):
            col_now = np.zeros(900)
            idx_change = [
                90 * j + 3 * i,
                90 * j + 3 * i + 1,
                90 * j + 3 * i + 2,
                90 * j + 3 * i + 30,
                90 * j + 3 * i + 31,
                90 * j + 3 * i + 32,
                90 * j + 3 * i + 60,
                90 * j + 3 * i + 61,
                90 * j + 3 * i + 62,
            ]
            locs_changed = locs_hf[idx_change]
            x = torch.mean(locs_changed[:, 0])
            y = torch.mean(locs_changed[:, 1])
            locs_now = [x.item(), y.item()]
            col_now[idx_change] = 1 / 9
            list_of_cols.append(col_now)
            list_of_coords.append(locs_now)

    A = np.stack(list_of_cols, axis=0)

    # Second averaging matrix

    list_of_cols = []
    list_of_coords = []
    for j in range(5):
        for i in range(5):
            col_now = np.zeros(100)
            idx_change = [
                20 * j + 2 * i,
                20 * j + 2 * i + 1,
                20 * j + 2 * i + 10,
                20 * j + 2 * i + 11,
            ]
            locs_changed = locs_mf[idx_change]
            x = torch.mean(locs_changed[:, 0])
            y = torch.mean(locs_changed[:, 1])
            locs_now = [x.item(), y.item()]
            col_now[idx_change] = 1 / 4
            list_of_cols.append(col_now)
            list_of_coords.append(locs_now)

    A2 = np.stack(list_of_cols, axis=0)

    train_hf = obs_hf[:n, :]

    test_slice = slice(200, 250)

    test_hf = obs_hf[test_slice, :]
    test_mf = obs_mf[test_slice, :]
    test_lf = obs_lf[test_slice, :]

    par = matern_param(locs_hf, train_hf.T, test_hf.T)
    dists_hf = cdist(locs_hf, locs_hf)
    cov_hf = matern_covariance(dists_hf, par)

    left_out_hf = []
    for j in range(1, 11):
        for i in range(1, 11):
            idx = 90 * (j - 1) + 3 * (i - 1) + 1  # R 1-based
            left_out_hf.append(idx)
    left_out_hf = np.array(left_out_hf, dtype=int) - 1

    left_out_mf = []
    for j in range(1, 6):
        for i in range(1, 6):
            idx = 20 * (j - 1) + 2 * (i - 1) + 1  # R 1-based
            left_out_mf.append(idx)
    left_out_mf = np.array(left_out_mf, dtype=int) - 1  # 0-based

    cov_mf = A @ cov_hf @ A.T
    cov_lf = A2 @ cov_mf @ A2.T

    cov_hfmf = cov_hf @ A.T
    cov_hflf = cov_hf @ A.T @ A2.T
    cov_mflf = A @ cov_hf @ A.T @ A2.T

    # HF
    cov_hf_del = np.delete(np.delete(cov_hf, left_out_hf, axis=0), left_out_hf, axis=1)
    cov_hfmf_del = np.delete(
        np.delete(cov_hfmf, left_out_hf, axis=0), left_out_mf, axis=1
    )
    cov_hflf_del = np.delete(cov_hflf, left_out_hf, axis=0)

    # MF
    cov_mf_del = np.delete(np.delete(cov_mf, left_out_mf, axis=0), left_out_mf, axis=1)
    cov_mflf_del = np.delete(cov_mflf, left_out_mf, axis=0)

    # --------------------------
    # Full covariance (HF + MF + LF)
    # --------------------------
    row1 = np.hstack([cov_hf_del, cov_hfmf_del, cov_hflf_del])
    row2 = np.hstack([cov_hfmf_del.T, cov_mf_del, cov_mflf_del])
    row3 = np.hstack([cov_hflf_del.T, cov_mflf_del.T, cov_lf])

    cov_full = np.vstack([row1, row2, row3])

    test_hf_del = np.delete(test_hf, left_out_hf, axis=1)
    test_mf_del = np.delete(test_mf, left_out_mf, axis=1)
    test_full = np.hstack([test_hf_del, test_mf_del, test_lf])

    mvn_full = multivariate_normal(
        mean=np.zeros(cov_full.shape[0]), cov=cov_full, allow_singular=True
    )
    full_nll = -np.mean(mvn_full.logpdf(test_full))
    print("Full (HF + MF + LF) -mean loglik:", full_nll)
    return full_nll


# Load data
data_fp = Path("../tests/data/data_mf.pkl")
with open(data_fp, "rb") as fh:
    data_pkl: dict[str, np.ndarray] = pickle.load(fh)

ns = [5, 10, 20, 30, 50, 100, 200]
n_list = []
ls = []
for n in ns:
    print("With ensemble size")
    print(n)
    ls_ = get_logscore(n, data_pkl)
    n_list.append(n)
    ls.append(ls_)
    print("n")
    print(n)

    print("log score")
    print(ls_)

    my_dict = {"n": n_list, "logscore": ls}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv("./results/logscores_matern_linear.csv", index=False)
