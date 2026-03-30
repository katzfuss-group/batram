from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Matérn baseline experiments.")
    parser.add_argument("--experiment", choices=["linear", "min"], required=True)
    parser.add_argument("--include-all-logscores", action="store_true")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--ns", default=None, help="Comma-separated list, e.g. 5,10,20")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def default_ns(experiment: str) -> list[int]:
    return [5, 10, 20, 30, 50, 100, 200]


def parse_ns(ns_arg: str | None, experiment: str) -> list[int]:
    if ns_arg is None:
        return default_ns(experiment)
    return [int(x.strip()) for x in ns_arg.split(",") if x.strip()]


def save_rows(
    rows: list[dict],
    output_dir: str | Path,
    model_name: str,
    experiment: str,
    include_all: bool,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_allfields" if include_all else ""
    out_fp = out_dir / f"logscores_{model_name}_{experiment}{suffix}.csv"
    df = pd.DataFrame(rows)
    if include_all:
        df.to_csv(out_fp, index=False)
    else:
        value_cols = [c for c in df.columns if c not in {"n", "test_idx"}]
        df.groupby("n", as_index=False)[value_cols].mean().to_csv(out_fp, index=False)
    return out_fp


# Implementation of Matérn covariance and negative log-likelihood,
# not using GPyTorch, since we want to fit smoothness nu as well.
def matern_correlation(dists: np.ndarray, rho: float, nu: float) -> np.ndarray:
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
    mvn = multivariate_normal(mean=np.zeros(n_dim), cov=cov_mat, allow_singular=True)
    logpdf_vals = mvn.logpdf(dat.T)
    return -np.mean(logpdf_vals)


def matern_param(locs: np.ndarray, data_train: np.ndarray) -> np.ndarray:
    dists = cdist(locs, locs)
    init = np.log(np.array([1.0, 0.1 * np.max(dists), 1.0]))
    res = minimize(
        lambda p: matern_nloglik(p, data_train, dists),
        init,
        method="L-BFGS-B",
        options=dict(maxiter=100, ftol=1e-3),
    )
    return res.x


def get_scores_multifidelity(n: int, locs_pkl: dict, obs_pkl: dict) -> np.ndarray:
    locs_hf = locs_pkl["locs_hf"]

    obs_lf = obs_pkl["obs_lf"]
    obs_mf = obs_pkl["obs_mf"]
    obs_hf = obs_pkl["obs_hf"]

    list_of_cols = []
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
            col_now[idx_change] = 1 / 9
            list_of_cols.append(col_now)
    A = np.stack(list_of_cols, axis=0)

    list_of_cols = []
    for j in range(5):
        for i in range(5):
            col_now = np.zeros(100)
            idx_change = [
                20 * j + 2 * i,
                20 * j + 2 * i + 1,
                20 * j + 2 * i + 10,
                20 * j + 2 * i + 11,
            ]
            col_now[idx_change] = 1 / 4
            list_of_cols.append(col_now)
    A2 = np.stack(list_of_cols, axis=0)

    train_hf = obs_hf[:n, :]
    test_slice = slice(200, 250)
    test_hf = obs_hf[test_slice, :]
    test_mf = obs_mf[test_slice, :]
    test_lf = obs_lf[test_slice, :]

    par = matern_param(locs_hf, train_hf.T)
    dists_hf = cdist(locs_hf, locs_hf)
    cov_hf = matern_covariance(dists_hf, par)

    left_out_hf = (
        np.array(
            [
                90 * (j - 1) + 3 * (i - 1) + 1
                for j in range(1, 11)
                for i in range(1, 11)
            ],
            dtype=int,
        )
        - 1
    )
    left_out_mf = (
        np.array(
            [20 * (j - 1) + 2 * (i - 1) + 1 for j in range(1, 6) for i in range(1, 6)],
            dtype=int,
        )
        - 1
    )

    cov_mf = A @ cov_hf @ A.T
    cov_lf = A2 @ cov_mf @ A2.T
    cov_hfmf = cov_hf @ A.T
    cov_hflf = cov_hf @ A.T @ A2.T
    cov_mflf = A @ cov_hf @ A.T @ A2.T

    cov_hf_del = np.delete(np.delete(cov_hf, left_out_hf, axis=0), left_out_hf, axis=1)
    cov_hfmf_del = np.delete(
        np.delete(cov_hfmf, left_out_hf, axis=0), left_out_mf, axis=1
    )
    cov_hflf_del = np.delete(cov_hflf, left_out_hf, axis=0)
    cov_mf_del = np.delete(np.delete(cov_mf, left_out_mf, axis=0), left_out_mf, axis=1)
    cov_mflf_del = np.delete(cov_mflf, left_out_mf, axis=0)

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
    return -np.atleast_1d(mvn_full.logpdf(test_full)).astype(float)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    locs_fp = Path("../tests/data/data_mf.pkl")
    with open(locs_fp, "rb") as fh:
        locs_pkl = pickle.load(fh)

    obs_fp = (
        Path("../tests/data/data_mf.pkl")
        if args.experiment == "linear"
        else Path("../tests/data/data_mf_min.pkl")
    )
    with open(obs_fp, "rb") as fh:
        obs_pkl = pickle.load(fh)

    rows: list[dict] = []
    for n in parse_ns(args.ns, args.experiment):
        print("With ensemble size")
        print(n)
        scores = get_scores_multifidelity(n, locs_pkl, obs_pkl)
        for i, score in enumerate(scores):
            rows.append({"n": n, "test_idx": i, "logscore_total": float(score)})
        print("mean log score")
        print(float(np.mean(scores)))
        save_rows(
            rows, args.output_dir, "matern", args.experiment, args.include_all_logscores
        )


if __name__ == "__main__":
    main()
