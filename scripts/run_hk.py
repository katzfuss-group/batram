from __future__ import annotations

import argparse
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import GridInterpolationKernel, MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HK baseline experiments.")
    parser.add_argument(
        "--experiment", choices=["linear", "min", "climate"], required=True
    )
    parser.add_argument("--include-all-logscores", action="store_true")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--ns", default=None, help="Comma-separated list, e.g. 5,10,20")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def default_ns(experiment: str) -> list[int]:
    return (
        [5, 10, 15, 20, 25, 30, 35, 40]
        if experiment == "climate"
        else [5, 10, 20, 30, 50, 100, 200]
    )


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


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def chol_with_jitter(
    K: torch.Tensor, jitter: float = 1e-6, max_tries: int = 6
) -> torch.Tensor:
    assert K.ndim == 2 and K.shape[0] == K.shape[1]
    eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    j = jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + j * eye)
        except RuntimeError:
            j *= 10.0
    return torch.linalg.cholesky(K + j * eye)


def matern52_r(r: torch.Tensor) -> torch.Tensor:
    sqrt5 = math.sqrt(5.0)
    return (1.0 + sqrt5 * r + (5.0 / 3.0) * (r**2)) * torch.exp(-sqrt5 * r)


def matern52_kernel(
    X: torch.Tensor, Z: torch.Tensor, ell: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    diff = (X[:, None, :] - Z[None, :, :]) / ell[None, None, :]
    r = torch.sqrt((diff**2).sum(dim=-1).clamp_min(1e-12))
    return sigma2 * matern52_r(r)


def fitc_log_prob(
    Y: torch.Tensor,
    X: torch.Tensor,
    Z: torch.Tensor,
    ell: torch.Tensor,
    sigma2: torch.Tensor,
    noise2: torch.Tensor,
    diag_floor: float = 1e-4,
    jitter: float = 1e-6,
) -> torch.Tensor:
    E, N = Y.shape
    Kuu = matern52_kernel(Z, Z, ell, sigma2)
    Luu = chol_with_jitter(Kuu, jitter=jitter)
    Kxu = matern52_kernel(X, Z, ell, sigma2)
    Kux = Kxu.transpose(0, 1).contiguous()
    A = torch.linalg.solve_triangular(Luu, Kux, upper=False)
    diagQ = (A**2).sum(dim=0)
    diagK = sigma2.expand_as(diagQ)
    D = (diagK - diagQ + noise2).clamp_min(diag_floor)
    Dinv = 1.0 / D
    Ad = A * Dinv.unsqueeze(0)
    Mmat = torch.eye(Z.shape[0], device=Y.device, dtype=Y.dtype) + Ad @ A.transpose(
        0, 1
    )
    LM = chol_with_jitter(Mmat, jitter=jitter)
    logdet = torch.log(D).sum() + 2.0 * torch.log(torch.diagonal(LM)).sum()
    YDinv = Y * Dinv.unsqueeze(0)
    base_quad = (Y * YDinv).sum(dim=1)
    V = YDinv @ A.transpose(0, 1)
    VT = V.transpose(0, 1).contiguous()
    t = torch.linalg.solve_triangular(LM, VT, upper=False)
    W = torch.linalg.solve_triangular(LM.transpose(0, 1), t, upper=True)
    corr = (V * W.transpose(0, 1)).sum(dim=1)
    invquad = base_quad - corr
    const = N * math.log(2.0 * math.pi)
    return -0.5 * (invquad + logdet + const)


def fitc_log_prob_chunked(
    Y: torch.Tensor,
    X: torch.Tensor,
    Z: torch.Tensor,
    ell: torch.Tensor,
    sigma2: torch.Tensor,
    noise2: torch.Tensor,
    diag_floor: float = 1e-4,
    jitter: float = 1e-6,
    chunk_size: int = 20000,
) -> torch.Tensor:
    E, N = Y.shape
    M = Z.shape[0]
    Kuu = matern52_kernel(Z, Z, ell, sigma2)
    Luu = chol_with_jitter(Kuu, jitter=jitter)
    Mmat = torch.eye(M, device=Y.device, dtype=Y.dtype)
    logdet_D = torch.tensor(0.0, device=Y.device, dtype=Y.dtype)
    base_quad = torch.zeros(E, device=Y.device, dtype=Y.dtype)
    V = torch.zeros(E, M, device=Y.device, dtype=Y.dtype)
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        Xc = X[s:e]
        Yc = Y[:, s:e]
        Kxu = matern52_kernel(Xc, Z, ell, sigma2)
        Kux = Kxu.transpose(0, 1).contiguous()
        A = torch.linalg.solve_triangular(Luu, Kux, upper=False)
        diagQ = (A**2).sum(dim=0)
        D = (sigma2 - diagQ + noise2).clamp_min(diag_floor)
        Dinv = 1.0 / D
        logdet_D = logdet_D + torch.log(D).sum()
        Ad = A * Dinv.unsqueeze(0)
        Mmat = Mmat + Ad @ A.transpose(0, 1)
        YDinv = Yc * Dinv.unsqueeze(0)
        base_quad = base_quad + (Yc * YDinv).sum(dim=1)
        V = V + (YDinv @ A.transpose(0, 1))
    LM = chol_with_jitter(Mmat, jitter=jitter)
    logdet = logdet_D + 2.0 * torch.log(torch.diagonal(LM)).sum()
    VT = V.transpose(0, 1).contiguous()
    t = torch.linalg.solve_triangular(LM, VT, upper=False)
    W = torch.linalg.solve_triangular(LM.transpose(0, 1), t, upper=True)
    corr = (V * W.transpose(0, 1)).sum(dim=1)
    invquad = base_quad - corr
    const = N * math.log(2.0 * math.pi)
    return -0.5 * (invquad + logdet + const)


@dataclass
class HFConfig:
    m_induce: int = 256
    batch_size: int = 4096
    niters_hf: int = 1500
    lr_hf: float = 5e-2
    clip_grad: float = 10.0
    noise_floor: float = 5e-3
    diag_floor: float = 5e-4
    jitter: float = 1e-6
    eval_chunk_size: int = 20000
    early_stop: bool = False
    patience: int = 75
    min_iters: int = 300


class GPMatern(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = self.covar_sp_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HKClimateKernel(torch.nn.Module):
    def __init__(
        self, grid_size=64, bounds=(-0.1, 1.1), dtype=torch.float32, device="cpu"
    ):
        super().__init__()
        self.mean_module = ZeroMean()
        base_k = MaternKernel(nu=0.5)
        self.covar_module = ScaleKernel(
            GridInterpolationKernel(
                base_k,
                grid_size=grid_size,
                grid_bounds=[bounds, bounds],
                active_dims=[0, 1],
            )
        )
        self.covar_module.outputscale = torch.tensor(1.0, dtype=dtype, device=device)
        for m in self.covar_module.modules():
            if isinstance(m, MaternKernel):
                m.lengthscale = torch.tensor(0.2, dtype=dtype, device=device)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def mvn_log_prob_lazy(covar_op, y_mat):
    if y_mat.dim() != 2:
        raise ValueError("y_mat must be 2D")
    if y_mat.shape[0] < y_mat.shape[1]:
        rhs = y_mat.T.contiguous()
        n = y_mat.shape[1]
    else:
        rhs = y_mat.contiguous()
        n = y_mat.shape[0]
    inv_quad, logdet = covar_op.inv_quad_logdet(inv_quad_rhs=rhs, logdet=True)
    const = n * math.log(2 * math.pi)
    log_prob = -0.5 * (inv_quad + logdet + const)
    return log_prob, logdet


class EarlyStopper2:
    def __init__(self, min_diff: float, patients: int, min_steps: int = 0):
        self.min_diff = min_diff
        self.patients = patients
        self.patients_counter = 0
        self.step_counter = 0
        self.min_steps = min_steps
        self.min_loss = math.inf

    def stop_test(self, loss):
        self.step_counter += 1
        if loss + self.min_diff >= self.min_loss:
            self.patients_counter += 1
        else:
            self.patients_counter = 0
        if loss < self.min_loss:
            self.min_loss = loss
        return (
            self.patients_counter > self.patients and self.step_counter > self.min_steps
        )


def get_multifidelity_scores(
    n: int, locs_pkl: dict, obs_pkl: dict
) -> dict[str, torch.Tensor]:
    left_out_hf = [int(90 * j + 3 * i) for j in range(10) for i in range(10)]
    left_out_mf = [int(20 * j + 2 * i) for j in range(5) for i in range(5)]

    locs_lf = locs_pkl["locs_lf"]
    locs_mf = locs_pkl["locs_mf"]
    locs_hf = locs_pkl["locs_hf"]

    obs_lf = obs_pkl["obs_lf"].to(torch.float64)
    obs_mf = obs_pkl["obs_mf"].to(torch.float64)
    obs_hf = obs_pkl["obs_hf"].to(torch.float64)

    ord_lf = np.lexsort((locs_lf[:, 1], locs_lf[:, 0]))
    ord_mf = np.lexsort((locs_mf[:, 1], locs_mf[:, 0]))
    ord_hf = np.lexsort((locs_hf[:, 1], locs_hf[:, 0]))

    locs_lf = locs_lf[ord_lf]
    locs_mf = locs_mf[ord_mf]
    locs_hf = locs_hf[ord_hf]
    obs_lf = obs_lf[:, ord_lf]
    obs_mf = obs_mf[:, ord_mf]
    obs_hf = obs_hf[:, ord_hf]

    obs_train_lf = obs_lf[0:200, :]
    obs_train_mf = obs_mf[0:200, :]
    obs_train_hf = obs_hf[0:200, :]
    obs_test_lf = obs_lf[200:250, :]
    obs_test_mf = obs_mf[200:250, :]
    obs_test_hf = obs_hf[200:250, :]

    niters = 1000
    train_lf = obs_train_lf[:n, :]
    mean = train_lf.mean(axis=0)
    test_lf = obs_test_lf - mean
    train_lf = train_lf - train_lf.mean(axis=0)

    likelihood_lf = gpytorch.likelihoods.GaussianLikelihood()
    model_lf = GPMatern(locs_lf, train_lf[0], likelihood_lf)
    model_lf.train()
    likelihood_lf.train()
    optimizer = torch.optim.Adam(model_lf.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_lf, model_lf)
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=False)):
        optimizer.zero_grad()
        output = model_lf(locs_lf)
        loss = torch.zeros(())
        for j in range(train_lf.shape[0]):
            loss += -mll(output, train_lf[j])
        loss /= train_lf.shape[0]
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")
        if stopper.stop_test(loss.item()):
            print(f"stopping early at iteration {i}")
            break
        tqdm_iter.set_description(f"loss: {loss.item():.3f}")
        optimizer.step()
        scheduler.step()

    model_lf.eval()
    likelihood_lf.eval()

    train_mf = obs_train_mf[:n, :]
    mean = train_mf.mean(axis=0)
    test_mf = obs_test_mf - mean
    train_mf = train_mf - mean

    likelihood_mf = gpytorch.likelihoods.GaussianLikelihood()
    model_mf = GPMatern(locs_mf, train_mf[0], likelihood_mf)
    model_mf.train()
    likelihood_mf.train()
    optimizer = torch.optim.Adam(model_mf.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_mf, model_mf)
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=False)):
        optimizer.zero_grad()
        output = model_mf(locs_mf)
        loss = torch.zeros(())
        for j in range(train_mf.shape[0]):
            loss += -mll(output, train_mf[j])
        loss /= train_mf.shape[0]
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")
        if stopper.stop_test(loss.item()):
            print(f"stopping early at iteration {i}")
            break
        tqdm_iter.set_description(f"loss: {loss.item():.3f}")
        optimizer.step()
        scheduler.step()

    train_hf = obs_train_hf[:n, :]
    mean = train_hf.mean(axis=0)
    test_hf = obs_test_hf - mean
    train_hf = train_hf - mean

    model_mf.eval()
    likelihood_mf.eval()

    likelihood_hf = gpytorch.likelihoods.GaussianLikelihood()
    model_hf = GPMatern(locs_hf, train_hf[0], likelihood_hf)
    model_hf.train()
    likelihood_hf.train()
    optimizer = torch.optim.Adam(model_hf.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_hf, model_hf)
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=False)):
        optimizer.zero_grad()
        output = model_hf(locs_hf)
        loss = torch.zeros(())
        for j in range(train_hf.shape[0]):
            loss += -mll(output, train_hf[j])
        loss /= train_hf.shape[0]
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")
        if stopper.stop_test(loss.item()):
            print(f"stopping early at iteration {i}")
            break
        tqdm_iter.set_description(f"loss: {loss.item():.3f}")
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        model_lf.train()
        model_lf_at_lf = likelihood_lf(model_lf(locs_lf))
        ls_lf = (
            torch.stack([-model_lf_at_lf.log_prob(test) for test in test_lf])
            .detach()
            .cpu()
        )

        model_mf.train()
        model_mf_at_mf = likelihood_mf(model_mf(locs_mf))
        cov_mf = model_mf_at_mf.covariance_matrix.detach().numpy()
        cov_mf_del = np.delete(
            np.delete(cov_mf, left_out_mf, axis=0), left_out_mf, axis=1
        )
        test_mf_del = np.delete(test_mf.detach().numpy(), left_out_mf, axis=1)
        fit_mf = MultivariateNormal(
            mean=torch.zeros(75), covariance_matrix=torch.from_numpy(cov_mf_del)
        )
        ls_mf = -fit_mf.log_prob(torch.tensor(test_mf_del)).detach().cpu()

        model_hf.train()
        model_hf_at_hf = likelihood_hf(model_hf(locs_hf))
        cov_hf = model_hf_at_hf.covariance_matrix.detach().numpy()
        cov_hf_del = np.delete(
            np.delete(cov_hf, left_out_hf, axis=0), left_out_hf, axis=1
        )
        test_hf_del = np.delete(test_hf.detach().numpy(), left_out_hf, axis=1)
        fit_hf = MultivariateNormal(
            mean=torch.zeros(800), covariance_matrix=torch.from_numpy(cov_hf_del)
        )
        ls_hf = -fit_hf.log_prob(torch.tensor(test_hf_del)).detach().cpu()

    return {"lf": ls_lf, "mf": ls_mf, "hf": ls_hf, "total": ls_lf + ls_mf + ls_hf}


def force_latlon_first(locs):
    locs = np.asarray(locs)
    if locs.shape[1] == 2:
        out = locs.copy()
    elif locs.shape[0] == 2:
        out = locs.T.copy()
    else:
        raise ValueError("locs must be (N,2) or (2,N)")
    lat, lon = out[:, 0], out[:, 1]
    if (np.abs(lat) > 90).any() and (np.abs(lon) <= 90).all():
        out = out[:, ::-1]
    return out


def get_climate_scores(n: int) -> dict[str, np.ndarray | list[int]]:
    locs_gcm = force_latlon_first(np.load("../tests/data/locs_gcm.npy"))
    locs_rcm = force_latlon_first(np.load("../tests/data/locs_rcm.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    locs_gcm = torch.tensor(locs_gcm, device=device, dtype=dtype)
    locs_rcm = torch.tensor(locs_rcm, device=device, dtype=dtype)
    locs_gcm = (locs_gcm - locs_gcm.min(0, keepdim=True).values) / (
        locs_gcm.max(0, keepdim=True).values - locs_gcm.min(0, keepdim=True).values
    )
    locs_rcm = (locs_rcm - locs_rcm.min(0, keepdim=True).values) / (
        locs_rcm.max(0, keepdim=True).values - locs_rcm.min(0, keepdim=True).values
    )

    obs_gcm = torch.tensor(
        np.load("../tests/data/obs_gcm.npy"), device=device, dtype=dtype
    )
    obs_rcm = torch.tensor(
        np.load("../tests/data/obs_rcm.npy"), device=device, dtype=dtype
    )

    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    obs_gcm_train = obs_gcm[train_idx[:n], :]
    obs_rcm_train = obs_rcm[train_idx[:n], :]
    obs_gcm_test = obs_gcm[test_idx, :]
    obs_rcm_test = obs_rcm[test_idx, :]

    mean_gcm = obs_gcm_train.mean(0, keepdim=True)
    sd_gcm = obs_gcm_train.std(0, keepdim=True).clamp_min(1e-6)
    mean_rcm = obs_rcm_train.mean(0, keepdim=True)
    sd_rcm = obs_rcm_train.std(0, keepdim=True).clamp_min(1e-6)

    train_gcm = (obs_gcm_train - mean_gcm) / sd_gcm
    test_gcm = (obs_gcm_test - mean_gcm) / sd_gcm
    train_rcm = (obs_rcm_train - mean_rcm) / sd_rcm
    test_rcm = (obs_rcm_test - mean_rcm) / sd_rcm

    likelihood_lf = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=GreaterThan(1e-5)
    ).to(device=device, dtype=dtype)
    likelihood_lf.noise = torch.tensor(5e-3, device=device, dtype=dtype)
    model_lf = GPMatern(locs_gcm, train_gcm[0], likelihood_lf).to(
        device=device, dtype=dtype
    )
    model_lf.train()
    likelihood_lf.train()
    opt_lf = torch.optim.Adam(
        list(model_lf.parameters()) + list(likelihood_lf.parameters()), lr=1e-2
    )
    mll_lf = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_lf, model_lf)

    for _ in tqdm(range(150), disable=False, desc=f"LF train (n={n})"):
        opt_lf.zero_grad()
        out = model_lf(locs_gcm)
        loss = -mll_lf(out, train_gcm[0])
        loss.backward()
        opt_lf.step()

    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        y_hat_lf = model_lf(locs_rcm).loc

    design = y_hat_lf.reshape(-1, 1)
    y_train = train_rcm.T.contiguous()
    y_test = test_rcm.T.contiguous()
    denom = (design.T @ design).clamp_min(1e-6)
    beta_train = (design.T @ y_train) / denom
    beta_test = (design.T @ y_test) / denom
    train_resid = (y_train - design @ beta_train).T.contiguous()
    test_resid = (y_test - design @ beta_test).T.contiguous()

    hf_cfg = HFConfig()
    torch.manual_seed(0)
    n_rcm = locs_rcm.shape[0]
    m_induce = min(hf_cfg.m_induce, n_rcm)
    idx_z = torch.randperm(n_rcm, device=device)[:m_induce]
    Z = locs_rcm[idx_z].detach()

    raw_ell = nn.Parameter(
        softplus_inv(torch.full((2,), 0.25, device=device, dtype=dtype))
    )
    raw_sigma2 = nn.Parameter(
        softplus_inv(torch.tensor(1.0, device=device, dtype=dtype))
    )
    raw_noise2 = nn.Parameter(
        softplus_inv(torch.tensor(0.05, device=device, dtype=dtype))
    )
    opt_hf = torch.optim.Adam([raw_ell, raw_sigma2, raw_noise2], lr=hf_cfg.lr_hf)
    best = float("inf")
    bad = 0

    for it in tqdm(range(hf_cfg.niters_hf), disable=False, desc=f"HF train (n={n})"):
        opt_hf.zero_grad()
        bsz = min(hf_cfg.batch_size, n_rcm)
        idx_pts = torch.randperm(n_rcm, device=device)[:bsz]
        x_b = locs_rcm[idx_pts]
        y_b = train_resid[:, idx_pts]

        ell = F.softplus(raw_ell) + 2e-2
        sigma2 = F.softplus(raw_sigma2) + 1e-4
        noise2 = F.softplus(raw_noise2) + hf_cfg.noise_floor

        logp = fitc_log_prob(
            y_b,
            x_b,
            Z,
            ell=ell,
            sigma2=sigma2,
            noise2=noise2,
            diag_floor=hf_cfg.diag_floor,
            jitter=hf_cfg.jitter,
        )
        loss = -logp.mean()
        loss.backward()
        if hf_cfg.clip_grad is not None and hf_cfg.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                [raw_ell, raw_sigma2, raw_noise2], hf_cfg.clip_grad
            )
        opt_hf.step()

        if hf_cfg.early_stop:
            cur = float(loss.item())
            if cur < best - 1e-4:
                best = cur
                bad = 0
            else:
                bad += 1
                if bad >= hf_cfg.patience and it >= hf_cfg.min_iters:
                    break

    with torch.no_grad():
        pred_lf = likelihood_lf(model_lf(locs_gcm))
        ls_lf = (
            torch.stack(
                [-pred_lf.log_prob(test_gcm[k]) for k in range(test_gcm.shape[0])],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )

        ell = F.softplus(raw_ell) + 2e-2
        sigma2 = F.softplus(raw_sigma2) + 1e-4
        noise2 = F.softplus(raw_noise2) + hf_cfg.noise_floor
        logp_hf = fitc_log_prob_chunked(
            test_resid,
            locs_rcm,
            Z,
            ell=ell,
            sigma2=sigma2,
            noise2=noise2,
            diag_floor=hf_cfg.diag_floor,
            jitter=hf_cfg.jitter,
            chunk_size=hf_cfg.eval_chunk_size,
        )
        ls_hf = (-logp_hf).detach().cpu().numpy()

    return {"test_idx": test_idx, "lf": ls_lf, "hf": ls_hf, "total": ls_lf + ls_hf}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows: list[dict] = []

    if args.experiment == "climate":
        for n in parse_ns(args.ns, args.experiment):
            print("With ensemble size")
            print(n)
            out = get_climate_scores(n)
            for k, idx in enumerate(out["test_idx"]):
                rows.append(
                    {
                        "n": n,
                        "test_idx": int(idx),
                        "logscore_total": float(out["total"][k]),
                        "logscore_lf": float(out["lf"][k]),
                        "logscore_hf": float(out["hf"][k]),
                    }
                )
            save_rows(
                rows, args.output_dir, "hk", args.experiment, args.include_all_logscores
            )
            print("mean log score")
            print(float(np.mean(out["total"])))
        return

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

    for n in parse_ns(args.ns, args.experiment):
        print("With ensemble size")
        print(n)
        scores = get_multifidelity_scores(n, locs_pkl, obs_pkl)
        for i in range(50):
            rows.append(
                {
                    "n": n,
                    "test_idx": i,
                    "logscore_total": scores["total"][i].item(),
                    "logscore_lf": scores["lf"][i].item(),
                    "logscore_mf": scores["mf"][i].item(),
                    "logscore_hf": scores["hf"][i].item(),
                }
            )
        save_rows(
            rows, args.output_dir, "hk", args.experiment, args.include_all_logscores
        )
        print("mean log score")
        print(scores["total"].mean().item())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="A not p.d., added jitter")
    main()
