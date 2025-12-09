import math
import os
import warnings

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    AdditiveKernel,
    GridInterpolationKernel,
    MaternKernel,
    ProductKernel,
    ScaleKernel,
)
from gpytorch.means import ConstantMean
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def force_latlon_first(locs):
    """
    Return a copy shaped (N, 2), column-0 = latitude, column-1 = longitude.
    Accepts input in any of these shapes/orientations:
        (N, 2)  → either [lat, lon] or [lon, lat]
        (2, N)  → either [[lat...],[lon...]] or [[lon...],[lat...]]
    Uses value ranges to detect & correct 'lon-lat' layouts.
    """
    locs = np.asarray(locs)

    # --- 1. Ensure shape (N, 2) -------------------------------------------
    if locs.shape[1] == 2:  # already (N, 2)
        out = locs.copy()
    elif locs.shape[0] == 2:  # transpose from (2, N)
        out = locs.T.copy()
    else:
        raise ValueError("locs must be (N,2) or (2,N)")

    # --- 2. Put latitude in column-0, longitude in column-1 ---------------
    lat, lon = out[:, 0], out[:, 1]

    # Latitude must live in ±90°; longitude often exceeds that.
    if (np.abs(lat) > 90).any() and (np.abs(lon) <= 90).all():
        out = out[:, ::-1]  # swap columns

    return out


class GPMatern(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = self.covar_sp_module = ScaleKernel(MaternKernel(nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k1 = MaternKernel(nu=0.5, active_dims=[1, 2])
        self.k2 = MaternKernel(nu=0.5, active_dims=[0])
        self.k3 = MaternKernel(nu=0.5, active_dims=[1, 2])

    def forward(self, x1, x2, **params):
        return (self.k1(x1, x2) * self.k2(x1, x2)) + self.k3(x1, x2)


class NarGP(ApproximateGP):
    def __init__(self, train_x):
        GRID = 32
        # super(NarGP, self).__init__(train_x)
        # ----- kernels (unchanged) -----------------------------------------
        base_k1 = MaternKernel(nu=0.5)
        base_k2 = MaternKernel(nu=0.5)
        base_k3 = MaternKernel(nu=0.5)

        k1 = GridInterpolationKernel(
            base_k1,
            grid_size=GRID,
            grid_bounds=[(-0.2, 1.2), (-0.2, 1.2)],
            active_dims=[1, 2],
        )
        k2 = GridInterpolationKernel(
            base_k2, grid_size=GRID, grid_bounds=[(-0.2, 1.2)], active_dims=[0]
        )
        k3 = GridInterpolationKernel(
            base_k3,
            grid_size=GRID,
            grid_bounds=[(-0.2, 1.2), (-0.2, 1.2)],
            active_dims=[1, 2],
        )

        # nugget = WhiteNoiseKernel(noise_prior=None,
        #                          noise_constraint=Positive(transform=None),
        #                          batch_shape=torch.Size())   # scalar

        signal_kernel = AdditiveKernel(ProductKernel(k1, k2), k3)

        # ----- variational layer  ------------------------------------------
        num_inducing = 64  # start small, tune later
        # random inducing locations inside [0,1]³  (matches your normalised x)
        inducing_points = torch.rand(
            num_inducing, train_x.size(-1), dtype=train_x.dtype, device=train_x.device
        )

        variational_dist = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=True,  # let the optimiser move them
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = signal_kernel  # + nugget

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


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

        if self.patients_counter > self.patients and self.step_counter > self.min_steps:
            return True
        else:
            return False

    def reset(self):
        self.patients_counter = 0
        self.step_counter = 0
        self.min_loss = math.inf


def make_bounds(x, pad=0.05):  # pad = 5 % of range
    # x: [N, D] tensor in [0,1] after normalisation
    mins = x.min(0).values
    maxs = x.max(0).values
    span = maxs - mins
    lower = (mins - pad * span).clamp_min(0.0)
    upper = (maxs + pad * span).clamp_max(1.0)
    return [(lo.item(), u.item()) for lo, u in zip(lower, upper)]


def get_logscore(n):
    locs_gcm = force_latlon_first(np.load("../tests/data/locs_gcm.npy"))
    locs_rcm = force_latlon_first(np.load("../tests/data/locs_rcm.npy"))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32  # float32 is plenty for SKI

    locs_gcm = torch.Tensor(locs_gcm).to(DEVICE, DTYPE)
    locs_rcm = torch.Tensor(locs_rcm).to(DEVICE, DTYPE)

    locs_min = locs_gcm.min(dim=0, keepdim=True).values
    locs_max = locs_gcm.max(dim=0, keepdim=True).values
    locs_gcm = (locs_gcm - locs_min) / (locs_max - locs_min)

    locs_min = locs_rcm.min(dim=0, keepdim=True).values
    locs_max = locs_rcm.max(dim=0, keepdim=True).values
    locs_rcm = (locs_rcm - locs_min) / (locs_max - locs_min)

    obs_gcm = torch.from_numpy(np.load("../tests/data/obs_gcm.npy")).to(DEVICE, DTYPE)
    obs_rcm = torch.from_numpy(np.load("../tests/data/obs_rcm.npy")).to(DEVICE, DTYPE)

    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    obs_gcm_train = torch.Tensor(obs_gcm[train_idx[0:n], :]).to(
        device=DEVICE, dtype=DTYPE
    )
    obs_rcm_train = torch.Tensor(obs_rcm[train_idx[0:n], :]).to(
        device=DEVICE, dtype=DTYPE
    )

    obs_gcm_test = torch.Tensor(obs_gcm[test_idx, :]).to(device=DEVICE, dtype=DTYPE)
    obs_rcm_test = torch.Tensor(obs_rcm[test_idx, :]).to(device=DEVICE, dtype=DTYPE)
    niters = 200
    silent = False

    mean_gcm = obs_gcm_train.mean(dim=0, keepdim=True)
    sd_gcm = obs_gcm_train.std(dim=0, keepdim=True)
    mean_rcm = obs_rcm_train.mean(dim=0, keepdim=True)
    sd_rcm = obs_rcm_train.std(dim=0, keepdim=True)
    train_gcm = (obs_gcm_train - mean_gcm) / sd_gcm
    train_rcm = (obs_rcm_train - mean_rcm) / sd_rcm
    test_gcm = (obs_gcm_test - mean_gcm) / sd_gcm
    test_rcm = (obs_rcm_test - mean_rcm) / sd_rcm

    likelihood_lf = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE, DTYPE)
    model_lf = GPMatern(locs_gcm, train_gcm[0], likelihood_lf).to(DEVICE, DTYPE)
    model_lf.train()
    likelihood_lf.train()

    optimizer = torch.optim.Adam(
        model_lf.parameters(), lr=0.01
    )  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_lf, model_lf)

    losses = []
    stopper = EarlyStopper2(1e-6, 50)

    for i in (tqdm_iter := tqdm(range(niters), disable=silent)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_lf(locs_gcm)

        # Calc loss and backprop gradients
        loss = torch.zeros((), device=DEVICE)
        for j in range(train_gcm.shape[0]):
            loss = loss + (-mll(output, train_gcm[j]))  # keep as tensor
        loss = loss / train_gcm.shape[0]

        losses.append(loss.item())
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("loss is NaN")

        # consider early stopping
        if stopper.stop_test(loss.item()):
            print(f"stopping early at iteration {i}")
            break

        tqdm_iter.set_description(f"loss: {loss.item():.3f}")

        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        model_lf_eval = model_lf(locs_rcm)
        y_hat_lf = model_lf_eval.loc

    E, N = train_rcm.shape
    batch_size = 256
    num_data = E * N

    # Build inputs
    a_norm = (y_hat_lf - y_hat_lf.min()) / (
        y_hat_lf.max() - y_hat_lf.min()
    )  # shape [N]
    loc_min = locs_rcm.min(0, keepdim=True).values
    loc_max = locs_rcm.max(0, keepdim=True).values
    locs_rcm_norm = (locs_rcm - loc_min) / (loc_max - loc_min)
    x_full = torch.hstack((a_norm.view(-1, 1), locs_rcm_norm)).to(DEVICE, DTYPE)

    likelihood_hf = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE, DTYPE)
    model_hf = NarGP(x_full).to(DEVICE, DTYPE)  # "a" now lives in [0,1]
    mll = VariationalELBO(likelihood_hf, model_hf, num_data=num_data)
    optimizer = torch.optim.Adam(
        [
            {"params": model_hf.parameters()},
            {"params": likelihood_hf.parameters()},
        ],
        lr=1e-2,
    )
    warnings.filterwarnings("ignore", message="A not p.d., added jitter of 1.0e-04")
    # Dataset where each item is *one spatial location* (same for all ensemble members)
    dataset = TensorDataset(x_full, train_rcm.T)  # [N, 3]

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True
    )  # keep in RAM

    for epoch in (pbar := tqdm(range(niters), disable=silent)):
        model_hf.train()
        likelihood_hf.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            # xb: [B, 3]          yb: [B, E]   (B = batch_size or smaller last batch)
            xb.clamp_(0, 1)

            # Expand xb so model sees the ensemble as a batch dimension
            xb_batched = xb.unsqueeze(0).expand(E, -1, -1)  # [E, B, 3]

            yb_batched = yb.T.contiguous()  # [E, B]

            optimizer.zero_grad()
            with gpytorch.settings.cg_tolerance(
                1e-2
            ), gpytorch.settings.max_cg_iterations(
                50
            ), gpytorch.settings.max_preconditioner_size(
                10
            ):
                output = model_hf(xb_batched)  # batched MvNormal
                loss = -mll(output, yb_batched).mean()  # scalar ELBO (already averaged)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)  # accumulate for logging

        epoch_loss /= N
        losses.append(epoch_loss)
        pbar.set_description(f"loss: {epoch_loss: .3f}")

        # ---- early stopping check ------------------------------------------------
        if stopper.stop_test(epoch_loss):
            print(f"Stopping early at epoch {epoch}")
            break

        with torch.no_grad():
            model_lf.eval()
            likelihood_lf.eval()
            pred_lf = likelihood_lf(model_lf(locs_gcm))
            ls_lf = -pred_lf.log_prob(test_gcm).mean()

            model_hf.eval()
            likelihood_hf.eval()
            pred_hf = likelihood_hf(model_hf(x_full))
            lss = -pred_hf.log_prob(test_rcm).mean()

        return ls_lf + lss


np.random.seed(0)
torch.manual_seed(0)

ns = [5, 10, 15, 20, 25, 30, 35, 40]
n_list = []
ls_list = []
for n in ns:
    print("With ensemble size")
    print(n)
    ls = get_logscore(n)
    n_list.append(n)
    ls_list.append(ls.item())
    print("n")
    print(n)

    print("log score")
    print(ls.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv("./results/logscores_nargp_climate.csv", index=False)
