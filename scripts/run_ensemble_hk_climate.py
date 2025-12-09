import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gpytorch
import torch
from tqdm import tqdm
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
import math
import pandas as pd
import numpy as np

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import GridInterpolationKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.distributions import MultivariateNormal

from gpytorch.mlls.variational_elbo import VariationalELBO
from torch.utils.data import DataLoader, TensorDataset

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
    if locs.shape[1] == 2:                  # already (N, 2)
        out = locs.copy()
    elif locs.shape[0] == 2:                # transpose from (2, N)
        out = locs.T.copy()
    else:
        raise ValueError("locs must be (N,2) or (2,N)")

    # --- 2. Put latitude in column-0, longitude in column-1 ---------------
    lat, lon = out[:, 0], out[:, 1]

    # Latitude must live in ±90°; longitude often exceeds that.
    if (np.abs(lat) > 90).any() and (np.abs(lon) <= 90).all():
        out = out[:, ::-1]                  # swap columns

    return out
    
from gpytorch.kernels import GridInterpolationKernel, MaternKernel

class MaternApprox(ApproximateGP):
    def __init__(self, train_x, grid_size=32, num_inducing=128):

        #super(NarGP, self).__init__(train_x)
        # ----- kernels (unchanged) -----------------------------------------
        base_k1 = MaternKernel(nu=0.5)


        k1 = GridInterpolationKernel(base_k1, grid_size=grid_size,
                                     grid_bounds=[(-1, 2), (-1, 2)],
                                     active_dims=[0, 1])

        signal_kernel = k1

        # ----- variational layer  ------------------------------------------
        #num_inducing = 64                      # start small, tune later
        # random inducing locations inside [0,1]³  (matches your normalised x)
        inducing_points = torch.rand(num_inducing,
                                     train_x.size(-1),
                                     dtype=train_x.dtype,
                                     device=train_x.device)

        variational_dist = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=True,      # let the optimiser move them
        )

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean()
        self.covar_module = signal_kernel #+ nugget

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

def get_logscore(n):

    locs_gcm = np.load("../tests/data/locs_gcm.npy")
    locs_rcm = np.load("../tests/data/locs_rcm.npy")

    locs_gcm = force_latlon_first(locs_gcm)
    locs_rcm = force_latlon_first(locs_rcm)

    obs_gcm = np.load("../tests/data/obs_gcm.npy")
    obs_rcm = np.load("../tests/data/obs_rcm.npy")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE  = torch.float32        # float32 is plenty for SKI

    locs_gcm = torch.Tensor(locs_gcm).to(DEVICE, DTYPE)
    locs_rcm = torch.Tensor(locs_rcm).to(DEVICE, DTYPE)

    locs_min = locs_gcm.min(dim=0, keepdim=True).values
    locs_max = locs_gcm.max(dim=0, keepdim=True).values
    locs_gcm = (locs_gcm - locs_min) / (locs_max - locs_min)

    locs_min = locs_rcm.min(dim=0, keepdim=True).values
    locs_max = locs_rcm.max(dim=0, keepdim=True).values
    locs_rcm = (locs_rcm - locs_min) / (locs_max - locs_min)

    print('Shape of RCM and GCM data')
    print(locs_gcm.shape, locs_rcm.shape, obs_gcm.shape, obs_rcm.shape)

    niters = 1000
    silent = False

    print('Shape of RCM and GCM data')
    print(locs_gcm.shape, locs_rcm.shape, obs_gcm.shape, obs_rcm.shape)

    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    obs_gcm_train = torch.Tensor(obs_gcm[train_idx[0:n], :]).to(device=DEVICE, dtype=DTYPE)
    obs_rcm_train = torch.Tensor(obs_rcm[train_idx[0:n], :]).to(device=DEVICE, dtype=DTYPE)

    obs_gcm_test = torch.Tensor(obs_gcm[test_idx, :]).to(device=DEVICE, dtype=DTYPE)
    obs_rcm_test = torch.Tensor(obs_rcm[test_idx, :]).to(device=DEVICE, dtype=DTYPE)

    mean_gcm = obs_gcm_train.mean(dim=0, keepdim = True)
    mean_rcm = obs_rcm_train.mean(dim=0, keepdim=True)
    train_gcm = (obs_gcm_train - mean_gcm)#/sd_gcm
    test_gcm = (obs_gcm_test - mean_gcm)#/sd_gcm
    E, N = train_gcm.shape
    batch_size = 8192
    num_data = E * N
    silent = True

    likelihood_lf = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE, DTYPE)
    model_lf = MaternApprox(locs_gcm).to(DEVICE, DTYPE)
    mll = VariationalELBO(likelihood_lf, model_lf, num_data=num_data)
    model_lf.train()
    likelihood_lf.train()

    optimizer = torch.optim.Adam(
        [
            {'params': model_lf.parameters()},
            {'params': likelihood_lf.parameters()},
        ],
        lr=1e-2,
    )

    dataset = TensorDataset(locs_gcm, train_gcm.T)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20 * steps_per_epoch, T_mult=2, eta_min=1e-4
    )

    stopper = EarlyStopper2(1e-4, 40)
    losses = []
    #niters = 200
    for epoch in (pbar := tqdm(range(1000), disable=silent)):
        model_lf.train(); likelihood_lf.train()
        epoch_loss = 0.0
        print(epoch)

        for xb, yb in loader:
            # xb: [B, 3]          yb: [B, E]   (B = batch_size or smaller last batch)
            xb.clamp_(0, 1)

            # Expand xb so model sees the ensemble as a batch dimension
            xb_batched = xb.unsqueeze(0).expand(E, -1, -1)       # [E, B, 3]
            
            yb_batched = yb.T.contiguous()                       # [E, B]

            optimizer.zero_grad()
            with gpytorch.settings.cg_tolerance(1e-2), \
                gpytorch.settings.max_cg_iterations(50), \
                gpytorch.settings.max_preconditioner_size(10):
                output = model_lf(xb_batched)                        # batched MvNormal
                loss   = -mll(output, yb_batched).mean()                    # scalar ELBO (already averaged)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * xb.size(0)               # accumulate for logging

        epoch_loss /= N
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(epoch)
            print(f"loss: {epoch_loss: .3f}")
        pbar.set_description(f"loss: {epoch_loss: .3f}")

        # ---- early stopping check ------------------------------------------------
        if stopper.stop_test(epoch_loss):
            print(f"Stopping early at epoch {epoch}")
            break

    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        model_lf_eval = likelihood_lf(model_lf(locs_rcm))
        y_hat_lf = model_lf_eval.loc + mean_rcm
        F = y_hat_lf.reshape(-1, 1)
        Y = obs_rcm_train.T                     # (N,n_train)

        # --- CHANGE: call the *whole* covar module, not .base_kernel -------
        R_op = model_lf.covar_module(locs_rcm)/torch.exp(model_lf.covar_module.base_kernel.raw_lengthscale)    # Lazy SKI operator (N×N)

        RinvF = R_op.solve(F)               # O(N log N)   no huge mat
        RinvY = R_op.solve(Y)               # batched solve

        FtRinvF  = (F * RinvF).sum()        # scalar
        beta_all = (F.T @ RinvY) / FtRinvF  # (1, n_train)

        train_rcm      = (Y - F @ beta_all).T  # (n_train, N)

        # --- test set ------------------------------------------------------
        Y_test        = obs_rcm_test.T
        RinvY_test    = R_op.solve(Y_test)
        beta_test     = (F.T @ RinvY_test) / FtRinvF
        test_rcm = (Y_test - F @ beta_test).T

    E, N = train_rcm.shape
    batch_size = 8192
    num_data = E * N

    likelihood_hf = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE, DTYPE)
    model_hf = MaternApprox(locs_rcm).to(DEVICE, DTYPE)
    mll = VariationalELBO(likelihood_hf, model_hf, num_data=num_data)
    model_hf.train()
    likelihood_hf.train()

    optimizer = torch.optim.Adam([
        {'params': model_hf.variational_strategy.parameters(), 'lr': 1e-2},
        {'params': model_hf.covar_module.parameters(),         'lr': 1e-3},
        {'params': model_hf.mean_module.parameters(),          'lr': 1e-3},
        {'params': likelihood_hf.parameters(),                 'lr': 3e-2},
    ])

    dataset = TensorDataset(locs_rcm, train_rcm.T)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20 * steps_per_epoch, T_mult=2, eta_min=1e-4
    )

    stopper = EarlyStopper2(1e-4, 40)
    losses = []
    for epoch in (pbar := tqdm(range(200), disable=silent)):
        model_hf.train(); likelihood_hf.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            # xb: [B, 3]          yb: [B, E]   (B = batch_size or smaller last batch)
            xb.clamp_(0, 1)

            # Expand xb so model sees the ensemble as a batch dimension
            xb_batched = xb.unsqueeze(0).expand(E, -1, -1)       # [E, B, 3]
            
            yb_batched = yb.T.contiguous()                       # [E, B]

            optimizer.zero_grad()
            with gpytorch.settings.cg_tolerance(1e-2), \
                gpytorch.settings.max_cg_iterations(50), \
                gpytorch.settings.max_preconditioner_size(10):
                output = model_hf(xb_batched)                        # batched MvNormal
                loss   = -mll(output, yb_batched).mean()                    # scalar ELBO (already averaged)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * xb.size(0)               # accumulate for logging

        epoch_loss /= N
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(epoch)
            print(f"loss: {epoch_loss: .3f}")
        pbar.set_description(f"loss: {epoch_loss: .3f}")

        # ---- early stopping check ------------------------------------------------
        if stopper.stop_test(epoch_loss):
            print(f"Stopping early at epoch {epoch}")
            break

    with torch.no_grad():
        model_lf.eval(); likelihood_lf.eval()
        pred_lf = likelihood_lf(model_lf(locs_gcm))
        ls_lf = -pred_lf.log_prob(test_gcm).mean()

        model_hf.eval(); likelihood_hf.eval()
        pred_hf = likelihood_hf(model_hf(locs_rcm))
        ls_hf = -pred_hf.log_prob(test_rcm).mean()
    
    return ls_lf + ls_hf

np.random.seed(0)
torch.manual_seed(0)

ns = [5, 10, 15, 20, 25, 30, 35, 40]
n_list = []
ls_list = []
for n in ns:

    print('With ensemble size')
    print(n)
    ls = get_logscore(n)
    n_list.append(n)
    ls_list.append(ls.item())
    print('n')
    print(n)

    print('log score')
    print(ls.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_hk_climate.csv', index=False)
