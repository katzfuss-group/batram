import gpytorch
import torch
from tqdm import tqdm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import pickle

class GPMatern(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPMatern, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = self.covar_sp_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
import math

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


def get_logscore(n, data_pkl):

    left_out_hf = []
    for j in range(10):
        for i in range(10):
            left_out_hf.append(int(90*j + 3*i))

    left_out_mf = []
    for j in range(5):
        for i in range(5):
            left_out_mf.append(int(20*j + 2*i))

    locs_lf = data_pkl["locs_lf"]
    locs_mf = data_pkl["locs_mf"]
    locs_hf = data_pkl["locs_hf"]

    obs_lf = data_pkl["obs_lf"]
    obs_mf = data_pkl["obs_mf"]
    obs_hf = data_pkl["obs_hf"]

    ord_lf = np.lexsort((locs_lf[:, 1], locs_lf[:, 0]))
    ord_mf = np.lexsort((locs_mf[:, 1], locs_mf[:, 0]))
    ord_hf = np.lexsort((locs_hf[:, 1], locs_hf[:, 0]))
    locs_lf = locs_lf[ord_lf]
    obs_lf = obs_lf[:, ord_lf]
    locs_mf = locs_mf[ord_mf]
    obs_mf = obs_mf[:, ord_mf]
    locs_hf = locs_hf[ord_hf]
    obs_hf = obs_hf[:, ord_hf]

    obs_train_hf = obs_hf[0:200, :]
    obs_train_mf = obs_mf[0:200, :]
    obs_train_lf = obs_lf[0:200, :]

    obs_test_hf = obs_hf[200:250, :]
    obs_test_mf = obs_mf[200:250, :]
    obs_test_lf = obs_lf[200:250, :]

    n_ensemble = n
    niters = 1000
    silent = False
    train_lf = obs_train_lf[0:n_ensemble,:]
    mean =  train_lf.mean(axis=0)
    test_lf = obs_test_lf - mean
    train_lf = train_lf - train_lf.mean(axis=0)

    locs_train_lf = locs_lf

    likelihood_lf = GaussianLikelihood()
    model_lf = GPMatern(locs_train_lf, train_lf[0], likelihood_lf)
    model_lf.train()
    likelihood_lf.train()

    optimizer = torch.optim.Adam(
        model_lf.parameters(), lr=0.01
    )  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_lf, model_lf)

    losses = []
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=silent)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_lf(locs_train_lf)

        # Calc loss and backprop gradients
        loss = torch.zeros(())
        for j in range(train_lf.shape[0]):
            loss += -mll(output, train_lf[j])
        loss /= train_lf.shape[0]

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

    obs_mf = obs_train_mf[0:n_ensemble,:]
    mean_mf =  obs_mf.mean(axis=0)
    test_mf = obs_test_mf
    train_mf = obs_mf

    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        model_lf_eval = likelihood_lf(model_lf(locs_mf))
        y_hat_lf = model_lf_eval.loc
        F = y_hat_lf.reshape(-1, 1)
        Y = train_mf.T                     # (N,n_train)

        R_op = model_lf.covar_module(locs_mf)/torch.exp(model_lf.covar_module.base_kernel.raw_lengthscale)    # Lazy SKI operator (N×N)

        RinvF = R_op.solve(F)               # O(N log N)   no huge mat
        RinvY = R_op.solve(Y)               # batched solve

        FtRinvF  = (F * RinvF).sum()        # scalar
        beta_all = (F.T @ RinvY) / FtRinvF  # (1, n_train)

        train_mf      = (Y - F @ beta_all).T  # (n_train, N)

        # test set
        Y_test        = test_mf.T
        RinvY_test    = R_op.solve(Y_test)
        beta_test     = (F.T @ RinvY_test) / FtRinvF
        test_mf       = (Y_test - F @ beta_test).T

    likelihood_mf = GaussianLikelihood()
    model_mf = GPMatern(locs_mf, train_mf[0], likelihood_mf)
    model_mf.train()
    likelihood_mf.train()

    optimizer = torch.optim.Adam(
        model_mf.parameters(), lr=0.01
    )  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_mf, model_mf)

    losses = []
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=silent)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_mf(locs_mf)

        # Calc loss and backprop gradients
        loss = torch.zeros(())
        for j in range(train_mf.shape[0]):
            loss += -mll(output, train_mf[j])
        loss /= train_mf.shape[0]

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

    obs_hf = obs_train_hf[0:n_ensemble,:]
    mean_hf =  obs_hf.mean(axis=0)
    test_hf = obs_test_hf
    train_hf = obs_hf

    with torch.no_grad():
        model_mf.eval()
        likelihood_mf.eval()
        model_mf_eval = likelihood_mf(model_mf(locs_hf))
        y_hat_mf = model_mf_eval.loc
        F = y_hat_mf.reshape(-1, 1)
        Y = train_hf.T                     # (N,n_train)

        R_op = model_mf.covar_module(locs_hf)/torch.exp(model_mf.covar_module.base_kernel.raw_lengthscale)   

        RinvF = R_op.solve(F)               # O(N log N)   no huge mat
        RinvY = R_op.solve(Y)               # batched solve

        FtRinvF  = (F * RinvF).sum()        # scalar
        beta_all = (F.T @ RinvY) / FtRinvF  # (1, n_train)

        train_hf      = (Y - F @ beta_all).T  # (n_train, N)

        # test set
        Y_test        = test_hf.T
        RinvY_test    = R_op.solve(Y_test)
        beta_test     = (F.T @ RinvY_test) / FtRinvF
        test_hf       = (Y_test - F @ beta_test).T

    likelihood_hf = gpytorch.likelihoods.GaussianLikelihood()
    model_hf = GPMatern(locs_hf, train_hf[0], likelihood_hf)
    model_hf.train()
    likelihood_hf.train()

    optimizer = torch.optim.Adam(
        model_hf.parameters(), lr=0.01
    )  # Includes GaussianLikelihood parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, niters)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_hf, model_hf)

    losses = []
    stopper = EarlyStopper2(1e-5, 100)

    for i in (tqdm_iter := tqdm(range(niters), disable=silent)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_hf(locs_hf)

        # Calc loss and backprop gradients
        loss = torch.zeros(())
        for j in range(train_hf.shape[0]):
            loss += -mll(output, train_hf[j])
        loss /= train_hf.shape[0]

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

    log_scores_lf = []
    with torch.no_grad():
        model_lf.train()
        model_lf_at_lf = likelihood_lf(model_lf(locs_lf))
        for test in test_lf:
            ls = model_lf_at_lf.log_prob(test)
            log_scores_lf.append(ls)

    ls1 = -np.mean(log_scores_lf)
    with torch.no_grad():
        model_mf.train()
        model_mf_at_mf = likelihood_mf(model_mf(locs_mf))

    cov_mf = model_mf_at_mf.covariance_matrix.detach().numpy()
    cov_mf_ = np.delete(cov_mf, left_out_mf, axis=0)
    cov_mf_del = np.delete(cov_mf_, left_out_mf, axis=1)
    test_mf_ = test_mf.detach().numpy()
    test_mf_del = np.delete(test_mf_, left_out_mf, axis=1)

    fit_mf = MultivariateNormal(mean = torch.zeros(75), covariance_matrix = torch.from_numpy(cov_mf_del))
    ls2 = torch.mean(-fit_mf.log_prob(torch.Tensor(test_mf_del)))

    with torch.no_grad():
        model_hf.train()
        model_hf_at_hf = likelihood_hf(model_hf(locs_hf))

    cov_hf = model_hf_at_hf.covariance_matrix.detach().numpy()
    cov_hf_ = np.delete(cov_hf, left_out_hf, axis=0)
    cov_hf_del = np.delete(cov_hf_, left_out_hf, axis=1)
    test_hf_ = test_hf.detach().numpy()
    test_hf_del = np.delete(test_hf_, left_out_hf, axis=1)

    fit_mf = MultivariateNormal(mean = torch.zeros(800), covariance_matrix = torch.from_numpy(cov_hf_del))
    ls3 = torch.mean(-fit_mf.log_prob(torch.Tensor(test_hf_del)))

    return ls1 + ls2 + ls3
# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

# Load data
data_fp = Path("../tests/data/data_mf.pkl")
with open(data_fp, "rb") as fh:
    data_pkl: dict[str, np.ndarray] = pickle.load(fh)


ns = [5, 10, 20, 30, 50, 100, 200]
n_list = []
ls = []
for n in ns:

    print('With ensemble size')
    print(n)
    ls_ = get_logscore(n, data_pkl)
    n_list.append(n)
    ls.append(ls_.item())
    print('n')
    print(n)

    print('log score')
    print(ls_.item())

    my_dict = {"n": n_list, "logscore": ls}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_hk_linear.csv', index=False)
