import pandas as pd
import gpytorch
import torch
from tqdm import tqdm
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

import pathlib
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import math

class GPMatern(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPMatern, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = self.covar_sp_module = ScaleKernel(
            MaternKernel(nu=0.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(CustomKernel, self).__init__(**kwargs)
        # Kernel k1 acts on the second and third columns (indices 1 and 2)
        self.k1 = ScaleKernel(MaternKernel(nu=0.5, active_dims=[1, 2]))
        # Kernel k2 acts on the first column only (index 0)
        self.k2 = ScaleKernel(MaternKernel(nu=0.5, active_dims=[0]))
        # Kernel k3 acts on the second and third columns (indices 1 and 2)
        self.k3 = ScaleKernel(MaternKernel(nu = 0.5, active_dims=[1, 2]))

    def forward(self, x1, x2, **params):
        return self.k1(x1, x2) * self.k2(x1, x2) + self.k3(x1, x2)
    
class NarGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(NarGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = CustomKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
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

def get_logscore(n_ensemble, data_pkl, data_pkl_min):
    locs_lf = data_pkl["locs_lf"]
    locs_mf = data_pkl["locs_mf"]
    locs_hf = data_pkl["locs_hf"]

    obs_lf = data_pkl_min["obs_lf"]
    obs_mf = data_pkl_min["obs_mf"]
    obs_hf = data_pkl_min["obs_hf"]
    obs_train_hf = obs_hf[0:200, :]
    obs_train_mf = obs_mf[0:200, :]
    obs_train_lf = obs_lf[0:200, :]

    obs_test_hf = obs_hf[200:250, :]
    obs_test_mf = obs_mf[200:250, :]
    obs_test_lf = obs_lf[200:250, :]

    niters = 1000
    silent = False
    train_lf = obs_train_lf[0:n_ensemble,:]
    mean_lf =  train_lf.mean(dim=0, keepdim=True)
    std_lf = train_lf.std(dim=0, keepdim=True)
    train_lf = (train_lf - mean_lf)/std_lf
    test_lf = (obs_test_lf - mean_lf)/std_lf

    locs_train_lf = locs_lf

    likelihood_lf = GaussianLikelihood().double()
    model_lf = GPMatern(locs_train_lf, train_lf[0], likelihood_lf).double()
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
    
    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        model_lf_eval = likelihood_lf(model_lf(locs_mf))
        y_hat_lf = model_lf_eval.loc

    x_mf = torch.hstack((y_hat_lf.reshape(100,1), locs_mf))

    train_mf = obs_train_mf[0:n_ensemble,:]
    test_mf = obs_test_mf
    mean_mf =  train_mf.mean(dim=0, keepdim=True)
    std_mf = train_mf.std(dim=0, keepdim=True)
    train_mf = (train_mf - mean_mf)/std_mf
    test_mf = (obs_test_mf - mean_mf)/std_mf

    likelihood_mf = GaussianLikelihood().double()
    model_mf = NarGP(x_mf, train_mf[0], likelihood_mf).double()
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
        output = model_mf(x_mf)

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

    train_hf = obs_train_hf[0:n_ensemble,:]
    test_hf = obs_test_hf
    mean_hf =  train_hf.mean(dim=0, keepdim=True)
    std_hf = train_hf.std(dim=0, keepdim=True)
    train_hf = (train_hf - mean_hf)/std_hf
    test_hf = (obs_test_hf - mean_hf)/std_hf

    # Prepare for level 3: sample f_1 at X3
    num_samples = 100
    ntest = locs_hf.shape[0]

    #model_lf_eval.covariance_matrix
    with torch.no_grad():
        model_lf.eval()
        likelihood_lf.eval()
        model_lf_eval = model_lf(locs_hf)
        posterior = likelihood_lf(model_lf_eval)
        Z = posterior.rsample(torch.Size([num_samples]))

        tmp_m = torch.zeros((num_samples, ntest))
        tmp_v = torch.zeros((num_samples, ntest))

        # push samples through f_2
        for i in range(num_samples):
            with torch.no_grad():
                model_mf.eval()
                likelihood_mf.eval()
                #model_lf_eval = model_lf(locs_mf)
                #y_hat_lf = model_lf_eval.loc
                x_hf = torch.hstack((Z[i,:][:, None], locs_hf))
                model_mf_eval = model_mf(x_hf)
                tmp_m[i,:] = model_mf_eval.loc.flatten()

        # get mean and variance at X3
        y_hat_mf = torch.mean(tmp_m, axis = 0)

        x_hf = torch.hstack((y_hat_mf.reshape(900,1), locs_hf))

    likelihood_hf = GaussianLikelihood().double()
    model_hf = NarGP(x_hf, train_hf[0], likelihood_hf).double()
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
        output = model_hf(x_hf)

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

    left_out_hf = []
    for j in range(10):
        for i in range(10):
            left_out_hf.append(int(90*j + 3*i))

    left_out_mf = []
    for j in range(5):
        for i in range(5):
            left_out_mf.append(int(20*j + 2*i))

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
        model_mf_at_mf = likelihood_mf(model_mf(x_mf))

    cov_mf = model_mf_at_mf.covariance_matrix.detach().numpy()
    cov_mf_ = np.delete(cov_mf, left_out_mf, axis=0)
    cov_mf_del = np.delete(cov_mf_, left_out_mf, axis=1)
    test_mf_ = test_mf.detach().numpy()
    test_mf_del = np.delete(test_mf_, left_out_mf, axis=1)

    fit_mf = MultivariateNormal(mean = torch.zeros(75), covariance_matrix = torch.from_numpy(cov_mf_del))
    ls2 = torch.mean(-fit_mf.log_prob(torch.Tensor(test_mf_del)))

    with torch.no_grad():
        model_hf.train()
        model_hf_at_hf = likelihood_hf(model_hf(x_hf))

    cov_hf = model_hf_at_hf.covariance_matrix.detach().numpy()
    cov_hf_ = np.delete(cov_hf, left_out_hf, axis=0)
    cov_hf_del = np.delete(cov_hf_, left_out_hf, axis=1)
    test_hf_ = test_hf.detach().numpy()
    test_hf_del = np.delete(test_hf_, left_out_hf, axis=1)

    fit_hf = MultivariateNormal(mean = torch.zeros(800), covariance_matrix = torch.from_numpy(cov_hf_del))
    ls3 = torch.mean(-fit_hf.log_prob(torch.Tensor(test_hf_del)))

    return ls1.item() + ls2.item() + ls3.item()

# Load data
data_fp = pathlib.Path("../tests/data/data_mf.pkl")
with open(data_fp, "rb") as fh:
    data_pkl: dict[str, np.ndarray] = pickle.load(fh)

data_min = pathlib.Path("../tests/data/data_mf_min.pkl")
with open(data_min, "rb") as fh:
    data_pkl_min: dict[str, np.ndarray] = pickle.load(fh)

ns = [5, 10, 20, 30, 50, 100, 200]
n_list = []
ls = []
for n in ns:

    print('With ensemble size')
    print(n)
    ls_ = get_logscore(n, data_pkl, data_pkl_min)
    n_list.append(n)
    ls.append(ls_)
    print('n')
    print(n)

    print('log score')
    print(ls_)

    my_dict = {"n": n_list, "logscore": ls}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_nargp_min.csv', index=False)