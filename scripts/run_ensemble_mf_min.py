import os
import gc
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Packages for data loading
import pathlib
import pickle
import csv

# Packages for working with array data and tensors]
import numpy as np
import matplotlib.pyplot as plt
import torch

# Packages for building transport maps
from veccs.orderings import maxmin_pred_cpp, find_nns_l2_mf
import scipy
import scipy.spatial.distance

from scipy.spatial.distance import cdist
from batram.data import MultiFidelityData, AugmentDataMF
#from batram.mf_train_separately import MultiFidelityTM
from batram.mf import MultiFidelityTM
import pandas as pd

def get_logscore(n, data_pkl, data_pkl_min):
    # no locs in the min pkl file so we need both, they share the same
    # locations
    print('Loading Data')
    locs_lf = data_pkl["locs_lf"]
    locs_mf = data_pkl["locs_mf"]
    locs_hf = data_pkl["locs_hf"]

    obs_lf = data_pkl_min["obs_lf"]
    obs_mf = data_pkl_min["obs_mf"]
    obs_hf = data_pkl_min["obs_hf"]

    # The loaded data contained 200 replicate spatial fields (200 samples, 900 locs).
    # We will subset the data to use the first 10 samples for each fidelity
    obs_lf_train = obs_lf[:n, :]
    obs_mf_train = obs_mf[:n, :]
    obs_hf_train = obs_hf[:n, :]

    vmin, vmax = obs_hf.min(), obs_hf.max()
    plt.set_cmap('RdBu')
    obs_lf_test = obs_lf[200:250, :]
    obs_mf_test = obs_mf[200:250, :]
    obs_hf_test = obs_hf[200:250, :]

    print(obs_lf_train.shape)
    print(obs_mf_train.shape)
    print(obs_hf_train.shape)

    # Pixel-wise data centering, so that each pixel is mean zero
    # variance 1)

    mean_lf = obs_lf.mean(dim=0, keepdim = True)
    sd_lf = obs_lf.std(dim=0, keepdim= True)
    mean_mf = obs_mf.mean(dim=0, keepdim=True)
    sd_mf = obs_mf.std(dim = 0, keepdim=True)
    mean_hf = obs_hf.mean(dim = 0, keepdim=True)
    sd_hf = obs_hf.std(dim=0, keepdim=True)
    obs_lf = (obs_lf_train - mean_lf)/sd_lf
    obs_mf = (obs_mf_train - mean_mf)/sd_mf
    obs_hf = (obs_hf_train - mean_hf)/sd_hf

    obs_lf_test = (obs_lf_test - mean_lf)/sd_lf
    obs_mf_test = (obs_mf_test - mean_mf)/sd_mf
    obs_hf_test = (obs_hf_test - mean_hf)/sd_hf

    obs = torch.hstack((obs_lf, obs_mf, obs_hf))
    obs_test = torch.hstack((obs_lf_test, obs_mf_test, obs_hf_test))

    # Epsilons

    dist_mf = cdist(locs_mf, locs_mf)
    epsilon_1 = dist_mf[dist_mf!=0.0].min()
    epsilon_1_col = torch.zeros(25, 1) + epsilon_1
    locs_lf = torch.hstack((locs_lf, epsilon_1_col))

    dist_hf = cdist(locs_hf, locs_hf)
    epsilon_2 = dist_hf[dist_hf!=0.0].min()
    epsilon_2_col = torch.zeros(100, 1) + epsilon_2
    locs_mf = torch.hstack((locs_mf, epsilon_2_col))

    locs_hf = torch.hstack((locs_hf, torch.zeros(900,1)))

    print('ordering')

    ord_lfmf = maxmin_pred_cpp(locs_lf.detach().numpy(), locs_mf.detach().numpy())

    locs_lfmf = np.vstack((locs_lf, locs_mf))
    ord_hf = maxmin_pred_cpp(locs_lfmf, locs_hf.detach().numpy())

    ord = np.concatenate((ord_lfmf, ord_hf[125:]))
    locs = torch.vstack((locs_lf, locs_mf, locs_hf))
    locs_ord = locs[ord, ...]

    locs_all = [locs_ord[:25], locs_ord[25:125], locs_ord[125:]]
    largest_conditioning_set = 50
    nn = find_nns_l2_mf(locs_all, largest_conditioning_set)

    ord_lf = ord[:25]
    ord_mf = ord[25:125] - 100
    ord_hf = ord[125:] - 125

    obs_ord = obs[..., ord]
    obs_test_ord = obs_test[..., ord]

    locs_ord = locs[ord, ...]
    obs_ord = obs[..., ord]
    obs_test_ord = obs_test[..., ord]

    locs_all = [locs_ord[:25].detach().numpy(), locs_ord[25:125].detach().numpy(), locs_ord[125:].detach().numpy()]
    largest_conditioning_set = 50
    nn = find_nns_l2_mf(locs_all, largest_conditioning_set)

    fidelity_sizes = torch.as_tensor(list(map(len, locs_all)))
    data = MultiFidelityData.new(locs_ord, obs_ord, torch.as_tensor(nn), fidelity_sizes)

    R = 3
    theta_init = torch.zeros((9*R-2))
    log_2m = data.response[:, 0].square().mean().log()
    # Nugget 1
    theta_init[:R] = log_2m
    # Nugget 2
    theta_init[R:2*R] = 0.2
    # Sigma 1
    theta_init[2*R:3*R] = 0.0
    # Sigma 2
    theta_init[3*R:4*R] = 0.0
    # Theta q 0 within
    theta_init[4*R:5*R] = 0.0
    # Theta q 1 within
    theta_init[5*R:6*R] = 0.0
    # Theta q 0 between
    theta_init[6*R:7*R-1] = 0.0
    # Theta q 1 between
    theta_init[7*R-1:8*R-2]
    # Theta gammas
    theta_init[8*R-2:9*R-2] = -1.0

    print('Training Model')

    tm = MultiFidelityTM(data, theta_init, nug_mult = 4.0)

    nsteps = 200
    opt = torch.optim.Adam(tm.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps)
    res = tm.fit(
        nsteps, 0.1, test_data=tm.data, optimizer=opt, scheduler=sched
    )

    print('Evaluating log-scores')

    left_out_hf = []
    for j in range(10):
        for i in range(10):
            left_out_hf.append(int(90*j + 3*i))
        
    left_out_mf = []
    for j in range(5):
        for i in range(5):
            left_out_mf.append(int(20*j + 2*i))
    
    ls = torch.zeros(50,)
    gspec = {"wspace": 0.1, "hspace": 0.1}
    for i, test in enumerate(obs_test_ord):
        gc.collect()
        if i % 10 == 0:
            print(i)
        sc = tm.score(test, 0)
        sc1 = -sc[:25].sum()
        sc_hf = sc[125:].detach().numpy()
        sc2 = -np.delete(sc_hf, ord_hf[left_out_hf]-125).sum()
        sc_mf = sc[25:125].detach().numpy()
        sc3 = -np.delete(sc_mf, ord_mf[left_out_mf]-25).sum()
        plt.close("all")
        ls[i] = sc1 + sc2 + sc3

    ls_mean = ls.mean()

    return ls_mean
# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

# Load data
print(os.getcwd())
data_fp = pathlib.Path('../tests/data/data_mf.pkl')
with open(data_fp, "rb") as fh:
    data_pkl: dict[str, np.ndarray] = pickle.load(fh)

data_min = pathlib.Path("../tests/data/data_mf_min.pkl")
with open(data_min, "rb") as fh:
    data_pkl_min: dict[str, np.ndarray] = pickle.load(fh)

ns = [5, 10, 20, 30, 50, 100, 200]
n_list = []
ls_list = []
for n in ns:

    print('With ensemble size')
    print(n)
    ls = get_logscore(n, data_pkl, data_pkl_min)
    n_list.append(n)
    ls_list.append(ls.item())
    print('n')
    print(n)

    print('log score')
    print(ls.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_mf_min.csv', index=False)