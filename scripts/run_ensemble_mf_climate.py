import os
import gc
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Packages for working with array data and tensors]
import numpy as np
import torch
import numpy as np

# Packages for building transport maps
from veccs.orderings import maxmin_pred_cpp, find_nns_l2_mf
import scipy.spatial.distance

from batram.data import MultiFidelityData, AugmentDataMF
from batram.mf import MultiFidelityTM
import pandas as pd

def get_logscore(n, locs_gcm, locs_rcm, obs_gcm, obs_rcm):
    
    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    obs_gcm_train = torch.Tensor(obs_gcm[train_idx[0:n], :])
    obs_rcm_train = torch.Tensor(obs_rcm[train_idx[0:n], :])

    obs_gcm_test = torch.Tensor(obs_gcm[test_idx, :])
    obs_rcm_test = torch.Tensor(obs_rcm[test_idx, :])

    mean_gcm = obs_gcm_train.mean(dim=0, keepdim = True)
    sd_gcm = obs_gcm_train.std(dim=0, keepdim= True)
    mean_rcm = obs_rcm_train.mean(dim=0, keepdim=True)
    sd_rcm = obs_rcm_train.std(dim = 0, keepdim=True)
    train_gcm = (obs_gcm_train - mean_gcm)/sd_gcm
    train_rcm = (obs_rcm_train - mean_rcm)/sd_rcm
    test_gcm = (obs_gcm_test - mean_gcm)/sd_gcm
    test_rcm = (obs_rcm_test - mean_rcm)/sd_rcm
    train = torch.hstack((train_gcm, train_rcm))
    test = torch.hstack((test_gcm, test_rcm))

    print('Ordering locations...')

    locs = torch.vstack((torch.Tensor(locs_gcm), torch.Tensor(locs_rcm)))

    # Maximin ordering of the locations using veccs
    ord = maxmin_pred_cpp(locs_gcm, locs_rcm)

    ord_gcm = ord[:336]
    ord_rcm = ord[336:]

    locs_ord = locs[ord, ...]
    obs_ord = train[..., ord]
    obs_test_ord = test[..., ord]
    obs_ord_gcm = train_gcm[..., ord_gcm]
    # indexes from 0 for easier indexing
    obs_ord_hf = train_rcm[..., ord_rcm - 336]

    # Finding nearest neighbors using the `veccs` package.
    # For this example let's use 20 max nearest neighbors. 
    # For the multifidelity version, pass a list of the ordered locs from
    # lowest to highest fidelity
    locs_all = [locs_ord[:336].detach().numpy(), locs_ord[336:].detach().numpy()]
    largest_conditioning_set = 50
    nn = find_nns_l2_mf(locs_all, largest_conditioning_set)

    fidelity_sizes = torch.as_tensor(list(map(len, locs_all)))
    data = MultiFidelityData.new(locs_ord, obs_ord, torch.as_tensor(nn), fidelity_sizes)

    R = 2  # Number of fidelities
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
    theta_init[7*R-1:8*R-2] = 0.0
    # Theta gammas
    theta_init[8*R-2:9*R-2] = -1.0

    print('Training transport map...')

    # nug_mult = 4 as in the paper
    tm = MultiFidelityTM(data, theta_init, nug_mult = 4.0)
    # The `nsteps` argument is always required. When using a user-defined optimizer
    # we ignore the initial learning rate. The `batch_size` specifies how to perform
    # minibatch gradient descent. The `test_data` argument is optional and is used
    # to compute the test loss at each step.
    nsteps = 200
    opt = torch.optim.Adam(tm.parameters(), lr=0.01, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps, eta_min = 0.001)
    res = tm.fit(
        nsteps, 0.01
    )

    print('Computing log-scores...')

    ls = torch.zeros(50,)
    gspec = {"wspace": 0.1, "hspace": 0.1}
    tm.eval()
    for i in range(10):
        gc.collect()
        if i % 3 == 0:
            print(i)
        obs_now = obs_test_ord[i].to(torch.float64)
        sc = tm.score(obs_now, 0)
        ls[i] = -sc.sum()

    ls_mean = ls.mean()

    return ls_mean

# Weird behavior if lat/lon are not in expected order
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

# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

locs_gcm = np.load("../tests/data/locs_gcm.npy")
locs_rcm = np.load("../tests/data/locs_rcm.npy")

locs_gcm = force_latlon_first(locs_gcm)
locs_rcm = force_latlon_first(locs_rcm)

obs_gcm = np.load("../tests/data/obs_gcm.npy")
obs_rcm = np.load("../tests/data/obs_rcm.npy")

ns = [5, 10, 15, 20, 25, 30, 35, 40]
n_list = []
ls_list = []

for n in ns:

    print('With ensemble size')
    print(n)
    ls = get_logscore(n, locs_gcm, locs_rcm, obs_gcm, obs_rcm)
    n_list.append(n)
    ls_list.append(ls.item())
    print('n')
    print(n)

    print('log score')
    print(ls.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_mf_climate.csv', index=False)