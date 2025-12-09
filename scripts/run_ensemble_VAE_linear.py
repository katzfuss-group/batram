import pathlib
import pickle
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from torch.distributions import Normal, kl_divergence   

import pandas as pd


def _conv_flat_dim(conv_block, in_shape):
    with torch.no_grad():
        return conv_block(torch.zeros(1, *in_shape)).numel()

class SEncoder(nn.Module):           
    def __init__(self, z_dim=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3)                       
        )
        flat = _conv_flat_dim(self.conv, (1, 5, 5))
        self.fc_mu     = nn.Linear(flat, z_dim)
        self.fc_logvar = nn.Linear(flat, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(logvar, -20, 20)


class MEncoder(nn.Module):           
    def __init__(self, z_dim=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3)                       
        )
        flat = _conv_flat_dim(self.conv, (1, 10, 10))
        self.fc_mu     = nn.Linear(flat, z_dim)
        self.fc_logvar = nn.Linear(flat, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(logvar, -20, 20)


class HEncoder(nn.Module):           
    def __init__(self, z_dim=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64,128, 4, stride=2, padding=1), 
            nn.ReLU()
        )
        flat = _conv_flat_dim(self.conv, (1, 30, 30))
        self.fc_mu     = nn.Linear(flat, z_dim)
        self.fc_logvar = nn.Linear(flat, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(logvar, -20, 20)
    
class JointDecoder3(nn.Module):
    """
    Input  : concatenated z = [z_s | z_m | z_h]  (5+10+100 = 115)
    Outputs:   sf 1×5×5,   mf 1×10×10,   hf 1×30×30
    """
    def __init__(self, z_s=5, z_m=10, z_h=100):
        super().__init__()
        z_tot = z_s + z_m + z_h
        self.fc = nn.Linear(z_tot, 256*4*4)             

        # heads
        self.sf_head = nn.Sequential(                   
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),            
            nn.Upsample(size=(5,5), mode='bilinear', align_corners=False)
        )

        self.mf_head = nn.Sequential(                   
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),            
            nn.Upsample(size=(10,10), mode='bilinear', align_corners=False)
        )

        self.hf_head = nn.Sequential(                   
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(256,128, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64,1,3,padding=1),
            nn.Upsample(size=(30,30), mode='bilinear', align_corners=False)
        )

    def forward(self, z_s, z_m, z_h):
        z = torch.cat([z_s, z_m, z_h], dim=1)           # B×115
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.sf_head(h), self.mf_head(h), self.hf_head(h)
    
class TFVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sf_enc = SEncoder()     # 5D
        self.mf_enc = MEncoder()     # 10D
        self.hf_enc = HEncoder()     # 100D
        self.dec    = JointDecoder3()

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, y_s, y_m, y_h):
        s_mu, s_lv = self.sf_enc(y_s)
        m_mu, m_lv = self.mf_enc(y_m)
        h_mu, h_lv = self.hf_enc(y_h)

        z_s = self.reparam(s_mu, s_lv)
        z_m = self.reparam(m_mu, m_lv)
        z_h = self.reparam(h_mu, h_lv)

        s_rec, m_rec, h_rec = self.dec(z_s, z_m, z_h)
        return (s_rec, m_rec, h_rec,
                s_mu, s_lv, m_mu, m_lv, h_mu, h_lv)

    # pixel-wise MSE + weighted KLs (tune the weights if needed)
    def loss_fn(self, s_rec, m_rec, h_rec,
                s_mu, s_lv, m_mu, m_lv, h_mu, h_lv,
                y_s, y_m, y_h):
        mse_s = F.mse_loss(s_rec, y_s)
        mse_m = F.mse_loss(m_rec, y_m)
        mse_h = F.mse_loss(h_rec, y_h)

        kl   = lambda mu, lv: -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        kl_s = kl(s_mu, s_lv)
        kl_m = kl(m_mu, m_lv)
        kl_h = kl(h_mu, h_lv)

        return mse_s + mse_m + mse_h + 0.1*(kl_s + kl_m + kl_h)
    
class TriFidelityDataset(Dataset):
    def __init__(self, y_sf, y_mf, y_hf,         # sf=low, mf=mid, hf=high
                 sf_shape=(5, 5),
                 mf_shape=(10, 10),
                 hf_shape=(30, 30)):
        assert len(y_sf) == len(y_mf) == len(y_hf)
        self.y_sf, self.y_mf, self.y_hf = [t.float().contiguous()
                                           for t in (y_sf, y_mf, y_hf)]
        self.sf_shape, self.mf_shape, self.hf_shape = sf_shape, mf_shape, hf_shape

    def __len__(self): return len(self.y_sf)

    def __getitem__(self, idx):
        sf = self.y_sf[idx].view(1, *self.sf_shape)   
        mf = self.y_mf[idx].view(1, *self.mf_shape)   
        hf = self.y_hf[idx].view(1, *self.hf_shape)   
        return sf, mf, hf
    
def compute_log_likelihood_tri(model,
                               y_s, y_m, y_h,          # vectors: 25 / 100 / 900
                               num_samples=5_000,
                               device=None):
    """
    Importance-sampled log p(y_s, y_m, y_h).

    y_s : (25,)   small-fidelity vector
    y_m : (100,)  mid-fidelity vector
    y_h : (900,)  high-fidelity vector
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # reshape to the image layout expected by the encoders
    y_s_img = y_s.to(device).view(1, 1,  5,  5)
    y_m_img = y_m.to(device).view(1, 1, 10, 10)
    y_h_img = y_h.to(device).view(1, 1, 30, 30)

    with torch.no_grad():
        # posterior q(z | y)  
        s_mu, s_lv = model.sf_enc(y_s_img)          # (1, z_s)
        m_mu, m_lv = model.mf_enc(y_m_img)          # (1, z_m)
        h_mu, h_lv = model.hf_enc(y_h_img)          # (1, z_h)
        z_sdim, z_mdim, z_hdim = s_mu.size(1), m_mu.size(1), h_mu.size(1)

        # broadcast mu, log variance
        s_mu  = s_mu.expand(num_samples, -1)          
        s_lv  = s_lv.expand(num_samples, -1)
        m_mu  = m_mu.expand(num_samples, -1)
        m_lv  = m_lv.expand(num_samples, -1)
        h_mu  = h_mu.expand(num_samples, -1)
        h_lv  = h_lv.expand(num_samples, -1)

        # reparameterization trick
        eps_s = torch.randn_like(s_mu)
        eps_m = torch.randn_like(m_mu)
        eps_h = torch.randn_like(h_mu)
        z_s = s_mu + eps_s * torch.exp(0.5 * s_lv)   # (N, z_s)
        z_m = m_mu + eps_m * torch.exp(0.5 * m_lv)   # (N, z_m)
        z_h = h_mu + eps_h * torch.exp(0.5 * h_lv)   # (N, z_h)

        # log q(z|y) (evaluate posterior density at sampled z)
        def diag_normal_log_prob(z, mu, lv):
            return (-0.5 * ( (z - mu)**2 / lv.exp() + lv + math.log(2*math.pi))
                   ).sum(dim=1)                      # (N,)
        log_q_z = (diag_normal_log_prob(z_s, s_mu, s_lv) +
                   diag_normal_log_prob(z_m, m_mu, m_lv) +
                   diag_normal_log_prob(z_h, h_mu, h_lv))           # (N,)

        # prior log p(z)
        log_p_z = (-0.5 * (z_s**2).sum(dim=1) - 0.5*z_sdim*math.log(2*math.pi) +
                   -0.5 * (z_m**2).sum(dim=1) - 0.5*z_mdim*math.log(2*math.pi) +
                   -0.5 * (z_h**2).sum(dim=1) - 0.5*z_hdim*math.log(2*math.pi))

        # decode & log p(y | z)  (unit-variance pixels)
        s_rec, m_rec, h_rec = model.dec(z_s, z_m, z_h)       # batched (N,…)

        # replicate observations to (N, …) for broadcasting
        y_s_exp = y_s_img.expand(num_samples, -1, -1, -1)
        y_m_exp = y_m_img.expand(num_samples, -1, -1, -1)
        y_h_exp = y_h_img.expand(num_samples, -1, -1, -1)

        mse_s = ((y_s_exp - s_rec)**2).view(num_samples, -1).sum(dim=1)  # (N,)
        mse_m = ((y_m_exp - m_rec)**2).view(num_samples, -1).sum(dim=1)
        mse_h = ((y_h_exp - h_rec)**2).view(num_samples, -1).sum(dim=1)

        const_s = -0.5 * 25  * math.log(2*math.pi)
        const_m = -0.5 * 100 * math.log(2*math.pi)
        const_h = -0.5 * 900 * math.log(2*math.pi)

        log_p_y_given_z = (-0.5*mse_s + const_s +
                           -0.5*mse_m + const_m +
                           -0.5*mse_h + const_h)                  # (N,)

        #  importance weights
        log_w = log_p_y_given_z + log_p_z - log_q_z               # (N,)

        # log-likelihood  log(1/N Σ exp(log_w))
        max_log_w = torch.max(log_w)                              # numerical stabilisation
        log_ll = max_log_w + torch.log(torch.exp(log_w - max_log_w).mean())

        return log_ll.item()


def get_logscore(n, data_pkl):
    obs_lf = data_pkl["obs_lf"]
    obs_mf = data_pkl["obs_mf"]
    obs_hf = data_pkl["obs_hf"]

    mean_lf = obs_lf[:n, :].mean(dim=0, keepdim = True)
    sd_lf = obs_lf[:n, :].std(dim=0, keepdim= True)

    mean_mf = obs_mf[:n, :].mean(dim=0, keepdim = True)
    sd_mf = obs_mf[:n, :].std(dim=0, keepdim= True)
    
    mean_hf = obs_hf[:n, :].mean(dim=0, keepdim = True)
    sd_hf = obs_hf[:n, :].std(dim=0, keepdim= True)

    y_hf_train = (obs_hf[:n, :].to(torch.float) - mean_hf)/sd_hf
    y_hf_test = (obs_hf[200:250, :].to(torch.float) - mean_hf)/sd_hf

    y_mf_train = (obs_mf[:n, :].to(torch.float) - mean_mf)/sd_mf
    y_mf_test = (obs_mf[200:250, :].to(torch.float) - mean_mf)/sd_mf

    y_lf_train = (obs_lf[:n, :].to(torch.float) - mean_lf)/sd_lf
    y_lf_test = (obs_lf[200:250, :].to(torch.float) - mean_lf)/sd_lf

    print('Low fidelity shape sanity check')
    print(y_lf_train.shape)

    train_ds  = TriFidelityDataset(y_lf_train, y_mf_train, y_hf_train)
    train_ld  = DataLoader(train_ds, batch_size=64,
                           shuffle=True, num_workers=0, pin_memory=True)
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = TFVAE().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('Training the model')
    for epoch in range(2000):
        for y_s, y_m, y_h in train_ld:
            y_s, y_m, y_h = [t.to(device) for t in (y_s, y_m, y_h)]

            opt.zero_grad()
            out = model(y_s, y_m, y_h)
            loss = model.loss_fn(*out, y_s, y_m, y_h)
            loss.backward()
            opt.step()

        if (epoch + 1) % 200 == 0:
            print(f'Epoch {epoch+1:03d}  loss {loss.item():.4f}')

    log_lls = []
    print('Computing log likelihoods')
    for i, (y_s, y_m, y_h) in enumerate(zip(y_lf_test, y_mf_test, y_hf_test)):
        print(i)
        ll = compute_log_likelihood_tri(model,
                                        y_s.to(torch.float32),
                                        y_m.to(torch.float32),
                                        y_h.to(torch.float32),
                                        num_samples=5000)
        log_lls.append(ll)

    print('Log likelihoods computed')
    return -torch.Tensor(log_lls).mean()


# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

data_fp = pathlib.Path("../tests/data/data_mf.pkl")
with open(data_fp, "rb") as fh:
    data_pkl: dict[str, np.ndarray] = pickle.load(fh)

ns = [5, 10, 20, 30, 50, 100, 200]
n_list = []
ls_list = []
for n in ns:

    print('With ensemble size')
    print(n)
    sc = get_logscore(n, data_pkl)
    n_list.append(n)
    ls_list.append(sc.item())
    print('n')
    print(n)

    print('Log score')
    print(sc.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv('./results/logscores_VAE_linear.csv', index=False)
