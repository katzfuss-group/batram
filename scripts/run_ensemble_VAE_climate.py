import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class LFEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(  # output ≈ 3×2
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 24×14 → 12×7
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 12×7  → 6×4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 6×4   → 3×2
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 3×2 → 1×1
        )
        self._flat = 128  # channels after pool
        self.fc_mu = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x):  # (B,1,24,14)
        h = self.conv(x).view(-1, self._flat)
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(lv, -20, 20)


# ─────────────────── HF : 280 × 280  →  z = 256
class HFEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(  # 280 → 9 after 5 stride-2 blocks
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 280→140
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 140→70
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 70 →35
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 35 →18
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),  # 18 →9
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 9 × 9 → 1 × 1
        )
        self._flat = 256
        self.fc_mu = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x):  # (B,1,280,280)
        h = self.conv(x).view(-1, self._flat)
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(lv, -20, 20)


class JointDecoder(nn.Module):
    """
    Input  : z = [z_lf | z_hf]   →  (lf_latent + hf_latent)
    Output :                       lf 1×24×14   and   hf 1×280×280
    """

    def __init__(self, lf_latent=16, hf_latent=256):
        super().__init__()
        z_dim = lf_latent + hf_latent  # 272
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)  # shared seed 4×4

        # LF head -> 24×14
        self.lf_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),  # 16×16
            nn.Upsample(
                size=(24, 14), mode="bilinear", align_corners=False
            ),  # 16×16→24×14
        )

        # HF head -> 280×280
        self.hf_head = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 128→256
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),  # 256×256
            nn.Upsample(
                size=(280, 280), mode="bilinear", align_corners=False
            ),  # 256→280
        )

    def forward(self, z_lf, z_hf):
        z = torch.cat([z_lf, z_hf], dim=1)
        h0 = self.fc(z).view(-1, 512, 4, 4)
        return self.lf_head(h0), self.hf_head(h0)


class BFVAE(nn.Module):
    def __init__(self, lf_latent=16, hf_latent=256):
        super().__init__()
        self.lf_encoder = LFEncoder(lf_latent)
        self.hf_encoder = HFEncoder(hf_latent)
        self.decoder = JointDecoder(lf_latent, hf_latent)

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + torch.randn_like(std) * std

    def forward(self, y_lf, y_hf):
        lf_mu, lf_lv = self.lf_encoder(y_lf)
        hf_mu, hf_lv = self.hf_encoder(y_hf)

        z_lf = self.reparam(lf_mu, lf_lv)
        z_hf = self.reparam(hf_mu, hf_lv)

        lf_rec, hf_rec = self.decoder(z_lf, z_hf)
        return (lf_rec, hf_rec, lf_mu, lf_lv, hf_mu, hf_lv)

    # pixel-wise MSE (unit variance likelihood) + KL
    def loss_fn(
        self, lf_rec, hf_rec, lf_mu, lf_lv, hf_mu, hf_lv, y_lf, y_hf, kl_weight=0.1
    ):
        mse_lf = F.mse_loss(lf_rec, y_lf)
        mse_hf = F.mse_loss(hf_rec, y_hf)

        def kl(mu, lv):
            return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

        kl_lf, kl_hf = kl(lf_mu, lf_lv), kl(hf_mu, hf_lv)

        return mse_lf + mse_hf + kl_weight * (kl_lf + kl_hf)


class BiFiDataset(Dataset):
    def __init__(self, lf_vectors, hf_vectors, lf_shape=(24, 14), hf_shape=(280, 280)):
        assert len(lf_vectors) == len(hf_vectors)
        self.lf = lf_vectors.float().contiguous()
        self.hf = hf_vectors.float().contiguous()
        self.lf_shape, self.hf_shape = lf_shape, hf_shape

    def __len__(self):
        return len(self.lf)

    def __getitem__(self, idx):
        x_lf = self.lf[idx].view(1, *self.lf_shape)  # (1,24,14)
        x_hf = self.hf[idx].view(1, *self.hf_shape)  # (1,280,280)
        return x_lf, x_hf


@torch.no_grad()
def log_score(model, y_lf_vec, y_hf_vec, N=5_000, device=None):
    """
    importance-sampled  log p(y_lf, y_hf)  for *one* pair
    y_lf_vec : (336,)  -- flattened 24×14
    y_hf_vec : (~78k,) -- flattened 280×280
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # reshape to images
    y_lf = y_lf_vec.to(device).view(1, 1, 24, 14)
    y_hf = y_hf_vec.to(device).view(1, 1, 280, 280)

    # encoder posteriors
    μ_lf, lv_lf = model.lf_encoder(y_lf)  # (1, z_lf)
    μ_hf, lv_hf = model.hf_encoder(y_hf)  # (1, z_hf)
    z_ldim, z_hdim = μ_lf.size(1), μ_hf.size(1)

    # expand to (N, z_dim)
    μ_lf = μ_lf.expand(N, -1)
    lv_lf = lv_lf.expand(N, -1)
    μ_hf = μ_hf.expand(N, -1)
    lv_hf = lv_hf.expand(N, -1)

    std_lf = torch.exp(0.5 * lv_lf)
    std_hf = torch.exp(0.5 * lv_hf)

    # reparameterised samples
    z_lf = μ_lf + std_lf * torch.randn_like(std_lf)
    z_hf = μ_hf + std_hf * torch.randn_like(std_hf)

    # log q(z|y)
    def diag_log_prob(z, μ, lv):
        return (-0.5 * (((z - μ) ** 2) / lv.exp() + lv + math.log(2 * math.pi))).sum(1)

    log_q = diag_log_prob(z_lf, μ_lf, lv_lf) + diag_log_prob(z_hf, μ_hf, lv_hf)

    # log p(z)  (standard normals)
    log_pz = (
        -0.5 * (z_lf**2).sum(1)
        - 0.5 * z_ldim * math.log(2 * math.pi)
        + -0.5 * (z_hf**2).sum(1)
        - 0.5 * z_hdim * math.log(2 * math.pi)
    )

    # decode
    lf_rec, hf_rec = model.decoder(z_lf, z_hf)

    # broadcast observations
    y_lf_b = y_lf.expand(N, -1, -1, -1)
    y_hf_b = y_hf.expand(N, -1, -1, -1)

    mse_lf = ((y_lf_b - lf_rec) ** 2).view(N, -1).sum(1)
    mse_hf = ((y_hf_b - hf_rec) ** 2).view(N, -1).sum(1)

    const_lf = -0.5 * 336 * math.log(2 * math.pi)
    const_hf = -0.5 * 78400 * math.log(2 * math.pi)
    log_py_z = -0.5 * mse_lf + const_lf + -0.5 * mse_hf + const_hf

    # importance weights
    log_w = log_py_z + log_pz - log_q
    max_log_w = torch.max(log_w)
    log_score = max_log_w + torch.log(torch.exp(log_w - max_log_w).mean())

    return log_score.item()


def get_logscore(n):
    obs_gcm = torch.from_numpy(np.load("../tests/data/obs_gcm.npy"))
    obs_rcm = torch.from_numpy(np.load("../tests/data/obs_rcm.npy"))
    obs_gcm = obs_gcm.to(torch.float32)
    obs_rcm = obs_rcm.to(torch.float32)

    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    mean_gcm = obs_gcm[train_idx[0:n], :].mean(dim=0, keepdim=True)
    sd_gcm = obs_gcm[train_idx[0:n], :].std(dim=0, keepdim=True)

    mean_rcm = obs_rcm[train_idx[0:n], :].mean(dim=0, keepdim=True)
    sd_rcm = obs_rcm[train_idx[0:n], :].std(dim=0, keepdim=True)

    y_rcm_train = (obs_rcm[train_idx[0:n], :].to(torch.float) - mean_rcm) / sd_rcm
    y_rcm_test = (obs_rcm[test_idx, :].to(torch.float) - mean_rcm) / sd_rcm

    y_gcm_train = (obs_gcm[train_idx[0:n], :].to(torch.float) - mean_gcm) / sd_gcm
    y_gcm_test = (obs_gcm[test_idx, :].to(torch.float) - mean_gcm) / sd_gcm

    print("Low fidelity shape sanity check")
    print(y_gcm_train.shape)

    batch_size = 16
    train_ds = BiFiDataset(y_gcm_train, y_rcm_train)  # your tensors
    train_ld = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BFVAE(lf_latent=16, hf_latent=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = 500

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for y_lf, y_hf in train_ld:
            y_lf, y_hf = y_lf.to(device), y_hf.to(device)

            opt.zero_grad()
            out = model(y_lf, y_hf)
            loss = model.loss_fn(*out, y_lf, y_hf, kl_weight=0.1)
            loss.backward()
            opt.step()
            running += loss.item() * y_lf.size(0)

        print(f"epoch {ep:03d}  train-loss {running/len(train_ds):.4f}")

    scores = []
    print("Computing log likelihoods")
    for i, (v_lf, v_hf) in enumerate(zip(y_gcm_test, y_rcm_test)):
        print(i)
        s = log_score(model, v_lf, v_hf, N=2_000)
        scores.append(s)

    print("Log likelihoods computed")
    return -torch.Tensor(scores).mean()


# after the imports set a seed for reproducibility
# anyhow, the results will be different on different machines
# cf. https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.manual_seed(0)

ns = [5, 10, 15, 20, 25, 30, 35, 40]
n_list = []
ls_list = []
for n in ns:
    print("With ensemble size")
    print(n)
    sc = get_logscore(n)
    n_list.append(n)
    ls_list.append(sc.item())
    print("n")
    print(n)

    print("Log score")
    print(sc.item())

    my_dict = {"n": n_list, "logscore": ls_list}
    df = pd.DataFrame.from_dict(my_dict)
    df.to_csv("./results/logscores_VAE_climate.csv", index=False)
