from __future__ import annotations

import argparse
import math
import pathlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAE baseline experiments.")
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
            nn.Conv2d(64, 64, 3),
        )
        flat = _conv_flat_dim(self.conv, (1, 5, 5))
        self.fc_mu = nn.Linear(flat, z_dim)
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
            nn.Conv2d(64, 64, 3),
        )
        flat = _conv_flat_dim(self.conv, (1, 10, 10))
        self.fc_mu = nn.Linear(flat, z_dim)
        self.fc_logvar = nn.Linear(flat, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(logvar, -20, 20)


class HEncoder(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        flat = _conv_flat_dim(self.conv, (1, 30, 30))
        self.fc_mu = nn.Linear(flat, z_dim)
        self.fc_logvar = nn.Linear(flat, z_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(logvar, -20, 20)


class JointDecoder3(nn.Module):
    def __init__(self, z_s=5, z_m=10, z_h=100):
        super().__init__()
        z_tot = z_s + z_m + z_h
        self.fc = nn.Linear(z_tot, 256 * 4 * 4)
        self.sf_head = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Upsample(size=(5, 5), mode="bilinear", align_corners=False),
        )
        self.mf_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Upsample(size=(10, 10), mode="bilinear", align_corners=False),
        )
        self.hf_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Upsample(size=(30, 30), mode="bilinear", align_corners=False),
        )

    def forward(self, z_s, z_m, z_h):
        z = torch.cat([z_s, z_m, z_h], dim=1)
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.sf_head(h), self.mf_head(h), self.hf_head(h)


class TFVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sf_enc = SEncoder()
        self.mf_enc = MEncoder()
        self.hf_enc = HEncoder()
        self.dec = JointDecoder3()

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
        return (*self.dec(z_s, z_m, z_h), s_mu, s_lv, m_mu, m_lv, h_mu, h_lv)

    def loss_fn(
        self, s_rec, m_rec, h_rec, s_mu, s_lv, m_mu, m_lv, h_mu, h_lv, y_s, y_m, y_h
    ):
        mse_s = F.mse_loss(s_rec, y_s)
        mse_m = F.mse_loss(m_rec, y_m)
        mse_h = F.mse_loss(h_rec, y_h)

        def kl(mu, lv):
            return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

        return (
            mse_s
            + mse_m
            + mse_h
            + 0.1 * (kl(s_mu, s_lv) + kl(m_mu, m_lv) + kl(h_mu, h_lv))
        )


class TriFidelityDataset(Dataset):
    def __init__(
        self, y_sf, y_mf, y_hf, sf_shape=(5, 5), mf_shape=(10, 10), hf_shape=(30, 30)
    ):
        assert len(y_sf) == len(y_mf) == len(y_hf)
        self.y_sf, self.y_mf, self.y_hf = (
            t.float().contiguous() for t in (y_sf, y_mf, y_hf)
        )
        self.sf_shape, self.mf_shape, self.hf_shape = sf_shape, mf_shape, hf_shape

    def __len__(self):
        return len(self.y_sf)

    def __getitem__(self, idx):
        return (
            self.y_sf[idx].view(1, *self.sf_shape),
            self.y_mf[idx].view(1, *self.mf_shape),
            self.y_hf[idx].view(1, *self.hf_shape),
        )


def compute_log_likelihood_tri(model, y_s, y_m, y_h, num_samples=5000, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    y_s_img = y_s.to(device).view(1, 1, 5, 5)
    y_m_img = y_m.to(device).view(1, 1, 10, 10)
    y_h_img = y_h.to(device).view(1, 1, 30, 30)
    with torch.no_grad():
        s_mu, s_lv = model.sf_enc(y_s_img)
        m_mu, m_lv = model.mf_enc(y_m_img)
        h_mu, h_lv = model.hf_enc(y_h_img)
        z_sdim, z_mdim, z_hdim = s_mu.size(1), m_mu.size(1), h_mu.size(1)
        s_mu, s_lv = s_mu.expand(num_samples, -1), s_lv.expand(num_samples, -1)
        m_mu, m_lv = m_mu.expand(num_samples, -1), m_lv.expand(num_samples, -1)
        h_mu, h_lv = h_mu.expand(num_samples, -1), h_lv.expand(num_samples, -1)
        z_s = s_mu + torch.randn_like(s_mu) * torch.exp(0.5 * s_lv)
        z_m = m_mu + torch.randn_like(m_mu) * torch.exp(0.5 * m_lv)
        z_h = h_mu + torch.randn_like(h_mu) * torch.exp(0.5 * h_lv)

        def diag_log_prob(z, mu, lv):
            return -0.5 * ((z - mu) ** 2 / lv.exp() + lv + math.log(2 * math.pi)).sum(
                dim=1
            )

        log_q_z = (
            diag_log_prob(z_s, s_mu, s_lv)
            + diag_log_prob(z_m, m_mu, m_lv)
            + diag_log_prob(z_h, h_mu, h_lv)
        )
        log_p_z = (
            -0.5 * (z_s**2).sum(dim=1)
            - 0.5 * z_sdim * math.log(2 * math.pi)
            + -0.5 * (z_m**2).sum(dim=1)
            - 0.5 * z_mdim * math.log(2 * math.pi)
            + -0.5 * (z_h**2).sum(dim=1)
            - 0.5 * z_hdim * math.log(2 * math.pi)
        )
        s_rec, m_rec, h_rec = model.dec(z_s, z_m, z_h)
        y_s_exp, y_m_exp, y_h_exp = (
            y_s_img.expand(num_samples, -1, -1, -1),
            y_m_img.expand(num_samples, -1, -1, -1),
            y_h_img.expand(num_samples, -1, -1, -1),
        )
        mse_s = ((y_s_exp - s_rec) ** 2).view(num_samples, -1).sum(dim=1)
        mse_m = ((y_m_exp - m_rec) ** 2).view(num_samples, -1).sum(dim=1)
        mse_h = ((y_h_exp - h_rec) ** 2).view(num_samples, -1).sum(dim=1)
        log_p_y_given_z = (
            -0.5 * mse_s
            - 0.5 * 25 * math.log(2 * math.pi)
            + -0.5 * mse_m
            - 0.5 * 100 * math.log(2 * math.pi)
            + -0.5 * mse_h
            - 0.5 * 900 * math.log(2 * math.pi)
        )
        log_w = log_p_y_given_z + log_p_z - log_q_z
        max_log_w = torch.max(log_w)
        return (max_log_w + torch.log(torch.exp(log_w - max_log_w).mean())).item()


def get_multifidelity_scores(n: int, data_pkl: dict) -> torch.Tensor:
    obs_lf = data_pkl["obs_lf"]
    obs_mf = data_pkl["obs_mf"]
    obs_hf = data_pkl["obs_hf"]
    mean_lf, sd_lf = obs_lf[:n, :].mean(dim=0, keepdim=True), obs_lf[:n, :].std(
        dim=0, keepdim=True
    )
    mean_mf, sd_mf = obs_mf[:n, :].mean(dim=0, keepdim=True), obs_mf[:n, :].std(
        dim=0, keepdim=True
    )
    mean_hf, sd_hf = obs_hf[:n, :].mean(dim=0, keepdim=True), obs_hf[:n, :].std(
        dim=0, keepdim=True
    )
    y_hf_train = (obs_hf[:n, :].float() - mean_hf) / sd_hf
    y_hf_test = (obs_hf[200:250, :].float() - mean_hf) / sd_hf
    y_mf_train = (obs_mf[:n, :].float() - mean_mf) / sd_mf
    y_mf_test = (obs_mf[200:250, :].float() - mean_mf) / sd_mf
    y_lf_train = (obs_lf[:n, :].float() - mean_lf) / sd_lf
    y_lf_test = (obs_lf[200:250, :].float() - mean_lf) / sd_lf
    train_ld = DataLoader(
        TriFidelityDataset(y_lf_train, y_mf_train, y_hf_train),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2000):
        for y_s, y_m, y_h in train_ld:
            y_s, y_m, y_h = (t.to(device) for t in (y_s, y_m, y_h))
            opt.zero_grad()
            out = model(y_s, y_m, y_h)
            loss = model.loss_fn(*out, y_s, y_m, y_h)
            loss.backward()
            opt.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:03d}  loss {loss.item():.4f}")
    log_lls = []
    for i, (y_s, y_m, y_h) in enumerate(zip(y_lf_test, y_mf_test, y_hf_test)):
        print(i)
        log_lls.append(
            compute_log_likelihood_tri(
                model, y_s.float(), y_m.float(), y_h.float(), num_samples=5000
            )
        )
    return -torch.tensor(log_lls)


class LFEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._flat = 128
        self.fc_mu = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(-1, self._flat)
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(lv, -20, 20)


class HFEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._flat = 256
        self.fc_mu = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(-1, self._flat)
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        return mu, torch.clamp(lv, -20, 20)


class JointDecoder(nn.Module):
    def __init__(self, lf_latent=16, hf_latent=256):
        super().__init__()
        z_dim = lf_latent + hf_latent
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)
        self.lf_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Upsample(size=(24, 14), mode="bilinear", align_corners=False),
        )
        self.hf_head = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Upsample(size=(280, 280), mode="bilinear", align_corners=False),
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
        return (*self.decoder(z_lf, z_hf), lf_mu, lf_lv, hf_mu, hf_lv)

    def loss_fn(
        self, lf_rec, hf_rec, lf_mu, lf_lv, hf_mu, hf_lv, y_lf, y_hf, kl_weight=0.1
    ):
        mse_lf = F.mse_loss(lf_rec, y_lf)
        mse_hf = F.mse_loss(hf_rec, y_hf)

        def kl(mu, lv):
            return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

        return mse_lf + mse_hf + kl_weight * (kl(lf_mu, lf_lv) + kl(hf_mu, hf_lv))


class BiFiDataset(Dataset):
    def __init__(self, lf_vectors, hf_vectors, lf_shape=(24, 14), hf_shape=(280, 280)):
        assert len(lf_vectors) == len(hf_vectors)
        self.lf = lf_vectors.float().contiguous()
        self.hf = hf_vectors.float().contiguous()
        self.lf_shape, self.hf_shape = lf_shape, hf_shape

    def __len__(self):
        return len(self.lf)

    def __getitem__(self, idx):
        return self.lf[idx].view(1, *self.lf_shape), self.hf[idx].view(
            1, *self.hf_shape
        )


def climate_log_score(model, y_lf_vec, y_hf_vec, N=2000, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    dtype = next(model.parameters()).dtype
    y_lf = y_lf_vec.to(device=device, dtype=dtype).view(1, 1, 24, 14)
    y_hf = y_hf_vec.to(device=device, dtype=dtype).view(1, 1, 280, 280)
    mu_lf, lv_lf = model.lf_encoder(y_lf)
    mu_hf, lv_hf = model.hf_encoder(y_hf)
    z_ldim, z_hdim = mu_lf.size(1), mu_hf.size(1)
    mu_lf, lv_lf = mu_lf.expand(N, -1), lv_lf.expand(N, -1)
    mu_hf, lv_hf = mu_hf.expand(N, -1), lv_hf.expand(N, -1)
    std_lf, std_hf = torch.exp(0.5 * lv_lf), torch.exp(0.5 * lv_hf)
    z_lf = mu_lf + std_lf * torch.randn_like(std_lf)
    z_hf = mu_hf + std_hf * torch.randn_like(std_hf)

    def diag_log_prob(z, mu, lv):
        return -0.5 * ((z - mu) ** 2 / lv.exp() + lv + math.log(2 * math.pi)).sum(dim=1)

    log_q = diag_log_prob(z_lf, mu_lf, lv_lf) + diag_log_prob(z_hf, mu_hf, lv_hf)
    log_pz = (
        -0.5 * (z_lf**2).sum(1)
        - 0.5 * z_ldim * math.log(2 * math.pi)
        + -0.5 * (z_hf**2).sum(1)
        - 0.5 * z_hdim * math.log(2 * math.pi)
    )
    lf_rec, hf_rec = model.decoder(z_lf, z_hf)
    y_lf_b, y_hf_b = y_lf.expand(N, -1, -1, -1), y_hf.expand(N, -1, -1, -1)
    mse_lf = ((y_lf_b - lf_rec) ** 2).view(N, -1).sum(1)
    mse_hf = ((y_hf_b - hf_rec) ** 2).view(N, -1).sum(1)
    log_p_y_given_z = (
        -0.5 * mse_lf
        - 0.5 * 336 * math.log(2 * math.pi)
        + -0.5 * mse_hf
        - 0.5 * 78400 * math.log(2 * math.pi)
    )
    log_w = log_p_y_given_z + log_pz - log_q
    max_log_w = torch.max(log_w)
    return (max_log_w + torch.log(torch.exp(log_w - max_log_w).mean())).item()


def get_climate_scores(n: int) -> torch.Tensor:
    obs_gcm = torch.from_numpy(np.load("../tests/data/obs_gcm.npy")).float()
    obs_rcm = torch.from_numpy(np.load("../tests/data/obs_rcm.npy")).float()
    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))
    mean_gcm = obs_gcm[train_idx[:n], :].mean(dim=0, keepdim=True)
    sd_gcm = obs_gcm[train_idx[:n], :].std(dim=0, keepdim=True)
    mean_rcm = obs_rcm[train_idx[:n], :].mean(dim=0, keepdim=True)
    sd_rcm = obs_rcm[train_idx[:n], :].std(dim=0, keepdim=True)
    y_rcm_train = (obs_rcm[train_idx[:n], :].float() - mean_rcm) / sd_rcm
    y_rcm_test = (obs_rcm[test_idx, :].float() - mean_rcm) / sd_rcm
    y_gcm_train = (obs_gcm[train_idx[:n], :].float() - mean_gcm) / sd_gcm
    y_gcm_test = (obs_gcm[test_idx, :].float() - mean_gcm) / sd_gcm
    train_ld = DataLoader(
        BiFiDataset(y_gcm_train, y_rcm_train),
        16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BFVAE(lf_latent=16, hf_latent=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for ep in range(500):
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
        print(f"epoch {ep+1:03d}  train-loss {running/len(train_ld.dataset):.4f}")
    scores = []
    for i, (v_lf, v_hf) in enumerate(zip(y_gcm_test, y_rcm_test)):
        print(i)
        scores.append(climate_log_score(model, v_lf, v_hf, N=2000))
    return -torch.tensor(scores)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows: list[dict] = []

    if args.experiment == "climate":
        test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
        for n in parse_ns(args.ns, args.experiment):
            print("With ensemble size")
            print(n)
            scores = get_climate_scores(n)
            for i, idx in enumerate(test_idx):
                rows.append(
                    {"n": n, "test_idx": int(idx), "logscore_total": scores[i].item()}
                )
            save_rows(
                rows,
                args.output_dir,
                "VAE",
                args.experiment,
                args.include_all_logscores,
            )
            print("mean log score")
            print(scores.mean().item())
        return

    data_fp = (
        pathlib.Path("../tests/data/data_mf.pkl")
        if args.experiment == "linear"
        else pathlib.Path("../tests/data/data_mf_min.pkl")
    )
    with open(data_fp, "rb") as fh:
        data_pkl = pickle.load(fh)

    for n in parse_ns(args.ns, args.experiment):
        print("With ensemble size")
        print(n)
        scores = get_multifidelity_scores(n, data_pkl)
        for i in range(scores.numel()):
            rows.append({"n": n, "test_idx": i, "logscore_total": scores[i].item()})
        save_rows(
            rows, args.output_dir, "VAE", args.experiment, args.include_all_logscores
        )
        print("mean log score")
        print(scores.mean().item())


if __name__ == "__main__":
    main()
