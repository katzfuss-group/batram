from __future__ import annotations

import argparse
import gc
import os
import pathlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from veccs.orderings import find_nns_l2_mf, maxmin_pred_cpp

from batram.data import MultiFidelityData
from batram.mf import MultiFidelityTM

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MultiFidelityTM experiments.")
    parser.add_argument(
        "--experiment", choices=["linear", "min", "climate"], required=True
    )
    parser.add_argument("--variant", choices=["mf", "mflinear"], default="mf")
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


def build_theta_init(data: MultiFidelityData, R: int, linear: bool) -> torch.Tensor:
    theta_init = torch.zeros(9 * R - 2)
    log_2m = data.response[:, 0].square().mean().log()
    theta_init[:R] = log_2m
    theta_init[R : 2 * R] = 0.2
    theta_init[2 * R : 3 * R] = 0.0
    theta_init[3 * R : 4 * R] = 0.0
    theta_init[4 * R : 5 * R] = 0.0
    theta_init[5 * R : 6 * R] = 0.0
    theta_init[6 * R : 7 * R - 1] = 0.0
    if linear:
        theta_init[7 * R - 1 : 8 * R - 2] = 0.0
    theta_init[8 * R - 2 : 9 * R - 2] = -1.0
    return theta_init


def get_multifidelity_scores(
    n: int,
    data_pkl: dict,
    variant: str,
    locs_pkl: dict | None = None,
) -> dict[str, torch.Tensor]:
    linear_variant = variant == "mflinear"

    if locs_pkl is None:
        locs_pkl = data_pkl

    locs_lf = locs_pkl["locs_lf"]
    locs_mf = locs_pkl["locs_mf"]
    locs_hf = locs_pkl["locs_hf"]
    obs_lf = data_pkl["obs_lf"]
    obs_mf = data_pkl["obs_mf"]
    obs_hf = data_pkl["obs_hf"]

    obs_lf_train = obs_lf[:n, :]
    obs_mf_train = obs_mf[:n, :]
    obs_hf_train = obs_hf[:n, :]
    obs_lf_test = obs_lf[200:250, :]
    obs_mf_test = obs_mf[200:250, :]
    obs_hf_test = obs_hf[200:250, :]

    mean_lf = obs_lf.mean(dim=0, keepdim=True)
    sd_lf = obs_lf.std(dim=0, keepdim=True)
    mean_mf = obs_mf.mean(dim=0, keepdim=True)
    sd_mf = obs_mf.std(dim=0, keepdim=True)
    mean_hf = obs_hf.mean(dim=0, keepdim=True)
    sd_hf = obs_hf.std(dim=0, keepdim=True)

    obs_lf_train = (obs_lf_train - mean_lf) / sd_lf
    obs_mf_train = (obs_mf_train - mean_mf) / sd_mf
    obs_hf_train = (obs_hf_train - mean_hf) / sd_hf
    obs_lf_test = (obs_lf_test - mean_lf) / sd_lf
    obs_mf_test = (obs_mf_test - mean_mf) / sd_mf
    obs_hf_test = (obs_hf_test - mean_hf) / sd_hf

    obs = torch.hstack((obs_lf_train, obs_mf_train, obs_hf_train))
    obs_test = torch.hstack((obs_lf_test, obs_mf_test, obs_hf_test))

    epsilon_1 = cdist(locs_mf, locs_mf)
    epsilon_1 = epsilon_1[epsilon_1 != 0.0].min()
    locs_lf = torch.hstack((locs_lf, torch.zeros(25, 1) + epsilon_1))

    epsilon_2 = cdist(locs_hf, locs_hf)
    epsilon_2 = epsilon_2[epsilon_2 != 0.0].min()
    locs_mf = torch.hstack((locs_mf, torch.zeros(100, 1) + epsilon_2))
    locs_hf = torch.hstack((locs_hf, torch.zeros(900, 1)))

    ord_lfmf = maxmin_pred_cpp(locs_lf.detach().numpy(), locs_mf.detach().numpy())
    locs_lfmf = np.vstack((locs_lf, locs_mf))
    ord_hf = maxmin_pred_cpp(locs_lfmf, locs_hf.detach().numpy())
    ord = np.concatenate((ord_lfmf, ord_hf[125:]))
    locs = torch.vstack((locs_lf, locs_mf, locs_hf))

    ord_mf = ord[25:125] - 100
    ord_hf = ord[125:] - 125
    obs_ord = obs[..., ord]
    obs_test_ord = obs_test[..., ord]
    locs_ord = locs[ord, ...]

    locs_all = [
        locs_ord[:25].detach().numpy(),
        locs_ord[25:125].detach().numpy(),
        locs_ord[125:].detach().numpy(),
    ]
    nn = find_nns_l2_mf(locs_all, 30 if linear_variant else 50)
    fidelity_sizes = torch.as_tensor(list(map(len, locs_all)))
    data = MultiFidelityData.new(locs_ord, obs_ord, torch.as_tensor(nn), fidelity_sizes)

    theta_init = build_theta_init(data, R=3, linear=linear_variant)
    tm = MultiFidelityTM(data, theta_init, nug_mult=4.0, linear=linear_variant)
    tm.fit(200, 0.01)

    left_out_hf = [int(90 * j + 3 * i) for j in range(10) for i in range(10)]
    left_out_mf = [int(20 * j + 2 * i) for j in range(5) for i in range(5)]

    ls_total = torch.zeros(50)
    ls_lf = torch.zeros(50)
    ls_mf = torch.zeros(50)
    ls_hf = torch.zeros(50)

    for i, test in enumerate(obs_test_ord):
        gc.collect()
        if i % 10 == 0:
            print(i)
        sc = tm.score(test, 0)
        sc1 = sc[:25].sum()
        sc_mf = sc[25:125].detach().cpu().numpy()
        sc2 = np.delete(sc_mf, ord_mf[left_out_mf] - 25).sum()
        sc_hf = sc[125:].detach().cpu().numpy()
        sc3 = np.delete(sc_hf, ord_hf[left_out_hf] - 125).sum()
        plt.close("all")
        ls_lf[i] = sc1
        ls_mf[i] = sc2
        ls_hf[i] = sc3
        ls_total[i] = sc1 + sc2 + sc3

    return {"total": ls_total, "lf": ls_lf, "mf": ls_mf, "hf": ls_hf}


def force_latlon_first(locs: np.ndarray) -> np.ndarray:
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


def get_climate_scores(n: int, variant: str) -> list[dict]:
    linear_variant = variant == "mflinear"

    locs_gcm = force_latlon_first(np.load("../tests/data/locs_gcm.npy"))
    locs_rcm = force_latlon_first(np.load("../tests/data/locs_rcm.npy"))
    obs_gcm = np.load("../tests/data/obs_gcm.npy")
    obs_rcm = np.load("../tests/data/obs_rcm.npy")

    n_gcm = locs_gcm.shape[0]
    test_idx = [44, 33, 27, 1, 10, 18, 12, 29, 37, 47]
    train_idx = list(set(range(50)) - set(test_idx))

    obs_gcm_train = torch.tensor(obs_gcm[train_idx[:n], :])
    obs_rcm_train = torch.tensor(obs_rcm[train_idx[:n], :])
    obs_gcm_test = torch.tensor(obs_gcm[test_idx, :])
    obs_rcm_test = torch.tensor(obs_rcm[test_idx, :])

    mean_gcm = obs_gcm_train.mean(dim=0, keepdim=True)
    sd_gcm = obs_gcm_train.std(dim=0, keepdim=True)
    mean_rcm = obs_rcm_train.mean(dim=0, keepdim=True)
    sd_rcm = obs_rcm_train.std(dim=0, keepdim=True)

    train_gcm = (obs_gcm_train - mean_gcm) / sd_gcm
    train_rcm = (obs_rcm_train - mean_rcm) / sd_rcm
    test_gcm = (obs_gcm_test - mean_gcm) / sd_gcm
    test_rcm = (obs_rcm_test - mean_rcm) / sd_rcm

    train = torch.hstack((train_gcm, train_rcm))
    test = torch.hstack((test_gcm, test_rcm))
    locs = torch.vstack((torch.tensor(locs_gcm), torch.tensor(locs_rcm)))

    ord_idx = maxmin_pred_cpp(locs_gcm, locs_rcm)
    locs_ord = locs[ord_idx, ...]
    obs_ord = train[..., ord_idx]
    obs_test_ord = test[..., ord_idx]

    locs_all = [locs_ord[:n_gcm].detach().numpy(), locs_ord[n_gcm:].detach().numpy()]
    nn = find_nns_l2_mf(locs_all, 20 if linear_variant else 50)
    fidelity_sizes = torch.as_tensor(list(map(len, locs_all)))
    data = MultiFidelityData.new(locs_ord, obs_ord, torch.as_tensor(nn), fidelity_sizes)

    theta_init = build_theta_init(data, R=2, linear=linear_variant)
    tm = MultiFidelityTM(data, theta_init, nug_mult=4.0, linear=linear_variant)
    tm.fit(200, 0.01)

    batch_results = []
    tm.eval()
    for i in range(10):
        gc.collect()
        if i % 3 == 0:
            print(i)
        sc = tm.score(obs_test_ord[i].to(torch.float64), 0)
        ls_gcm = sc[:n_gcm].sum().item()
        ls_rcm = sc[n_gcm:].sum().item()
        batch_results.append(
            {
                "n": n,
                "test_idx": int(test_idx[i]),
                "logscore_total": float(sc.sum().item()),
                "logscore_gcm": float(ls_gcm),
                "logscore_rcm": float(ls_rcm),
            }
        )
    return batch_results


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.variant
    rows: list[dict] = []

    if args.experiment == "climate":
        for n in parse_ns(args.ns, args.experiment):
            print("With ensemble size")
            print(n)
            rows.extend(get_climate_scores(n, args.variant))
            save_rows(
                rows,
                args.output_dir,
                model_name,
                args.experiment,
                args.include_all_logscores,
            )
            current = [r["logscore_total"] for r in rows if r["n"] == n]
            print("mean log score")
            print(float(np.mean(current)))
        return

    if args.experiment == "linear":
        data_fp = pathlib.Path("../tests/data/data_mf.pkl")
        locs_pkl = None
    else:
        data_fp = pathlib.Path("../tests/data/data_mf_min.pkl")
        with open(pathlib.Path("../tests/data/data_mf.pkl"), "rb") as fh:
            locs_pkl = pickle.load(fh)

    with open(data_fp, "rb") as fh:
        data_pkl = pickle.load(fh)

    for n in parse_ns(args.ns, args.experiment):
        print("With ensemble size")
        print(n)
        scores = get_multifidelity_scores(n, data_pkl, args.variant, locs_pkl=locs_pkl)
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
            rows,
            args.output_dir,
            model_name,
            args.experiment,
            args.include_all_logscores,
        )
        print("Means for n =", n)
        print("  total:", scores["total"].mean().item())
        print("  lf:", scores["lf"].mean().item())
        print("  mf:", scores["mf"].mean().item())
        print("  hf:", scores["hf"].mean().item())


if __name__ == "__main__":
    main()
