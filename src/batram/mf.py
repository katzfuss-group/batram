import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from gpytorch.kernels import MaternKernel
from matplotlib import pyplot as plt
from matplotlib.axes import Axes as MPLAxes
from pyro.distributions import InverseGamma
from torch.distributions import Normal
from torch.distributions.studentT import StudentT
from tqdm import tqdm

from .data import AugmentDataMF, AugmentedDataMF, MultiFidelityData
from .legmods import KernelResult, _PreCalcLogLik


def scaling_fun(k, theta_0, theta_1):
    theta_pos = torch.exp(theta_1)
    return torch.sqrt(torch.exp(theta_0 - k * theta_pos))


def range_fun(theta):
    return torch.exp(theta)


def kernel_fun(
    X1_now,
    X1_preb,
    theta_0,
    theta_1,
    theta_0_preb,
    theta_1_preb,
    theta_ls,
    smooth,
    sigma,
    nugget_mean,
    X2_now=None,
    X2_preb=None,
):
    if X2_now is None:
        X2_now = X1_now
    if X2_preb is None:
        X2_preb = X1_preb
    N_now = X1_now.shape[-1]
    N_preb = X1_preb.shape[-1]
    scaling_now = scaling_fun(torch.arange(1, N_now + 1), theta_0, theta_1)
    scaling_preb = scaling_fun(torch.arange(1, N_preb + 1), theta_0_preb, theta_1_preb)
    X1s_now = X1_now.mul(scaling_now.unsqueeze(0))
    X1s_preb = X1_preb.mul(scaling_preb.unsqueeze(0))
    X1s = torch.cat((X1s_now, X1s_preb), dim=1)
    X2s_now = X2_now.mul(scaling_now.unsqueeze(0))
    X2s_preb = X2_preb.mul(scaling_preb.unsqueeze(0))
    X2s = torch.cat((X2s_now, X2s_preb), dim=1)
    lin = X1s @ X2s.mT
    MaternObj = MaternKernel(smooth)
    MaternObj._set_lengthscale(torch.tensor(1.0))
    MaternObj.requires_grad_(False)
    lenScal = range_fun(theta_ls) * math.sqrt(2 * smooth)
    nonlin = MaternObj.forward(X1s.div(lenScal), X2s.div(lenScal))
    nonlin = sigma.pow(2).reshape(-1, 1, 1) * nonlin
    out = (lin + nonlin).div(nugget_mean)
    return out


class NuggetMultiFidelity(torch.nn.Module):
    """
    To write
    """

    def __init__(self, nugget_params: torch.Tensor, R: int) -> None:
        super().__init__()
        assert nugget_params.shape == (2 * R,)
        self.nugget_params = torch.nn.Parameter(nugget_params)
        self.R = R

    def forward(self, data: AugmentedDataMF, r: int) -> torch.Tensor:
        theta = self.nugget_params
        sigma_1 = theta[: self.R]
        sigma_2 = theta[self.R :]
        nugget_mean_now = (sigma_1[r] + sigma_2[r] * data.scales.log()).exp()
        nugget_mean = torch.relu(nugget_mean_now.sub(1e-5)).add(1e-5)
        return nugget_mean


class TMKernelMF(torch.nn.Module):
    """
    This implements the Multi-scale Transport Map kernel for multi-scale data,
    as described in the paper. It includes the logic for
    determining the conditioning set sizes m1 and m2 based on the kernel parameters,
    and for computing the kernel matrix given the augmented data and nugget means.
    """

    def __init__(
        self,
        kernel_params: torch.Tensor,
        R: int,
        smooth: float = 1.5,
        fix_m: int | None = None,
        linear: bool = False,
    ) -> None:
        super().__init__()

        assert kernel_params.numel() == 7 * R - 2  # 2R for sigma params, #4R-2 for
        # Theta_params, R for theta_gamma param

        self.fix_m = fix_m
        self.smooth = smooth
        self.linear = linear
        self._tracked_values: dict["str", torch.Tensor] = {}

        sigma_init = kernel_params[: 2 * R].clone()
        if self.linear:
            sigma_init[:R] = -torch.inf
            self.register_buffer("sigma_params", sigma_init)
        else:
            self.sigma_params = torch.nn.Parameter(sigma_init)

        self.theta_q = torch.nn.Parameter(kernel_params[2 * R : 6 * R - 2])
        self.lengthscale = torch.nn.Parameter(kernel_params[6 * R - 2 : 7 * R - 2])
        self.R = R

        matern = MaternKernel(smooth)
        matern._set_lengthscale(torch.tensor(1.0))
        matern.requires_grad_(False)
        self._kernel = matern

    def _range(self, r: int) -> torch.Tensor:
        "Computes lenghtscale of the kernel"
        return self.lengthscale[r].exp()

    def _kernel_fun(
        self,
        x1_now: torch.Tensor,
        x1_preb: torch.Tensor,
        sigmas: torch.Tensor,
        nug_mean: torch.Tensor,
        r: int,
        x2_now: torch.Tensor | None = None,
        x2_preb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the transport map kernel."""
        k1 = torch.arange(x1_now.shape[-1], device=x1_now.device) + 1
        k2 = torch.arange(x1_preb.shape[-1], device=x1_preb.device) + 1  # k1.max()
        scaling_now = self._scale(k1, r, False)
        scaling_preb = self._scale(k2, r, True)

        x1 = torch.cat((x1_now, x1_preb), dim=-1)
        scale1 = torch.cat((scaling_now, scaling_preb), dim=0)
        _x1 = x1 * scale1

        if x2_now is None and x2_preb is None:
            _x2 = _x1
        else:
            parts: list[torch.Tensor] = []
            scales: list[torch.Tensor] = []
            if x2_now is not None:
                parts.append(x2_now)
                scales.append(scaling_now)
            if x2_preb is not None:
                parts.append(x2_preb)
                scales.append(scaling_preb)

            x2 = torch.cat(parts, dim=-1)
            scale2 = torch.cat(scales, dim=0)
            _x2 = x2 * scale2

        linear = _x1 @ _x2.mT
        if self.linear:
            return linear / nug_mean

        ls = self._range(r) * math.sqrt(2 * self.smooth)
        nonlinear = self._kernel(_x1 / ls, _x2 / ls).to_dense()
        out = (linear + sigmas**2 * nonlinear) / nug_mean
        return out

    def _m_threshold(
        self, max_m: int, r: int, preb: bool, past_m: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes the size of the conditiong sets"""
        # if preb:
        #    rng = torch.arange(past_m + 1, past_m + max_m + 1)
        # else:
        rng = torch.arange(max_m) + 1
        scales = self._scale(rng, r, preb)
        mask = scales >= 0.01
        m = mask.sum()
        if m > max_m:
            logging.warning(
                f"Required size of the conditioning sets m = {m.item()} is "
                f"greater than the maximum number of neighbors {max_m = } in "
                "the pre-calculated conditioning sets."
            )
            m = torch.tensor(max_m, dtype=torch.int)
        if r == 0 and preb:
            m = torch.tensor(0, dtype=torch.int)
        return m

    def _scale(self, k: torch.Tensor, r: int, preb: bool) -> torch.Tensor:
        """Compute scaling with respect to the same fidelity and
        the previous fidelity"""
        if preb:
            theta_q_0 = self.theta_q[r + 2 * self.R - 1]
            theta_q_1 = self.theta_q[r + 3 * self.R - 2]
        else:
            theta_q_0 = self.theta_q[r]
            theta_q_1 = self.theta_q[r + self.R]
        theta_q_1 = torch.exp(theta_q_1)
        scales_now = torch.exp(theta_q_0 - 0.5 * theta_q_1 * k)
        return scales_now

    def _determine_m(
        self, max_m: int, r: int, preb: bool, past_m: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Determine m at each fidelity and wrt previous fidelity"""
        m: torch.Tensor
        if self.fix_m is None:
            m = self._m_threshold(max_m, r, preb, past_m)
        else:
            m = torch.tensor(max_m)
        return m

    def _sigmas(self, scales: torch.Tensor, r: int) -> torch.Tensor:
        if self.linear:
            return torch.zeros_like(scales)
        sigma_0_now = self.sigma_params[r]
        sigma_1_now = self.sigma_params[r + self.R]
        return torch.exp(sigma_0_now + sigma_1_now * scales.log())

    def forward(
        self, data: AugmentedDataMF, nug_means: torch.Tensor, r: int
    ) -> KernelResult:
        """Computes with Kernel params"""
        max_m = data.data.max_m
        # this is list of ms
        m1 = self._determine_m(max_m, r, False)
        m2 = self._determine_m(max_m, r, True, m1)
        assert m1 <= max_m
        assert m2 <= max_m
        self._tracked_values[f"m1_{r}"] = m1
        self._tracked_values[f"m2_{r}"] = m2
        # data.augmented_response: (n, batch, 2*max_m+1)
        x1 = data.augmented_response[..., 1 : (m1 + 1)]
        x2 = data.augmented_response[..., (max_m + 1) : (max_m + m2 + 1)]

        x1 = torch.where(torch.isnan(x1), 0.0, x1).permute(-2, -3, -1)  # (batch, n, m1)
        x2 = torch.where(torch.isnan(x2), 0.0, x2).permute(-2, -3, -1)  # (batch, n, m2)

        scales = data.scales.clone()
        if torch.any(scales == 0.0):
            scales[scales == 0.0] = scales[scales != 0.0].min() / 2

        sigmas = self._sigmas(scales, r).reshape(-1, 1, 1)
        nug_mean_reshaped = nug_means.reshape(-1, 1, 1)
        k = self._kernel_fun(x1, x2, sigmas, nug_mean_reshaped, r)
        eye = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype).expand_as(k)
        g = k + eye
        mask0 = data.batch_idx == 0
        if mask0.any():
            Id = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
            g[mask0] = Id

        g_chol = robust_cholesky(g)

        return KernelResult(g, g_chol, nug_means)


def robust_cholesky(g, max_attempts=10, eps=1e-6):
    """Compute the Cholesky decomposition of g,
    adding a small value to the diagonal if it fails to
    ensure numerical stability."""
    for attempt in range(max_attempts):
        try:
            return torch.linalg.cholesky(g)
        except RuntimeError:
            if attempt == max_attempts - 1:
                print(f"Cholesky failed, adding {eps*(10**attempt):.1e} to diagonal")
            g = g + torch.eye(g.shape[-1], device=g.device) * (eps * (10**attempt))
    raise RuntimeError(f"Cholesky failed after {max_attempts} attempts")


class IntLogLikMF(torch.nn.Module):
    """
    Computes the integrated log-likelihood for the multi-scale model,
    given the kernel results and the current response.
    """

    def __init__(self, R: int, nug_mult: float = 4.0):
        super().__init__()
        self.nug_mult = torch.tensor(nug_mult)
        self.R = R

    def precalc(
        self, kernel_result: KernelResult, response_now: torch.Tensor
    ) -> _PreCalcLogLik:
        # response_now: (n, batch)
        nug_mean = kernel_result.nug_mean
        nug_sd = nug_mean.mul(self.nug_mult)
        alpha = nug_mean.pow(2).div(nug_sd.pow(2)).add(2)
        beta = nug_mean.mul(alpha.sub(1))

        n = response_now.shape[0]
        y_tilde = torch.linalg.solve_triangular(
            kernel_result.GChol, response_now.t().unsqueeze(-1), upper=False
        ).squeeze()
        alpha_post = alpha.add(n / 2)
        beta_post = beta + y_tilde.square().sum(dim=1).div(2)

        return _PreCalcLogLik(
            nug_sd=nug_sd,
            alpha=alpha,
            beta=beta,
            alpha_post=alpha_post,
            beta_post=beta_post,
            y_tilde=y_tilde,
        )

    def forward(
        self, data: AugmentedDataMF, kernel_result: KernelResult, r: int
    ) -> torch.Tensor:
        tmp_res = self.precalc(kernel_result, data.response)
        Gchol_diag = kernel_result.GChol.diagonal(dim1=-1, dim2=-2)
        logdet = Gchol_diag.log().sum(dim=1)
        loglik = (
            -logdet
            + tmp_res.alpha.mul(tmp_res.beta.log())
            - tmp_res.alpha_post.mul(tmp_res.beta_post.log())
            + tmp_res.alpha_post.lgamma()
            - tmp_res.alpha.lgamma()
        )
        return loglik


class MultiFidelityTM(torch.nn.Module):
    """
    Wrapper class that combines the nugget, kernel, and integrated log-likelihood
    modules. Call .fit() to train the model on the data, and .score() to
    compute the log-score values for new observations. Use cond_sample()
    to draw conditional (or unconditional) samples from the model.
    """

    def __init__(
        self,
        data: MultiFidelityData,
        theta_init: None | torch.Tensor = None,
        smooth: float = 1.5,
        nug_mult: float = 4.0,
        linear: bool = False,
    ) -> None:
        super().__init__()

        self.R = len(data.fidelity_sizes)

        self.augment_data = AugmentDataMF()
        self.data = data
        if theta_init is None:
            theta_init = torch.zeros(9 * self.R - 2)
            log_2m = data.response[:, 0].square().mean().log()
            # Nugget 1
            theta_init[: self.R] = log_2m
            # Nugget 2
            theta_init[self.R : 2 * self.R] = 0.2
            # Sigma 1
            theta_init[2 * self.R : 3 * self.R] = 0.0
            # Sigma 2
            theta_init[3 * self.R : 4 * self.R] = 0.0
            # Theta q 0 within
            theta_init[4 * self.R : 5 * self.R] = 0.0
            # Theta q 1 within
            theta_init[5 * self.R : 6 * self.R] = 0.0
            # Theta q 0 between
            theta_init[6 * self.R : 7 * self.R - 1] = 0.0
            # Theta q 1 between
            theta_init[7 * self.R - 1 : 8 * self.R - 2] = 0.0
            # Theta gammas
            theta_init[8 * self.R - 2 : 9 * self.R - 2] = -1.0
        self.nugget = NuggetMultiFidelity(theta_init[: 2 * self.R], self.R)
        self.kernel = TMKernelMF(
            theta_init[2 * self.R :], R=self.R, smooth=smooth, linear=linear
        )
        self.intloglik = IntLogLikMF(R=self.R, nug_mult=nug_mult)
        self._tracked_values: dict[str, torch.Tensor] = {}
        self.smooth = smooth

    def _fid_range(self, r: int, fs: torch.Tensor) -> tuple[int, int]:
        start = int(fs[:r].sum().item())
        end = int(fs[: r + 1].sum().item())
        return start, end

    def forward(
        self,
        r: int,
        batch_idx: torch.Tensor | None = None,
        data: MultiFidelityData | None = None,
    ) -> torch.Tensor:
        if data is None:
            data = self.data

        fs = data.fidelity_sizes
        start, end = self._fid_range(r, fs)
        n_r = end - start

        if batch_idx is None:
            batch_idx = torch.arange(start, end, device=data.response.device)

        if not torch.all((batch_idx >= start) & (batch_idx < end)):
            raise ValueError(
                f"Batch indices must be in the range [{start}, {end}) for fidelity {r}."
            )
        batch_size = batch_idx.shape[0]
        batch_size_r = n_r if batch_size is None else min(batch_size, n_r)

        aug_data = self.augment_data(data, batch_idx)
        nugget = self.nugget(aug_data, r)
        kernel_result = self.kernel(aug_data, nugget, r)
        intloglik = self.intloglik(aug_data, kernel_result, r)

        loss = -n_r / batch_size_r * intloglik.sum()
        return loss

    def named_tracked_values(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        gen = self._named_members(
            lambda module: getattr(module, "_tracked_values", {}).items(),
            prefix=prefix,
            recurse=recurse,
        )
        yield from gen

    def _freeze_indices_for_r(self, r: int, dev: torch.device):
        R = self.R
        freeze = []

        # nugget: length 2R
        if r > 0:
            freeze.append((self.nugget.nugget_params, torch.arange(0, r, device=dev)))
            freeze.append(
                (self.nugget.nugget_params, torch.arange(R, R + r, device=dev))
            )

            if not self.kernel.linear:
                # sigma: length 2R
                freeze.append(
                    (self.kernel.sigma_params, torch.arange(0, r, device=dev))
                )
                freeze.append(
                    (self.kernel.sigma_params, torch.arange(R, R + r, device=dev))
                )

            # theta_q: length 4R-2 (within: R + R, between: (R-1) + (R-1))
            freeze.append(
                (self.kernel.theta_q, torch.arange(0, r, device=dev))
            )  # q0 within
            freeze.append(
                (self.kernel.theta_q, torch.arange(R, R + r, device=dev))
            )  # q1 within

            # between blocks correspond to fidelities 1..R-1; freeze those < r
            if r > 1:
                freeze.append(
                    (
                        self.kernel.theta_q,
                        torch.arange(2 * R, 2 * R + (r - 1), device=dev),
                    )
                )  # q0 between for 1..r-1
                freeze.append(
                    (
                        self.kernel.theta_q,
                        torch.arange(3 * R - 1, 3 * R - 1 + (r - 1), device=dev),
                    )
                )  # q1 between for 1..r-1

            # lengthscale: length R
            freeze.append((self.kernel.lengthscale, torch.arange(0, r, device=dev)))

        return freeze

    def _save_frozen(self, freeze):
        saved = []
        for p, idx in freeze:
            if idx.numel() == 0:
                continue
            saved.append((p, idx, p.data[idx].detach().clone()))
        return saved

    def _restore_frozen(self, saved):
        for p, idx, val in saved:
            p.data[idx].copy_(val)

    def fit(
        self,
        num_iter,
        init_lr: float,
        batch_size: None | int = None,
        test_data: MultiFidelityData | None = None,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        num_iter
            Number of iterations to run the optimizer.
        init_lr
            Initial learning rate. Only used if optimizer is None.
        batch_size
            Batch size for training. If None, use all data.
        test_data
            Data to use for testing. If None, do not test.
        optimizer
            Optimizer to use. If None, use Adam.
        scheduler
            Learning rate scheduler to use. If None, CosineAnnealingLR
            is used with default optimizer.
        stopper
            An early stopper. If None, no early stopping is used. Requires test data.
        silent
            If True, do not print progress.
        """

        data_size = self.data.response.shape[1]
        if batch_size is not None:
            if batch_size > data_size:
                raise ValueError(
                    f"Batch size {batch_size} is larger than data size {data_size}."
                )

        losses: list[Any] = []
        test_losses: list[Any] = []
        parameters = []
        values = []
        for r in range(self.R):
            # Restart optimizer/scheduler for each fidelity
            dev = self.data.response.device
            start, end = self._fid_range(r, self.data.fidelity_sizes)
            n_r = end - start
            batch_size_r = n_r if batch_size is None else min(batch_size, n_r)
            if n_r <= 5 * batch_size_r:  # heuristic
                batch_size_r = n_r
            perm = torch.randperm(n_r, device=self.data.response.device) + start
            idxes = perm.split(batch_size_r)

            if idxes and idxes[-1].shape[0] < batch_size_r:
                idxes = idxes[:-1]

            lr_scaled = init_lr * batch_size_r / n_r
            param_list = list(
                set(self.parameters()) - {self.kernel._kernel.raw_lengthscale}
            )

            optimizer = torch.optim.Adam(param_list, lr=lr_scaled, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_iter, eta_min=lr_scaled * 0.01
            )
            test_losses.append(
                [] if test_data is None else [self(r, data=test_data).item()]
            )
            parameters.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
            )
            values.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_tracked_values()}
            )
            for _ in (tqdm_obj := tqdm(range(num_iter))):
                # create batches
                perm = torch.randperm(n_r, device=dev) + start
                idxes = perm.split(batch_size_r)  # keep last partial batch too

                # update for each batch
                epoch_losses = []
                epoch_weights = []
                freeze = self._freeze_indices_for_r(r, dev)
                for idx in idxes:
                    optimizer.zero_grad(set_to_none=True)

                    loss = self(r, batch_idx=idx)  # forward already checks idx range
                    loss.backward()

                    saved = self._save_frozen(freeze)  # save frozen values BEFORE step
                    optimizer.step()
                    self._restore_frozen(saved)  # restore frozen after step

                    bs = idx.numel()
                    epoch_losses.append(loss.item())
                    epoch_weights.append(bs)
                losses.append(float(np.average(epoch_losses, weights=epoch_weights)))

                if scheduler is not None:
                    scheduler.step()

                desc = f"Train Loss: {losses[-1]:.3f}"
                # validate
                if test_data is not None:
                    with torch.no_grad():
                        test_losses.append(self(r, data=test_data).item())
                    desc += f", Test Loss: {test_losses[-1]:.3f}"

                # store parameters and values
                parameters.append(
                    {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
                )
                values.append(
                    {
                        k: np.copy(v.detach().numpy())
                        for k, v in self.named_tracked_values()
                    }
                )

                tqdm_obj.set_description(desc)

        param_chain = {}
        for k in parameters[0].keys():
            param_chain[k] = np.stack([d[k] for d in parameters], axis=0)

        tracked_chain = {}
        for k in values[0].keys():
            tracked_chain[k] = np.stack([d[k] for d in values], axis=0)

        return FitResultMF(
            model=self,
            max_m=self.data.max_m,
            losses=np.array(losses),
            parameters=parameters[-1],
            test_losses=np.array(test_losses) if test_data is not None else None,
            param_chain=param_chain,
            tracked_chain=tracked_chain,
        )

    @torch.no_grad()
    def score(self, obs, r=0, last_ind=None):
        """Compute log-score values for the
        given observations, strarting at fidelity r."""
        if self.kernel.linear:
            return self.score_linear_nouq(obs, r=r, last_ind=last_ind)
        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        score = torch.zeros(N, dtype=torch.float64)

        if last_ind is None:
            last_ind = N

        for j in range(r, self.R):
            start_, end_ = self._fid_range(j, fs)
            batch_idx = torch.arange(start_, end_, device=self.data.response.device)
            aug_j = self.augment_data(self.data, batch_idx)
            scales = aug_j.scales
            nug_means = self.nugget(aug_j, j)
            kernel_results = self.kernel(aug_j, nug_means, j)
            tmp_res = self.intloglik.precalc(kernel_results, aug_j.response)
            y_tilde = tmp_res.y_tilde
            beta_post = tmp_res.beta_post
            alpha_post = tmp_res.alpha_post
            sigmas = self.kernel._sigmas(scales, j)
            nugget_mean = kernel_results.nug_mean
            chol = kernel_results.GChol
            m1 = self.kernel._determine_m(max_m, j, False)
            m2 = self.kernel._determine_m(max_m, j, True)
            for i in range(fs[j]):
                idx = start_ + i
                if idx == 0:
                    cStar = torch.zeros(n, device=data.device, dtype=torch.float64)
                    prVar = torch.tensor(0.0, device=data.device, dtype=torch.float64)
                else:
                    ncol = min(i, m1)
                    nn1 = NN[idx, :ncol]
                    nn2 = NN[idx, max_m : max_m + m2]
                    nn1 = nn1[nn1 >= 0]
                    nn2 = nn2[nn2 >= 0]
                    if nn1.numel() == 0 and nn2.numel() == 0:
                        cStar = torch.zeros(n, device=data.device, dtype=torch.float64)
                        prVar = torch.tensor(
                            0.0, device=data.device, dtype=torch.float64
                        )
                    else:
                        XPred_now = obs[nn1].unsqueeze(0).unsqueeze(0)
                        XPred_prev = obs[nn2].unsqueeze(0).unsqueeze(0)
                        X_now = data[:, nn1].to(torch.float64)
                        X_preb = data[:, nn2].to(torch.float64)
                        cStar = self.kernel._kernel_fun(
                            XPred_now,
                            XPred_prev,
                            sigmas[i],
                            nugget_mean[i],
                            j,
                            X_now,
                            X_preb,
                        ).squeeze(1)
                        prVar = self.kernel._kernel_fun(
                            XPred_now, XPred_prev, sigmas[i], nugget_mean[i], j
                        ).squeeze()
                cChol = torch.linalg.solve_triangular(
                    chol[i, :, :], cStar.unsqueeze(-1), upper=False
                ).squeeze(-1)
                meanPred = y_tilde[i, :].unsqueeze(0).mul(cChol).sum()
                varPredNoNug = prVar - cChol.square().sum()
                initVar = beta_post[i] / alpha_post[i] * (1 + varPredNoNug)
                STDist = StudentT(2 * alpha_post[i])
                score[idx] = (
                    STDist.log_prob((obs[idx] - meanPred) / initVar.sqrt())
                    - 0.5 * initVar.log()
                )
        return -score[sum(fs[:r]) : last_ind]

    @torch.no_grad()
    def cond_sample(self, obs, r=0, last_ind=None):
        """
        Draw a conditional sample from the model, given observations obs up
        to fidelity r-1. If r = 0, draws an unconditional sample from
        the model.
        """
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = self.data.fidelity_sizes
        n, _ = data.shape
        N = int(fs.sum().item())
        known = int(fs[:r].sum().item())
        max_m = int(self.data.max_m)

        obs = obs.to(device=data.device, dtype=torch.float64)
        if obs.numel() < known:
            raise ValueError(
                f"obs has length {obs.numel()} but need at least {known} for r={r}"
            )
        x_fix = obs[:known]

        if last_ind is None:
            last_ind = N

        x_new = torch.full((1, N), float("nan"), device=obs.device, dtype=obs.dtype)
        x_new[0, :known] = x_fix

        for j in range(r, self.R):
            start_ = int(fs[:j].sum().item())
            end_ = int(fs[: j + 1].sum().item())

            batch_idx = torch.arange(start_, end_, device=self.data.locs.device)
            aug = self.augment_data(self.data, batch_idx)

            # ---- scale clamp BEFORE nugget/sigmas ----
            scales = aug.scales.clone()
            if (scales <= 0).any():
                pos = scales[scales > 0]
                scales[scales <= 0] = (pos.min() / 2) if pos.numel() else 1.0
            scales = scales.clamp(min=1e-12)
            aug.scales = scales

            nug_means = self.nugget(aug, j)
            kernel_results = self.kernel(aug, nug_means, j)
            tmp_res = self.intloglik.precalc(kernel_results, aug.response)

            y_tilde = tmp_res.y_tilde
            beta_post = tmp_res.beta_post
            alpha_post = tmp_res.alpha_post

            sigmas = self.kernel._sigmas(scales, j)  # (batch,)
            nugget_mean = kernel_results.nug_mean  # (batch,)
            chol = kernel_results.GChol  # (batch,n,n)

            m1 = int(self.kernel._determine_m(max_m, j, False).item())
            m2 = int(self.kernel._determine_m(max_m, j, True).item())

            for i_local in range(end_ - start_):
                idx = start_ + i_local

                # only the very first global point needs the "nuke"
                if j == 0 and i_local == 0:
                    cStar = torch.zeros((1, n), device=data.device, dtype=torch.float64)
                    prVar = torch.zeros((1,), device=data.device, dtype=torch.float64)
                else:
                    # ---- FIXED-LENGTH neighbor slots (match training) ----
                    nn1 = NN[idx, :m1]
                    nn2 = NN[idx, max_m : max_m + m2]

                    # optional sanity checks (highly recommended while debugging)
                    nn1v = nn1[nn1 >= 0]
                    nn2v = nn2[nn2 >= 0]
                    if nn1v.numel():
                        assert (
                            nn1v < idx
                        ).all(), f"within-fid NN points forward at idx={idx}"
                    if nn2v.numel():
                        assert (
                            nn2v < start_
                        ).all(), f"cross-fid NN not in previous fidelities at idx={idx}"

                    # pad -1 -> 0 indices, then zero-out those columns
                    nn1_safe = nn1.clone()
                    nn2_safe = nn2.clone()
                    m1_mask = nn1_safe < 0
                    m2_mask = nn2_safe < 0
                    nn1_safe[m1_mask] = 0
                    nn2_safe[m2_mask] = 0

                    # predictor features (1,1,m)
                    XPred_now = x_new[:, nn1_safe].unsqueeze(1)  # (1,1,m1)
                    XPred_prev = x_new[:, nn2_safe].unsqueeze(1)  # (1,1,m2)

                    # if any of these are still NaN, you are conditioning
                    # on future/unset points
                    if torch.isnan(XPred_now).any() or torch.isnan(XPred_prev).any():
                        raise RuntimeError(
                            f"NaN in predictors at idx={idx} (bad NN ordering?)"
                        )

                    XPred_now[..., m1_mask] = 0.0
                    XPred_prev[..., m2_mask] = 0.0

                    # training features (n,m)
                    X_now = data[:, nn1_safe].to(torch.float64)
                    X_preb = data[:, nn2_safe].to(torch.float64)
                    X_now[:, m1_mask] = 0.0
                    X_preb[:, m2_mask] = 0.0

                    cStar = self.kernel._kernel_fun(
                        XPred_now,
                        XPred_prev,
                        sigmas[i_local],
                        nugget_mean[i_local],
                        j,
                        X_now,
                        X_preb,
                    ).squeeze(
                        1
                    )  # (1,n)

                    prVar = self.kernel._kernel_fun(
                        XPred_now, XPred_prev, sigmas[i_local], nugget_mean[i_local], j
                    ).squeeze(
                        (1, 2)
                    )  # (1,)

                cChol = torch.linalg.solve_triangular(
                    chol[i_local, :, :], cStar.unsqueeze(-1), upper=False
                ).squeeze(
                    -1
                )  # (1,n)

                meanPred = y_tilde[i_local, :].unsqueeze(0).mul(cChol).sum(1)  # (1,)
                varPredNoNug = prVar - cChol.square().sum(1)  # (1,)
                varPredNoNug = torch.clamp(varPredNoNug, min=0.0)

                invGDist = InverseGamma(
                    concentration=alpha_post[i_local], rate=beta_post[i_local]
                )
                nugget = invGDist.sample((1,))  # (1,)
                uniNDist = Normal(
                    loc=meanPred, scale=nugget.mul(1.0 + varPredNoNug).sqrt()
                )
                x_new[:, idx] = uniNDist.sample()[0]

        return x_new

    @torch.no_grad()
    def score_linear_nouq(self, obs, r=0, last_ind=None):
        """
        Plug-in Gaussian score for the linear model (no predictive UQ).

        log p(x_i | past) = log N(x_i ; meanPred, nugget_hat)
        where
        nugget_hat = beta_post / (alpha_post - 1)

        Unlike `score`, this does NOT include:
            - the Student-t correction
            - the function uncertainty term (1 + varPredNoNug)
        """
        if not self.kernel.linear:
            raise ValueError(
                "score_linear_nouq is intended for models with linear=True."
            )

        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        score = torch.zeros(N, dtype=torch.float64)

        if last_ind is None:
            last_ind = N

        for j in range(r, self.R):
            start_, end_ = self._fid_range(j, fs)
            batch_idx = torch.arange(start_, end_, device=self.data.response.device)
            aug_j = self.augment_data(self.data, batch_idx)
            scales = aug_j.scales
            nug_means = self.nugget(aug_j, j)
            kernel_results = self.kernel(aug_j, nug_means, j)
            tmp_res = self.intloglik.precalc(kernel_results, aug_j.response)
            y_tilde = tmp_res.y_tilde
            beta_post = tmp_res.beta_post
            alpha_post = tmp_res.alpha_post
            sigmas = self.kernel._sigmas(scales, j)
            nugget_mean = kernel_results.nug_mean
            chol = kernel_results.GChol
            m1 = self.kernel._determine_m(max_m, j, False)
            m2 = self.kernel._determine_m(max_m, j, True)
            for i in range(fs[j]):
                idx = start_ + i
                if idx == 0:
                    cStar = torch.zeros(n, device=data.device, dtype=torch.float64)
                else:
                    ncol = min(i, m1)
                    nn1 = NN[idx, :ncol]
                    nn2 = NN[idx, max_m : max_m + m2]
                    nn1 = nn1[nn1 >= 0]
                    nn2 = nn2[nn2 >= 0]
                    if nn1.numel() == 0 and nn2.numel() == 0:
                        cStar = torch.zeros(n, device=data.device, dtype=torch.float64)
                    else:
                        XPred_now = obs[nn1].unsqueeze(0).unsqueeze(0)
                        XPred_prev = obs[nn2].unsqueeze(0).unsqueeze(0)
                        X_now = data[:, nn1].to(torch.float64)
                        X_preb = data[:, nn2].to(torch.float64)
                        cStar = self.kernel._kernel_fun(
                            XPred_now,
                            XPred_prev,
                            sigmas[i],
                            nugget_mean[i],
                            j,
                            X_now,
                            X_preb,
                        ).squeeze(1)
                cChol = torch.linalg.solve_triangular(
                    chol[i, :, :], cStar.unsqueeze(-1), upper=False
                ).squeeze(-1)
                meanPred = y_tilde[i, :].unsqueeze(0).mul(cChol).sum()
                # Plug-in posterior mean of nugget variance
                nugget_hat = beta_post[i] / (alpha_post[i] - 1.0)
                nugget_hat = torch.clamp(nugget_hat, min=1e-12)

                score[idx] = Normal(meanPred, nugget_hat.sqrt()).log_prob(obs[idx])

        return -score[sum(fs[:r]) : last_ind]


@dataclass
class FitResultMF:
    model: MultiFidelityTM
    max_m: int
    losses: np.ndarray
    test_losses: None | np.ndarray
    parameters: dict[str, np.ndarray]
    param_chain: dict[str, np.ndarray]
    tracked_chain: dict[str, np.ndarray]

    def plot_loss(
        self,
        ax: None | plt.Axes = None,
        use_inset: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the loss curve.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        (p1,) = ax.plot(self.losses, "C0", label="Train Loss", **kwargs)
        legend_handle = [p1]

        if self.test_losses is not None:
            twin = cast(MPLAxes, ax.twinx())
            (p2,) = twin.plot(self.test_losses, "C1", label="Test Loss", **kwargs)
            legend_handle.append(p2)
            twin.set_ylabel("Test Loss")

        if use_inset:
            end_idx = len(self.losses)
            start_idx = int(0.8 * end_idx)
            inset_iterations = np.arange(start_idx, end_idx)

            inset = ax.inset_axes((0.5, 0.5, 0.45, 0.45))
            inset.plot(inset_iterations, self.losses[start_idx:], "C0", **kwargs)

            if self.test_losses is not None:
                insert_twim = cast(MPLAxes, inset.twinx())
                insert_twim.plot(
                    inset_iterations, self.test_losses[start_idx:], "C1", **kwargs
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Train Loss")
        ax.legend(handles=legend_handle, loc="lower right", bbox_to_anchor=(0.925, 0.2))

        return ax
