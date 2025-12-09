import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from gpytorch.kernels import MaternKernel
from matplotlib import pyplot as plt
from matplotlib.axes import Axes as MPLAxes
from pyro.distributions import InverseGamma
from torch.distributions import Normal
from torch.distributions.studentT import StudentT
from tqdm import tqdm

from .data import AugmentDataMF, MultiFidelityData
from .legmods import AugmentedData, Data, KernelResult, _PreCalcLogLik
from .stopper import PEarlyStopper


@dataclass
class AugmentedDataMF:
    """Augmented data class

    Holds $n$ replicates of spatial field observed at $N$ locations. The data in
    this class has been normalized.


    Attributes
    ----------
    data_size
        Size of the original data set
    batch_size
        Size of the batch
    batch_idx
        Index of the batch in the original data set
    locs
        Locations of the data. shape (N, d)
    augmented_response
        Augmented response of the data. Shape (n, N, m + 1)$
    scales
        Scales of the data. Shape (N, )
    data
        Original data

    Notes
    -----

    m is the maximum size of the conditioning sets.

    scales is called $l_i$ in the paper.
    """

    data_size: int
    batch_size: int
    batch_idx: torch.Tensor
    locs: torch.Tensor
    augmented_response: torch.Tensor
    scales: torch.Tensor
    data: Data
    fidelity_sizes: torch.Tensor


class _PreCalcLogLikMF(NamedTuple):
    nug_sd: list[torch.Tensor]
    alpha: list[torch.Tensor]
    beta: list[torch.Tensor]
    alpha_post: list[torch.Tensor]
    beta_post: list[torch.Tensor]
    y_tilde: list[torch.Tensor]


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
        fs = data.fidelity_sizes
        theta = self.nugget_params
        sigma_1 = theta[: self.R]
        sigma_2 = theta[self.R :]
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        if r == self.R:
            nugget_mean_now = (
                sigma_1[r] + sigma_2[r] * data.scales[start:end].log()
            ).exp()
        nugget_mean_now = (sigma_1[r] + sigma_2[r] * data.scales[start:end].log()).exp()
        nugget_mean = torch.relu(nugget_mean_now.sub(1e-5)).add(1e-5)
        return nugget_mean


class TMKernelMF(torch.nn.Module):
    """
    To Write
    """

    def __init__(
        self,
        kernel_params: torch.Tensor,
        R: int,
        smooth: float = 1.5,
        fix_m: int | None = None,
    ) -> None:
        super().__init__()

        assert kernel_params.numel() == 7 * R - 2  # 2R for sigma params, #4R-2 for
        # Theta_params, R for theta_gamma param
        self.sigma_params = torch.nn.Parameter(kernel_params[: 2 * R])
        self.theta_q = torch.nn.Parameter(kernel_params[2 * R : 6 * R - 2])
        self.lengthscale = torch.nn.Parameter(kernel_params[6 * R - 2 : 7 * R - 2])
        self.R = R

        self.fix_m = fix_m
        self.smooth = smooth
        self._tracked_values: dict["str", torch.Tensor] = {}

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
        k1 = torch.arange(x1_now.shape[-1]) + 1
        k2 = torch.arange(x1_preb.shape[-1]) + 1  # k1.max()
        scaling_now = self._scale(k1, r, False)
        scaling_preb = self._scale(k2, r, True)
        scaling = torch.cat((scaling_now, scaling_preb), dim=0)
        x = torch.cat((x1_now, x1_preb), dim=2)
        _x1 = x * scaling

        # Handle all combinations in a way mypy can follow
        if x2_now is None:
            if x2_preb is None:
                # both None → fall back to _x1
                _x2 = _x1
            else:
                # only previous provided
                x2 = x2_preb
                _x2 = x2 * scaling
        else:
            if x2_preb is None:
                # only current provided
                x2 = x2_now
            else:
                # both provided → concatenate
                x2 = torch.cat((x2_now, x2_preb), dim=1)
            _x2 = x2 * scaling

        linear = _x1 @ _x2.mT

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
        sigma_0_now = self.sigma_params[r]
        sigma_1_now = self.sigma_params[r + self.R]
        return torch.exp(sigma_0_now + sigma_1_now * scales.log())

    def forward(
        self, data: AugmentedDataMF, nug_means: torch.Tensor, r: int
    ) -> KernelResult:
        """Computes with Kernel params"""
        max_m = data.data.max_m
        # this is list of ms
        fs = data.fidelity_sizes
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        m1 = self._determine_m(max_m, r, False)
        m2 = self._determine_m(max_m, r, True, m1)
        assert m1 <= max_m
        assert m2 <= max_m
        self._tracked_values["m1_" + str(r)] = m1
        self._tracked_values["m2_" + str(r)] = m2
        x1 = data.augmented_response[..., start:end, 1 : (m1 + 1)]
        x2 = data.augmented_response[..., start:end, (max_m + 1) : (max_m + m2 + 1)]
        x1 = torch.where(torch.isnan(x1), 0.0, x1)
        x2 = torch.where(torch.isnan(x2), 0.0, x2)
        x1 = x1.permute(-2, -3, -1)
        x2 = x2.permute(-2, -3, -1)
        nug_mean_reshaped = nug_means.reshape(-1, 1, 1)
        scales = data.scales[start:end]
        scales[scales == 0.0] = scales[scales != 0.0].min() / 2
        sigmas = self._sigmas(scales, r).reshape(-1, 1, 1)
        k = self._kernel_fun(x1, x2, sigmas, nug_mean_reshaped, r)
        eyes = torch.eye(k.shape[-1]).expand_as(k)
        g = k + eyes
        g[0] = torch.eye(k.shape[-1])

        g_chol = robust_cholesky(g)

        return KernelResult(g, g_chol, nug_means)


def robust_cholesky(g, max_attempts=10, eps=1e-6):
    for attempt in range(max_attempts):
        try:
            return torch.linalg.cholesky(g)
        except RuntimeError:
            if attempt == max_attempts - 1:
                print(f"Cholesky failed, adding {eps*(10**attempt):.1e} to diagonal")
            g = g + torch.eye(g.shape[-1], device=g.device) * (eps * (10**attempt))
    raise RuntimeError(f"Cholesky failed after {max_attempts} attempts")


class IntLogLikMF(torch.nn.Module):
    def __init__(self, R: int, nug_mult: float = 4.0):
        super().__init__()
        self.nug_mult = torch.tensor(nug_mult)
        self.R = R

    def precalc(
        self, kernel_result: KernelResult, response, fidelity_sizes: torch.Tensor, r
    ) -> _PreCalcLogLik:
        fs = fidelity_sizes
        nug_mean = kernel_result.nug_mean
        nug_sd = nug_mean.mul(self.nug_mult)
        alpha = nug_mean.pow(2).div(nug_sd.pow(2)).add(2)
        beta = nug_mean.mul(alpha.sub(1))
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        response_now = response[..., start:end]

        assert nug_sd.shape == (response_now.shape[1],)
        assert alpha.shape == (response_now.shape[1],)
        assert beta.shape == (response_now.shape[1],)

        n = response_now.shape[0]
        y_tilde = torch.linalg.solve_triangular(
            kernel_result.GChol, response_now.t().unsqueeze(-1), upper=False
        ).squeeze()
        alpha_post = alpha.add(n / 2)
        beta_post = beta + y_tilde.square().sum(dim=1).div(2)

        assert alpha_post.shape == (response_now.shape[1],)
        assert beta_post.shape == (response_now.shape[1],)

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
        tmp_res = self.precalc(
            kernel_result, data.augmented_response[:, :, 0], data.fidelity_sizes, r
        )
        aux = kernel_result.GChol
        Gchol_now = aux.diagonal(dim1=-1, dim2=-2)
        logdet = Gchol_now.log().sum(dim=1)
        loglik = (
            -logdet
            + tmp_res.alpha.mul(tmp_res.beta.log())
            - tmp_res.alpha_post.mul(tmp_res.beta_post.log())
            + tmp_res.alpha_post.lgamma()
            - tmp_res.alpha.lgamma()
        )

        assert loglik.isfinite().all().item()

        return loglik


class MultiFidelityTM(torch.nn.Module):
    """
    To write
    """

    def __init__(
        self,
        data: MultiFidelityData,
        theta_init: None | torch.Tensor = None,
        smooth: float = 1.5,
        nug_mult: float = 4.0,
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
        self.kernel = TMKernelMF(theta_init[2 * self.R :], R=self.R, smooth=smooth)
        self.intloglik = IntLogLikMF(R=self.R, nug_mult=nug_mult)
        self._tracked_values: dict[str, torch.Tensor] = {}
        self.smooth = smooth

    def forward(
        self,
        r: int,
        batch_idx: None | torch.Tensor = None,
        data: None | MultiFidelityData = None,
    ) -> torch.Tensor:
        if data is None:
            data = self.data

        fs = data.fidelity_sizes

        aug_data: AugmentedData = self.augment_data(data, batch_idx)
        # Currently, batch_idx is always set to None
        nugget = self.nugget(aug_data, r)
        kernel_result = self.kernel(aug_data, nugget, r)
        intloglik = self.intloglik(aug_data, kernel_result, r)
        loss = -aug_data.data_size / fs[r] * intloglik.sum()
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

    def fit(
        self,
        num_iter,
        init_lr: float,
        batch_size: None | int = None,
        test_data: MultiFidelityData | None = None,
        optimizer: None | torch.optim.Optimizer = None,
        scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
        stopper: None | PEarlyStopper = None,
        silent: bool = False,
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
        if optimizer is None:
            if scheduler is not None:
                raise ValueError(
                    "Cannot specify scheduler without speicifying an optimizer."
                )
            optimizer = torch.optim.Adam(
                list(set(self.parameters()) - {self.kernel._kernel.raw_lengthscale}),
                lr=init_lr,
            )
            if scheduler is None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_iter
                )

        if stopper is not None and test_data is None:
            raise ValueError("Cannot use stopper without test data.")

        if batch_size is None:
            batch_size = self.data.response.shape[1]

        data_size = self.data.response.shape[1]
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
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_iter, eta_min=0.001
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
            for _ in (tqdm_obj := tqdm(range(num_iter), disable=silent)):
                # create batches
                if batch_size == data_size:
                    idxes = [torch.arange(data_size)]
                else:
                    idxes = torch.randperm(data_size).split(batch_size)
                    # skip non-full batch
                    if idxes[-1].shape[0] < batch_size:
                        idxes = idxes[:-1]

                # update for each batch
                epoch_losses = np.zeros(len(idxes))
                for j, idx in enumerate(idxes):
                    with torch.autograd.set_detect_anomaly(True):

                        def closure():
                            optimizer.zero_grad()  # type: ignore
                            # optimizer is not None
                            loss = self(r, batch_idx=idx)
                            loss.backward()
                            # Adam has some momentums that will make some parameters
                            # keep updating even if the training in that fidelity is
                            # done, so I have to manually paste the final trained
                            # parameter here, and zero the gradients and momentums
                            for r_i in range(r):
                                if j != 0:
                                    # Nugget 1
                                    self.nugget.nugget_params.grad[r_i] = 0
                                    print(optimizer.state[self.nugget.nugget_params])
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg"
                                    ][r_i] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg_sq"
                                    ][r_i] = 0

                                    # Nugget 2
                                    self.nugget.nugget_params.grad[r_i + self.R] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg"
                                    ][r_i + self.R] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg_sq"
                                    ][r_i + self.R] = 0

                                    # Sigma 1
                                    self.kernel.sigma_params.grad[r_i] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg"
                                    ][r_i] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg_sq"
                                    ][r_i] = 0

                                    # Sigma 2
                                    self.kernel.sigma_params.grad[r_i + self.R] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg"
                                    ][r_i + self.R] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg_sq"
                                    ][r_i + self.R] = 0

                                    # Theta q 0
                                    self.kernel.theta_q.grad[r_i] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg"][
                                        r_i
                                    ] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg_sq"][
                                        r_i
                                    ] = 0

                                    # Theta q 1
                                    self.kernel.theta_q.grad[r_i + self.R] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg"][
                                        r_i + self.R
                                    ] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg_sq"][
                                        r_i + self.R
                                    ] = 0

                                    # Lengthscales
                                    self.kernel.lengthscale.grad[r_i] = 0
                                    optimizer.state[self.kernel.lengthscale]["exp_avg"][
                                        r_i
                                    ] = 0
                                    optimizer.state[self.kernel.lengthscale][
                                        "exp_avg_sq"
                                    ][r_i] = 0

                                    if r_i > 0:
                                        # theta q 0 preb
                                        self.kernel.theta_q.grad[
                                            r_i + 2 * self.R - 1
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q]["exp_avg"][
                                            r_i + 2 * self.R - 1
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q][
                                            "exp_avg_sq"
                                        ][r_i + 2 * self.R - 1] = 0
                                        # Theta q 1 preb
                                        self.kernel.theta_q.grad[
                                            r_i + 3 * self.R - 2
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q]["exp_avg"][
                                            r_i + 3 * self.R - 2
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q][
                                            "exp_avg_sq"
                                        ][r_i + 3 * self.R - 2] = 0
                            return loss

                    # closure returns a tensor (which is needed for backprop).
                    # pytorch's type signature is wrong
                    loss: float = optimizer.step(closure).item()  # type: ignore
                    epoch_losses[j] = loss
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(np.mean(epoch_losses)))

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

                if stopper is not None:
                    state = {
                        k: v.detach().clone() for k, v in self.state_dict().items()
                    }
                    stop = stopper.step(test_losses[-1], state)
                    if stop:
                        # restore best state
                        self.load_state_dict(stopper.best_state())
                        # and break
                        break

                # if i == num_iter - 1:
                #    sd = self.state_dict()
                #    nugget_par[r] = sd['nugget.nugget_params'][r]
                #    nugget_par[r+self.R] = sd['nugget.nugget_params'][r+self.R]

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
        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        score = torch.zeros(sum(fs[r:]))

        if last_ind is None:
            last_ind = N

        for j in range(r, self.R):
            start_ = sum(fs[:j])
            end_ = sum(fs[: j + 1])
            scales = augmented_data.scales[start_:end_]
            nug_means = self.nugget(augmented_data, j)
            kernel_results = self.kernel.forward(augmented_data, nug_means, j)
            tmp_res = self.intloglik.precalc(
                kernel_results, augmented_data.response, fs, j
            )
            y_tilde = tmp_res.y_tilde
            beta_post = tmp_res.beta_post
            alpha_post = tmp_res.alpha_post
            sigmas = self.kernel._sigmas(scales, j)
            nugget_mean = kernel_results.nug_mean
            chol = kernel_results.GChol
            m1 = self.kernel._determine_m(max_m, j, False)
            m2 = self.kernel._determine_m(max_m, j, True)
            for i in range(fs[j]):
                idx = sum(fs[:j]) + i
                if i == 0:
                    cStar = torch.zeros(n)
                    prVar = torch.tensor(0.0)
                else:
                    ncol = min(i, m1)
                    nn1 = NN[idx, :ncol]
                    start = int(NN.shape[-1] / 2)
                    end = int(NN.shape[-1] / 2 + m2)
                    nn2 = NN[idx, start:end]
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
        return score

    @torch.no_grad()
    def cond_sample(self, obs, res, r=0, last_ind=None):
        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        x_fix = obs[: sum(fs[:r])]

        if last_ind is None:
            last_ind = N
        x_new = torch.empty((1, N), dtype=torch.float64)
        x_new[:, : sum(fs[0:r])] = x_fix.repeat(1, 1)
        x_new[:, sum(fs[0:r]) :] = 0.0
        for j in range(r, self.R):
            start_ = sum(fs[:j])
            end_ = sum(fs[: j + 1])
            scales = augmented_data.scales[start_:end_]
            nug_means = self.nugget(augmented_data, j)
            kernel_results = self.kernel.forward(augmented_data, nug_means, j)
            tmp_res = self.intloglik.precalc(
                kernel_results, augmented_data.response, fs, j
            )
            y_tilde = tmp_res.y_tilde
            beta_post = tmp_res.beta_post
            alpha_post = tmp_res.alpha_post
            sigmas = self.kernel._sigmas(scales, j)
            nugget_mean = kernel_results.nug_mean
            chol = kernel_results.GChol
            m1 = self.kernel._determine_m(max_m, j, False)
            m2 = self.kernel._determine_m(max_m, j, True)
            for i in range(fs[j]):
                idx = sum(fs[:j]) + i
                if i == 0:
                    cStar = torch.zeros((1, n))
                    prVar = torch.zeros((1,))
                else:
                    ncol = min(i, m1)
                    nn1 = NN[idx, :ncol]
                    start = int(NN.shape[-1] / 2)
                    end = int(NN.shape[-1] / 2 + m2)
                    nn2 = NN[idx, start:end]
                    XPred_now = x_new[:, nn1].unsqueeze(1)
                    XPred_prev = x_new[:, nn2].unsqueeze(1)
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
                    ).squeeze((1, 2))
                cChol = torch.linalg.solve_triangular(
                    chol[i, :, :], cStar.unsqueeze(-1), upper=False
                ).squeeze(-1)
                meanPred = y_tilde[i, :].unsqueeze(0).mul(cChol).sum(1)
                varPredNoNug = prVar - cChol.square().sum(1)

                if torch.any(varPredNoNug < 0.0):
                    varPredNoNug[varPredNoNug < 0.0] = 0.0
                invGDist = InverseGamma(concentration=alpha_post[i], rate=beta_post[i])
                nugget = invGDist.sample((1,))
                uniNDist = Normal(
                    loc=meanPred, scale=nugget.mul(1.0 + varPredNoNug).sqrt()
                )
                x_new[:, idx] = uniNDist.sample()[0]
        return x_new


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
