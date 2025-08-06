import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, cast

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
from .legmods import AugmentedData, KernelResult
from .stopper import PEarlyStopper


@dataclass
class KernelMFResult:
    G: list[torch.Tensor]
    GChol: list[torch.Tensor]
    nug_mean: list[torch.Tensor]


class _PreCalcLogLikMF(NamedTuple):
    nug_sd: list[torch.Tensor]
    alpha: list[torch.Tensor]
    beta: list[torch.Tensor]
    alpha_post: list[torch.Tensor]
    beta_post: list[torch.Tensor]
    y_tilde: list[torch.Tensor]

class NuggetMultiFidelity(torch.nn.Module):
    """
    Class for storing the nugget parameters for all fidelities 
    and computing the nugget means. 
    """

    def __init__(self, nugget_params: torch.Tensor, R: int) -> None:
        super().__init__()
        # 2 nugget parameters per fidelity
        assert nugget_params.shape == (2 * R,)
        self.nugget_params = torch.nn.Parameter(nugget_params)
        self.R = R

    def forward(self, data: AugmentedData, r: int) -> torch.Tensor:
        fs = data.data.fidelity_sizes
        theta = self.nugget_params
        sigma_1 = theta[: self.R]
        sigma_2 = theta[self.R :]
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        # If last fidelity the logic changes a bit
        if r == self.R:
            nugget_mean_now = (
                sigma_1[r] + sigma_2[r] * data.scales[start:end].log()
            ).exp()
        nugget_mean_now = (sigma_1[r] + sigma_2[r] * data.scales[start:end].log()).exp()
        # numerical statbility
        nugget_mean = torch.relu(nugget_mean_now.sub(1e-5)).add(1e-5)
        return nugget_mean


class TMKernelMF(torch.nn.Module):
    """
    Class for storing the kernel parameters for all fidelities
    and computing the kernel matrix for the transport map.
    """

    def __init__(
        self,
        kernel_params: torch.Tensor,
        R: int,
        smooth: float = 1.5,
        fix_m: int | None = None,
    ) -> None:
        super().__init__()
        # 7 kernel parameters per fidelity, except for the first fidelity 
        # which does not have the parameters for the previous fidelity
        assert kernel_params.numel() == 7 * R - 2  # 2R for sigma params, #4R-2 for
        # Theta_params, R for theta_gamma param
        # 2 sigma parameters per fidelity, 1 lengthscale parameter per fidelity,
        # 4 relevance parameters per fidelity (except for the first fidelity)
        self.sigma_params = torch.nn.Parameter(kernel_params[: 2 * R])
        self.theta_q = torch.nn.Parameter(kernel_params[2 * R : 6 * R - 2])
        self.lengthscale = torch.nn.Parameter(kernel_params[6 * R - 2 : 7 * R - 2])
        self.R = R

        self.fix_m = fix_m
        self.smooth = smooth
        self._tracked_values: dict["str", torch.Tensor] = {}

        matern = MaternKernel(smooth)
        # We set the nonlinear part of the kernel to be a Matern kernel
        # with lengthscale 1.0
        matern._set_lengthscale(torch.tensor(1.0))
        matern.requires_grad_(False)
        self._kernel = matern

    def _range(self, r: int) -> torch.Tensor:
        "Computes lenghtscales of the kernel at fidelity r"
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
        """Computes the transport map kernel at fidelity r"""
        k1 = torch.arange(x1_now.shape[-1]) + 1
        k2 = torch.arange(x1_preb.shape[-1]) + 1  # k1.max()
        scaling_now = self._scale(k1, r, False)
        scaling_preb = self._scale(k2, r, True)
        scaling = torch.cat((scaling_now, scaling_preb), dim=0)
        x = torch.cat((x1_now, x1_preb), dim=2)
        _x1 = x * scaling
        if x2_now is None:
            _x2 = _x1
        else:
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
        # If first fidelity, we do not condition on the previous fidelity
        if r == 0 and preb:
            m = torch.tensor(0, dtype=torch.int)
        return m

    def _scale(self, k: torch.Tensor, r: int, preb: bool) -> torch.Tensor:
        """Compute scaling with respect to the same fidelity and
        the previous fidelity"""
        # We access different parameters depending on whether we are
        # computing scaling with respect to current fidelity or the previous one
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
        """Computes nonlinear scaling for the kernel at fidelity r"""
        sigma_0_now = self.sigma_params[r]
        sigma_1_now = self.sigma_params[r + self.R]
        return torch.exp(sigma_0_now + sigma_1_now * scales.log())

    def forward(
        self, data: AugmentedData, nug_means: torch.Tensor, r: int
    ) -> KernelMFResult:
        """Computes the kernel at fidelity r"""
        max_m = data.data.max_m
        # this is list of ms
        fs = data.data.fidelity_sizes
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        # Get the conditioning set sizes wrt the current 
        # and the previous fidelity
        m1 = self._determine_m(max_m, r, False)
        m2 = self._determine_m(max_m, r, True, m1)
        assert m1 <= max_m
        assert m2 <= max_m
        # Track the appropriate values
        self._tracked_values["m1_" + str(r)] = m1
        self._tracked_values["m2_" + str(r)] = m2
        # Get the conditioning sets with m1, m2
        x1 = data.augmented_response[..., start:end, 1 : (m1 + 1)]
        x2 = data.augmented_response[..., start:end, (max_m + 1) : (max_m + m2 + 1)]
        x1 = torch.where(torch.isnan(x1), 0.0, x1)
        x2 = torch.where(torch.isnan(x2), 0.0, x2)
        x1 = x1.permute(-2, -3, -1)
        x2 = x2.permute(-2, -3, -1)
        nug_mean_reshaped = nug_means.reshape(-1, 1, 1)
        scales = data.scales[start:end]
        # This helps with numerical stability
        scales[scales == 0.0] = scales[scales != 0.0].min() / 2
        # Compute the kernel matrix
        sigmas = self._sigmas(scales, r).reshape(-1, 1, 1)
        k = self._kernel_fun(x1, x2, sigmas, nug_mean_reshaped, r)
        eyes = torch.eye(k.shape[-1]).expand_as(k)
        g = k + eyes
        g[0] = torch.eye(k.shape[-1])
        # Apply robust Cholesky decomposition for numerical stability
        g_chol = self._robust_cholesky(g, 10, 1e-6)

        return KernelResult(g, g_chol, nug_means)

    def _robust_cholesky(self, g, max_attempts: int, eps: float) -> torch.Tensor:
        """Computes the Cholesky decomposition, if it fails it adds a tiny value to the diagonal.
        Tries 10 times maximum to ensure it is quick enough. This helps with numerical stability.
        Usually works at first try."""
        for attempt in range(max_attempts):
            try:
                return torch.linalg.cholesky(g)
            except RuntimeError:
                if attempt == max_attempts - 1:
                    print(f"Cholesky failed, adding {eps*(10**attempt):.1e} to diagonal")
                g = g + torch.eye(g.shape[-1], device=g.device) * (eps * (10**attempt))
        raise RuntimeError(f"Cholesky failed after {max_attempts} attempts")

class IntLogLikMF(torch.nn.Module):
    "Class for computing the integrated log-likelihood for multifidelity data."
    def __init__(self, R: int, nug_mult: float = 4.0):
        super().__init__()
        self.nug_mult = torch.tensor(nug_mult)
        self.R = R

    def precalc(
        self, kernel_result: KernelMFResult, response, fidelity_sizes: torch.Tensor, r
    ) -> _PreCalcLogLikMF:
        """Precomputes the values needed for the integrated log-likelihood at fidelity r."""
        fs = fidelity_sizes
        nug_mean = kernel_result.nug_mean
        nug_sd = nug_mean.mul(self.nug_mult)
        alpha = nug_mean.pow(2).div(nug_sd.pow(2)).add(2)
        beta = nug_mean.mul(alpha.sub(1))
        start = sum(fs[:r])
        end = sum(fs[: r + 1])
        # Get appropriate responses for fidelity r
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

        return _PreCalcLogLikMF(
            nug_sd=nug_sd,
            alpha=alpha,
            beta=beta,
            alpha_post=alpha_post,
            beta_post=beta_post,
            y_tilde=y_tilde,
        )

    def forward(
        self, data: AugmentedData, kernel_result: KernelMFResult, r: int
    ) -> torch.Tensor:
        """Computes the integrated log-likelihood for multifidelity data at fidelity r."""
        tmp_res = self.precalc(
            kernel_result, data.augmented_response[:, :, 0], data.data.fidelity_sizes, r
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
    Wrap everything, this is the full transport map model for multifidelity data.
    It contains the nugget, kernel and integrated log-likelihood modules. 
    This is what you initialize and use for training and inference.
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
        # Augmented data is the data with the conditioning sets
        self.augment_data = AugmentDataMF()
        self.data = data
        # Pass parameters to the different modules appropriately
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
        """Computes the loss for fidelity r."""
        if data is None:
            data = self.data

        fs = data.fidelity_sizes

        aug_data: AugmentedData = self.augment_data(data, batch_idx)
        # Currently, batch_idx is always set to None
        nugget = self.nugget(aug_data, r)
        kernel_result = self.kernel(aug_data, nugget, r)
        intloglik = self.intloglik(aug_data, kernel_result, r)
        # Normalize by data size and fidelity size
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
        num_iter: int,
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

        losses: list[float] = []
        test_losses: list[float] = []
        parameters = []
        values = []
        for r in range(self.R):
            # Restart optimizer/scheduler for each fidelity
            optimizer = torch.optim.Adam(self.parameters(), lr=init_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_iter, eta_min=0.001
            )
            losses.append(self(r).item())
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
                            optimizer.zero_grad()  # type: ignore # optimizer is not None
                            loss = self(r, batch_idx=idx)
                            loss.backward()
                            # Adam has some momentums that will make some parameters keep updating
                            # even if the training in that fidelity is done, so I have to manually
                            # paste the final trained parameter here. Very ugly, but it works.
                            for l in range(r):
                                if j != 0:
                                    # Nugget 1
                                    self.nugget.nugget_params.grad[l] = 0
                                    print(optimizer.state[self.nugget.nugget_params])
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg"
                                    ][l] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg_sq"
                                    ][l] = 0

                                    # Nugget 2
                                    self.nugget.nugget_params.grad[l + self.R] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg"
                                    ][l + self.R] = 0
                                    optimizer.state[self.nugget.nugget_params][
                                        "exp_avg_sq"
                                    ][l + self.R] = 0

                                    # Sigma 1
                                    self.kernel.sigma_params.grad[l] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg"
                                    ][l] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg_sq"
                                    ][l] = 0

                                    # Sigma 2
                                    self.kernel.sigma_params.grad[l + self.R] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg"
                                    ][l + self.R] = 0
                                    optimizer.state[self.kernel.sigma_params][
                                        "exp_avg_sq"
                                    ][l + self.R] = 0

                                    # Theta q 0
                                    self.kernel.theta_q.grad[l] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg"][
                                        l
                                    ] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg_sq"][
                                        l
                                    ] = 0

                                    # Theta q 1
                                    self.kernel.theta_q.grad[l + self.R] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg"][
                                        l + self.R
                                    ] = 0
                                    optimizer.state[self.kernel.theta_q]["exp_avg_sq"][
                                        l + self.R
                                    ] = 0

                                    # Lengthscales
                                    self.kernel.lengthscale.grad[l] = 0
                                    optimizer.state[self.kernel.lengthscale]["exp_avg"][
                                        l
                                    ] = 0
                                    optimizer.state[self.kernel.lengthscale][
                                        "exp_avg_sq"
                                    ][l] = 0

                                    if l > 0:
                                        # theta q 0 preb
                                        self.kernel.theta_q.grad[l + 2 * self.R - 1] = 0
                                        optimizer.state[self.kernel.theta_q]["exp_avg"][
                                            l + 2 * self.R - 1
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q][
                                            "exp_avg_sq"
                                        ][l + 2 * self.R - 1] = 0

                                        # Theta q 1 preb
                                        self.kernel.theta_q.grad[l + 3 * self.R - 2] = 0
                                        optimizer.state[self.kernel.theta_q]["exp_avg"][
                                            l + 3 * self.R - 2
                                        ] = 0
                                        optimizer.state[self.kernel.theta_q][
                                            "exp_avg_sq"
                                        ][l + 3 * self.R - 2] = 0

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
            test_losses=test_losses if test_data is not None else None,
            param_chain=param_chain,
            tracked_chain=tracked_chain,
        )
    
    @torch.no_grad()
    def score(self, obs: torch.Tensor, r: int = 0, last_ind: int | None =None):
        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        # Score at every pixel for all fidelities
        scores = torch.zeros(sum(fs))

        if last_ind is None:
            last_ind = N

        # Compute scores from fidelity r to the last fidelity

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
                scores[idx] = (
                    STDist.log_prob((obs[idx] - meanPred) / initVar.sqrt())
                    - 0.5 * initVar.log()
                )
        # Return the negative log-likelihood scores
        # We sum over all fidelities from r to the last fidelity, the scores
        return -scores[sum(fs[:r]) :].sum()  # , scores

    def cond_sample(self, obs: torch.Tensor, r: int = 0, 
                    last_ind: int | None = None, num_samples: int = 1):
        augmented_data = self.augment_data(self.data, None)
        data = self.data.response
        NN = self.data.conditioning_sets
        fs = augmented_data.data.fidelity_sizes
        n, N = data.shape
        max_m = self.data.max_m

        x_fix = obs[: sum(fs[:r])]

        if last_ind is None:
            last_ind = N
        x_new = torch.empty((num_samples, N), dtype=torch.float64)
        x_new[:, : sum(fs[0:r])] = x_fix.repeat(num_samples, 1)
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
                    cStar = torch.zeros((num_samples, n))
                    prVar = torch.zeros((num_samples,))
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
                nugget = invGDist.sample((num_samples,))
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
