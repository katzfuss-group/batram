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

from .base_functions import compute_scale
from .stopper import PEarlyStopper


def nug_fun(i, theta, scales):
    """Scales nugget (d) at location i."""
    return torch.exp(torch.log(scales[i]).mul(theta[1]).add(theta[0]))


def scaling_fun(k, theta):
    """Scales distance to kth nearest neighbors."""
    theta_pos = torch.exp(theta[2])
    return torch.sqrt(torch.exp(-k * theta_pos))


def sigma_fun(i, theta, scales):
    """Scales nonlinearities (sigma) at location i."""
    return torch.exp(torch.log(scales[i]).mul(theta[4]).add(theta[3]))


def range_fun(theta):
    """Sets lengthscale of nonlinear kernel transformation."""
    return torch.exp(theta[5])


def varscale_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[7]).add(theta[6]))


def con_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[9]).add(theta[8]))


def m_threshold(theta, m_max) -> torch.Tensor:
    """Determines a threshold for m nearest neighbors."""
    rng = torch.arange(m_max + 1) + 1
    mask = scaling_fun(rng, theta) >= 0.01
    m = mask.sum()

    if m <= 0:
        raise RuntimeError(
            f"Size of conditioning set is less than or equal to zero; m = {m.item()}."
        )
    if m > m_max:
        logging.warning(
            f"Required size of the conditioning sets m = {m.item()} is greater than "
            f"the maximum number of neighbors {m_max = } in the pre-calculated "
            "conditioning sets."
        )
        m = torch.tensor(m_max)
    return m


def kernel_fun(X1, theta, sigma, smooth, nuggetMean=None, X2=None):
    N = X1.shape[-1]  # size of the conditioning set

    if X2 is None:
        X2 = X1
    if nuggetMean is None:
        nuggetMean = 1
    X1s = X1.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    X2s = X2.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    lin = X1s @ X2s.mT
    MaternObj = MaternKernel(smooth)
    MaternObj._set_lengthscale(torch.tensor(1.0))
    MaternObj.requires_grad_(False)
    lenScal = range_fun(theta) * math.sqrt(2 * smooth)
    nonlin = MaternObj.forward(X1s.div(lenScal), X2s.div(lenScal))
    nonlin = sigma.pow(2).reshape(-1, 1, 1) * nonlin
    return (lin + nonlin).div(nuggetMean)


@dataclass
class Data:
    """Data class

    Holds $n$ replicates of spatial field observed at $N$ locations. The data in
    this class has not been normalized.

    Note
    ----
    scales refers scaled distance to the nearest neighbor.

    Attributes
    ----------
    locs
        Locations of the data. shape (N, d)
    response
        Response of the data. Shape (n, N)

    augmented_response
        Augmented response of the data. Shape (n, N, m + 1). nan indicates no
        conditioning.

    conditioning_sets
        Conditioning sets of the data. Shape (N, m)
    """

    locs: torch.Tensor
    response: torch.Tensor
    augmented_response: torch.Tensor
    conditioning_sets: torch.Tensor

    @staticmethod
    def new(locs, response, conditioning_set):
        """Creates a new data object."""
        nlocs = locs.shape[0]
        ecs = torch.hstack([torch.arange(nlocs).reshape(-1, 1), conditioning_set])
        augmented_response = torch.where(ecs == -1, torch.nan, response[:, ecs])

        return Data(locs, response, augmented_response, conditioning_set)


@dataclass
class AugmentedData:
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

    @property
    def response(self) -> torch.Tensor:
        return self.augmented_response[:, :, 0]

    @property
    def response_neighbor_values(self) -> torch.Tensor:
        return self.augmented_response[:, :, 1:]

    @property
    def max_m(self) -> int:
        return self.augmented_response.shape[-1] - 1


class AugmentData(torch.nn.Module):
    """Augments data

    Right now this just adds the scales to the data and creates a batch based on
    the provided index.

    Calculating the scales could be cached.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, data: Data, batch_idx: None | torch.Tensor = None
    ) -> AugmentedData:
        if batch_idx is None:
            batch_idx = torch.arange(data.response.shape[1])
        scales = compute_scale(data.locs, data.conditioning_sets)

        return AugmentedData(
            data_size=data.response.shape[1],
            batch_size=batch_idx.shape[0],
            batch_idx=batch_idx,
            locs=data.locs[batch_idx, :],
            augmented_response=data.augmented_response[:, batch_idx, :],
            scales=scales[batch_idx],
            data=data,
        )


@dataclass
class KernelResult:
    G: torch.Tensor
    GChol: torch.Tensor
    nug_mean: torch.Tensor


class Nugget(torch.nn.Module):
    def __init__(self, nugget_params: torch.Tensor) -> None:
        super().__init__()
        assert nugget_params.shape == (2,)
        self.nugget_params = torch.nn.Parameter(nugget_params)

    def forward(self, data: AugmentedData) -> torch.Tensor:
        theta = self.nugget_params
        nugget_mean = (theta[0] + theta[1] * data.scales.log()).exp()
        nugget_mean = torch.relu(nugget_mean.sub(1e-5)).add(1e-5)
        return nugget_mean


class TransportMapKernel(torch.nn.Module):
    """A temporary class to refactor the `TransportMapKernel`.

    Once the refactor is complete, replace the `TransportMapKernel` with this
    class. To complete the refactor, we will need to modify the `SimpleTM`
    constructor slightly.
    """

    def __init__(
        self,
        kernel_params: torch.Tensor,
        smooth: float = 1.5,
        fix_m: int | None = None,
    ) -> None:
        super().__init__()

        assert kernel_params.numel() == 4
        self.theta_q = torch.nn.Parameter(kernel_params[0])
        self.sigma_params = torch.nn.Parameter(kernel_params[1:3])
        self.lengthscale = torch.nn.Parameter(kernel_params[-1])

        self.fix_m = fix_m
        self.smooth = smooth
        self._tracked_values: dict["str", torch.Tensor] = {}

        matern = MaternKernel(smooth)
        matern._set_lengthscale(torch.tensor(1.0))
        matern.requires_grad_(False)
        self._kernel = matern

    def _sigmas(self, scales: torch.Tensor) -> torch.Tensor:
        """Computes nonlinear scales for the kernel."""
        params = self.sigma_params
        return torch.exp(params[0] + params[1] * scales.log())

    def _scale(self, k: torch.Tensor) -> torch.Tensor:
        """Computes scales of nearest neighbors in the kernel."""
        theta_pos = torch.exp(self.theta_q)
        return torch.exp(-0.5 * k * theta_pos)

    def _range(self) -> torch.Tensor:
        """Computes the lengthscale of the kernel."""
        return self.lengthscale.exp()

    def _m_threshold(self, max_m: int) -> torch.Tensor:
        """Computes the size of the conditioning sets."""
        rng = torch.arange(max_m) + 1
        mask = self._scale(rng) >= 0.01
        m = mask.sum()

        if m <= 0:
            raise RuntimeError(
                f"Size of conditioning set not positive; m = {m.item()}."
            )
        if m > max_m:
            logging.warning(
                f"Required size of the conditioning sets m = {m.item()} is "
                f"greater than the maximum number of neighbors {max_m = } in "
                "the pre-calculated conditioning sets."
            )
            m = torch.tensor(max_m)
        return m

    def _determine_m(self, max_m: int) -> torch.Tensor:
        m: torch.Tensor
        if self.fix_m is None:
            m = self._m_threshold(max_m)
        else:
            m = torch.tensor(self.fix_m)

        return m

    def _kernel_fun(
        self,
        x1: torch.Tensor,
        sigmas: torch.Tensor,
        nug_mean: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the transport map kernel."""
        k = torch.arange(x1.shape[-1]) + 1
        scaling = self._scale(k)

        _x1 = x1 * scaling
        _x2 = _x1 if x2 is None else x2 * scaling
        linear = _x1 @ _x2.mT

        ls = self._range() * math.sqrt(2 * self.smooth)
        nonlinear = self._kernel(_x1 / ls, _x2 / ls).to_dense()
        out = (linear + sigmas**2 * nonlinear) / nug_mean
        return out

    def forward(self, data: AugmentedData, nug_mean: torch.Tensor) -> KernelResult:
        max_m = data.max_m
        m = self._determine_m(max_m)
        self._tracked_values["m"] = m
        assert m <= max_m

        x = data.augmented_response[..., 1 : (m + 1)]
        x = torch.where(torch.isnan(x), 0.0, x)
        # Want the spatial dim in the first position for kernel computations,
        # so data follow (..., N, n, fixed_m) instead of (..., n, N, m) as in
        # the original kernel implementation. Doing this with an eye towards
        # parallelism.
        x = x.permute(-2, -3, -1)

        nug_mean_reshaped = nug_mean.reshape(-1, 1, 1)
        sigmas = self._sigmas(data.scales).reshape(-1, 1, 1)
        k = self._kernel_fun(x, sigmas, nug_mean_reshaped)

        eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
        g = k + eye

        g[data.batch_idx == 0] = eye
        try:
            g_chol = torch.linalg.cholesky(g)
        except RuntimeError as e:
            # TODO: Use torch.linalg.LinAlgError to produce a better error message.
            raise RuntimeError("Failed to compute Cholesky decomposition of G.") from e

        return KernelResult(g, g_chol, nug_mean)


class _PreCalcLogLik(NamedTuple):
    nug_sd: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    alpha_post: torch.Tensor
    beta_post: torch.Tensor
    y_tilde: torch.Tensor


class IntLogLik(torch.nn.Module):
    def __init__(self, nug_mult: float = 4.0):
        super().__init__()
        self.nug_mult = torch.tensor(nug_mult)

    def precalc(self, kernel_result: KernelResult, response) -> _PreCalcLogLik:
        nug_mean = kernel_result.nug_mean.squeeze()  # had shape (N, 1, 1)
        nug_sd = nug_mean.mul(self.nug_mult).squeeze()  # shape (N,)
        alpha = nug_mean.pow(2).div(nug_sd.pow(2)).add(2)  # shape (N,)
        beta = nug_mean.mul(alpha.sub(1))  # shape (N,)

        assert nug_sd.shape == (response.shape[1],)
        assert alpha.shape == (response.shape[1],)
        assert beta.shape == (response.shape[1],)

        n = response.shape[0]
        y_tilde = torch.linalg.solve_triangular(
            kernel_result.GChol, response.t().unsqueeze(-1), upper=False
        ).squeeze()  # (N, n)
        alpha_post = alpha.add(n / 2)  # (N),
        beta_post = beta + y_tilde.square().sum(dim=1).div(2)  # (N,)

        assert alpha_post.shape == (response.shape[1],)
        assert beta_post.shape == (response.shape[1],)
        return _PreCalcLogLik(
            nug_sd=nug_sd,
            alpha=alpha,
            beta=beta,
            alpha_post=alpha_post,
            beta_post=beta_post,
            y_tilde=y_tilde,
        )

    def forward(self, data: AugmentedData, kernel_result: KernelResult) -> torch.Tensor:
        """Computes the integrated log likelihood."""
        tmp_res = self.precalc(kernel_result, data.augmented_response[:, :, 0])

        logdet = kernel_result.GChol.diagonal(dim1=-1, dim2=-2).log().sum(dim=1)  # (N,)

        # Has shape (N, )
        loglik = (
            -logdet
            + tmp_res.alpha.mul(tmp_res.beta.log())
            - tmp_res.alpha_post.mul(tmp_res.beta_post.log())
            + tmp_res.alpha_post.lgamma()
            - tmp_res.alpha.lgamma()
        )

        assert loglik.isfinite().all().item(), (
            "Log-likelihood contains non finite values."
        )

        return loglik


@dataclass
class _PredictionSamplingContex:
    """Intermediate values for predictive density computations.

    This is a module used inside of `SimpleTM` and is not intended for external
    use. It provides helpers to instantiate some useful values with named
    access.
    """

    augmented_data: AugmentedData
    theta: torch.Tensor
    scales: torch.Tensor
    sigmas: torch.Tensor
    kernel_result: KernelResult
    precalc_ll: _PreCalcLogLik


class SimpleTM(torch.nn.Module):
    """TODO: Add docs"""

    def __init__(
        self,
        data: Data,
        theta_init: None | torch.Tensor = None,
        linear: bool = False,
        smooth: float = 1.5,
        nug_mult: float = 4.0,
    ) -> None:
        super().__init__()

        if linear:
            raise ValueError("Linear TM not implemented yet.")

        if theta_init is None:
            # This is essentially \log E[y^2] over the spatial dim
            # to initialize the nugget mean.
            log_2m = data.response[:, 0].square().mean().log()
            theta_init = torch.tensor([log_2m, 0.2, 0.0, 0.0, 0.0, -1.0])

        self.augment_data = AugmentData()
        self.nugget = Nugget(theta_init[:2])
        self.kernel = TransportMapKernel(theta_init[2:], smooth=smooth)
        self.intloglik = IntLogLik(nug_mult=nug_mult)
        self.data = data
        self._tracked_values: dict[str, torch.Tensor] = {}

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

    def forward(
        self, batch_idx: None | torch.Tensor = None, data: None | Data = None
    ) -> torch.Tensor:
        if data is None:
            data = self.data

        aug_data: AugmentedData = self.augment_data(data, batch_idx)
        nugget = self.nugget(aug_data)
        kernel_result = self.kernel(aug_data, nugget)
        intloglik = self.intloglik(aug_data, kernel_result)

        loss = -aug_data.data_size / aug_data.batch_size * intloglik.sum()
        return loss

    def _init_ctx(self):
        """Initializes context we need for conditional sampling and scoring.

        NOTE: This is an internal-only piece of code. It should not be required
        to use it externally ever!
        """
        theta = torch.tensor(
            [
                *self.nugget.nugget_params.detach(),
                self.kernel.theta_q.detach(),
                *self.kernel.sigma_params.detach(),
                self.kernel.lengthscale.detach(),
            ]
        )

        augmented_data: AugmentedData = self.augment_data(self.data, None)
        scales = augmented_data.scales
        sigmas = self.kernel._sigmas(scales)
        nug_mean = self.nugget(augmented_data)
        kernel_result = self.kernel.forward(augmented_data, nug_mean)
        precalc_ll = self.intloglik.precalc(kernel_result, augmented_data.response)

        return _PredictionSamplingContex(
            augmented_data=augmented_data,
            theta=theta,
            scales=scales,
            sigmas=sigmas,
            kernel_result=kernel_result,
            precalc_ll=precalc_ll,
        )

    def _update_posi(self, i, yn, ctx):
        """Calc parameters of the ith density for conditional sampling and scoring."""
        n, _ = self.data.response.shape
        nbrs = self.data.conditioning_sets
        m = nbrs.shape[1]
        nugget_mean = ctx.kernel_result.nug_mean
        sigmas = ctx.sigmas

        # Notation:
        #   - c00: The covariance for (y0, y0) for training data y0
        #   - c11: The covariance for (y1, y1) for prediction data y1
        #   - c10: The covariance for (y1, y0)
        if i == 0:
            c10 = torch.zeros((yn.shape[0], n))
            c11 = torch.zeros(yn.shape[0])
        else:
            ncol = min(i, m)
            y0 = self.data.response[:, nbrs[i, :ncol]]
            y1 = yn[:, nbrs[i, :ncol]]
            c10 = self.kernel._kernel_fun(y1, sigmas[i], nugget_mean[i], y0)
            c11 = self.kernel._kernel_fun(y1, sigmas[i], nugget_mean[i], y1)
            c11 = torch.diag(c11)

        L = ctx.kernel_result.GChol[i]
        v = torch.linalg.solve_triangular(L, c10.mT, upper=False)

        y_tilde = ctx.precalc_ll.y_tilde[i, :]
        mean_pred = torch.sum(v * y_tilde[:, None], dim=0)
        var_pred_no_nugget = c11 - torch.sum(v**2, dim=0)

        return mean_pred, var_pred_no_nugget

    def cond_sample(
        self,
        x_fix=torch.tensor([]),
        last_ind=None,
        num_samples: int = 1,
    ):
        """
        I'm not sure where this should exactly be implemented.

        I guess, best ist in the likelihood nn.Module but it needs access to the
        kernel module as well.

        In any case, this class should expose an interface.
        """
        _, N = self.data.response.shape
        if last_ind is None:
            last_ind = N

        x_new = torch.empty((num_samples, N))
        x_new[:, : x_fix.size(0)] = x_fix.repeat(num_samples, 1)
        x_new[:, x_fix.size(0) :] = 0.0

        ctx = self._init_ctx()

        for i in range(x_fix.size(0), last_ind):
            alpha_post = ctx.precalc_ll.alpha_post
            beta_post = ctx.precalc_ll.beta_post
            mean_pred, var_pred_no_nugget = self._update_posi(i, x_new, ctx)

            invGDist = InverseGamma(concentration=alpha_post[i], rate=beta_post[i])
            nugget = invGDist.sample((num_samples,))
            var_pred = torch.sqrt(nugget * (1.0 + var_pred_no_nugget))
            uniNDist = Normal(loc=mean_pred, scale=var_pred)
            x_new[:, i] = uniNDist.sample()

        return x_new

    # QUESTION: Can scoring be done completely in parallel? We should know all
    # of the data values necessary for scoring in complete parallel. We only
    # need a for loop when we're generating samples because none of the data is
    # initialized until we create it.
    #
    # ANSWER: Yes, scoring can be done in parallel!
    def score(self, obs, x_fix=torch.tensor([]), last_ind=None):
        """TODO: Add proper docs

        Parameters
        ----------
        obs
            Sample fields to score. Assumes shape
        x_fix
            Initial learning rate. Only used if optimizer is None.
        last_ind
            The last index to calculate scores up to (typically less than the
            size of the field being scored). This is useful for doing quick
            summaries such as checking the first 30 values of the field instead
            of scoring all N values.
        """

        if isinstance(last_ind, int) and last_ind < x_fix.size(-1):
            raise ValueError("last_ind must be larger than conditioned field x_fix.")

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if x_fix.ndim < 2:
            x_fix = x_fix.reshape(1, -1)

        if last_ind is None:
            _, N = self.data.response.shape
            last_ind = N

        ctx = self._init_ctx()
        score = torch.zeros_like(obs)

        for i in range(x_fix.shape[1], last_ind):
            mean_pred, var_pred_no_nugget = self._update_posi(i, obs, ctx)
            alpha_post = ctx.precalc_ll.alpha_post[i]
            beta_post = ctx.precalc_ll.beta_post[i]

            init_var = beta_post / alpha_post * (1 + var_pred_no_nugget)
            z = (obs[..., i] - mean_pred) / torch.sqrt(init_var)
            tval = StudentT(2 * alpha_post).log_prob(z)
            score[..., i] = tval - 0.5 * init_var.log()

        return score[..., x_fix.size(0) :].sum(-1)

    def fit(
        self,
        num_iter,
        init_lr: float,
        batch_size: None | int = None,
        validation_data: Data | None = None,
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
        validation_data
            Data to use for convergence monitoring. If None, do not validate.
        optimizer
            Optimizer to use. If None, use Adam.
        scheduler
            Learning rate scheduler to use. If None, CosineAnnealingLR
            is used with default optimizer.
        stopper
            An early stopper. If None, no early stopping is used. Requires test data.
        silent
            If True, do not print progress.

        NOTE: The choice of `init_lr` should be informed by the `batch_size`.
        Starting from a batch size of 1, increasing the `batch_size` by a factor
        k means roughly that you should increase the learning rate by the same
        factor k to approximate the same learning rate. The reason for this is
        that minibatching leads to multiple gradient (parameter) updates per
        iteration of the fit method, so increasing the learning rate scales the
        gradients proportionally. See

          Goyal et al. (2017)
          https://arxiv.org/pdf/1706.02677

        for more discussion of how this applies to deep neural networks. The
        findings from that paper apply to fitting this method also. As a
        concrete example:

        ```python
        # Assuming a transport map tm is instantiated with data containing 1024
        # locations. For batch size 32, this implies we can make 32 passes
        # through the data per iteration.
        # tm_batch = legmods.SimpleTM(...)
        # tm_full = legmods.SimpleTM(...)

        # Using a small batch size to fit
        batch_fit = tm.fit(num_iter=100, init_lr=0.01, batch_size=32)

        # Now compare with no minibatching but the learning rate scaled
        full_fit = tm_full.fit(num_iter=100, init_lr=32*0.01, batch_size=None)

        # The loss curves from these will be different because minibatching
        # applies stochastic updates at every step. However, the resulting
        # parameters should be similar.
        ```

        The parameter estimates will be different for these, but the amount by
        which they were scaled should be similar. In the first case, we use 32
        gradient updates, each with learning rate `lr`. In the second case, we
        use one gradient update with learning rate `32 * lr`. (Note the `lr` is
        generally changing from `init_lr` using a learning rate scheduler.
        """

        if optimizer is None:
            if scheduler is not None:
                raise ValueError(
                    "Cannot specify scheduler without speicifying an optimizer."
                )
            optimizer = torch.optim.Adam(self.parameters(), lr=init_lr)
            if scheduler is None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_iter
                )

        if stopper is not None and validation_data is None:
            raise ValueError("Cannot use stopper without test data.")

        if batch_size is None:
            batch_size = self.data.response.shape[1]

        data_size = self.data.response.shape[1]
        if batch_size > data_size:
            raise ValueError(
                f"Batch size {batch_size} is larger than data size {data_size}."
            )

        losses: list[float] = [self().item()]
        validation_losses: list[float] = (
            [] if validation_data is None else [self(data=validation_data).item()]
        )
        parameters = [
            {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
        ]
        values = [
            {k: np.copy(v.detach().numpy()) for k, v in self.named_tracked_values()}
        ]

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

                def closure():
                    optimizer.zero_grad()  # type: ignore # optimizer is not None
                    loss = self(batch_idx=idx)
                    loss.backward()
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
            if validation_data is not None:
                with torch.no_grad():
                    validation_losses.append(self(data=validation_data).item())
                desc += f", Test Loss: {validation_losses[-1]:.3f}"

            # store parameters and values
            parameters.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
            )
            values.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_tracked_values()}
            )

            tqdm_obj.set_description(desc)

            if stopper is not None:
                state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                stop = stopper.step(validation_losses[-1], state)
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

        return FitResult(
            model=self,
            max_m=self.data.conditioning_sets.shape[-1],
            losses=np.array(losses),
            parameters=parameters[-1],
            # NOTE: This is left as `test_losses` for compatibility. We can
            # modify it later if we choose to.
            test_losses=np.array(validation_losses)
            if validation_data is not None
            else None,
            param_chain=param_chain,
            tracked_chain=tracked_chain,
        )


@dataclass
class FitResult:
    """
    Result of a fit.
    """

    model: SimpleTM
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

    def plot_params(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(
            self.param_chain["nugget.nugget_params"],
            label=[f"nugget {i}" for i in range(2)],
            **kwargs,
        )
        ax.plot(
            self.param_chain["kernel.theta_q"],
            label="neighbors scale",
            **kwargs,
        )
        ax.plot(
            self.param_chain["kernel.sigma_params"],
            label=[f"sigma {i}" for i in range(2)],
            **kwargs,
        )
        ax.plot(
            self.param_chain["kernel.lengthscale"],
            label="lengthscale",
            **kwargs,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Raw Parameter Value")
        ax.legend()

        return ax

    def plot_neighbors(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(1, 1)

        m = self.tracked_chain["kernel.m"]
        epochs = np.arange(m.size, **kwargs)
        ax.step(epochs, m)
        ax.set_title("Nearest neighbors through optimization")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of nearest neighbors (m)")
        ax.set_ylim(0, self.max_m)

        return ax
