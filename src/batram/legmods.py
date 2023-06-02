import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from gpytorch.kernels import MaternKernel
from matplotlib import pyplot as plt
from pyro.distributions import InverseGamma
from torch.distributions import Normal
from torch.distributions.studentT import StudentT
from tqdm import tqdm


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
    MaternObj._set_lengthscale(1.0)
    MaternObj.requires_grad_(False)
    lenScal = range_fun(theta) * math.sqrt(2 * smooth)
    nonlin = MaternObj.forward(X1s.div(lenScal), X2s.div(lenScal))
    nonlin = sigma.pow(2).reshape(-1, 1, 1) * nonlin
    return (lin + nonlin).div(nuggetMean)


def compute_scale(ordered_locs, NN):
    """Computes scaling for the data.

    Args:
        ordered_locs: Locations of the data. shape (N, d)
            Each row is one location in a d-dimensional space.

        NN: Conditioning sets of the data. Shape (N, m)
            Each row represents one location with references to the m nearest
            neighbors. -1 indicates not to condition on more neighbors.
    """
    ordered_locs = torch.as_tensor(ordered_locs)
    scal = (ordered_locs[1:, :] - ordered_locs[NN[1:, 0], :]).square().sum(1).sqrt()
    scal = torch.cat((scal[0].square().div(scal[4]).unsqueeze(0), scal))
    scal = scal.div(scal[0])
    return scal


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
    def response(self):
        return self.augmented_response[:, :, 0]

    @property
    def response_neighbor_values(self):
        return self.augmented_response[:, :, 1:]


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
        # batched_data = data[batch_idx]
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


class ParameterBox(torch.nn.Module):
    def __init__(self, theta) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(theta)

    def forward(self) -> torch.Tensor:
        return self.theta


class Nugget(torch.nn.Module):
    def __init__(self, theta: ParameterBox) -> None:
        super().__init__()
        self.theta = theta

    def forward(self, data: AugmentedData) -> torch.Tensor:
        theta = self.theta()
        index = torch.arange(data.scales.numel())
        nugget_mean = nug_fun(index, theta, data.scales)
        nugget_mean = torch.relu(nugget_mean.sub(1e-5)).add(1e-5)
        return nugget_mean


class TransportMapKernel(torch.nn.Module):
    def __init__(
        self, theta: ParameterBox, smooth: float = 1.5, fix_m: int | None = None
    ) -> None:
        super().__init__()
        self.fix_m = fix_m
        self.theta = theta
        self.smooth = smooth
        self._tracked_values: dict["str", torch.Tensor] = {}

    def _determin_m(self, theta, max_m):
        m: torch.Tensor
        if self.fix_m is None:
            m = m_threshold(theta, max_m)
        else:
            m = torch.tensor(self.fix_m)

        return m

    def kernel_fun(self, X1, theta, sigma, smooth, nuggetMean=None, X2=None):
        """
        The kernel function should probably be part of the kernel. However, the
        kernel itself does more than just evaluating the kernel, e.g.,
        rescaling. To sample or calculate the (out of sample score), we need
        access to the plain kernel function.
        """

        return kernel_fun(X1, theta, sigma, smooth, nuggetMean, X2)

    def forward(
        self, data: AugmentedData, nug_mean: torch.Tensor, new_method: bool = True
    ) -> KernelResult:  # FIXME: Why is this linting with inconclusive types?
        if new_method:
            return self.new_forward(data, nug_mean)
        else:
            return self.old_forward(data, nug_mean)

    def new_forward(self, data: AugmentedData, nug_mean: torch.Tensor) -> KernelResult:
        """A drop-in, parallel replacement for the legacy kernel."""
        theta = self.theta()
        max_m = data.augmented_response.shape[-1] - 1
        m = self._determin_m(theta, max_m)
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
        sigmas = sigma_fun(torch.arange(data.scales.numel()), theta, data.scales)
        k = kernel_fun(x, theta, sigmas, self.smooth, nug_mean_reshaped)
        eyes = torch.eye(k.shape[-1]).expand_as(k)
        g = k + eyes

        g[data.batch_idx == 0] = torch.eye(k.shape[-1])
        try:
            g_chol = torch.linalg.cholesky(g)
        except RuntimeError as e:
            # One contrast between the errors we return here and the ones in the
            # other function is that here we don't know which Cholesky factor
            # failed based on this message. It would be good to inherit the
            # torch.linalg.LinAlgError and make a more informative error message
            # with it.
            raise RuntimeError("Failed to compute Cholesky decomposition of G.") from e

        # Here we have talked about changing the response to be only the g
        # matrices or simply the kernel. This requires further thought still.
        return KernelResult(g, g_chol, nug_mean)

    def old_forward(self, data: AugmentedData, nug_mean: torch.Tensor) -> KernelResult:
        """Roughly equivalent to the legacy kernel ops."""
        theta = self.theta()
        n = data.augmented_response.shape[0]
        N = data.batch_size
        m = self._determin_m(theta, data.augmented_response.shape[-1] - 1)
        self._tracked_values["m"] = m

        K = torch.zeros(N, n, n)
        G = torch.zeros(N, n, n)
        # Prior vars
        nugMean = nug_mean.squeeze()

        # Compute G, GChol
        for i in range(N):
            if data.batch_idx[i] == 0:
                G[i, :, :] = torch.eye(n)
            else:
                ncol = torch.minimum(data.batch_idx[i], m) + 1
                X = data.augmented_response[:, i, 1:ncol]  # type: ignore[misc]

                K[i, :, :] = self.kernel_fun(
                    X,
                    theta,
                    sigma_fun(i, theta, data.scales),
                    self.smooth,
                    nugMean[i],
                )  # n X n
                G[i, :, :] = K[i, :, :] + torch.eye(n)  # n X n
        try:
            GChol = torch.linalg.cholesky(G)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to compute Cholesky decomposition for observation "
                f"{data.batch_idx[i]}."
            ) from e

        return KernelResult(G=G, GChol=GChol, nug_mean=nugMean)


class _PreCalcLogLik(NamedTuple):
    nug_sd: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    alpha_post: torch.Tensor
    beta_post: torch.Tensor
    y_tilde: torch.Tensor


class IntLogLik(torch.nn.Module):
    def __init__(self, theta: ParameterBox, nugMult: float = 4.0):
        super().__init__()
        self.nugMult = torch.tensor(4.0)
        self.theta = theta

    def precalc(self, kernel_result: KernelResult, response) -> _PreCalcLogLik:
        nugSd = kernel_result.nug_mean.mul(self.nugMult)  # shape (N,)
        alpha = kernel_result.nug_mean.pow(2).div(nugSd.pow(2)).add(2)  # shape (N,)
        beta = kernel_result.nug_mean.mul(alpha.sub(1))  # shape (N,)

        n = response.shape[0]
        yTilde = torch.linalg.solve_triangular(
            kernel_result.GChol, response.t().unsqueeze(-1), upper=False
        ).squeeze()  # (N, n)
        alphaPost = alpha.add(n / 2)  # (N),
        betaPost = beta + yTilde.square().sum(dim=1).div(2)  # (N,)
        return _PreCalcLogLik(
            nug_sd=nugSd,
            alpha=alpha,
            beta=beta,
            alpha_post=alphaPost,
            beta_post=betaPost,
            y_tilde=yTilde,
        )

    def forward(self, data: AugmentedData, kernel_result: KernelResult) -> torch.Tensor:
        tmp_res = self.precalc(kernel_result, data.augmented_response[:, :, 0])

        # integrated likelihood
        logdet = kernel_result.GChol.diagonal(dim1=-1, dim2=-2).log().sum(dim=1)  # (N,)
        loglik = (
            -logdet
            + tmp_res.alpha.mul(tmp_res.beta.log())
            - tmp_res.alpha_post.mul(tmp_res.beta_post.log())
            + tmp_res.alpha_post.lgamma()
            - tmp_res.alpha.lgamma()
        )  # (N,)

        assert (
            loglik.isfinite().all().item()
        ), "Log-likelihood contains non finite values."

        return loglik


class SimpleTM(torch.nn.Module):
    """"""

    def __init__(
        self,
        data: Data,
        theta_init: torch.Tensor,
        linear=False,
        smooth: float = 1.5,
        nugMult: float = 4.0,
        new_method: bool = True,
    ) -> None:
        super().__init__()

        assert linear is False, "Linear TM not implemented yet."

        self.theta = ParameterBox(theta_init)
        self.augment_data = AugmentData()
        self.nugget = Nugget(self.theta)
        self.transport_map_kernel = TransportMapKernel(self.theta, smooth=smooth)
        self.intloglik = IntLogLik(self.theta, nugMult=nugMult)
        self.data = data
        self._new_method = new_method
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
        kernel_result = self.transport_map_kernel(
            aug_data, nugget, new_method=self._new_method
        )
        intloglik = self.intloglik(aug_data, kernel_result)

        loss = -aug_data.data_size / aug_data.batch_size * intloglik.sum()
        return loss

    def cond_sample(
        self,
        obs=None,
        xFix=torch.tensor([]),
        indLast=None,
        mode: str = "bayes",
        num_samples: int = 1,
    ):
        """
        I'm not sure where this should exactly be implemented.

        I guess, best ist in the likelihood nn.Module but it needs access to the
        kernel module as well.

        In any case, this class should expose an interface.
        """

        if obs is not None:
            raise ValueError(
                "The argument obs is not used and will be removed in a future version."
            )

        if mode != "bayes":
            raise NotImplementedError("No modes other than bayes implemented")

        augmented_data: AugmentedData = self.augment_data(self.data, None)

        data = self.data.response
        NN = self.data.conditioning_sets
        theta = self.theta()
        scal = augmented_data.scales
        self.intloglik.nugMult
        # nugMult = self.intloglik.nugMult  # not used
        smooth = self.transport_map_kernel.smooth

        nug_mean = self.nugget(augmented_data)
        kernel_result = self.transport_map_kernel.forward(augmented_data, nug_mean)
        nugMean = kernel_result.nug_mean
        chol = kernel_result.GChol
        tmp_res = self.intloglik.precalc(kernel_result, augmented_data.response)
        yTilde = tmp_res.y_tilde
        betaPost = tmp_res.beta_post
        alphaPost = tmp_res.alpha_post
        n, N = data.shape
        m = NN.shape[1]
        if indLast is None:
            indLast = N
        # loop over variables/locations
        xNew = torch.empty((num_samples, N))
        xNew[:, : xFix.size(0)] = xFix.repeat(num_samples, 1)
        xNew[:, xFix.size(0) :] = 0.0
        for i in range(xFix.size(0), indLast):
            # predictive distribution for current sample
            if i == 0:
                cStar = torch.zeros((num_samples, n))
                prVar = torch.zeros((num_samples,))
            else:
                ncol = min(i, m)
                X = data[:, NN[i, :ncol]]
                XPred = xNew[:, NN[i, :ncol]].unsqueeze(1)
                cStar = self.transport_map_kernel.kernel_fun(
                    XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i], X
                ).squeeze(1)
                prVar = self.transport_map_kernel.kernel_fun(
                    XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i]
                ).squeeze((1, 2))
            cChol = torch.linalg.solve_triangular(
                chol[i, :, :], cStar.unsqueeze(-1), upper=False
            ).squeeze(-1)
            meanPred = yTilde[i, :].unsqueeze(0).mul(cChol).sum(1)
            varPredNoNug = prVar - cChol.square().sum(1)

            # sample
            invGDist = InverseGamma(concentration=alphaPost[i], rate=betaPost[i])
            nugget = invGDist.sample((num_samples,))
            uniNDist = Normal(loc=meanPred, scale=nugget.mul(1.0 + varPredNoNug).sqrt())
            xNew[:, i] = uniNDist.sample()

        return xNew

    def score(self, obs, xFix=torch.tensor([]), indLast=None, mode: str = "score"):
        """
        I'm not sure where this should exactly be implemented.

        I guess, best ist in the likelihood nn.Module but it needs access to the
        kernel module as well.

        In any case, this class should expose an interface.

        Also, this function shares a lot of code with cond sample. that should
        be refactored
        """

        if mode != "score":
            raise NotImplementedError("No modes other than 'score' implemented.")

        augmented_data: AugmentedData = self.augment_data(self.data, None)

        data = self.data.response
        NN = self.data.conditioning_sets
        # response = augmented_data.response
        # response_neighbor_values = augmented_data.response_neighbor_values
        theta = self.theta()
        scal = augmented_data.scales
        self.intloglik.nugMult
        # nugMult = self.intloglik.nugMult  # not used
        smooth = self.transport_map_kernel.smooth

        nug_mean = self.nugget(augmented_data)
        kernel_result = self.transport_map_kernel.forward(augmented_data, nug_mean)
        nugMean = kernel_result.nug_mean
        chol = kernel_result.GChol
        tmp_res = self.intloglik.precalc(kernel_result, augmented_data.response)
        yTilde = tmp_res.y_tilde
        betaPost = tmp_res.beta_post
        alphaPost = tmp_res.alpha_post
        n, N = data.shape
        m = NN.shape[1]
        if indLast is None:
            indLast = N
        # loop over variables/locations
        scr = torch.zeros(N)
        for i in range(xFix.size(0), indLast):
            # predictive distribution for current sample
            if i == 0:
                cStar = torch.zeros(n)
                prVar = torch.tensor(0.0)
            else:
                ncol = min(i, m)
                X = data[:, NN[i, :ncol]]
                XPred = obs[NN[i, :ncol]].unsqueeze(
                    0
                )  # this line is different than in cond_sampl
                cStar = self.transport_map_kernel.kernel_fun(
                    XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i], X
                ).squeeze()
                prVar = self.transport_map_kernel.kernel_fun(
                    XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i]
                ).squeeze()
            cChol = torch.linalg.solve_triangular(
                chol[i, :, :], cStar.unsqueeze(1), upper=False
            ).squeeze()
            meanPred = yTilde[i, :].mul(cChol).sum()
            varPredNoNug = prVar - cChol.square().sum()

            # score
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            STDist = StudentT(2 * alphaPost[i])
            scr[i] = (
                STDist.log_prob((obs[i] - meanPred) / initVar.sqrt())
                - 0.5 * initVar.log()
            )

        return scr[xFix.size(0) :].sum()

    def fit(
        self,
        num_iter,
        init_lr: float,
        batch_size: None | int = None,
        test_data: Data | None = None,
        optimizer: None | torch.optim.Optimizer = None,
        scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
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
        silent
            If True, do not print progress.
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

        if batch_size is None:
            batch_size = self.data.response.shape[1]

        data_size = self.data.response.shape[1]
        if batch_size > data_size:
            raise ValueError(
                f"Batch size {batch_size} is larger than data size {data_size}."
            )

        losses: list[float] = [self().item()]
        test_losses: list[float] = (
            [] if test_data is None else [self(data=test_data).item()]
        )
        parameters = [
            {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
        ]
        values = [
            {k: np.copy(v.detach().numpy()) for k, v in self.named_tracked_values()}
        ]

        tqdm_obj = tqdm(range(num_iter), disable=silent)
        for _ in tqdm_obj:
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
            if test_data is not None:
                with torch.no_grad():
                    test_losses.append(self(data=test_data).item())
                desc += f", Test Loss: {test_losses[-1]:.3f}"

            # store parameters and values
            parameters.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_parameters()}
            )
            values.append(
                {k: np.copy(v.detach().numpy()) for k, v in self.named_tracked_values()}
            )

            tqdm_obj.set_description(desc)

        param_chain = {}
        for k in parameters[0].keys():
            param_chain[k] = np.stack([d[k] for d in parameters], axis=0)

        tracked_chain = {}
        for k in values[0].keys():
            tracked_chain[k] = np.stack([d[k] for d in values], axis=0)

        return FitResult(
            model=self,
            losses=np.array(losses),
            parameters=parameters[-1],
            test_losses=np.array(test_losses) if test_data is not None else None,
            param_chain=param_chain,
            tracked_chain=tracked_chain,
        )


@dataclass
class FitResult:
    """
    Result of a fit.
    """

    model: SimpleTM
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
            twin = ax.twinx()
            (p2,) = twin.plot(self.test_losses, "C1", label="Test Loss", **kwargs)
            legend_handle.append(p2)
            twin.set_ylabel("Test Loss")

        if use_inset:
            end_idx = len(self.losses)
            start_idx = int(0.8 * end_idx)
            inset_iterations = np.arange(start_idx, end_idx)

            inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
            inset.plot(inset_iterations, self.losses[start_idx:], "C0", **kwargs)

            if self.test_losses is not None:
                insert_twim = inset.twinx()
                insert_twim.plot(
                    inset_iterations, self.test_losses[start_idx:], "C1", **kwargs
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Train Loss")
        ax.legend(handles=legend_handle, loc="lower right", bbox_to_anchor=(0.925, 0.2))

        return ax
