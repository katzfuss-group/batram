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
from .legmods import Data, AugmentData, AugmentedData

__doc__ = (

"""
This module contains the implementation of the transport map kernel where
the mean and variance of the individual regressions are centered around
the given mean and variances. This mean and variances are values derived
from the parametric kernels.
"""
)

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

############################ HAS BEEN CHANGED FROM BASE FUNCTION ##############################
def kernel_fun(X1, theta, sigma, smooth, nuggetMean=None, X2=None):
    N = X1.shape[-1]  # size of the conditioning set

    if X2 is None:
        X2 = X1
    if nuggetMean is None:
        nuggetMean = 1
    X1s = X1.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    X2s = X2.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    #lin = X1s @ X2s.mT
    MaternObj = MaternKernel(smooth)
    MaternObj._set_lengthscale(torch.tensor(1.0))
    MaternObj.requires_grad_(False)
    lenScal = range_fun(theta) * math.sqrt(2 * smooth)
    nonlin = MaternObj.forward(X1s.div(lenScal), X2s.div(lenScal))
    nonlin = sigma.pow(2).reshape(-1, 1, 1) * nonlin
    return (#lin + 
        nonlin).div(nuggetMean)



def compute_shrinkage_means(data: Data, mean_factor: torch.Tensor) -> torch.Tensor:
    """Computes the means of the individual regressions."""
    # make sure the dimensions match properly
    previous_ordered_responses = data.augmented_response[..., 1:]
    assert mean_factor.shape[1] == previous_ordered_responses.shape[2]
    assert mean_factor.shape[0] == previous_ordered_responses.shape[1]
    previous_ordered_responses = previous_ordered_responses.nan_to_num()
    mean_factor = mean_factor.unsqueeze(-1)
    previous_ordered_responses = previous_ordered_responses.permute(1, 0, 2)
    mean_values = (torch.bmm(previous_ordered_responses, mean_factor)).squeeze().mT
    return mean_values
    

@dataclass
class KernelResult:
    G: torch.Tensor
    GChol: torch.Tensor
    nug_mean: torch.Tensor


# TODO: Deprecate / remove
# The nugget and kernel refactors make this obsolete.
class ParameterBox(torch.nn.Module):
    def __init__(self, theta) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(theta)

    def forward(self) -> torch.Tensor:
        return self.theta

class CalcShrinkFactors(torch.nn.Module):
    def __init__(self, kernel_factor: torch.Tensor, nugget_fector: torch.Tensor) -> None:
        super().__init__()
        self.nugget_shrinkage_factor = torch.nn.Parameter(nugget_fector)
        self.kernel_shrinkage_factor = torch.nn.Parameter(kernel_factor)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        def transform(x: torch.Tensor) -> torch.Tensor:
            y = x.exp().add(1)
            y = 1 - 1 / y
            return y
        return transform(self.nugget_shrinkage_factor), transform(self.kernel_shrinkage_factor)

# TODO: Promote to TransportMapKernel
# This was put in to preserve the existing test suite. We can remove the old
# test suite once we are satisfied with this implementation. This integration
# does not eliminate the need for the helper functions at the start of the file
# but it is a step in that direction. (The helpers are still used in the
# `cond_samp` and `score` methods of `SimpleTM`.) This rewrite allows us to
# more easily extend the covariates case and simplifies how parameters are
# managed throughout the model.
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
        #we consider that fraction shrink is in log(c_f/(1-c_f) 
        #format so range is (-inf, inf).

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
    
    ########################### HAS BEEN CHANGED FROM BASE FUNCTION (ANIRBAN) ###########################
    def _kernel_fun(
        self,
        x1: torch.Tensor,
        sigmas: torch.Tensor,
        nug_mean: torch.Tensor,
        fraction_shrink: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the transport map kernel."""
        k = torch.arange(x1.shape[-1]) + 1
        scaling = self._scale(k)

        _x1 = x1 * scaling
        _x2 = _x1 if x2 is None else x2 * scaling
        #linear = _x1 @ _x2.mT

        ls = self._range() * math.sqrt(2 * self.smooth)
        nonlinear = self._kernel(_x1 / ls, _x2 / ls).to_dense()
        out = (#linear + 
            sigmas**2 * nonlinear) / nug_mean
        
        return fraction_shrink * out

    def forward(self, data: AugmentedData, nug_mean: torch.Tensor, 
                fraction_shrink: torch.Tensor) -> KernelResult:
        """Computes with internalized kernel params instead of ParameterBox."""
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
        k = self._kernel_fun(x, sigmas, nug_mean_reshaped, fraction_shrink)
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


class _PreCalcLogLik(NamedTuple):
    nug_sd: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    alpha_post: torch.Tensor
    beta_post: torch.Tensor
    y_tilde: torch.Tensor


class IntLogLik(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def precalc(self, kernel_result: KernelResult, response: torch.Tensor, 
                f_mean: torch.Tensor, nug_mult: torch.Tensor) -> _PreCalcLogLik:
        nug_mean = kernel_result.nug_mean.squeeze()  # had shape (N, 1, 1)
        nug_sd = nug_mean.mul(nug_mult).squeeze()  # shape (N,)
        alpha = nug_mean.pow(2).div(nug_sd.pow(2)).add(2)  # shape (N,)
        beta = nug_mean.mul(alpha.sub(1))  # shape (N,)

        assert nug_sd.shape == (response.shape[1],)
        assert alpha.shape == (response.shape[1],)
        assert beta.shape == (response.shape[1],)
######################### CHANGED BY ANIRBAN #########################################
        n = response.shape[0]
        y_tilde = torch.linalg.solve_triangular(
            kernel_result.GChol, (response - f_mean).t().unsqueeze(-1), upper=False
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

    def forward(self, data: AugmentedData, kernel_result: KernelResult, 
                nug_mult: torch.Tensor, f_mean: torch.Tensor) -> torch.Tensor:
        tmp_res = self.precalc(kernel_result = kernel_result, 
                               response = data.augmented_response[:, :, 0], 
                               nug_mult = nug_mult, f_mean = f_mean)

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


class ShrinkTM(torch.nn.Module):
    """"""

    def __init__(
        self,
        data: Data,
        shrinkage_mean_factor: torch.Tensor,
        shrinkage_var: torch.Tensor,
        theta_init: None | torch.Tensor = None,
        linear: bool = False,
        smooth: float = 1.5,
        #nug_mult: float = 4.0,
    ) -> None:
        super().__init__()

        if linear:
            raise ValueError("Linear TM not implemented yet.")

        if theta_init is None:
            # This is essentially \log E[y^2] over the spatial dim
            # to initialize the nugget mean.
            #log_2m = data.response[:, 0].square().mean().log()
            theta_init = torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0, -1.0])
            #alignment is (log(c_f/(1-c_f)), log(c_d/(1-c_d)), 
            #\theta_q, \theta_{\sigma,1}, \theta_{\sigma,2}, \theta_{\gamma})
            #alignment of SimpleTM: (\theta_{d,1}, \theta_{d,2}, 
            #\theta_q, \theta_{\sigma,1}, \theta_{\sigma,2}, \theta_{\gamma})

        self.augment_data = AugmentData()
        self.shrinkage_mean = compute_shrinkage_means(data, shrinkage_mean_factor)
        self.shrinkage_var = shrinkage_var
        #self.nugget = Nugget(theta_init[:2])
        self.nugget_shrinkage_factor = torch.nn.Parameter(theta_init[1])
        self.kernel_shrinkage_factor = torch.nn.Parameter(theta_init[0])
        self.kernel = TransportMapKernel(theta_init[2:], smooth=smooth)
        self.intloglik = IntLogLik()
        self.data = data
        self.shrinkage_mean_factor = shrinkage_mean_factor
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
        if batch_idx is None:   
            nugget_mean = self.shrinkage_var
            kernel_mean = self.shrinkage_mean
        else:
            nugget_mean = self.shrinkage_var[batch_idx]
            kernel_mean = self.shrinkage_mean[: ,batch_idx]
        aug_data: AugmentedData = self.augment_data(data, batch_idx)
        
        kernel_mult = self.kernel_shrinkage_factor.exp().add(1)
        kernel_mult = 1 - 1/kernel_mult
        kernel_result = self.kernel(data = aug_data, 
                                    nug_mean = nugget_mean, 
                                    fraction_shrink = kernel_mult)
        nug_mult = self.nugget_shrinkage_factor.exp().add(1)
        nug_mult = 1 - 1/nug_mult
        intloglik = self.intloglik(data = aug_data, 
                                   kernel_result = kernel_result, 
                                   nug_mult = nug_mult, 
                                   f_mean = kernel_mean)

        loss = -aug_data.data_size / aug_data.batch_size * intloglik.sum()
        return loss

    def fit(
        self,
        num_iter,
        init_lr: float,
        batch_size: None | int = None,
        test_data: Data | None = None,
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
            optimizer = torch.optim.Adam(self.parameters(), lr=init_lr)
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

            if stopper is not None:
                state = {k: v.detach().clone() for k, v in self.state_dict().items()}
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

        return FitResult(
            model=self,
            max_m=self.data.conditioning_sets.shape[-1],
            losses=np.array(losses),
            parameters=parameters[-1],
            test_losses=np.array(test_losses) if test_data is not None else None,
            param_chain=param_chain,
            tracked_chain=tracked_chain,
        )

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

        augmented_data: AugmentedData = self.augment_data(self.data, None)

        data = self.data.response
        NN = self.data.conditioning_sets
        scales = augmented_data.scales
        sigmas = self.kernel._sigmas(scales)

        nug_mean = self.shrinkage_var

        #####################################################################################
        ################### PLEASE CHANGE ASAP. THIS IS NOT CORRECT ##########################
        nug_mult = self.nugget_shrinkage_factor.exp().add(1)
        nug_mult = 1 - 1/nug_mult
        kernel_mult = self.kernel_shrinkage_factor.exp().add(1)
        kernel_mult = 1 - 1/kernel_mult
        kernel_result = self.kernel.forward(augmented_data, nug_mean, nug_mult)
        nugget_mean = kernel_result.nug_mean
        chol = kernel_result.GChol

        tmp_res = self.intloglik.precalc(kernel_result, augmented_data.response, self.shrinkage_mean, 
                                         nug_mult)
        y_tilde = tmp_res.y_tilde
        beta_post = tmp_res.beta_post
        alpha_post = tmp_res.alpha_post
        n, N = data.shape
        m = NN.shape[1]
        if last_ind is None:
            last_ind = N
        # loop over variables/locations
        x_new = torch.empty((num_samples, N))
        x_new[:, : x_fix.size(0)] = x_fix.repeat(num_samples, 1)
        x_new[:, x_fix.size(0) :] = 0.0
        for i in range(x_fix.size(0), last_ind):
            # predictive distribution for current sample
            if i == 0:
                cStar = torch.zeros((num_samples, n))
                prVar = torch.zeros((num_samples,))
            else:
                ncol = min(i, m)
                X = data[:, NN[i, :ncol]]
                XPred = x_new[:, NN[i, :ncol]].unsqueeze(1)
                cStar = self.kernel._kernel_fun(
                    XPred, sigmas[i], nugget_mean[i], kernel_mult, X
                ).squeeze(1)
                ######### CHANGED BY ANIRBAN. PLEASE DOUBLE-CHECK. IS NOT CORRECT. ############################
                ######### PLEASE CHANGE ASAP. ############################
                ######### CHANGES MADE BY ANIRBAN ON May 6. ############################
                #### AFFECTED VARIABLES: cStar, cChol, meanPred, varPredNoNug
                prVar = self.kernel._kernel_fun(
                    XPred, sigmas[i], nugget_mean[i],
                kernel_mult).squeeze((1, 2))
            
            cChol = torch.linalg.solve_triangular(
                chol[i, :, :], cStar.unsqueeze(-1), upper=False
            ).squeeze(-1)
            meanPred = y_tilde[i, :].unsqueeze(0).mul(cChol).sum(1)
            varPredNoNug = prVar - cChol.square().sum(1)

            if i > 0:
                meanPred = (meanPred + 
                            self.shrinkage_mean_factor[i, :ncol].mul(XPred.squeeze()).sum())

            # sample
            invGDist = InverseGamma(concentration=alpha_post[i], rate=beta_post[i])
            nugget = invGDist.sample((num_samples,))
            uniNDist = Normal(loc=meanPred, scale=nugget.mul(1.0 + varPredNoNug).sqrt())
            x_new[:, i] = uniNDist.sample()

        return x_new

@dataclass
class FitResult:
    """
    Result of a fit.
    """

    model: ShrinkTM
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
