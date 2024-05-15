from scipy import linalg, stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from sklearn.gaussian_process import kernels as gpkernel
import numpy as np
import torch
from math import sqrt

from gpytorch import add_jitter, kernels
from linear_operator import to_linear_operator

def make_grid(nlocs: int, ndims: int) -> np.ndarray:
    """
    Make a grid of equally spaced points in a unit hypercube.
    """
    _ = np.linspace(0, 1, nlocs)
    return np.stack(np.meshgrid(*[_] * ndims), axis=-1).reshape(-1, ndims)


class GaussianProcessGenerator:
    """
    Numpy-based Gausssian Process data generator.
    """

    def __init__(self, locs: np.ndarray, kernel: gpkernel.Kernel, sd_noise: float):
        self.locs = locs
        self.kernel = kernel
        self.sd_noise = sd_noise

    def update_kernel(self, **params):
        self.kernel.set_params(**params)

    def score(self, y: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data under the Gaussian Process.
        """
        cov = self.kernel(self.locs)
        cov += self.sd_noise**2 * np.eye(cov.shape[0])
        return stats.multivariate_normal(cov=cov).logpdf(y)

    def sample(self, num_reps: int = 1):
        """
        Sample from the Gaussian Process.
        """
        cov = self.kernel(self.locs)
        cov += self.sd_noise**2 * np.eye(cov.shape[0])
        chol = linalg.cholesky(cov, lower=True)
        z = stats.multivariate_normal(cov=np.eye(chol.shape[0])).rvs(num_reps)
        return np.dot(chol, z.T).T
    
    def sample_seed(self, num_reps: int = 1, seed: int = 42):
        """
        Sample from the Gaussian Process.
        Used only because otherwise optimization is breaking down.
        """
        covar = self.kernel(self.locs)
        covar += self.sd_noise**2 * np.eye(covar.shape[0])
        chol = linalg.cholesky(covar, lower=True)
        z = stats.multivariate_normal(cov=np.eye(chol.shape[0]), seed=np.random.seed(seed)).rvs(num_reps)
        return np.dot(chol, z.T).T
    

class CustomMaternKernel(torch.nn.Module):
    """Initializes a Matern Kernel."""
    def __init__(self, **kwargs):
        super().__init__()
        fix_nu = kwargs.get("fix_nu", 1.5)
        self.nu = fix_nu
        
    
    def forward(self, length_scale: torch.Tensor, locs_X:torch.Tensor, locs_Y: torch.Tensor | None):

        if locs_Y is None:
            distmatrix = torch.pdist(locs_X)
        else:
            assert locs_X.shape[1] == locs_Y.shape[1]
            distmatrix = torch.cdist(locs_X, locs_Y)
        distmatrix =  distmatrix.div(length_scale)

        ## this part is mostly copied from sklearn.gaussian_process.kernels.py
        if (self.nu == 0.5):
            K = torch.exp(-distmatrix)
        elif (self.nu == 1.5):
            K = distmatrix.mul((torch.Tensor([3])).sqrt())
            K = (1.0 + K) * torch.exp(-K)
        elif (self.nu == 2.5):
            K = distmatrix.mul((torch.Tensor([5])).sqrt())
            K = (1.0 + K + (K**2).div(torch.Tensor([3.0]))) * torch.exp(-K)
        elif self.nu == torch.inf:
            K = torch.exp(-(distmatrix**2).div(torch.Tensor([2.0])))
        return K
        

class BaseKernel(kernels.Kernel):
    """A flexible Matern or RBF (Gaussian) kernel parameterized by `nu`.

    This kernel contains no torch parameters, so it is easy to build on top of
    to compose parameterizations on top of.
    """

    def __init__(self, nu: float | str, eps: float = 0.0):
        """Initialize the kernel with smoothness and pre-defined jitter.

        Args:
        -----
        nu: float | "inf"
            The smoothness parameter of the Matern kernel. Must be one of
            {0.5, 1.5, 2.5, "inf"}. When "inf" is passed, the kernel is defined
            as an RBF (Gaussian) kernel.

        eps: float
            The jitter to add to the diagonal of the kernel matrix. This is not
            always used, but can be used to stabilize computations in some cases.
        """
        super().__init__()

        if nu not in (0.5, 1.5, 2.5, "inf"):
            raise ValueError(f"nu must be one of {{0.5, 1.5, 2.5, 'inf'}}, got {nu}")

        self.nu = nu
        self.eps = eps

    def forward(  # type: ignore
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None,
        **params,
    ):
        x2 = x1 if x2 is None else x2
        mean = x1.mean(-2, keepdim=True)
        x1_ = x1 - mean
        x2_ = x2 - mean

        match self.nu:
            case 0.5:
                dist = self.covar_dist(x1_, x2_, square_dist=False, **params)
                s = 1.0
                exp = torch.exp(-dist)
            case 1.5:
                dist = self.covar_dist(x1_, x2_, square_dist=False, **params)
                s = 1.0 + sqrt(3) * dist
                exp = torch.exp(-sqrt(3) * dist)
            case 2.5:
                dist = self.covar_dist(x1_, x2_, square_dist=False, **params)
                s = 1.0 + sqrt(5) * dist + 5 / 3 * dist**2
                exp = torch.exp(-sqrt(5) * dist)
            case _:
                dist2 = self.covar_dist(x1_, x2_, square_dist=True, **params)
                s = 1.0
                exp = torch.exp(-0.5 * dist2)

        kernel = s * exp

        if torch.equal(x1, x2):
            return add_jitter(kernel, self.eps)
        else:
            return to_linear_operator(kernel)
