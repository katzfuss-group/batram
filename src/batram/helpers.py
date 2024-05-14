from scipy import linalg, stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from sklearn.gaussian_process import kernels
import numpy as np
import torch


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

    def __init__(self, locs: np.ndarray, kernel: kernels.Kernel, sd_noise: float):
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
    

###################### NOT IN USE YET (ANIRBAN, MAY 10, 2024) ###################################
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
        

