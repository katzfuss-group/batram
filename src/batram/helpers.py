from scipy import linalg, stats
from sklearn.gaussian_process import kernels
import numpy as np


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