import math
from dataclasses import dataclass

import gpytorch
import numpy as np
import scipy.spatial
import scipy.special
import torch
import torch.nn

from ..data import VanillaData


class EarlyStopper:
    def __init__(self, min_diff: float, patients: int, min_steps: int = 0):
        self.min_diff = min_diff
        self.patients = patients
        self.patients_counter = 0
        self.step_counter = 0
        self.min_steps = min_steps
        self.min_loss = math.inf

    def stop_test(self, loss):
        self.step_counter += 1
        if loss < self.min_loss:
            self.min_loss = loss
            self.patients_counter = 0
        elif loss > self.min_loss + self.min_diff:
            self.patients_counter += 1

        if self.patients_counter > self.patients and self.step_counter > self.min_steps:
            return True
        else:
            return False

    def reset(self):
        self.patients_counter = 0
        self.step_counter = 0
        self.min_loss = math.inf


class EarlyStopper2:
    def __init__(self, min_diff: float, patients: int, min_steps: int = 0):
        self.min_diff = min_diff
        self.patients = patients
        self.patients_counter = 0
        self.step_counter = 0
        self.min_steps = min_steps
        self.min_loss = math.inf

    def stop_test(self, loss):
        self.step_counter += 1
        if loss + self.min_diff >= self.min_loss:
            self.patients_counter += 1
        else:
            self.patients_counter = 0

        if loss < self.min_loss:
            self.min_loss = loss

        if self.patients_counter > self.patients and self.step_counter > self.min_steps:
            return True
        else:
            return False

    def reset(self):
        self.patients_counter = 0
        self.step_counter = 0
        self.min_loss = math.inf


@dataclass
class Data:
    resp: torch.Tensor  # rep, proc, sp --- k, m, n
    eta: torch.Tensor | None  #
    loc_sp: torch.Tensor  # n, 2

    _resp_long = None
    _loc_sp_long = None

    def to_long(self) -> tuple[torch.Tensor, torch.Tensor]:
        """resp, loc_sp in long format"""
        if self._resp_long is None:
            self._resp_long = self.resp.reshape(self.nrep, -1)
            self._loc_sp_long = self.loc_sp.repeat(self.nproc, 1)

        return (self._resp_long, self._loc_sp_long)

    @staticmethod
    def from_vanilla(data: VanillaData) -> "Data":
        return Data(
            resp=torch.as_tensor(data.response),
            eta=None,
            loc_sp=torch.as_tensor(data.locs),
        )

    @property
    def nrep(self) -> int:
        return self.resp.shape[0]

    @property
    def nproc(self) -> int:
        return self.resp.shape[1]

    @property
    def nsp(self) -> int:
        return self.resp.shape[2]


class CorModule(torch.nn.Module):
    """
    implements a fully flexible corelation matrix parametrized as described in
    https://mc-stan.org/docs/2_26/reference-manual/cholesky-factors-of-corelation-matrices-1.html
    """

    def __init__(self, size):
        super().__init__()
        npars = size * (size - 1) // 2
        self.size = size
        self.unconst_params = torch.nn.Parameter(torch.zeros(npars))

    def chol_fac(self) -> torch.Tensor:
        zvec = torch.tanh(self.unconst_params)
        idx = torch.tril_indices(self.size, self.size, offset=-1)
        z = torch.zeros(self.size, self.size)

        z[idx[0, :], idx[1, :]] = zvec
        x = torch.zeros_like(z)

        for i in range(self.size):
            for j in range(i + 1):
                if i == j:
                    x[i, j] = torch.sqrt(1.0 - torch.sum(x[i, :j].clone() ** 2))
                elif i > j:
                    x[i, j] = z[i, j] * torch.sqrt(1 - torch.sum(x[i, :j].clone() ** 2))
        return x

    def cor(self) -> torch.Tensor:
        lmat = self.chol_fac()
        return lmat @ lmat.T

    def forward(self) -> torch.Tensor:
        return self.cor()

    @staticmethod
    def from_correlation(cor) -> "CorModule":
        cor_chol = torch.linalg.cholesky(cor)
        y = torch.zeros_like(cor_chol)
        for i in range(1, cor.shape[0]):
            for j in range(i):
                z = cor_chol[i, j] / torch.sqrt(1 - torch.sum(cor_chol[i, :j] ** 2))
                y[i, j] = torch.arctanh(z)

        cm = CorModule(cor.shape[0])
        idx = torch.tril_indices(cm.size, cm.size, offset=-1)
        cm.unconst_params.data = y[idx[0, :], idx[1, :]]
        return cm


class SeperableGP(gpytorch.models.ExactGP):
    def __init__(self, data: Data, likelihood):
        self.data = data
        train_y, train_x = data.to_long()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_sp_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        self.cor_proc_module = CorModule(data.nproc)

    def cov(self, data: Data):
        covar_sp = self.covar_sp_module(data.loc_sp)
        cor_proc = gpytorch.linear_operator.to_linear_operator(self.cor_proc_module())
        cov = gpytorch.lazy.KroneckerProductLazyTensor(cor_proc, covar_sp)

        return cov

    def forward(self, data=None):
        if data is None:
            data = self.data
        mean = self.mean_module(data.to_long()[1])
        cov = self.cov(data)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

    def process_dist(self) -> np.ndarray:
        with torch.no_grad():
            length_scale = self.covar_sp_module.base_kernel.lengthscale.numpy()
            cor_process = self.cor_proc_module().numpy()

        dist = matern_one_half_inv(cor_process, length_scale).astype(np.float32)
        return dist

    def process_locs(self, tol: float = 1e-6) -> np.ndarray:
        dist = self.process_dist()
        locs = dist_to_coords(dist, tol)

        return locs

    def process_cors(self) -> np.ndarray:
        with torch.no_grad():
            cor_process = self.cor_proc_module().numpy()
        return cor_process


def dist_to_coords(dist_mat: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Following the procedure from
    https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix

    returns a matrix with coordinates dropping axis with evals smaller than or
    equal to tol

    """

    m_mat = np.zeros_like(dist_mat)
    for i in range(m_mat.shape[0]):
        for j in range(m_mat.shape[1]):
            m_mat[i, j] = (
                dist_mat[0, j] ** 2 + dist_mat[i, 0] ** 2 - dist_mat[i, j] ** 2
            ) / 2

    evals, evecs = np.linalg.eigh(m_mat)
    # if not np.alltrue(evals >= 0):
    #     print(evals)

    # sort evals and evecs in descanding order of evals
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # remove zero evs
    mask = evals > tol
    evals = evals[mask]
    evecs = evecs[:, mask]

    coords = evecs @ np.sqrt(np.diag(evals))
    return coords


def coords_to_dist(coords: np.ndarray) -> np.ndarray:
    dmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    return dmat


def matern_one_half(dist, lengthscale):
    dist_norm = dist / lengthscale
    a = np.sqrt(3.0)
    t0 = 1 + a * dist_norm
    t1 = np.exp(-a * dist_norm)
    return t0 * t1


def matern_one_half_inv(cor, lengthscale):
    z = math.sqrt(3) / lengthscale
    lw = np.real(scipy.special.lambertw(-cor * math.exp(-1), -1))
    dist = -(lw + 1) / z
    dist = np.where(cor == 1.0, 0, dist)
    return dist
