__doc__ = (

    """
    This script is used to generate the data for the project. The data description follows.
    LR900: The data is generated from exponential covariance kernel on 30x30 grid.
    NR900: Non-linear version of LR900 generated on 30x30 grid. Refer to K&S 2023
           for more details.
    """
)

from pathlib import Path
from argparse import ArgumentParser, Namespace
import pickle

import torch
import numpy as np
import math
from veccs import orderings
from gpytorch.kernels import MaternKernel
from sklearn.gaussian_process import kernels
from matplotlib import pyplot as plt

from batram.helpers import make_grid, GaussianProcessGenerator

DATAPATH = Path(__file__).resolve().parent.parent / "data"

parsed_args = ArgumentParser(
    description=(
        "Some arguments for generating the data."
    )
)
parsed_args.add_argument(
    "--spatial_grid_size", type=int, default=30,
    help=("Spatial grid size of underlying location space. Default is 30.")
)
parsed_args.add_argument(
    "--spatial_dim", type=int, default=2,
    help=("Spatial dimension of the location space. Default is 2.")
)
parsed_args.add_argument(
    "--n_samples", type=int, default=100,
    help=("Number of samples to generate. Default is 100.")
)
parsed_args.add_argument(
    "--nu", type=float, default=0.5,
    help=("Smoothness parameter for Matern kernel. Default is 0.5.")
)
parsed_args.add_argument(
    "--lengthscale", type=float, default=0.3,
    help=("Lengthscale parameter for Matern kernel. Default is 0.3.")
)
parsed_args.add_argument(
    "--sigmasq_f", type=float, default=1.0,
    help=("Signal variance for Matern kernel. Default is 1.0.")
)
parsed_args.add_argument(
    "--max_m", type=int, default=30,
    help=("Maximum number of nearest neighbours for Vecchia approximation. Default is 30.")
)
parsed_args.add_argument(
    "--sd_noise", type=float, default=1e-6,
    help=("Standard deviation to add to the diagonals of the covariance matrix. Default is 1e-6.")
)
parsed_args.add_argument(
    "--output", type = str, default="NR",
    help = ("Output data type based on data generating mechanism.")
)

def _calculate_weights(gp: GaussianProcessGenerator,
                       largest_conditioning_set: int) -> tuple[torch.Tensor, torch.Tensor]:
    ## defining the weights of shrinkage kernel

    locs = gp.locs
    nn = orderings.find_nns_l2(locs, largest_conditioning_set)
    gpkernel = gp.kernel
    sd_noise = gp.sd_noise

    parametric_mean_factors = torch.zeros((locs.shape[0], largest_conditioning_set))
    parametric_variances = torch.zeros(locs.shape[0])
    parametric_variances[0] = 1

    for i in range(1, locs.shape[0]):
        current_locs = locs[i, :]
        if i < largest_conditioning_set:
            previous_locs = locs[nn[i, 0:i], :]
        else:
            previous_locs = locs[nn[i], :]
        Sigma22 = torch.from_numpy(gpkernel(previous_locs, previous_locs)) + (sd_noise ** 2) * torch.eye(previous_locs.shape[0])
        Sigma12 = torch.from_numpy(gpkernel(current_locs, previous_locs))
        Sigma22inv = torch.linalg.solve(Sigma22, torch.eye(previous_locs.shape[0], dtype=torch.double))
        parametric_mean_factors[i, 0:min(i, largest_conditioning_set)] = Sigma12 @ Sigma22inv
        parametric_variances[i] = 1 - Sigma12 @ Sigma22inv @ Sigma12.T
    
    return parametric_mean_factors, (gp.variance * parametric_variances)

@torch.inference_mode()    
def _nrsamples(gp: GaussianProcessGenerator, largest_conditioning_set: int, num_reps: int) -> torch.Tensor:
    
    samples = torch.zeros((num_reps, gp.locs.shape[0]))
    rands = (torch.from_numpy(np.random.standard_normal((num_reps * 
                    gp.locs.shape[0])))).reshape(num_reps, gp.locs.shape[0])

    weights, variances = _calculate_weights(gp, largest_conditioning_set)
    nn = orderings.find_nns_l2(gp.locs, largest_conditioning_set)
    for i in range(gp.locs.shape[0]):
        if (i == 0):
            samples[:, i] = rands[:, i] * torch.sqrt(variances[i])
        else:
            _weights = weights[i, :]
            means = ((samples[:, nn[i, :]]) @ _weights.unsqueeze(-1)).squeeze()
            mean_shift = (_weights[0] * samples[ :, nn[i, 0]] + _weights[1] * samples[:, nn[i, 1]]).squeeze()
            mean_shift = 4.0 * mean_shift
            means += 2.0 * (mean_shift.sin())
            samples[:, i] = rands[:, i] * torch.sqrt(variances[i]) + means

    return samples

def main(args: Namespace) -> None:
    
    ## generate data

    locs = make_grid(args.spatial_grid_size, args.spatial_dim)
    locsorder = orderings.maxmin_cpp(locs=locs) #find maxmin-ordeing
    locs = locs[locsorder, ...]

    #sample data
    gpkernel = kernels.Matern(nu=args.nu, length_scale=args.lengthscale)
    gp = GaussianProcessGenerator(locs=locs, kernel=gpkernel, sd_noise=args.sd_noise, variance = args.sigmasq_f)

    ## ordering the locations
    N = int(math.pow(args.spatial_grid_size, args.spatial_dim))

    if args.output == "LR":
        numpydata = gp.sample(num_reps=args.n_samples)
        torchdata = torch.from_numpy(numpydata).float()
        torchdata = torchdata#[:, locsorder]
    elif args.output == "NR":
        torchdata = _nrsamples(gp, args.max_m, args.n_samples)


    ## save data
    

    masterfile = {"locs": locs, 
                  "order": locsorder,
                  "gp": gp,
                  "data": torchdata[:, locsorder.argsort()]}
    if args.nu == 0.5:
        KernName = "Exp"
    elif args.nu == 1.5:
        KernName = "MatThrbyT"
    elif args.nu == 2.5:
        KernName = "MatFivbyT"
    
    ls_transform = int(100 * args.lengthscale)

    datafilename = (str(args.output) + 
                    str(N) + 
                    str(KernName) + 
                    "LST" + 
                    str(ls_transform) + 
                    "SIGSQT" +
                    str(int(10 * args.sigmasq_f)) +
                    ".pkl")
    
    with open(DATAPATH / datafilename, "wb") as f:
        pickle.dump(masterfile, f)
        f.close()
    return None

if __name__ == "__main__":
    args = parsed_args.parse_args()
    main(args)