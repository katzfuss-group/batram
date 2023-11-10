import math

import matplotlib.pyplot as plt
import numpy as np

from .legmods import Data


def plot_data(data: Data, nrows: int = 1, figsize: tuple[int, int] = (12, 8)):
    if data.loc_dims != 2:
        raise ValueError("Only 2D locations are supported.")

    nx = data.locs[:, 0].unique().shape[0]
    ny = data.locs[:, 1].unique().shape[0]
    ncols = math.ceil(data.nreps / nrows)

    ord = np.lexsort((data.locs[:, 1], data.locs[:, 0]))
    locs_ord = data.locs[ord, :]
    responses_ord = data.response[:, ord]
    vmin, vmax = responses_ord.min(), responses_ord.max()

    gspec = {"wspace": 0.1, "hspace": 0.1}
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, gridspec_kw=gspec, squeeze=True, figsize=figsize
    )

    for i in range(data.nreps):
        ax = axs.reshape(-1)[i]
        pcm = ax.pcolormesh(
            locs_ord[:, 0].reshape(nx, ny),
            locs_ord[:, 1].reshape(nx, ny),
            responses_ord[i, :].reshape(nx, ny),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # del unused axes
    for ax in axs.reshape(-1)[data.nreps :]:
        fig.delaxes(ax)

    # add colorbar
    fig.subplots_adjust(right=0.9)
    cbar = fig.add_axes([0.125, 0.05, 0.775, 0.045])
    fig.colorbar(pcm, cax=cbar, orientation="horizontal")
