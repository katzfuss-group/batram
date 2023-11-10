import math

import matplotlib.pyplot as plt
import numpy as np

from .data import Data, VanillaData


def plot_data(data: Data, nrows: int = 1, figsize: tuple[int, int] = (12, 8)):
    if data.loc_dim != 2:
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


def plot_vanilla_data(
    data: VanillaData,
    nrows: int | None = 3,
    # fig: plt.Figure | None = None,
    # axs: plt.Axes | None = None,
    figsize: tuple[int, int] = (12, 8),
):
    if data.spdim != 2:
        raise ValueError("Only 2D locations are supported.")

    if nrows is None:
        nrows = data.nrep

    ncols = data.nproc if data.nproc > 1 else data.nrep // nrows

    # if fig is None:
    fig = plt.figure(figsize=figsize)

    # if axs is None:
    axs = np.atleast_1d(fig.subplots(nrows=nrows, ncols=ncols, squeeze=True))
    axs = axs.reshape(nrows, ncols)

    nx = np.unique(data.locs[:, 0]).shape[0]
    ny = np.unique(data.locs[:, 1]).shape[0]

    locsx = data.locs[:, 0].reshape(nx, ny)
    locsy = data.locs[:, 1].reshape(nx, ny)

    # plot multivariate response
    vmin, vmax = data.response.min(), data.response.max()
    for i in range(nrows):
        for j in range(ncols):
            if data.nproc > 1:
                resp = data.response[i, j, :]
            else:
                resp = data.response[i * ncols + j, 0, :]

            ax: plt.Axes = axs[i, j]
            pcm = ax.pcolormesh(
                locsx,
                locsy,
                resp.reshape(nx, ny),
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    # add colorbar
    fig.subplots_adjust(right=0.9)
    cbar = fig.add_axes([0.125, 0.05, 0.775, 0.045])
    fig.colorbar(pcm, cax=cbar, orientation="horizontal")


def _setdiff2d(a, b):
    aa = a.view([("", a.dtype)] * a.shape[1])
    bb = b.view([("", b.dtype)] * b.shape[1])
    return np.setdiff1d(aa, bb).view(a.dtype).reshape(-1, a.shape[1])


def plot_vanilla_data2(
    data: VanillaData,
    nrows: int | None = 3,
    # fig: plt.Figure | None = None,
    # axs: plt.Axes | None = None,
    figsize: tuple[int, int] = (12, 8),
):
    if data.spdim != 2:
        raise ValueError("Only 2D locations are supported.")

    if nrows is None:
        nrows = data.nrep

    ncols = data.nproc if data.nproc > 1 else data.nrep // nrows

    # if fig is None:
    fig = plt.figure(figsize=figsize)

    # if axs is None:
    axs = np.atleast_1d(fig.subplots(nrows=nrows, ncols=ncols, squeeze=True))
    axs = axs.reshape(nrows, ncols)

    # fill the grid with nan values
    locs_all = np.array(
        np.meshgrid(np.unique(data.locs[:, 0]), np.unique(data.locs[:, 1]))
    ).T.reshape(-1, 2)
    locs_nan = np.array(list(set(map(tuple, locs_all)) - set(map(tuple, data.locs))))
    # locs_nan = _setdiff2d(locs_all, locs)
    locs = np.concatenate((data.locs, locs_nan), axis=0)

    resp_nan = np.full(
        (
            data.nrep,
            data.nproc,
            locs_nan.shape[0],
        ),
        np.nan,
    )
    response = np.concatenate((data.response, resp_nan), axis=2)

    # order
    ord = np.lexsort((locs[:, 1], locs[:, 0]))
    locs = locs[ord, :]
    response = response[..., ord]

    nx = np.unique(locs[:, 0]).shape[0]
    ny = np.unique(locs[:, 1]).shape[0]

    locsx = locs[:, 0].reshape(nx, ny)
    locsy = locs[:, 1].reshape(nx, ny)

    # plot multivariate response
    vmin, vmax = np.nanmin(response), np.nanmax(response)
    for i in range(nrows):
        for j in range(ncols):
            if data.nproc > 1:
                resp = response[i, j, :]
            else:
                resp = response[i * ncols + j, 0, :]

            ax: plt.Axes = axs[i, j]
            pcm = ax.pcolormesh(
                locsx,
                locsy,
                resp.reshape(nx, ny),
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    # add colorbar
    fig.subplots_adjust(right=0.9)
    cbar = fig.add_axes([0.125, 0.05, 0.775, 0.045])
    fig.colorbar(pcm, cax=cbar, orientation="horizontal")
