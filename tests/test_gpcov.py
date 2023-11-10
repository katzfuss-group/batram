import numpy as np
import scipy.spatial
import torch

import batram.seperable.gpcov as gpcov


def test_inverse_cor_module_init():
    corr = torch.eye(5)
    corr[1:, 0] = 0.1
    corr[4, 3] = 0.5
    corr[0, 1:] = 0.1
    corr[3, 4] = 0.5

    cm = gpcov.CorModule.from_correlation(corr)
    assert torch.allclose(cm.cor(), corr)


def test_dist_to_coords():
    dmat = np.zeros((3, 3))
    dmat[0, 1] = dmat[1, 0] = 1
    dmat[0, 2] = dmat[2, 0] = 2
    dmat[1, 2] = dmat[2, 1] = 2.5

    coords = gpcov.dist_to_coords(dmat)

    dmat2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    assert np.allclose(dmat, dmat2)


def test_coords_to_coords():
    coords = np.array(
        [
            [0.0, 0, 0, 0],
            [1.0, 0, 0, 0],
            [1.0, 1, 0, 0],
            [0.0, 0, 1, 0],
        ]
    )
    dmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    coords2 = gpcov.dist_to_coords(dmat)

    # rank of coords is 3
    assert coords2.shape[1] == 3

    dmat2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords2))
    assert np.allclose(dmat, dmat2)


def test_matern_invert():
    lengthscale = 0.6
    coords = np.array(
        [
            [0, 2, 0],
            [1, 0, 0.01],
            [0, 0.1, 0.02],
            [-0.5, 0.5, 1.03],
        ]
    )

    dmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    cor = gpcov.matern_one_half(dmat, lengthscale)
    dmat2 = gpcov.matern_one_half_inv(cor, lengthscale)

    assert np.allclose(dmat, dmat2)
