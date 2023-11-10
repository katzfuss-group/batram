import itertools

import numpy as np


def cov_exponential(d, sigma, range):
    "exponential covariance function"
    return sigma**2 * np.exp(-d / range)


def rev_mat(x):
    """
    Calculates the reverse matrix.

    See Section 4.1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.
    """
    return x[-1::-1, -1::-1]


def rchol(x):
    """
    reverse cholesky factor

    Defined in Sec 4.1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.

    """
    return rev_mat(np.linalg.cholesky(rev_mat(x)))


def calc_u_d_b(x):
    """
    Calculates u, d, b as defined in Proposition 1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.
    """

    u_direct = rchol(x)
    d = u_direct.diagonal() ** (-2)
    u = u_direct * d**0.5
    return u, d, -u


def dict_product(dict_of_lists):
    """
    Turns a dictionary of lists into a list of dictionaries with all possible
    combinations.
    """
    return [
        dict(zip(dict_of_lists, x)) for x in itertools.product(*dict_of_lists.values())
    ]
