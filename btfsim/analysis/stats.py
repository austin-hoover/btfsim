"""Module for statistical beam analysis."""
import numpy as np


def twiss(Sigma, dim='x'):
    """Compute rms Twiss parameters from 6x6 covariance matrix."""
    i = 2 * ['x', 'y', 'z'].index(dim)
    sigma = Sigma[i:i+2, i:i+2]
    eps = np.sqrt(np.linalg.det(sigma))
    beta = sigma[0, 0] / eps
    alpha = -sigma[0, 1] / eps
    return alpha, beta


def emittance(Sigma, dim='x'):
    i = 2 * ['x', 'y', 'z'].index(dim)
    sigma = Sigma[i:i+2, i:i+2]
    eps = np.sqrt(np.linalg.det(sigma))
    return eps