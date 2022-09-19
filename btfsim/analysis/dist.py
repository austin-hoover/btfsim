"""Analysis of discrete particle distributions."""
import numpy as np


def radial_extent(X, fraction=1.0):
    """Return radius of sphere containing fraction of points.

    X : n-d array
        Coordinate array.
    fraction : float
        Fraction of points in sphere.
    """
    radii = np.linalg.norm(X, axis=0)
    radii = np.sort(radii)
    if (X.shape[0] * fraction < 1.0):
        imax = X.shape[0] - 1
    else:
        imax = int(np.round(X.shape[0] * fraction))
    try:
        radius = radii[imax]
    except:
        radius = 0.0
    return radius