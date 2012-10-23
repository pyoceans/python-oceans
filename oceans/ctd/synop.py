# -*- coding: utf-8 -*-
#
# synop.py
#
# purpose:  Legacy Synop-LaDO toolbox
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  13-Sep-2012
# modified: Sat 13 Oct 2012 01:53:58 AM BRT
#
# obs:  Mostly backwards compatible functions.  There are newer and better ways
# to do all this!!!
#

import numpy as np


# Objective analysis in the same fashion as `scaloa.m`.
def scaloa(xc, yc, x, y, t=None, corrlen=None, err=None):
    r"""Scalar objective analysis.  Interpolates t(x, y) into tp(xc, yc)
    Assumes spatial correlation function to be isotropic and Gaussian in the
    form of: C = (1 - err) * np.exp(-d**2 / corrlen**2) where:
    d : Radial distance from the observations.

    Parameters
    ----------
    corrlen : float
              Correlation length.
    err     : float
              Random error variance (epsilon in the papers).

    Return
    ------
    tp : array
         Gridded observations.
    ep : array
         Normalized mean error.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # OA parameters.
    >>> err = 0.09  # Sampling error.
    >>> err = err ** 2  # Variance.
    >>> corrlen = 1.5  # Correlation length scale.
    >>> # Grid
    >>> IM, JM, xg, yg = np.loadtxt('grade_abro2_50x50curv.dat', comments='%',
    ...                              usecols=(0, 1, 8, 9), unpack=True)
    >>> IM = IM.max()
    >>> JM = JM.max()
    >>> xgg = np.reshape(xg, (JM, IM))
    >>> ygg = np.reshape(yg, (JM, IM))
    >>> xc = np.reshape(xgg, (JM * IM, 1))
    >>> yc = np.reshape(ygg, (JM * IM, 1))
    >>> # Data.
    >>> x, y = np.loadtxt('grade_ambes.dat', usecols=(0, 1), comments='%',
    ...                   unpack=True)
    >>> aotemp, aoer = scaloa(xc, yc, x, y, t=None, corrlen=corrlen, err=err)
    Computing just the interpolation errors.
    >>> xgg = np.reshape(xgg, (JM, IM))
    >>> ygg = np.reshape(ygg, (JM, IM))
    >>> aoer = np.reshape(aoer, (JM, IM))
    >>> aoer = np.sqrt(aoer) * 100
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> cs = ax.contourf(xgg, ygg, aoer)
    >>> _ =fig.colorbar(cs)

    Notes
    -----
    The funcion `scaloa` assumes that the user knows `err` and `corrlen` or
    that these parameters where chosen arbitrary.  The usual guess are the
    first baroclinic Rossby radius for `corrlen` and 0.1 e 0.2 to the sampling
    error.
    """

    # TODO: Enter 2D matrix and output 2D matrix transforming
    # everything inside the function.
    #
    # Create an example with real data.

    n = len(x)
    x, y = np.reshape(x, (1, n)), np.reshape(y, (1, n))

    # Squared distance matrix between the observations.
    d2 = ((np.tile(x, (n, 1)).T - np.tile(x, (n, 1))) ** 2 +
          (np.tile(y, (n, 1)).T - np.tile(y, (n, 1))) ** 2)

    nv = len(xc)
    xc, yc = np.reshape(xc, (1, nv)), np.reshape(yc, (1, nv))

    # Squared distance between the observations and the grid points.
    dc2 = ((np.tile(xc, (n, 1)).T - np.tile(x, (nv, 1))) ** 2 +
           (np.tile(yc, (n, 1)).T - np.tile(y, (nv, 1))) ** 2)

    # Correlation matrix between stations (A) and cross correlation (stations
    # and grid points (C))
    A = (1 - err) * np.exp(-d2 / corrlen ** 2)
    C = (1 - err) * np.exp(-dc2 / corrlen ** 2)

    if 0:  # NOTE: If the parameter zc is used (`scaloa2.m`)
        A = (1 - d2 / zc ** 2) * np.exp(-d2 / corrlen ** 2)
        C = (1 - dc2 / zc ** 2) * np.exp(-dc2 / corrlen ** 2)

    # Add the diagonal matrix associated with the sampling error.  We use the
    # diagonal because the error is assumed to be random.  This means it just
    # correlates with itself at the same place.
    A = A + err * np.eye(len(A))

    # Gauss-Markov to get the weights that minimize the variance (OI).
    tp = None
    if t:
        t = np.reshape(t, (n, 1))
        tp = np.dot(C, np.linalg.solve(A, t))
        if 0:  # NOTE: `scaloa2.m`
            mD = (np.sum(np.linalg.solve(A, t)) /
                  np.sum(np.sum(np.linalg.inv(A))))
            t = t - mD
            tp = (C * (np.linalg.solve(A, t)))
            tp = tp + mD * np.ones(tp.shape)
    if not t:
        print("Computing just the interpolation errors.")

    # Normalized mean error.  Taking the squared root you can get the
    # interpolation error in percentage.
    ep = 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)

    return tp, ep

if __name__ == '__main__':
    import doctest
    doctest.testmod()
