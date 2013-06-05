# -*- coding: utf-8 -*-
#
# ocfis.py
#
# purpose:  Some misc PO functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Wed 05 Jun 2013 02:16:15 PM BRT
#
# obs:
#

from __future__ import division

import numpy as np
import numpy.ma as ma

import gsw

__all__ = [
           'mld',
           'lanc',
           'pcaben',
           'spdir2uv',
           'uv2spdir',
           'interp_nan',
           'del_eta_del_x'
           ]


def lanc(numwt, haf):
    r"""Generates a numwt + 1 + numwt lanczos cosine low pass filter with -6dB
    (1/4 power, 1/2 amplitude) point at haf

    Parameters
    ----------
    numwt : int
            number of points
    haf : float
          frequency (in 'cpi' of -6dB point, 'cpi' is cycles per interval.
          For hourly data cpi is cph,

    Examples
    --------
    >>> from datetime import datetime
    >>> import matplotlib.pyplot as plt
    >>> cols = ['j', 'u', 'v', 'temp', 'sal', 'y', 'mn', 'd', 'h', 'mi']
    >>> fname = '../test/15t30717.3f1'
    >>> df = read_table(fname , delim_whitespace=True, names=cols)
    >>> dates = [datetime(*x) for x in
    ...          zip(df['y'], df['mn'], df['d'], df['h'], df['mi'])]
    >>> df.index = dates
    _ = map(df.pop, ['y', 'mn', 'd', 'h', 'mi'])
    >>> wt = lanc(96+1+96, 1./40)
    >>> df['low'] = np.convolve(wt, df['v'], mode='same')
    >>> df['high'] = df['v'] - df['low']
    >>> fig, (ax0, ax1) = plt.subplots(nrows=2)
    >>> _ = ax0.plot(df['j'], df['high'], label='high')
    >>> _ = ax1.plot(df['j'], df['low'], label='low')
    >>> _ = ax0.legend(numpoints=1)
    >>> _ = ax1.legend(numpoints=1)
    """
    summ = 0
    numwt += 1
    wt = np.zeros(numwt)

    # Filter weights.
    ii = np.arange(numwt)
    wt = 0.5 * (1.0 + np.cos(np.pi * ii * 1. / numwt))
    ii = np.arange(1, numwt)
    xx = np.pi * 2 * haf * ii
    wt[1:numwt + 1] = wt[1:numwt + 1] * np.sin(xx) / xx
    summ = wt[1:numwt + 1].sum()
    xx = wt.sum() + summ
    wt /= xx
    return np.r_[wt[::-1], wt[1:numwt + 1]]


def spdir2uv(spd, ang, deg=False):
    r"""Computes u, v components from speed and direction.

    Parameters
    ----------
    spd : array_like
          speed [m s :sup:`-1`]
    ang : array_like
          direction [deg]
    deg : bool
          option, True if data is in degrees. Default is False

    Returns
    -------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]
    """

    if deg:
        ang = np.deg2rad(ang)

    # Calculate U (E-W) and V (N-S) components
    u = spd * np.sin(ang)
    v = spd * np.cos(ang)
    return u, v


def uv2spdir(u, v, mag=0, rot=0):
    r"""Computes speed and direction from u, v components.
    Converts rectangular to polar coordinate, geographic convention
    Allows for rotation and magnetic declination correction.

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]
    mag : float, array_like, optional
          Magnetic correction [deg]
    rot : float, array_like
          Angle for rotation [deg]

    Returns
    -------
    ang : array_like
          direction [deg]
    spd : array_like
          speed [m s :sup:`-1`]

    See Also
    --------
    TODO: spdir2uv

    Notes
    -----
    Zero degrees is north.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> from windrose import WindroseAxes
    >>> def new_axes():
    >>>     fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='w')
    >>>     rect = [0.1, 0.1, 0.8, 0.8]
    >>>     ax = WindroseAxes(fig, rect, axisbg='w')
    >>>     fig.add_axes(ax)
    >>>     return ax
    >>> def set_legend(ax):
    >>>     l = ax.legend(axespad=-0.10)
    >>>     plt.setp(l.get_texts(), fontsize=8)
    >>> u, v = [0., 1., -2., -1., 1.], [3., 1., 0., -1., -1.]
    >>> wd, ws = ff.uv2spdir(u,v)
    >>> ax = new_axes()
    >>> ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    >>> set_legend(ax)
    >>> plt.show()
    """

    u, v, mag, rot = map(np.asarray, (u, v, mag, rot))

    vec = u + 1j * v
    spd = np.abs(vec)
    ang = np.angle(vec, deg=True)
    ang = ang - mag + rot
    ang = np.mod(90 - ang, 360)  # Zero is North.

    return ang, spd


def interp_nan(data):
    r"""Linear interpolate of NaNs in a record.

    Parameters
    ----------
    y : 1d array
        array with NaNs

    Returns
    -------
    nans : bool array
           indices of NaNs
    idx_nan : logical
              indices of NaNs

    Examples
    --------
    FIXME
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    nans = np.isnan(data)

    x = lambda z: z.nonzero()[0]

    data[nans] = np.interp(x(nans), x(~nans), data[~nans])

    return nans, data


def del_eta_del_x(U, f, g, balance='geostrophic', R=None):
    r"""Calculate :mat: `\frac{\partial \eta} {\partial x}` for different
    force balances

    Parameters:
    ----------
    U : array_like
        velocity magnitude [m/s]
    f : float
        Coriolis parameter
    d : float
        Acceleration of gravity
    balance : str, optional
              geostrophic [default], gradient or max_gradient
    R : float, optional
        Radius
    """

    if balance == 'geostrophic':
        detadx = f * U / g

    elif balance == 'gradient':
        detadx = (U ** 2 / R + f * U) / g

    elif balance == 'max_gradient':
        detadx = (R * f ** 2) / (4 * g)

    return detadx


def mld(SA, CT, p, criterion='pdvar'):
    r"""Compute the mixed layer depth.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]
    criterion : str, optional
               MLD Criteria

    Mixed layer depth criteria are:

    'temperature' : Computed based on constant temperature difference
    criterion, CT(0) - T[mld] = 0.5 degree C.

    'density' : computed based on the constant potential density difference
    criterion, pd[0] - pd[mld] = 0.125 in sigma units.

    `pdvar` : computed based on variable potential density criterion
    pd[0] - pd[mld] = var(T[0], S[0]), where var is a variable potential
    density difference which corresponds to constant temperature difference of
    0.5 degree C.

    Returns
    -------
    MLD : array_like
          Mixed layer depth
    idx_mld : bool array
              Boolean array in the shape of p with MLD index.


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> import gsw
    >>> from gsw.utilities import Dict2Struc
    >>> # Read data file with check value profiles
    >>> datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
    >>> cv = Dict2Struc(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
    >>> SA, CT, p = (cv.SA_chck_cast[:, 0], cv.CT_chck_cast[:, 0],
    ...              cv.p_chck_cast[:, 0])
    >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)
    >>> ax1.plot(CT, -p, 'b.-')
    >>> MDL, idx = ff.mld(SA, CT, p, criterion='temperature')
    >>> ax1.plot(CT[idx], -p[idx], 'ro')
    >>> ax2.plot(CT, -p, 'b.-')
    >>> MDL, idx = ff.mld(SA, CT, p, criterion='density')
    >>> ax2.plot(CT[idx], -p[idx], 'ro')
    >>> ax3.plot(CT, -p, 'b.-')
    >>> MDL, idx = ff.mld(SA, CT, p, criterion='pdvar')
    >>> ax3.plot(CT[idx], -p[idx], 'ro')
    >>> ax3.set_ylim(-500, 0)
    >>> plt.show()

    References
    ----------
    .. [1] Monterey, G., and S. Levitus, 1997: Seasonal variability of mixed
    layer depth for the World Ocean. NOAA Atlas, NESDIS 14, 100 pp.
    Washington, D.C.
    """

    SA, CT, p = map(np.asanyarray, (SA, CT, p))
    SA, CT, p = np.broadcast_arrays(SA, CT, p)
    SA, CT, p = map(ma.masked_invalid, (SA, CT, p))

    p_min, idx = p.min(), p.argmin()

    sigma = gsw.rho(SA, CT, p_min) - 1000.

    # Temperature and Salinity at the surface,
    T0, S0, Sig0 = CT[idx], SA[idx], sigma[idx]

    # NOTE: The temperature difference criterion for MLD
    Tdiff = T0 - 0.5  # 0.8 on the matlab original

    if criterion == 'temperature':
        idx_mld = (CT > Tdiff)
    elif criterion == 'pdvar':
        pdvar_diff = gsw.rho(S0, Tdiff, p_min) - 1000.
        idx_mld = (sigma <= pdvar_diff)
    elif criterion == 'density':
        sig_diff = Sig0 + 0.125
        idx_mld = (sigma <= sig_diff)
    else:
        raise NameError("Unknown criteria %s" % criterion)

    MLD = ma.masked_all_like(p)
    MLD[idx_mld] = p[idx_mld]

    return MLD.max(axis=0), idx_mld


def pcaben(u, v):
    r"""Principal components of 2-d (e.g. current meter) data
    calculates ellipse parameters for currents.

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]

    Returns
    -------
    major axis (majrax)
    minor axis (minrax)
    major azimuth (majaz)
    minor azimuth (minaz)
    elipticity (elptcty)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> u = np.r_[(0., 1., -2., -1., 1.), np.random.randn(10)]
    >>> v = np.r_[(3., 1., 0., -1., -1.), np.random.randn(10)]
    >>> (mjrax, mjaz, mirax, miaz, el), (x1, x2, y1, y2) = ff.pcaben(u, v)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot(x1, y1,'r-', x2, y2, 'r-')
    >>> ax.set_aspect('equal')
    >>> _ = ax.set_xlabel('U component')
    >>> _ = ax.set_ylabel('V component')
    >>> _ = ax.plot(u, v, 'bo')
    >>> _ = ax.axis([-3.2, 3.2, -3.2, 3.2])
    >>> mdir, mspd = ff.uv2spdir(u.mean(), v.mean())
    >>> _ = ax.plot([0, u.mean()],[0, v.mean()], 'k-')
    >>> plt.show()
    >>> print('Mean u = %f\nMean v = %f\n' % (u.mean(), v.mean()))
    >>> print('Mean magnitude: %f\nDirection: %f\n' % (mspd, mdir))
    >>> print('Axis 1 Length: %f\nAzimuth: %f\n' % (mjrax, mjaz))
    >>> print('Axis 2 Length: %f\nAzimuth: %f\n' % (mirax, miaz))
    >>> print("elipticity is : %s" % el)
    >>> print("Negative (positive) means clockwise (anti-clockwise)")
    >>> flatness = 1 - mirax / mjrax
    >>> print("Flatness: %s" % flatness)

    Notes:
    http://pubs.usgs.gov/of/2002/of02-217/m-files/pcaben.m
    """

    u, v = np.broadcast_arrays(u, v)

    C = np.cov(u, v)
    D, V = np.linalg.eig(C)

    x1 = np.r_[0.5 * np.sqrt(D[0]) * V[0, 0], -0.5 * np.sqrt(D[0]) * V[0, 0]]
    y1 = np.r_[0.5 * np.sqrt(D[0]) * V[1, 0], -0.5 * np.sqrt(D[0]) * V[1, 0]]
    x2 = np.r_[0.5 * np.sqrt(D[1]) * V[0, 1], -0.5 * np.sqrt(D[1]) * V[0, 1]]
    y2 = np.r_[0.5 * np.sqrt(D[1]) * V[1, 1], -0.5 * np.sqrt(D[1]) * V[1, 1]]

    # Length and direction.
    az, leng = np.c_[uv2spdir(x1[0], y1[0]), uv2spdir(x2[1], y2[1])]

    if (leng[0] >= leng[1]):
        majrax, majaz = leng[0], az[0]
        minrax, minaz = leng[1], az[1]
    else:
        majrax, majaz = leng[1], az[1]
        minrax, minaz = leng[0], az[0]

    elptcty = minrax / majrax

    # Radius to diameter.
    majrax *= 2
    minrax *= 2

    return (majrax, majaz, minrax, minaz, elptcty), (x1, x2, y1, y2)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
