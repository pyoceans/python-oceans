#!/usr/bin/env python
#
#
# ocfis.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Fri 02 Dec 2011 11:58:58 AM EST
#
# obs:
#

from __future__ import division

import numpy as np
import scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook


def spdir2uv(spd, ang, deg=False):
    r"""
    Computes u, v components from speed and
    direction.

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
    u = spd * np.cos(ang)
    v = spd * np.sin(ang)
    return u, v


def uv2spdir(u, v, mag=0, rot=0):
    r"""
    Computes speed and direction from u, v components. Allows for rotation and
    magnetic declination correction.

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
    >>> import matplotlib.pyplot as plt
    >>> import ff_tools as ff
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
    >>> u, v = [0,1,-2], [3,1,0]
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


def despike(datain, slope):
    r"""
    De-spikes a time-series by calculating point-to-point slopes and
    determining whether a maximum allowable slope is exceeded.

    Parameters
    ----------
    datain : array_like
             any time series
    slope : float
            diff slope threshold [in data units]

    Returns
    -------
    cdata : array_like
            clean time series

    See Also
    --------
    TODO

    Notes
    -----
    Dangerous de-spiking technique, use with caution!
    Recommend only for highly noisy (lousy?) series.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import ff_tools as ff
    >>> time = np.linspace(-4, 4, 100)
    >>> series = np.sin(time)
    >>> spiked = series + np.random.randn( len(time) ) * 0.3
    >>> cdata = ff.despike(spiked, 0.15 )
    >>> plt.plot(time, series, 'k')
    >>> plt.plot(time, spiked, 'r.')
    >>> plt.plot(time, cdata, 'bo')
    >>> plt.show()

    References
    ----------
    TODO

    author:   Filipe P. A. Fernandes
    date:     23-Nov-2010
    modified: 23-Nov-2010
    """

    datain, slope = map(np.asarray, (datain, slope))

    cdata = np.zeros(datain.size)
    cdata[0] = datain[0]  # FIXME: Assume that the first point is not a spike.
    kk = 0
    npts = len(datain)

    for k in range(1, npts, 1):
        nslope = datain[k] - cdata[kk]
        # if the slope is okay, let the data through
        if abs(nslope) <= abs(slope):
            kk = kk + 1
            cdata[kk] = datain[k]
        # if slope is not okay, look for the next data point
        else:
            n = 0
            # TODO: add a limit option for npts
            while (abs(nslope) > abs(slope)) and (k + n < npts):
                n = n + 1  # keep an index for the offset from the test point
                try:
                    # FIXME: Index out of bound.
                    num = datain[k + n] - cdata[kk]
                    # TODO: Check original `snum` and `num` are defined but not
                    # used and undefined respectively.
                    dem = k + n - kk
                    nslope = num / dem
                    # If we have a "good" slope, calculate new point using
                    # linear interpolation:
                    # point = {[(ngp - lgp)/(deltax)]*(actual distance)} + lgp
                    # ngp = next good point
                    # lgp = last good point
                    # actual distance = 1, the distance between the last lgp
                    # and the point we want to interpolate.
                    # Otherwise, let the value through
                    # (i.e. we've run out of good data)
                    if (k + n) < npts:
                        pts = nslope + cdata[kk]
                        kk = kk + 1
                        cdata[kk] = pts
                    else:
                        kk = kk + 1
                        cdata[kk] = datain[k]
                except:  # TODO: raise
                    print("Index out of bounds")
    return cdata


def binave(datain, r):
    r"""
    Averages vector data in bins of length r. The last bin may be the average
    of less than r elements. Useful for computing daily average time series
    (with r=24 for hourly data).

    Parameters
    ----------
    data : array_like
    r : int
        bin length

    Returns
    -------
    bindata : array_like
              bin-averaged vector

    See Also
    --------
    TODO

    Notes
    -----
    Original from MATLAB AIRSEA TOOLBOX.

    Examples
    --------
    >>> import ff_tools as ff
    >>> data = [10., 11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123.,
        ...     3.2, 0.52, 18.2, 10., 11., 13., 2., 34., 21.5, 6.46, 6.27,
        ...     7.0867, 15., 123., 3.2, 0.52, 18.2, 10., 11., 13., 2., 34.,
        ...     21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2, 0.52, 18.2, 10.,
        ...     11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2,
        ...     0.52, 18.2]
    >>> ff.binave(data, 24)
    array([ 16.564725 ,  21.1523625,  22.4670875])

    References
    ----------
    TODO

    03/08/1997: version 1.0
    09/19/1998: version 1.1 (vectorized by RP)
    08/05/1999: version 2.0
    02/04/2010: Translated to python by FF
    """

    datain, r = np.asarray(datain), np.asarray(r, dtype=np.int)

    if datain.ndim != 1:
        raise ValueError("Must be a 1D array")

    if r <= 0:
        raise ValueError("Bin size R must be a positive integer.")

    # compute bin averaged series
    l = datain.size / r
    l = np.fix(l)
    z = datain[0:(l * r)].reshape(r, l, order='F')  # .copy()
    bindata = np.mean(z, axis=0)

    return bindata

"""
function [mu, bins] = bindata(x, y, numbins);
%function [mu, bins] = bindata(x, y, numbins);
% bins the data y according to x and returns the bins and the average
% value of y for that bin

bins = linspace(min(x), max(x), numbins);
[n,bin] = histc(x, bins);
mu = NaN*zeros(size(bins));
for k = [1:numbins],
  ind = find(bin==k);
  if (~isempty(ind))
    mu(k) = mean(y(ind));
  end
end

[n,bins] = histc(x,linspace(min(x),max(x),numbins));
ty = sparse(1:length(x),bin,y);
mu = full(sum(ty)./sum(ty~=0))

you may get NaN from 0/0 if you have an empty bin, but that can be
taken care of easily.
"""


def psu2ppt(psu):
    r"""Converts salinity from PSU units to PPT
    http://stommel.tamu.edu/~baum/paleo/ocean/node31.html#PracticalSalinityScale
    """

    a = [0.008, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081]
    ppt = (a[1] + a[2] * psu ** 0.5 + a[3] * psu + a[4] * psu ** 1.5 + a[5] *
                                                 psu ** 2 + a[6] * psu ** 2.5)
    return ppt


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


def spec_rot(u, v):
    r"""
    Compute the rotary spectra from u,v velocity components

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]

    Returns
    -------
    cw : array_like
         Clockwise spectrum [TODO]
    ccw : array_like
          Counter-clockwise spectrum [TODO]
    puv : array_like
          Cross spectra [TODO]
    quv : array_like
          Quadrature spectra [ TODO]

    Notes
    -----
    The spectral energy at some frequency can be decomposed into two circulaly
    polarized constituents, one rotating clockwise and other anti-clockwise.

    Examples
    --------
    TODO
    puv, quv, cw, ccw = spec_rot(u, v)

    References
    ----------
    .. [1] J. Gonella Deep Sea Res., 833-846, 1972.
    """

    # Individual components Fourier series.
    fu, fv = map(np.fft.fft, (u, v))

    # Auto-spectra of the scalar components.
    pu = fu * np.conj(fu)
    pv = fv * np.conj(fv)

    # Cross spectra.
    puv = fu.real * fv.real + fu.imag * fv.imag

    # Quadrature spectra.
    quv = -fu.real * fv.imag + fv.real * fu.imag

    # Rotatory components
    # TODO: Check the division, 4 or 8?
    cw = (pu + pv - 2 * quv) / 8.
    ccw = (pu + pv + 2 * quv) / 8.

    return puv, quv, cw, ccw


def lagcorr(x, y, M=None):
    r"""Compute lagged correlation between two series.
    Follow emery and Thomson book "summation" notation.

    Parameters
    ----------
    y : array
        time-series
    y : array
        time-series
    M : integer
        number of lags

    Returns
    -------
    Cxy : array
          normalized cross-correlation function

    Examples
    --------
    TODO
    """

    x, y = map(np.asanyarray, (x, y))
    try:
        np.broadcast(x, y)
    except ValueError:
        pass  # TODO: Print error and leave gracefully.

    if not M:
        M = x.size

    N = x.size
    Cxy = np.zeros(M)
    x_bar, y_bar = x.mean(), y.mean()

    for k in range(0, M, 1):
        a = 0.
        for i in range(N - k):
            a = a + (y[i] - y_bar) * (x[i + k] - x_bar)

        Cxy[k] = 1. / (N - k) * a

    return Cxy / (np.std(y) * np.std(x))


def mld(S, T, P):
    r"""Compute a mixed layer depth (MLD) from vertical profiles of salinity
    and temperature.

    There are many criteria out there to compute a MLD, the one used here
    defined MLD as the first depth for which:
    ST(T(z), S(z)) > ST(T(z=0) - 0.8, S(z=0))

    FIXME: Reference for this definition.

    Parameters
    ----------
    S : array
        Salinity [g/Kg]
    CT : array
        Conservative Temperature [degC]
    Z : array
        Depth [m]

    Returns
    -------
    mld : array
        Mixed layer depth [m]

    """

    # TODO: Use gsw
    #P = 0.09998 * 9.81 * np.abs(Z)
    #depth = gsw.z_from_p(?)
    #T = sw_ptmp(S, T, P, 0)

    zmin, iz0 = np.abs(Z).min(), np.abs(Z).argmin()

    if np.isnan(T[iz0]):
        iz0 = iz0 + 1

    SST = T[iz0]
    SSS = S[iz0]

    SST08 = SST - 0.8
    #SSS   = SSS + 35
    #gsw.dens
    Surfadens08 = densjmd95(SSS, SST08, P[iz0]) - 1000
    ST = densjmd95(S, T, P) - 1000

    mm = ST > Surfadens08
    if mm.any():
        mld = Z[mm]
    else:
        # TODO: raise
        mld = np.NaN

    return mld

def psd_ci(x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
                    window=mlab.window_hanning, noverlap=0, pad_to=None,
                    sides='default', scale_by_freq=None, smooth=None,
                    Confidencelevel=0.9):
    r"""Extention of matplotlib.mlab.psd The power spectral density with
    upper and lower limits within confidence level You should not use
    Welch's method here, instead you can use smoother in frequency domain
    with *smooth*

    Same input as matplotlib.mlab.psd except the following new inputs

    *smooth*
        smoothing window in frequency domain for example, 1-2-1 filter is
        [1,2,1] default is None.

    *Confidencelevel*
        Confidence level to estimate upper and lower (default 0.9).

    Returns the tuple (*Pxx*, *freqs*, *upper*, *lower*).
    upper and lower are limits of psd within confidence level.

    Original: http://oceansciencehack.blogspot.com/2010/04/psd.html

    Examples
    --------
    TODO
    """
    Pxxtemp, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap,
                              pad_to, sides, scale_by_freq)

    # Un-commented if you want to get Minobe-san's result
    # Pxxtemp = Pxxtemp / float(NFFT) / 2. * ((np.abs(window) ** 2).mean())

    # Smoothing.
    if smooth is not None:
        smooth = np.asarray(smooth)
        avfnc = smooth / np.float(np.sum(smooth))
        Pxx = np.convolve(Pxxtemp, avfnc, mode="same")
        #Pxx = np.convolve(Pxxtemp[:, 0], avfnc, mode="same")
    else:
        Pxx = Pxxtemp
        #Pxx = Pxxtemp[:, 0]
        avfnc = np.asarray([1.])

    # Estimate upper and lower estimate with equivalent degree of freedom.
    if pad_to is None:
        pad_to = NFFT

    if cbook.iterable(window):
        assert(len(window) == NFFT)
        windowVals = window
    else:
        windowVals = window(np.ones((NFFT,), x.dtype))

    # Equivalent degree of freedom.
    edof = (1. + (1. / np.sum(avfnc ** 2) - 1.) * np.float(NFFT) /
                                          np.float(pad_to) * windowVals.mean())

    a1 = (1. - Confidencelevel) / 2.
    a2 = 1. - a1

    lower = Pxx * stats.chi2.ppf(a1, 2 * edof) / stats.chi2.ppf(0.50, 2 * edof)
    upper = Pxx * stats.chi2.ppf(a2, 2 * edof) / stats.chi2.ppf(0.50, 2 * edof)

    cl = np.c_[upper, lower]

    return Pxx, freqs, cl
