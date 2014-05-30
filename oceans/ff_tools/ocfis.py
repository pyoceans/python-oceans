# -*- coding: utf-8 -*-
#
# ocfis.py
#
# purpose:  Some misc PO functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Sun 23 Jun 2013 04:29:36 PM BRT
#
# obs:
#

from __future__ import division

import gsw
import numpy as np
import numpy.ma as ma
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from scipy.stats import nanmean, nanstd, chi2
from pandas import Series, date_range, isnull
from scipy.interpolate import InterpolatedUnivariateSpline


__all__ = ['bin_dates',
           'binave',
           'binavg',
           'complex_demodulation',
           'del_eta_del_x',
           'despike',
           'fft_lowpass',
           'interp_nan',
           'lagcorr',
           'lanc',
           'medfilt1',
           'mld',
           'pcaben',
           'plot_spectrum',
           'psd_ci',
           'series_spline',
           'smoo1',
           'spdir2uv',
           'spec_rot',
           'uv2spdir']


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
    >>> from pandas import read_table
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
    >>> import matplotlib.pyplot as plt
    >>> from windrose import WindroseAxes
    >>> from oceans.ff_tools import uv2spdir
    >>> def new_axes():
    ...     fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='w')
    ...     rect = [0.1, 0.1, 0.8, 0.8]
    ...     ax = WindroseAxes(fig, rect, axisbg='w')
    ...     fig.add_axes(ax)
    ...     return ax
    >>> def set_legend(ax):
    ...     l = ax.legend(axespad=-0.10)
    ...     plt.setp(l.get_texts(), fontsize=8)
    >>> u, v = [0., 1., -2., -1., 1.], [3., 1., 0., -1., -1.]
    >>> wd, ws = uv2spdir(u,v)
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
    >>> import os
    >>> import gsw
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
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


def smoo1(datain, window_len=11, window='hanning'):
    r"""Smooth the data using a window with requested size.

    Parameters
    ----------
    datain : array_like
             input series
    window_len : int
                 size of the smoothing window; should be an odd integer
    window : str
             window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
             flat window will produce a moving average smoothing.

    Returns
    -------
    data_out : array_like
            smoothed signal

    See Also
    --------
    scipy.signal.lfilter

    Notes
    -----
    original from: http://www.scipy.org/Cookbook/SignalSmooth
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with
    the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output signal.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> time = np.linspace( -4, 4, 100 )
    >>> series = np.sin(time)
    >>> noise_series = series + np.random.randn( len(time) ) * 0.1
    >>> data_out = ff.smoo1(series)
    >>> ws = 31
    >>> ax = plt.subplot(211)
    >>> _ = ax.plot(np.ones(ws))
    >>> windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    >>> for w in windows[1:]:
    ...     _ = eval('plt.plot(np.' + w + '(ws) )')
    >>> _ = ax.axis([0, 30, 0, 1.1])
    >>> _ = ax.legend(windows)
    >>> _ = plt.title("The smoothing windows")
    >>> ax = plt.subplot(212)
    >>> _ = ax.plot(series)
    >>> _ = ax.plot(noise_series)
    >>> for w in windows:
    ...     _ = plt.plot(ff.smoo1(noise_series, 10, w))
    >>> l = ['original signal', 'signal with noise']
    >>> l.extend(windows)
    >>> leg = ax.legend(l)
    >>> _ = plt.title("Smoothing a noisy signal")
    >>> plt.show()

    TODO: window parameter can be the window itself (i.e. an array)
    instead of a string.
    """

    datain = np.asarray(datain)

    if datain.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if datain.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return datain

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("""Window is on of 'flat', 'hanning', 'hamming',
                         'bartlett', 'blackman'""")

    s = np.r_[2 * datain[0] - datain[window_len:1:-1], datain, 2 *
              datain[-1] - datain[-1:-window_len:-1]]

    if window == 'flat':  # Moving average.
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    data_out = np.convolve(w / w.sum(), s, mode='same')
    return data_out[window_len - 1:-window_len + 1]


def spec_rot(u, v):
    r"""Compute the rotary spectra from u,v velocity components

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
    The spectral energy at some frequency can be decomposed into two circularly
    polarized constituents, one rotating clockwise and other anti-clockwise.

    Examples
    --------
    TODO: puv, quv, cw, ccw = spec_rot(u, v)

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
    cw = (pu + pv - 2 * quv) / 4.
    ccw = (pu + pv + 2 * quv) / 4.
    F = np.arange(0, N) / N
    return puv, quv, cw, ccw, F


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
    TODO: Emery and Thomson.
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

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> fig = plt.figure()
    >>> for npnl in range(4):
    ...     tlng, nsmpl = 100, 1000
    ...     if npnl==0:
    ...         nfft = 100
    ...         taper = signal.boxcar(tlng)
    ...         avfnc = [1, 1, 1, 1, 1]
    ...         cttl = "no padding, boxcar, 5-running"
    ...     elif npnl==1:
    ...         nfft = 100
    ...         taper = signal.hamming(tlng)
    ...         avfnc = [1, 1, 1, 1, 1]
    ...         cttl = "no padding, hamming, 5-running"
    ...     elif npnl==2:
    ...         nfft = 200
    ...         taper = signal.boxcar(tlng)
    ...         avfnc = [1, 1, 1, 1, 1]
    ...         cttl = "double padding, boxcar, 5-running"
    ...     elif npnl==3:
    ...         nfft = 200
    ...         taper = signal.hamming(tlng)
    ...         avfnc = np.convolve([1, 2, 1], [1, 2, 1])
    ...         cttl = "double padding, hamming, double 1-2-1"
    ...     tsrs = np.random.randn(tlng, nsmpl)
    ...     ds_psd = np.zeros([nfft / 2 + 1, nsmpl])
    ...     upper_psd = np.zeros([nfft / 2 + 1, nsmpl])
    ...     lower_psd = np.zeros([nfft / 2 + 1, nsmpl])
    ...     for n in range(nsmpl):
    ...         a, b, ci = ff.psd_ci(tsrs[:,n], NFFT=tlng, pad_to=nfft,
    ...                              Fs=1, window=taper, smooth=avfnc,
    ...                              Confidencelevel=0.9)
    ...         ds_psd[:,n] = a[:]
    ...         upper_psd[:, n] = ci[:, 0]
    ...         lower_psd[:, n] = ci[:, 1]
    ...         frq = b[:]
    ...     # 90% confidence level by Monte-Carlo.
    ...     srt_psd = np.sort(ds_psd, axis=1)
    ...     c = np.zeros([frq.size, 2])
    ...     c[:, 0] = np.log10(srt_psd[:, nsmpl * 0.05])
    ...     c[:, 1] = np.log10(srt_psd[:, nsmpl * 0.95])
    ...     # Estimate from extended degree of freedom.
    ...     ce = np.zeros([frq.size, 2])
    ...     ce[:, 0] = np.log10(np.sort(upper_psd, axis=1)[:, nsmpl * 0.5])
    ...     ce[:, 1] = np.log10(np.sort(lower_psd, axis=1)[:, nsmpl * 0.5])
    ...     ax = plt.subplot(2, 2, npnl + 1)
    ...     _ = ax.plot(frq, c, 'b', frq, ce, 'r')
    ...     _ = plt.title(cttl)
    ...     _ = plt.xlabel('frq')
    ...     _ = plt.ylabel('psd')
    ...     if (npnl==0):
    ...         _ = plt.legend(('Monte-carlo', '', 'Theory'), 'lower center',
    ...                         labelspacing=0.05)
    >>> _ = plt.subplots_adjust(wspace=0.6, hspace=0.4)
    >>> plt.show()

    Notes
    -----
    Based on http://oceansciencehack.blogspot.com/2010/04/psd.html

    """
    Pxxtemp, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap,
                              pad_to, sides, scale_by_freq)

    # Un-commented if you want to get Minobe-san's result
    # Pxxtemp = Pxxtemp / float(NFFT) / 2. * ((np.abs(window) ** 2).mean())

    # Smoothing.
    if smooth is not None:
        smooth = np.asarray(smooth)
        avfnc = smooth / np.float(np.sum(smooth))
        #Pxx = np.convolve(Pxxtemp, avfnc, mode="same")
        Pxx = np.convolve(Pxxtemp[:, 0], avfnc, mode="same")
    else:
        #Pxx = Pxxtemp
        Pxx = Pxxtemp[:, 0]
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

    lower = Pxx * chi2.ppf(a1, 2 * edof) / chi2.ppf(0.50, 2 * edof)
    upper = Pxx * chi2.ppf(a2, 2 * edof) / chi2.ppf(0.50, 2 * edof)

    cl = np.c_[upper, lower]

    return Pxx, freqs, cl


def complex_demodulation(series, f, fc, axis=-1):
    r"""Perform a Complex Demodulation
    It acts as a bandpass filter for `f`.

    series => Time-Series object with data and datetime
    f => inertial frequency in rad/sec
    fc => normalized cutoff [0.2]

    math : series.data * np.exp(2 * np.pi * 1j * (1 / T) * time_in_seconds)
    """

    # Time period ie freq = 1 / T
    T = 2 * np.pi / f
    #fc = fc * 1 / T Filipe

    # De mean data.
    d = series.data - series.data.mean(axis=axis)
    dfs = d * np.exp(2 * np.pi * 1j * (1 / T) * series.time_in_seconds)

    # make a 5th order butter filter
    # FIXME: Why 5th order? Try signal.buttord
    Wn = fc / series.Nyq

    [b, a] = signal.butter(5, Wn, btype='low')

    # FIXME: These are a factor of a thousand different from Matlab, why?
    cc = signal.filtfilt(b, a, dfs)  # FIXME: * 1e3
    amplitude = 2 * np.abs(cc)

    # TODO: Fix the outputs
    #phase = np.arctan2(np.imag(cc), np.real(cc))

    filtered_series = amplitude * np.exp(-2 * np.pi * 1j *
                                         (1 / T) * series.time_in_seconds)

    new_series = filtered_series.real, series.time

    #return cc, amplitude, phase, dfs, filtered_series
    return new_series


def plot_spectrum(data, fs):
    r"""Plots a Single-Sided Amplitude Spectrum of y(t)."""
    n = len(data)  # Length of the signal.
    k = np.arange(n)
    T = n / fs
    frq = k / T  # Two sides frequency range.
    frq = frq[range(n // 2)]  # One side frequency range

    # fft computing and normalization
    Y = np.fft.fft(data) / n
    Y = Y[range(n // 2)]

    # Plotting the spectrum.
    plt.semilogx(frq, np.abs(Y), 'r')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()


def medfilt1(x, L=3):
    r"""Median filter for 1d arrays.

    Performs a discrete one-dimensional median filter with window length `L` to
    input vector `x`.  Produces a vector the same size as `x`.  Boundaries are
    handled by shrinking `L` at edges; no data outside of `x` is used in
    producing the median filtered output.

    Parameters
    ----------
    x : array_like
        Input 1D data
    L : integer
        Window length

    Returns
    -------
    xout : array_like
           Numpy 1d array of median filtered result; same size as x

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> # 100 pseudo-random integers ranging from 1 to 100, plus three large
    >>> # outliers for illustration.
    >>> x = np.r_[np.ceil(np.random.rand(25)*100), [1000],
    ...           np.ceil(np.random.rand(25)*100), [2000],
    ...           np.ceil(np.random.rand(25)*100), [3000],
    ...           np.ceil(np.random.rand(25)*100)]
    >>> L = 2
    >>> xout = ff.medfilt1(x=x, L=L)
    >>> ax = plt.subplot(211)
    >>> l1, l2 = ax.plot(x), ax.plot(xout)
    >>> ax.grid(True)
    >>> y1min, y1max = np.min(xout) * 0.5, np.max(xout) * 2.0
    >>> _ = ax.legend(['x (pseudo-random)','xout'])
    >>> _ = ax.set_title('''Median filter with window length %s.
    ...                 Removes outliers, tracks remaining signal)''' % L)
    >>> L = 103
    >>> xout = ff.medfilt1(x=x, L=L)
    >>> ax = plt.subplot(212)
    >>> l1, l2, = ax.plot(x), ax.plot(xout)
    >>> ax.grid(True)
    >>> y2min, y2max = np.min(xout) * 0.5, np.max(xout) * 2.0
    >>> _ = ax.legend(["Same x (pseudo-random)", "xout"])
    >>> _ = ax.set_title('''Median filter with window length %s.
    ...              Removes outliers and noise''' % L)
    >>> ax = plt.subplot(211)
    >>> _ = ax.set_ylim([min(y1min, y2min), max(y1max, y2max)])
    >>> ax = plt.subplot(212)
    >>> _ = ax.set_ylim([min(y1min, y2min), max(y1max, y2max)])
    >>> plt.show()

    Notes
    -----
    Based on: http://staff.washington.edu/bdjwww/medfilt.py
    """

    xin = np.atleast_1d(np.asanyarray(x))
    N = len(x)
    L = int(L)  # Ensure L is odd integer so median requires no interpolation.
    if L % 2 == 0:
        L += 1

    if N < 2:
        raise ValueError("Input sequence must be >= 2")
        return None

    if L < 2:
        raise ValueError("Input filter window length must be >=2")
        return None

    if L > N:
        raise ValueError('''Input filter window length must be shorter than
                         series: L = %d, len(x) = %d''' % (L, N))
        return None

    if xin.ndim > 1:
        raise ValueError("input sequence has to be 1d: ndim = %s" % xin.ndim)
        return None

    xout = np.zeros_like(xin) + np.NaN

    Lwing = (L - 1) // 2

    # NOTE: Use np.ndenumerate in case I expand to +1D case
    for i, xi in enumerate(xin):
        if i < Lwing:  # Left boundary.
            xout[i] = np.median(xin[0:i + Lwing + 1])   # (0 to i + Lwing)
        elif i >= N - Lwing:  # Right boundary.
            xout[i] = np.median(xin[i - Lwing:N])  # (i-Lwing to N-1)
        else:  # Middle (N-2*Lwing input vector and filter window overlap).
            xout[i] = np.median(xin[i - Lwing:i + Lwing + 1])
            # (i-Lwing to i+Lwing)

    return xout


def fft_lowpass(signal, low, high):
    r"""Performs a low pass filer on the series.
    low and high specifies the boundary of the filter.

    obs: From tappy's filters.py.
    """

    if len(signal) % 2:
        result = np.fft.rfft(signal, len(signal))
    else:
        result = np.fft.rfft(signal)

    freq = np.fft.fftfreq(len(signal))[:len(signal) / 2 + 1]
    factor = np.ones_like(freq)
    factor[freq > low] = 0.0
    sl = np.logical_and(high < freq, freq < low)
    a = factor[sl]

    # Create float array of required length and reverse.
    a = np.arange(len(a) + 2).astype(float)[::-1]

    # Ramp from 1 to 0 exclusive.
    a = (a / a[0])[1:-1]

    # Insert ramp into factor.
    factor[sl] = a

    result = result * factor

    return np.fft.irfft(result, len(signal))


def binave(datain, r):
    r"""Averages vector data in bins of length r. The last bin may be the
    average of less than r elements. Useful for computing daily average time
    series (with r=24 for hourly data).

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
    >>> from oceans.ff_tools import binave
    >>> data = [10., 11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123.,
    ...         3.2, 0.52, 18.2, 10., 11., 13., 2., 34., 21.5, 6.46, 6.27,
    ...         7.0867, 15., 123., 3.2, 0.52, 18.2, 10., 11., 13., 2., 34.,
    ...         21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2, 0.52, 18.2, 10.,
    ...         11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2,
    ...         0.52, 18.2]
    >>> binave(data, 24)
    array([ 16.564725 ,  21.1523625,  22.4670875])

    References
    ----------
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
    z = datain[0:(l * r)].reshape(r, l, order='F')
    bindata = np.mean(z, axis=0)

    return bindata


def binavg(x, y, db):
    r"""Bins y(x) into db spacing.  The spacing is given in `x` units.
    y = np.random.random(20)
    x = np.arange(len(y))
    xb, yb = binavg(x, y, 2)
    plt.figure()
    plt.plot(x, y, 'k.-')
    plt.plot(xb, yb, 'r.-')
    """
    # Cut the corners.
    x_min, x_max = np.ceil(x.min()), np.floor(x.max())
    x = x.clip(x_min, x_max)

    # This is used to get the `inds`.
    xbin = np.arange(x_min, x_max, db)
    inds = np.digitize(x, xbin)

    # But this is the center of the bins.
    xbin = xbin - (db / 2.)

    # FIXME there is an IndexError if I want to show this.
    #for n in range(x.size):
        #print xbin[inds[n]-1], "<=", x[n], "<", xbin[inds[n]]

    ybin = np.array([y[inds == i].mean() for i in range(0, len(xbin))])
    #xbin = np.array([x[inds == i].mean() for i in range(0, len(xbin))])

    return xbin, ybin


# Slowly converting all the Time-series to work with a Pandas Series.
def bin_dates(self, freq, tz=None):
    r"""Take a pandas time Series and return a new Series on the specified
    frequency.
    FIXME: There is a bug when I use tz that two means are reported!

    Examples
    --------
    >>> import numpy as np
    >>> from pandas import Series, date_range
    >>> n = 365
    >>> sig = np.random.rand(n) + 2 * np.cos(2 * np.pi * np.arange(n))
    >>> dates = date_range(start='1/1/2000', end='30/12/2000', periods=365,
    ...                    freq='D')
    >>> series = Series(data=sig, index=dates)
    """
    #closed='left', label='left'
    new_index = date_range(start=self.index[0], end=self.index[-1],
                           freq=freq, tz=tz)

    new_series = self.groupby(new_index.asof).mean()

    # I want the averages at the center.
    new_series.index = new_series.index + freq.delta // 2
    return new_series


def series_spline(self):
    r"""Fill NaNs using a spline interpolation."""

    inds, values = np.arange(len(self)), self.values

    invalid = isnull(values)
    valid = -invalid

    firstIndex = valid.argmax()
    valid = valid[firstIndex:]
    invalid = invalid[firstIndex:]
    inds = inds[firstIndex:]

    result = values.copy()
    s = InterpolatedUnivariateSpline(inds[valid], values[firstIndex:][valid])
    result[firstIndex:][invalid] = s(inds[invalid])

    return Series(result, index=self.index, name=self.name)


def despike(self, n=3, recursive=False, verbose=False):
    r"""Replace spikes with np.NaN.
    Removing spikes that are >= n * std.
    default n = 3."""

    result = self.values.copy()
    outliers = (np.abs(self.values - nanmean(self.values)) >= n *
                nanstd(self.values))

    removed = np.count_nonzero(outliers)
    result[outliers] = np.NaN

    if verbose and not recursive:
        print("Removing from %s\n # removed: %s" % (self.name, removed))

    counter = 0
    if recursive:
        while outliers.any():
            result[outliers] = np.NaN
            outliers = np.abs(result - nanmean(result)) >= n * nanstd(result)
            counter += 1
            removed += np.count_nonzero(outliers)
        if verbose:
            print("Removing from %s\nNumber of iterations: %s # removed: %s" %
                  (self.name, counter, removed))
    return Series(result, index=self.index, name=self.name)


def md_trenberth(x):
    r"""Returns the filtered series using the Trenberth filter as described
    on Monthly Weather Review, vol. 112, No. 2, Feb 1984.

    Input data: series x of dimension 1Xn (must be at least dimension 11)
    Output data: y = md_trenberth(x) where y has dimension 1X(n-10)
    """
    x = np.asanyarray(x)
    weight = np.array([0.02700, 0.05856, 0.09030, 0.11742, 0.13567, 0.14210,
                       0.13567, 0.11742, 0.09030, 0.05856, 0.02700])

    sz = len(x)
    y = np.zeros(sz - 10)
    for i in range(5, sz - 5):
        y[i - 5] = 0
        for j in range(11):
            y[i - 5] = y[i - 5] + x[i - 6 + j + 1] * weight[j]

    return y

if __name__ == '__main__':
    import doctest
    doctest.testmod()
