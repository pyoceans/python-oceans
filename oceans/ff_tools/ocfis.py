# -*- coding: utf-8 -*-
#
# ocfis.py
#
# purpose:  Some misc PO functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Fri 27 Feb 2015 05:41:59 PM BRT
#
# obs:
#

from __future__ import division

import gsw
import numpy as np
import numpy.ma as ma
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.stats import nanmean, nanstd
from pandas import Series, date_range, isnull
from scipy.interpolate import InterpolatedUnivariateSpline


__all__ = ['bin_dates',
           'binave',
           'binavg',
           'complex_demodulation',
           'del_eta_del_x',
           'despike',
           'fft_lowpass',
           'lagcorr',
           'lanc',
           'medfilt1',
           'mld',
           'pcaben',
           'plot_spectrum',
           'series_spline',
           'smoo1',
           'spdir2uv',
           'spec_rot',
           'uv2spdir']


def lanc(numwt, haf):
    """Generates a numwt + 1 + numwt lanczos cosine low pass filter with -6dB
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
    >>> t = np.arange(500)  # Time in hours.
    >>> h = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> h += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> h += 0.3 * np.random.randn(len(t))
    >>> wt = lanc(96+1+96, 1./40)
    >>> low = np.convolve(wt, h, mode='same')
    >>> high = h - low
    >>> fig, (ax0, ax1) = plt.subplots(nrows=2)
    >>> _ = ax0.plot(high, label='high')
    >>> _ = ax1.plot(low, label='low')
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
    """Computes u, v components from speed and direction.

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
    """Computes speed and direction from u, v components.
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

    Notes
    -----
    Zero degrees is north.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from oceans.ff_tools import uv2spdir
    >>> u = [+0, -0.5, -0.50, +0.90]
    >>> v = [+1, +0.5, -0.45, -0.85]
    >>> wd, ws = uv2spdir(u,v)
    >>> fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    >>> kw = dict(arrowstyle="->")
    >>> wd = np.deg2rad(wd)
    >>> lines = [ax.annotate("", xy=(d, s), xytext=(0, 0), arrowprops=kw)
    ...  for d, s in zip(wd, ws)]
    >>> _ = ax.set_ylim(0, np.max(ws))
    """

    u, v, mag, rot = map(np.asarray, (u, v, mag, rot))

    vec = u + 1j * v
    spd = np.abs(vec)
    ang = np.angle(vec, deg=True)
    ang = ang - mag + rot
    ang = np.mod(90. - ang, 360.)  # Zero is North.

    return ang, spd


def del_eta_del_x(U, f, g, balance='geostrophic', R=None):
    """Calculate :mat: `\frac{\partial \eta} {\partial x}` for different
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
    """Compute the mixed layer depth.

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
    >>> from oceans import mld
    >>> from gsw.utilities import Bunch
    >>> # Read data file with check value profiles
    >>> datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
    >>> cv = Bunch(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
    >>> SA, CT, p = (cv.SA_chck_cast[:, 0], cv.CT_chck_cast[:, 0],
    ...              cv.p_chck_cast[:, 0])
    >>> fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True)
    >>> l0 = ax0.plot(CT, -p, 'b.-')
    >>> MDL, idx = mld(SA, CT, p, criterion='temperature')
    >>> l1 = ax0.plot(CT[idx], -p[idx], 'ro')
    >>> l2 = ax1.plot(CT, -p, 'b.-')
    >>> MDL, idx = mld(SA, CT, p, criterion='density')
    >>> l3 = ax1.plot(CT[idx], -p[idx], 'ro')
    >>> l4 = ax2.plot(CT, -p, 'b.-')
    >>> MDL, idx = mld(SA, CT, p, criterion='pdvar')
    >>> l5 = ax2.plot(CT[idx], -p[idx], 'ro')
    >>> _ = ax2.set_ylim(-500, 0)

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
    """Principal components of 2-d (e.g. current meter) data
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
    >>> from oceans import pcaben, uv2spdir
    >>> u = np.r_[(0., 1., -2., -1., 1.), np.random.randn(10)]
    >>> v = np.r_[(3., 1., 0., -1., -1.), np.random.randn(10)]
    >>> (mjrax, mjaz, mirax, miaz, el), (x1, x2, y1, y2) = pcaben(u, v)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot(x1, y1,'r-', x2, y2, 'r-')
    >>> ax.set_aspect('equal')
    >>> _ = ax.set_xlabel('U component')
    >>> _ = ax.set_ylabel('V component')
    >>> _ = ax.plot(u, v, 'bo')
    >>> _ = ax.axis([-3.2, 3.2, -3.2, 3.2])
    >>> mdir, mspd = uv2spdir(u.mean(), v.mean())
    >>> _ = ax.plot([0, u.mean()],[0, v.mean()], 'k-')
    >>> flatness = 1 - mirax / mjrax
    >>> if False:
    ...     print('Mean u = {}'.format(u.mean()))
    ...     print('Mean v = {}'.format(v.mean()))
    ...     print('Mean magnitude: {}'.format(mspd))
    ...     print('Direction: {}'.format(mdir))
    ...     print('Axis 1 Length: {}'.format(mjrax))
    ...     print('Axis 1 Azimuth: {}'.format(mjaz))
    ...     print('Axis 2 Length: {}'.format.format(mirax))
    ...     print('Axis 2 Azimuth: {}'.format(miaz))
    ...     print('elipticity is : {}'.format(el))
    ...     print('Negative (positive) means clockwise (anti-clockwise)')
    ...     print('Flatness: {}'.format(flatness))

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
    """Smooth the data using a window with requested size.

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
    >>> from oceans import smoo1
    >>> time = np.linspace( -4, 4, 100 )
    >>> series = np.sin(time)
    >>> noise_series = series + np.random.randn( len(time) ) * 0.1
    >>> data_out = smoo1(series)
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
    ...     _ = plt.plot(smoo1(noise_series, 10, w))
    >>> l = ['original signal', 'signal with noise']
    >>> l.extend(windows)
    >>> leg = ax.legend(l)
    >>> _ = plt.title("Smoothing a noisy signal")

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

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
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
    """Compute the rotary spectra from u,v velocity components

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
    N = len(u)
    F = np.arange(0, N) / N
    return puv, quv, cw, ccw, F


def lagcorr(x, y, M=None):
    """Compute lagged correlation between two series.
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


def complex_demodulation(series, f, fc, axis=-1):
    """Perform a Complex Demodulation
    It acts as a bandpass filter for `f`.

    series => Time-Series object with data and datetime
    f => inertial frequency in rad/sec
    fc => normalized cutoff [0.2]

    math : series.data * np.exp(2 * np.pi * 1j * (1 / T) * time_in_seconds)
    """

    # Time period ie freq = 1 / T
    T = 2 * np.pi / f
    # fc = fc * 1 / T # Filipe

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
    # phase = np.arctan2(np.imag(cc), np.real(cc))

    filtered_series = amplitude * np.exp(-2 * np.pi * 1j *
                                         (1 / T) * series.time_in_seconds)
    new_series = filtered_series.real, series.time
    # return cc, amplitude, phase, dfs, filtered_series
    return new_series


def plot_spectrum(data, fs):
    """Plots a Single-Sided Amplitude Spectrum of y(t)."""
    n = len(data)  # Length of the signal.
    k = np.arange(n)
    T = n / fs
    frq = k / T  # Two sides frequency range.
    frq = frq[range(n // 2)]  # One side frequency range

    # FFT computing and normalization.
    Y = np.fft.fft(data) / n
    Y = Y[range(n // 2)]

    # Plotting the spectrum.
    plt.semilogx(frq, np.abs(Y), 'r')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()


def medfilt1(x, L=3):
    """Median filter for 1d arrays.

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
    >>> from oceans import medfilt1
    >>> # 100 pseudo-random integers ranging from 1 to 100, plus three large
    >>> # outliers for illustration.
    >>> x = np.r_[np.ceil(np.random.rand(25)*100), [1000],
    ...           np.ceil(np.random.rand(25)*100), [2000],
    ...           np.ceil(np.random.rand(25)*100), [3000],
    ...           np.ceil(np.random.rand(25)*100)]
    >>> L = 2
    >>> xout = medfilt1(x=x, L=L)
    >>> ax = plt.subplot(211)
    >>> l1, l2 = ax.plot(x), ax.plot(xout)
    >>> ax.grid(True)
    >>> y1min, y1max = np.min(xout) * 0.5, np.max(xout) * 2.0
    >>> _ = ax.legend(['x (pseudo-random)','xout'])
    >>> _ = ax.set_title('''Median filter with window length %s.
    ...                 Removes outliers, tracks remaining signal)''' % L)
    >>> L = 103
    >>> xout = medfilt1(x=x, L=L)
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
    """Performs a low pass filer on the series.
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
    """Averages vector data in bins of length r. The last bin may be the
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
    """

    datain, r = np.asarray(datain), np.asarray(r, dtype=np.int)

    if datain.ndim != 1:
        raise ValueError("Must be a 1D array")

    if r <= 0:
        raise ValueError("Bin size R must be a positive integer.")

    # Compute bin averaged series.
    l = datain.size // r
    z = datain[0:(l * r)].reshape(r, l, order='F')
    bindata = np.r_[np.mean(z, axis=0), np.mean(datain[(l * r):])]

    return bindata


def binavg(x, y, db):
    """Bins y(x) into db spacing.  The spacing is given in `x` units.
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
    # for n in range(x.size):
    #    print xbin[inds[n]-1], "<=", x[n], "<", xbin[inds[n]]

    ybin = np.array([y[inds == i].mean() for i in range(0, len(xbin))])
    # xbin = np.array([x[inds == i].mean() for i in range(0, len(xbin))])

    return xbin, ybin


def bin_dates(self, freq, tz=None):
    """Take a pandas time Series and return a new Series on the specified
    frequency.

    Examples
    --------
    >>> import numpy as np
    >>> from pandas import Series, date_range
    >>> n = 24*30
    >>> sig = np.random.rand(n) + 2 * np.cos(2 * np.pi * np.arange(n))
    >>> dates = date_range(start='1/1/2000', periods=n, freq='H')
    >>> series = Series(data=sig, index=dates)
    >>> new_series = bin_dates(series, freq='D', tz=None)
    """
    new_index = date_range(start=self.index[0], end=self.index[-1],
                           freq=freq, tz=tz)
    new_series = self.groupby(new_index.asof).mean()
    # Averages at the center.
    secs = new_index.freq.delta.total_seconds()
    new_series.index = new_series.index.values + int(secs // 2)
    return new_series


def series_spline(self):
    """Fill NaNs using a spline interpolation."""

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
    """Replace spikes with np.NaN.
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
    """Returns the filtered series using the Trenberth filter as described
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


def pol2cart(theta, radius, units='deg'):
    """Convert from polar to Cartesian coordinates
    **usage**:
        x, y = pol2cart(theta, radius, units='deg')."""
    if units in ['deg', 'degs']:
        theta = theta * np.pi / 180.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def cart2pol(x, y):
    """Convert from Cartesian to polar coordinates.

    Example
    -------
    >>> x = [+0, -0.5]
    >>> y = [+1, +0.5]
    >>> cart2pol(x, y)
    (array([ 1.57079633,  2.35619449]), array([ 1.        ,  0.70710678]))
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius


def compass(u, v, **arrowprops):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, -0.5, -0.50, +0.90]
    >>> v = [+1, +0.5, -0.45, -0.85]
    >>> fig, ax = compass(u, v)
    """

    # Create plot.
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    angles, radii = cart2pol(u, v)

    # Arrows or sticks?
    kw = dict(arrowstyle="->")
    kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax


if __name__ == '__main__':
    import doctest
    doctest.testmod()
