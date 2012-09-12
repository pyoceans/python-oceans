# -*- coding: utf-8 -*-
#
#
# time_series.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  12-Feb-2012
# modified: Wed 12 Sep 2012 11:52:51 AM BRT
#
# obs:
#

from __future__ import division

import numpy as np
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from scipy.stats import nanmean, nanstd, chi2
from pandas import Series, date_range, isnull
from scipy.interpolate import InterpolatedUnivariateSpline

__all__ = [
    'TimeSeries',
    'smoo1',
    'spec_rot',
    'lagcorr',
    'psd_ci',
    'complex_demodulation',
    'plot_spectrum',
    'medfilt1',
    'fft_lowpass',
    'despike_slope',
    'binave',
    'binavg',
    'bin_dates',
    'series_spline',
    'despike',
]


class TimeSeries(object):
    r""" Time-series object to store data and time information.
    Contains some handy methods... Still a work in progress.
    """
    def __init__(self, data, time):
        r"""data : array_like
            Just a data container
        time : datetime object
            The series time information.
            TODO: Must be regularly spaced.
            Changed this to a more robust method, maybe interpolate?
        """
        data, time = map(np.asanyarray, (data, time))

        # Derived information.
        time_in_seconds = [(t - time[0]).total_seconds() for t in time]
        dt = np.unique(np.diff(time_in_seconds))

        # TODO raise something if assertion does not pass.
        assert len(dt) == 1

        fs = 1.0 / dt  # Sampling frequency.
        Nyq = fs / 2.0

        self.data = data
        self.time = time
        self.fs = fs
        self.Nyq = Nyq
        self.dt = dt
        self.time_in_seconds = np.asanyarray(time_in_seconds)

    def plot_spectrum(self):
        r"""Plots a Single-Sided Amplitude Spectrum of y(t)."""
        n = len(self.data)  # Length of the signal.
        k = np.arange(n)
        T = n / self.fs
        frq = k / T  # Two sides frequency range.
        frq = frq[range(n // 2)]  # One side frequency range

        # fft computing and normalization
        Y = np.fft.fft(self.data) / n
        Y = Y[range(n // 2)]

        # Plotting the spectrum.
        plt.semilogx(frq, np.abs(Y), 'r')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')
        plt.show()


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
    The spectral energy at some frequency can be decomposed into two circulaly
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
    fc = fc * 1 / T

    # De mean data.
    d = series.data - series.data.mean(axis=axis)
    dfs = d * np.exp(2. * np.pi * 1j * (1. / T) * series.time_in_seconds)

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

    new_series = TimeSeries(filtered_series.real, series.time)

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


def despike_slope(datain, slope):
    r"""De-spikes a time-series by calculating point-to-point slopes and
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

    datain, slope = map(np.asanyarray, (datain, slope))

    cdata = np.zeros_like(datain) + np.NaN

    offset = datain.min()
    if offset < 0:
        datain = datain - offset
    else:
        offset = 0

    cdata[0] = datain[0]  # FIXME: Assume that the first point is not a spike.
    kk, npts = 0, len(datain)

    for k in range(1, npts, 1):
        try:
            nslope = datain[k] - cdata[kk]
            # if the slope is okay, let the data through
            if abs(nslope) <= abs(slope):
                kk = kk + 1
                cdata[kk] = datain[k]
            # if slope is not okay, look for the next data point
            else:
                n = 0
                # TODO: add a limit option for npts.
                while (abs(nslope) > abs(slope)) and (k + n < npts):
                    n = n + 1  # Keep index for the offset from the test point.
                    num = datain[k + n] - cdata[kk]
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
        except IndexError:
            print("Index out of bounds at %s" % k)
            continue
    return cdata + offset


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
    import numpy as np
    from pandas import Series, date_range
    n = 365
    sig = np.random.rand(n) + 2 * np.cos(2 * np.pi * np.arange(n))
    dates = date_range(start='1/1/2000', end='30/12/2000', periods=365, freq='D')
    series = Series(data=sig, index=dates)
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
