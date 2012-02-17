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
# modified: Fri 17 Feb 2012 10:29:03 AM EST
#
# obs:
#

from __future__ import division

import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


class TimeSeries(object):
    """ Time-series object to store data and time information.
    Contains some handy methods... Still a work in progress.
    """
    def __init__(self, data, time):
        """
        data : array_like
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
        """Plots a Single-Sided Amplitude Spectrum of y(t)."""
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
    r"""
    Smooth the data using a window with requested size.

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
    binave, binavg
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve,
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
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import ff_tools as ff
    >>> time = np.linspace( -4, 4, 100 )
    >>> series = np.sin(time)
    >>> noise_series = series + np.random.randn( len(time) ) * 0.1
    >>> data_out = ff.smoo1(series)
    >>> ws = 31
    >>> plt.subplot(211)
    >>> plt.plot( np.ones(ws) )
    >>> windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    >>> plt.hold(True)
    >>> for w in windows[1:]:
    >>>     eval('plt.plot(np.'+w+'(ws) )')
    >>> plt.axis([0,30,0,1.1])
    >>> plt.legend(windows)
    >>> plt.title("The smoothing windows")
    >>> plt.subplot(212)
    >>> plt.plot(series)
    >>> plt.plot(noise_series)
    >>> for w in windows:
    >>>     plt.plot( ff.smoo1(noise_series, 10, w) )
    >>> l = ['original signal', 'signal with noise']
    >>> l.extend(windows)
    >>> plt.legend(l)
    >>> plt.title("Smoothing a noisy signal")
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


def complex_demodulation(series, f, fc, axis=-1):
    """Perform a Complex Demodulation
    It acts as a bandpass filter for "f".

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
    """Plots a Single-Sided Amplitude Spectrum of y(t)."""
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
