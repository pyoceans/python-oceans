import numpy as np


def lanc(numwt, haf):
    """
    Generates a numwt + 1 + numwt lanczos cosine low pass filter with -6dB
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
    >>> from oceans.filters import lanc
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


def smoo1(datain, window_len=11, window='hanning'):
    """
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
    scipy.signal.lfilter

    Notes
    -----
    original from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with
    the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output signal.

    Examples
    --------
    >>> from oceans.filters import smoo1
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
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
    >>> leg = ax.legend(windows)
    >>> _ = plt.title('The smoothing windows')
    >>> ax = plt.subplot(212)
    >>> l1, = ax.plot(series)
    >>> l2, = ax.plot(noise_series)
    >>> for w in windows:
    ...     _ = plt.plot(smoo1(noise_series, 10, w))
    >>> l = ['original signal', 'signal with noise']
    >>> l.extend(windows)
    >>> leg = ax.legend(l)
    >>> _ = plt.title('Smoothing a noisy signal')

    TODO: window parameter can be the window itself (i.e. an array)
    instead of a string.

    """

    datain = np.asarray(datain)

    if datain.ndim != 1:
        raise ValueError('Smooth only accepts 1 dimension arrays.')

    if datain.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if window_len < 3:
        return datain

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        msg = "Window must be is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"  # noqa
        raise ValueError(msg)

    s = np.r_[2 * datain[0] - datain[window_len:1:-1], datain, 2 *
              datain[-1] - datain[-1:-window_len:-1]]

    if window == 'flat':  # Moving average.
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    data_out = np.convolve(w / w.sum(), s, mode='same')
    return data_out[window_len - 1:-window_len + 1]


def smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14):
    """
    Calculates the smoothed array 'As' from the original array 'A' using the
    specified window of type 'kind' and shape ('hei', 'wid').

    Usage:
    As = smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14)

    Parameters
    ----------
    A : 2D array
      Array to be smoothed.
    hei : integer
      Window height. Must be odd and greater than or equal to 3.
    wid : integer
      Window width. Must be odd and greater than or equal to 3.
    kind : string, optional
      Refer to Numpy for details about each window type.
    badflag : float, optional
      The bad data flag. Elements of the input array 'A' holding this value are ignored.
    beta : float, optional
      Shape parameter for the kaiser window.

    Returns
    -------
    As : 2D array
      The smoothed array.

    André Palóczy Filho (paloczy@gmail.com)
    April 2012

    """
    # Checking window type and dimensions
    kinds = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']
    if (kind not in kinds):
        raise ValueError('Invalid window type requested: %s' % kind)

    if (np.mod(hei, 2) == 0) or (np.mod(wid, 2) == 0):
        raise ValueError('Window dimensions must be odd')

    if (hei <= 1) or (wid <= 1):
        raise ValueError('Window shape must be (3,3) or greater')

    # Creating the 2D window.
    if (kind == 'kaiser'):  # If the window kind is kaiser (beta is required).
        wstr = 'np.outer(np.kaiser(hei, beta), np.kaiser(wid, beta))'
    # If the window kind is hann, hamming, blackman or bartlett
    # (beta is not required).
    else:
        if kind == 'hann':
            # Converting the correct window name (Hann) to the numpy function
            # name (numpy.hanning).
            kind = 'hanning'
        # Computing outer product to make a 2D window out of the original 1d
        # windows.
        # TODO: Get rid of this evil eval.
        wstr = 'np.outer(np.' + kind + '(hei), np.' + kind + '(wid))'
    wdw = eval(wstr)

    A = np.asanyarray(A)
    Fnan = np.isnan(A)
    imax, jmax = A.shape
    As = np.NaN * np.ones((imax, jmax))

    for i in range(imax):
        for j in range(jmax):
            # Default window parameters.
            wupp = 0
            wlow = hei
            wlef = 0
            wrig = wid
            lh = np.floor(hei / 2)
            lw = np.floor(wid / 2)

            # Default array ranges (functions of the i, j indices).
            upp = i - lh
            low = i + lh + 1
            lef = j - lw
            rig = j + lw + 1

            # Tiling window and input array at the edges.
            # Upper edge.
            if upp < 0:
                wupp = wupp - upp
                upp = 0

            # Left edge.
            if lef < 0:
                wlef = wlef - lef
                lef = 0

            # Bottom edge.
            if low > imax:
                ex = low - imax
                wlow = wlow - ex
                low = imax

            # Right edge.
            if rig > jmax:
                ex = rig - jmax
                wrig = wrig - ex
                rig = jmax

            # Computing smoothed value at point (i, j).
            Ac = A[upp:low, lef:rig]
            wdwc = wdw[wupp:wlow, wlef:wrig]
            fnan = np.isnan(Ac)
            Ac[fnan] = 0
            wdwc[fnan] = 0  # Eliminating NaNs from mean computation.
            fbad = Ac == badflag
            wdwc[fbad] = 0  # Eliminating bad data from mean computation.
            a = Ac * wdwc
            As[i, j] = a.sum() / wdwc.sum()
    # Assigning NaN to the positions holding NaNs in the original array.
    As[Fnan] = np.NaN

    return As


def weim(x, N, kind='hann', badflag=-9999, beta=14):
    """
    Calculates the smoothed array 'xs' from the original array 'x' using the
    specified window of type 'kind' and size 'N'. 'N' must be an odd number.

    Usage:
    xs = weim(x, N, kind='hann', badflag=-9999, beta=14)


    Parameters
    ----------
    x : 1D array
      Array to be smoothed.

    N : integer
      Window size. Must be odd.

    kind : string, optional
    Refer to Numpy for details about each window type.

    badflag : float, optional
              The bad data flag. Elements of the input array 'A' holding this
              value are ignored.

    beta    : float, optional
              Shape parameter for the kaiser window. For windows other than the
              kaiser window, this parameter does nothing.

    Returns
    -------
    xs      : 1D array
              The smoothed array.

    ---------------------------------------
    André Palóczy Filho (paloczy@gmail.com) June 2012

    """
    # Checking window type and dimensions.
    kinds = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']
    if (kind not in kinds):
        raise ValueError('Invalid window type requested: %s' % kind)

    if np.mod(N, 2) == 0:
        raise ValueError('Window size must be odd')

    # Creating the window.
    if (kind == 'kaiser'):  # If the window kind is kaiser (beta is required).
        wstr = 'np.kaiser(N, beta)'
    # If the window kind is hann, hamming, blackman or bartlett (beta is not
    # required).
    else:
        if kind == 'hann':
            # Converting the correct window name (Hann) to the numpy function
            # name (numpy.hanning).
            kind = 'hanning'
            # Computing outer product to make a 2D window out of the original
            # 1D windows.
        wstr = 'np.' + kind + '(N)'

    # FIXME: Do not use `eval`.
    w = eval(wstr)
    x = np.asarray(x).flatten()
    Fnan = np.isnan(x).flatten()

    ln = (N - 1) / 2
    lx = x.size
    lf = lx - ln
    xs = np.NaN * np.ones(lx)

    # Eliminating bad data from mean computation.
    fbad = x == badflag
    x[fbad] = np.nan

    for i in range(lx):
        if i <= ln:
            xx = x[:ln + i + 1]
            ww = w[ln - i:]
        elif i >= lf:
            xx = x[i - ln:]
            ww = w[:lf - i - 1]
        else:
            xx = x[i - ln:i + ln + 1]
            ww = w.copy()

        # Counting only NON-NaNs, both in the input array and in the window
        # points.
        f = ~np.isnan(xx)
        xx = xx[f]
        ww = ww[f]

        # Thou shalt not divide by zero.
        if f.sum() == 0:
            xs[i] = x[i]
        else:
            xs[i] = np.sum(xx * ww) / np.sum(ww)

    # Assigning NaN to the positions holding NaNs in the input array.
    xs[Fnan] = np.nan

    return xs


def medfilt1(x, L=3):
    """
    Median filter for 1d arrays.

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
    >>> from oceans.filters import medfilt1
    >>> import matplotlib.pyplot as plt
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
    >>> leg1 = ax.legend(['x (pseudo-random)','xout'])
    >>> t1 = ax.set_title('''Median filter with window length %s.
    ...                   Removes outliers, tracks remaining signal)''' % L)
    >>> L = 103
    >>> xout = medfilt1(x=x, L=L)
    >>> ax = plt.subplot(212)
    >>> l1, l2, = ax.plot(x), ax.plot(xout)
    >>> ax.grid(True)
    >>> y2min, y2max = np.min(xout) * 0.5, np.max(xout) * 2.0
    >>> leg2 = ax.legend(["Same x (pseudo-random)", "xout"])
    >>> t2 = ax.set_title('''Median filter with window length %s.
    ...                   Removes outliers and noise''' % L)
    >>> ax = plt.subplot(211)
    >>> lims1 = ax.set_ylim([min(y1min, y2min), max(y1max, y2max)])
    >>> ax = plt.subplot(212)
    >>> lims2 = ax.set_ylim([min(y1min, y2min), max(y1max, y2max)])

    """

    xin = np.atleast_1d(np.asanyarray(x))
    N = len(x)
    L = int(L)  # Ensure L is odd integer so median requires no interpolation.
    if L % 2 == 0:
        L += 1

    if N < 2:
        raise ValueError('Input sequence must be >= 2.')
        return None

    if L < 2:
        raise ValueError('Input filter window length must be >=2.')
        return None

    if L > N:
        msg = 'Input filter window length must be shorter than series: L = {0:d}, len(x) = {1:d}'.format  # noqa
        raise ValueError(msg(L, N))
        return None

    if xin.ndim > 1:
        msg = 'Input sequence has to be 1d: ndim = {}'.format
        raise ValueError(msg(xin.ndim))

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
    """
    Performs a low pass filer on the series.
    low and high specifies the boundary of the filter.

    >>> from oceans.filters import fft_lowpass
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(500)  # Time in hours.
    >>> x = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> x += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> x += 0.3 * np.random.randn(len(t))
    >>> filtered = fft_lowpass(x, low=1/30, high=1/40)
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(t, x, label='original')
    >>> l2, = ax.plot(t, filtered, label='filtered')
    >>> legend = ax.legend()

    """

    if len(signal) % 2:
        result = np.fft.rfft(signal, len(signal))
    else:
        result = np.fft.rfft(signal)

    freq = np.fft.fftfreq(len(signal))[:len(signal) // 2 + 1]
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


def md_trenberth(x):
    """
    Returns the filtered series using the Trenberth filter as described
    on Monthly Weather Review, vol. 112, No. 2, Feb 1984.

    Input data: series x of dimension 1Xn (must be at least dimension 11)
    Output data: y = md_trenberth(x) where y has dimension 1X(n-10)

    Examples
    --------
    >>> from oceans.filters import md_trenberth
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(500)  # Time in hours.
    >>> x = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> x += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> x += 0.3 * np.random.randn(len(t))
    >>> filtered = md_trenberth(x)
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(t, x, label='original')
    >>> pad = [np.NaN]*5
    >>> l2, = ax.plot(t, np.r_[pad, filtered, pad], label='filtered')
    >>> legend = ax.legend()

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


def pl33tn(x, dt=1.0, T=33.0, mode='valid'):
    """
    Computes low-passed series from `x` using pl33 filter, with optional
    sample interval `dt` (hours) and filter half-amplitude period T (hours)
    as input for non-hourly series.

    The PL33 filter is described on p. 21, Rosenfeld (1983), WHOI
    Technical Report 85-35.  Filter half amplitude period = 33 hrs.,
    half power period = 38 hrs.  The time series x is folded over
    and cosine tapered at each end to return a filtered time series
    xf of the same length.  Assumes length of x greater than 67.

    Examples
    --------
    >>> from oceans.filters import pl33tn
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(500)  # Time in hours.
    >>> x = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> x += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> x += 0.3 * np.random.randn(len(t))
    >>> filtered_33 = pl33tn(x, dt=4.0)   # 33 hr filter
    >>> filtered_33d3 = pl33tn(x, dt=4.0, T=72.0)  # 3 day filter
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(t, x, label='original')
    >>> pad = [np.NaN]*8
    >>> l2, = ax.plot(t, np.r_[pad, filtered_33, pad], label='33 hours')
    >>> pad = [np.NaN]*17
    >>> l3, = ax.plot(t, np.r_[pad, filtered_33d3, pad], label='3 days')
    >>> legend = ax.legend()


    """

    pl33 = np.array(
        [
            -0.00027, -0.00114, -0.00211, -0.00317, -0.00427, -0.00537,
            -0.00641, -0.00735, -0.00811, -0.00864, -0.00887, -0.00872,
            -0.00816, -0.00714, -0.00560, -0.00355, -0.00097, +0.00213,
            +0.00574, +0.00980, +0.01425, +0.01902, +0.02400, +0.02911,
            +0.03423, +0.03923, +0.04399, +0.04842, +0.05237, +0.05576,
            +0.05850, +0.06051, +0.06174, +0.06215, +0.06174, +0.06051,
            +0.05850, +0.05576, +0.05237, +0.04842, +0.04399, +0.03923,
            +0.03423, +0.02911, +0.02400, +0.01902, +0.01425, +0.00980,
            +0.00574, +0.00213, -0.00097, -0.00355, -0.00560, -0.00714,
            -0.00816, -0.00872, -0.00887, -0.00864, -0.00811, -0.00735,
            -0.00641, -0.00537, -0.00427, -0.00317, -0.00211, -0.00114,
            -0.00027
        ]
    )

    _dt = np.linspace(-33, 33, 67)

    dt = float(dt) * (33.0 / T)

    filter_time = np.arange(0.0, 33.0, dt, dtype='d')
    # N = len(filter_time)
    filter_time = np.hstack((-filter_time[-1:0:-1], filter_time))

    pl33 = np.interp(filter_time, _dt, pl33)
    pl33 /= pl33.sum()

    xf = np.convolve(x, pl33, mode=mode)
    return xf
