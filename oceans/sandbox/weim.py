import numpy as np


def weim(x, N, kind='hann', badflag=-9999, beta=14):
    r"""
    Usage
    -----
    xs = weim(x, N, kind='hann', badflag=-9999, beta=14)

    Description
    -----------
    Calculates the smoothed array 'xs' from the original array 'x' using the
    specified window of type 'kind' and size 'N'. 'N' must be an odd number.

    Parameters
    ----------
    x       : 1D array
              Array to be smoothed.

    N       : integer
              Window size. Must be odd.

    kind    : string, optional
              One of the window types available in the numpy module:

              hann (default) : Gaussian-like.  The weight decreases toward the
                               ends.  Its end-points are zeroed.
              hamming        : Similar to the hann window. Its end-points are
                               not zeroed, therefore it is discontinuous at the
                               edges, and may produce undesired artifacts.
              blackman       : Similar to the hann and hamming windows, with
                               sharper ends.
              bartlett       : Triangular-like. Its end-points are zeroed.
              kaiser         : Flexible shape. Takes the optional parameter
                               "beta" as a shape parameter.  For beta=0, the
                               window is rectangular. As beta increases, the
                               window gets narrower.

              Refer to the numpy functions for details about each window type.

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

    for i in xrange(lx):
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
