# -*- coding: utf-8 -*-

from __future__ import absolute_import, division


import numpy as np

from seawater import dist
from netCDF4 import Dataset
from scipy.io import loadmat

from ..RPSstuff import near

__all__ = ['ftopo',
           'fwoa',
           'weim',
           'smoo2']


def ftopo(x, y, topofile='gebco15-40s_30-52w_30seg.nc'):
    """
    Usage
    -----
    H, D, Xo, Yo = ftopo(x, y, topofile='gebco/gebco_08_30seg.nc')

    Description
    -----------
    Finds the depth of points of coordinates 'x','y' using GEBCO data set.

    Parameters
    ----------
    x         : 1D array
                Array containing longitudes of the points of unknown depth.

    y         : 1D array
                Array containing latitudes of the points of unknown depth.

    topofile  : string, optional
                String containing path to the GEBCO netCDF file.

    Returns
    -------
    H         : 1D array
                Array containing depths of points closest to the input X, Y
                coordinates.

    X         : 1D array
                Array of horizontal distance associated with array 'H'.

    D         : 1D array
                Array containing distances (in km) from the input X, Y
                coordinates to the data points.

    Xo        : 1D array
                Array containing longitudes of the data points.

    Yo        : 1D array
                Array containing latitudes of the data points.

    NOTES
    -------
    This function reads the entire netCDF file before extracting the wanted
    data.  Therefore, it does not handle the full GEBCO dataset (1.8 GB)
    efficiently.

    TODO
    -------
    Make it possible to work with the full gebco dataset, by extracting only
    the wanted indexes.

    Code History
    ---------------------------------------
    Author of the original Matlab code (ftopo.m, ETOPO2 dataset):
    Marcelo Andrioni <marceloandrioni@yahoo.com.br>
    December 2008: Modification performed by Cesar Rocha <cesar.rocha@usp.br>
    to handle ETOPO1 dataset.
    July 2012: Python Translation and modifications performed by André Palóczy
    Filho <paloczy@gmail.com>
    to handle GEBCO dataset (30 arc seconds resolution).

    """
    x, y = list(map(np.asanyarray, (x, y)))

    # Opening netCDF file and extracting data.
    grid = Dataset(topofile)
    yyr = grid.variables['y_range'][:]
    xxr = grid.variables['x_range'][:]
    spacing = grid.variables['spacing'][:]
    dx, dy = spacing[0], spacing[1]

    # Creating lon and lat 1D arrays.
    xx = np.arange(xxr[0], xxr[1], dx)
    xx = xx + dx / 2
    yy = np.arange(yyr[0], yyr[1], dy)
    yy = yy + dy / 2
    h = grid.variables['z'][:]
    grid.close()

    # Retrieving nearest point for each input coordinate.
    A = np.asanyarray([])
    xx, yy = np.meshgrid(xx, yy)
    ni, nj = xx.shape[0], yy.shape[1]
    h = np.reshape(h, (ni, nj))
    h = np.flipud(h)
    Xo = A.copy()
    Yo = A.copy()
    H = A.copy()
    D = A.copy()
    for I in range(x.size):
        ix = near(xx[0, :], x[I])
        iy = near(yy[:, 0], y[I])
        H = np.append(H, h[iy, ix])
        # Calculating distance between input and GEBCO points.
        D = np.append(D, dist([x[I], xx[0, ix]], [y[I], yy[iy, 0]],
                              units='km')[0])
        Xo = np.append(Xo, xx[0, ix])
        Yo = np.append(Yo, yy[iy, 0])
        # Calculating distance axis.
        X = np.append(0, np.cumsum(dist(Xo, Yo, units='km')[0]))

    return H, X, D, Xo, Yo


def fwoa(x, y, woafile='woa2009_annual.mat'):
    """
    Usage
    -----
    T,S,X,D,Xo,Yo = fwoa(x, y, woafile='woa2009_annual.mat')

    Description
    -----------
    Gets the TS profiles in the World Ocean Atlas (WOA) 2009 data set whose
    coordinates are closest to the input coordinates 'x', 'y'.

    Parameters
    ----------
    x         : 1D array
                Array containing longitudes of the points of unknown depth.

    y         : 1D array
                Array containing latitudes of the points of unknown depth.

    woafile   : string, optional
                String containing path to the WOA .mat file.

    Returns
    -------
    T         : 2D array
                Array containing the Temperature profiles closest to the input
                X, Y coordinates.

    S         : 2D array
                Array containing the Salinity (PSS-78) profiles closest to the
                input X,Y coordinates.

    X         : 1D array
                Array of horizontal distance associated with the TS profiles
                recovered.

    D         : 1D array
                Array containing distances (in km) from the input X, Y
                coordinates to the TS profiles.

    Xo        : 1D array
                Array containing longitudes of the WOA TS profiles.

    Yo        : 1D array
                Array containing latitudes of the WOA TS profiles.

    NOTES
    -------
    This function reads mat files, converted from the original netCDF ones.

    TODO
    -------
    Implement netCDF file reading (Original WOA 2009 format)
    Implement option to retrieve linearly interpolated profiles instead of
    nearest ones.

    """
    x, y = list(map(np.asanyarray, (x, y)))

    # Reading .mat file.
    d = loadmat(woafile)
    xx = d['lon']
    yy = d['lat']
    TT = d['temp']
    SS = d['salt']

    # Retrieving nearest profiles for each input coordinate.
    A = np.asanyarray([])
    B = np.NaN * np.ones((TT.shape[2], x.size))
    Xo = A.copy()
    Yo = A.copy()
    D = A.copy()
    T = B.copy()
    S = B.copy()

    for I in range(x.size):
        ix = near(xx[0, :], x[I])
        iy = near(yy[:, 0], y[I])
        T[:, I] = TT[iy, ix, :]
        S[:, I] = SS[iy, ix, :]
        # Calculating distance between input and nearest WOA points.
        D = np.append(D, dist([x[I], xx[0, ix]], [y[I], yy[iy, 0]],
                              units='km')[0])
        Xo = np.append(Xo, xx[0, ix])
        Yo = np.append(Yo, yy[iy, 0])

        # Calculating distance axis.
        X = np.append(0, np.cumsum(dist(Xo, Yo, units='km')[0]))

    return T, S, X, D, Xo, Yo


def weim(x, N, kind='hann', badflag=-9999, beta=14):
    """
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


def smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14):
    """
    Usage
    -----
    As = smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14)

    Description
    -----------
    Calculates the smoothed array 'As' from the original array 'A' using the
    specified window of type 'kind' and shape ('hei','wid').

    Parameters
    ----------
    A       : 2D array
              Array to be smoothed.

    hei     : integer
              Window height. Must be odd and greater than or equal to 3.

    wid     : integer
              Window width. Must be odd and greater than or equal to 3.

    kind    : string, optional
              One of the window types available in the numpy module:

              hann (default) : Gaussian-like. The weight decreases toward the
              ends. Its end-points are zeroed.
              hamming        : Similar to the hann window. Its end-points are
                               not zeroed, therefore it is discontinuous at
                               the edges, and may produce artifacts.
              blackman       : Similar to the hann and hamming windows, with
                               sharper ends.
              bartlett       : Triangular-like. Its end-points are zeroed.
              kaiser         : Flexible shape. Takes the optional parameter
                               "beta" as a shape parameter.  For beta=0, the
                               window is rectangular. As beta increases, the
                               window gets narrower.

              Refer to Numpy for details about each window type.

    badflag : float, optional
              The bad data flag. Elements of the input array 'A' holding this
              value are ignored.

    beta    : float, optional
              Shape parameter for the kaiser window. For windows other than
              the kaiser window, this parameter does nothing.

    Returns
    -------
    As      : 2D array
              The smoothed array.
    TODO
    ----
    This function definitely needs optimization.
    It is extremely computationally expensive.

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
