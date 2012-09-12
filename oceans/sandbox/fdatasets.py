import numpy as np

from scipy.io import loadmat
from netCDF4 import Dataset
from seawater.csiro import dist


def near(x, x0):
    r"""Function near(x, x0) Given an 1D array x and a scalar x0,
    returns the index of the element of x closest to x0."""
    nearest_value_idx = (abs(x - x0)).argmin()
    return nearest_value_idx


def ftopo(x, y, topofile='gebco15-40s_30-52w_30seg.nc'):
    r"""
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

    x, y = map(np.asanyarray, (x, y))

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
    for I in xrange(x.size):
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
    r"""
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

    x, y = map(np.asanyarray, (x, y))

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

    for I in xrange(x.size):
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
