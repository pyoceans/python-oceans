from datetime import datetime

import numpy as np
import numpy.ma as ma


earth_radius = 6371.e3


def h2hms(hours):
    """
    Converts hours to hours, minutes, and seconds.

    Examples
    --------
    >>> h2hms(12.51)
    (12.0, 30.0, 36.0)

    """
    hour = np.floor(hours)
    mins = np.remainder(hours, 1.) * 60.
    mn = np.floor(mins)
    secs = np.round(np.remainder(mins, 1.) * 60.)
    return hour, mn, secs


def hms2h(h, m=None, s=None):
    """
    Converts hours, minutes, and seconds to hours.

    Examples
    --------
    >>> hms2h(12., 30., 36.)
    12.51
    >>> # Or,
    >>> hms2h(123036)
    12.51

    """
    if not m and not s:
        hms = h
        h = np.floor(hms / 10000)
        ms = hms - h * 10000
        m = np.floor(ms / 100)
        s = ms - m * 100
        hours = h + m / 60 + s / 3600
    else:
        hours = h + (m + s / 60) / 60
    return hours


def ms2hms(millisecs):
    """
    Converts milliseconds to integer hour, minute, seconds.

    Examples
    --------
    >>> ms2hms(1e3 * 60)
    (0.0, 1.0, 0.0)

    """
    sec = np.round(millisecs / 1000)
    hour = np.floor(sec / 3600)
    mn = np.floor(np.remainder(sec, 3600) / 60)
    sec = np.round(np.remainder(sec, 60))
    return hour, mn, sec


def julian(y, m=0, d=0, h=0, mi=0, s=0, noon=False):
    """
    Converts Gregorian calendar dates to Julian dates

    USAGE: [j]=julian(y,m,d,h)

    DESCRIPTION:  Converts Gregorian dates to decimal Julian days using the
                  astronomical conversion, but with time zero starting at
                  midnight instead of noon.  In this convention, Julian day
                  2440000 begins at 0000 hours, May 23, 1968. The decimal
                  Julian day, with double precision, yields an accuracy of
                  decimal days of about 0.1 milliseconds.

    If you want Julian days to start and end at noon set `noon` to True.

    INPUT:
    y : year (e.g., 1979) component
    m : month (1-12) component
    d : day (1-31) component of Gregorian date
    hour : hours (0-23)
    min : minutes (0-59)
    sec : decimal seconds or
    h : decimal hours (assumed 0 if absent)

    OUTPUT:
    j : decimal Julian day number

    last revised 1/3/96 by Rich Signell (rsignell@usgs.gov)

    Examples
    --------
    >>> julian(1968, 5, 23, 0)
    array([2440000.])

    """
    y, m, d, h, mi, s = list(map(np.atleast_1d, (y, m, d, h, mi, s)))
    h = hms2h(h, mi, s)

    mo = m + 9
    yr = y - 1
    i = m > 2
    mo[i] = m[i] - 3
    yr[i] = y[i]
    c = np.floor(yr / 100.)
    yr = yr - c * 100
    j = (np.floor((146097 * c) / 4.) + np.floor((1461 * yr) / 4.) +
         np.floor((153 * mo + 2) / 5.) + d + 1721119)

    if noon:
        j = j + (h - 12) / 24
    else:
        j = j + h / 24

    return j


def jdrps2jdmat(jd):
    """
    Convert Signell's Julian days to Matlab's Serial day
    matlab's serial date = 1 at 0000 UTC, 1-Jan-0000.

    Examples
    --------
    >>> jdrps2jdmat(2440000)
    array([718941.])

    """
    return jd - julian(0000, 1, 1, 0, 0, 0) + 1


def jdmat2jdrps(jdmat):
    """
    Convert Matlab's Serial Day to Signell's Julian days
    matlab's serial date = 1 at 0000 UTC, 1-Jan-0000.

    Examples
    --------
    >>> jdmat2jdrps(718941)
    array([2440000.])

    """
    return jdmat + julian(0000, 1, 1, 0, 0, 0) - 1


def gregorian(jd, noon=False):
    """
    Converts decimal Julian days to Gregorian dates using the astronomical
    conversion, but with time zero starting at midnight instead of noon.  In
    this convention, Julian day 2440000 begins at 0000 hours, May 23, 1968.
    The Julian day does not have to be an integer, and with Matlab's double
    precision, the accuracy of decimal days is about 0.1 milliseconds.

    INPUT:   jd  = decimal Julian days

    OUTPUT:  gtime = six column Gregorian time matrix, where each row is:
    [yyyy mo da hr mi sec].
    yyyy = year (e.g., 1979)
    mo = month (1-12)
    da = day (1-31)
    hr = hour (0-23)
    mi = minute (0-59)
    sec = decimal seconds
    example: [1990 12 12 0 0 0] is midnight on Dec 12, 1990.

    Examples
    --------
    >>> gregorian(2440000)
    array([[1968.,    5.,   23.,    0.,    0.,    0.]])

    AUTHOR: Rich Signell  (rsignell@usgs.gov)

    """
    # Add 0.2 milliseconds before Gregorian calculation to prevent
    # roundoff error resulting from math operations on time
    # from occasionally representing midnight as
    # (for example) [1990 11 30 23 59 59.99...] instead of [1990 12 1 0 0 0]);
    # If adding a 0.2 ms to time (each time you go back and forth between
    # Julian and Gregorian) bothers you more than the inconvenient
    # representation of Gregorian time at midnight you can comment this
    # line out...

    jd = np.atleast_1d(jd)
    jd = jd + 2.e-9

    if noon:
        h = np.remainder(jd, 1) * 24 + 12
        i = (h >= 24)
        jd[i] = jd[i] + 1
        h[i] = h[i] - 24

    secs = np.remainder(jd, 1) * 24 * 3600

    j = np.floor(jd) - 1721119

    ini = 4 * j - 1
    y = np.floor(ini / 146097)
    j = ini - 146097 * y
    ini = np.floor(j / 4)
    ini = 4 * ini + 3
    j = np.floor(ini / 1461)
    d = np.floor(((ini - 1461 * j) + 4) / 4)
    ini = 5 * d - 3
    m = np.floor(ini / 153)
    d = np.floor(((ini - 153 * m) + 5) / 5)
    y = y * 100 + j
    mo = m - 9
    yr = y + 1
    i = (m < 10)
    mo[i] = m[i] + 3
    yr[i] = y[i]

    hr, mi, sc = s2hms(secs)
    return np.c_[yr, mo, d, hr, mi, sc]


def s2hms(secs):
    """
    Converts seconds to integer hour,minute,seconds
    Usage: hour, min, sec = s2hms(secs)

    Examples
    --------
    >>> s2hms(3600 + 60 + 1)
    (1.0, 1.0, 1)

    """
    hr = np.floor(secs / 3600)
    mi = np.floor(np.remainder(secs, 3600) / 60)
    sc = np.round(np.remainder(secs, 60))

    return hr, mi, sc


# FIXME: STOPPED HERE
def ss2(jd):
    """
    Return Gregorian start and stop dates of Julian day variable
    Usage:  start, stop = ss2(jd)

    """
    start = gregorian(jd[0])
    stop = gregorian(jd[-1])
    return start, stop


def angled(h):
    """
    ANGLED: Returns the phase angles in degrees of a matrix with complex
            elements.

    Usage:
        deg = angled(h)
        h = complex matrix
        deg = angle in math convention (degrees counterclockwise from "east")

    """
    pd = np.angle(h, deg=True)
    return pd


def ij2ind(a, i, j):
    m, n = a.shape
    return m * i - j + 1  # TODO: Check this +1


def ind2ij(a, ind):
    """
    ind2ij returns i, j indices of array.

    """
    m, n = a.shape
    j = np.ceil(ind / m)
    i = np.remainder(ind, m)
    i[i == 0] = m
    return i, j


def rms(u):
    """
    Compute root mean square for each column of matrix u.

    """
    # TODO: use an axis arg.
    if u.ndim > 1:
        m, n = u.shape
    else:
        m = u.size
    return np.sqrt(np.sum(u ** 2) / m)


def z0toCn(z0, H):
    """
    Convert roughness height z0 to Chezy "C" and Manning's "n" which is a
    function of the water depth

    Inputs:
        z0 = roughness height (meters)
        H = water depth (meters) (can be vector)
    Outputs:
        C = Chezy "C" (non-dimensional)
        n = Manning's "n" (non-dimensional)

    Examples
    --------
    >>> # finds vectors C and n corresponding to a z0=0.003
    >>> # and a range of water depths from 2--200 meters.
    >>> C, n = z0toCn(0.003, np.arange(2, 200))

    """

    k_s = 30 * z0
    C = 18 * np.log10(12 * H / k_s)
    n = (H ** (1.0 / 6.0)) / C

    return C, n


def z0tocd(z0, zr):
    """
    Calculates CD at a given ZR corresponding to Z0.

    """
    cd = (0.4 * np.ones(z0.size) / np.log(zr / z0)) ** 2
    return cd


def short_calc(amin, amax):
    rang = 32767 - (-32767)
    add_offset = (amax + amin) * 0.5
    scale_factor = (amax - amin) / rang
    return add_offset, scale_factor


def gsum(x, **kw):
    """
    Just like sum, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.sum(xnew, **kw)


def gmean(x, **kw):
    """
    Just like mean, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.mean(xnew, **kw)


def gmedian(x, **kw):
    """
    Just like median, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.median(xnew, **kw)


def gmin(x, **kw):
    """
    Just like min, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.min(xnew, **kw)


def gmax(x, **kw):
    """
    Just like max, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.max(xnew, **kw)


def gstd(x, **kw):
    """
    Just like std, except that it skips over bad points.

    """
    xnew = ma.masked_invalid(x)
    return np.std(xnew, **kw)


def near(x, x0, n=1):
    """
    Given an 1D array `x` and a scalar `x0`, returns the `n` indices of the
    element of `x` closest to `x0`.

    """
    distance = np.abs(x - x0)
    index = np.argsort(distance)
    return index[:n], distance[index[:n]]


def swantime(a):
    """
    Converts SWAN default time format to datetime object.

    """
    if isinstance(a, str):
        a = float(a)
        a = np.asanyarray(a)

    year = np.floor(a / 1e4)
    a = a - year * 1e4
    mon = np.floor(a / 1e2)
    a = a - mon * 1e2
    day = np.floor(a)
    a = a - day
    hour = np.floor(a * 1e2)
    a = a - hour / 1e2
    mn = np.floor(a * 1e4)
    a = a - mn / 1e4
    sec = np.floor(a * 1e6)

    return datetime(year, mon, day, hour, mn, sec)


def shift(a, b, n):
    """
    a and b are vectors
    n is number of points of a to cut off
    anew and bnew will be the same length.

    """
    # la, lb = a.size, lb = b.size

    anew = a[list(range(0 + n, len(a))), :]

    if len(anew) > len(b):
        anew = anew[list(range(0, len(b))), :]
        bnew = b
    else:
        bnew = b[list(range(0, len(anew))), :]
    return anew, bnew


def lagcor(a, b, n):
    """
    Finds lagged correlations between two series.
    a and b are two column vectors
    n is range of lags
    cor is correlation as fn of lag.

    """
    cor = []
    for k in range(0, n + 1):
        d1, d2 = shift(a, b, k)
        ind = ~np.isnan(d1 + d2)
        c = np.corrcoef(d1[ind], d2[ind])
        if len(c) > 1:
            cor.append(c[0, 1])

    return cor


def coast2bln(coast, bln_file):
    """
    Converts a matlab coast (two column array w/ nan for line breaks) into
    a Surfer blanking file.

    Where coast is a two column vector and bln_file is the output file name.

    Needs `fixcoast`.

    """

    c2 = fixcoast(coast)
    ind = np.where(np.isnan(c2[:, 0]))[0]
    n = len(ind) - 1
    bln = c2.copy()

    for k in range(0, n - 1):
        kk = list(range(ind[k] + 1, ind[k + 1]))
        NP = int(len(kk))
        bln[ind[k], 0] = NP
        bln[ind[k], 1] = int(1)

    bln = bln[:-1]
    np.savetxt(bln_file, bln, fmt='%g')


def fixcoast(coast):
    """
    FIXCOAST  Makes sure coastlines meet Signell's conventions.

    Fixes coastline is in the format we want.  Assumes that lon/lat are in the
    first two columns of the matrix coast, and that coastline segments are
    separated by rows of NaNs (or -99999s).  This routine ensures that only 1
    row of NaNs separates each segment, and makes sure that the first and last
    rows contain NaNs.

    """

    ind = coast == -99999.
    coast[ind] = np.NaN

    ind = np.where(np.isnan(coast[:, 0]))[0]
    dind = np.diff(ind)
    idup = np.where(dind == 1)[0]

    coast = np.delete(coast, ind[idup], axis=0)

    if not np.isnan(coast[0, 0]):
        coast = np.insert(coast, 0, np.NaN, axis=0)

    if not np.isnan(coast[-1, -1]):
        coast = np.append(coast, np.c_[np.NaN, np.NaN], axis=0)

    return coast
