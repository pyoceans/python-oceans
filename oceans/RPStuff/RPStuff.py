#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# RPStuff.py
#
# purpose:  RPStuff matlab routines in python
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2011
# modified: Tue 12 Feb 2013 11:53:11 AM BRST
#
# obs: This "legacy" package is intended for compatibility only.
#      Most of function should be re-written in a more pythonic way.
#

from __future__ import division
from datetime import datetime

import numpy as np
import numpy.ma as ma

from oceans.utilities import match_args_return

__all__ = ['hms2h',
           'julian',
           'gregorian',
           's2hms',
           'angled',
           'coast2bln',
           'fixcoast']


earth_radius = 6371.e3

# TODO: at_leats_1D, float decorator


@match_args_return
def h2hms(hours):
    """Converts hours to hours, minutes, and seconds."""
    hour = np.floor(hours)
    mins = np.remainder(hours, 1.) * 60.
    mn = np.floor(mins)
    secs = np.round(np.remainder(mins, 1.) * 60.)
    return hour, mn, secs


@match_args_return
def hms2h(h, m=None, s=None):
    """Converts hours, minutes, and seconds to hours.

    Usage:
        hours = hms2h(h,m,s
        hours = hms2h(hhmmss)
    """
    if not m and not s:
        hms = h
        h = np.floor(hms / 10000.)
        ms = hms - h * 10000.
        m = np.floor(ms / 100.)
        s = ms - m * 100
        hours = h + m / 60. + s / 3600.
    else:
        hours = h + (m + s / 60.) / 60.

    return hours


@match_args_return
def ms2hms(millisecs):
    """Converts milliseconds to integer hour,minute,seconds."""
    sec = np.round(millisecs / 1000)
    hour = np.floor(sec / 3600)
    mn = np.floor(np.remainder(sec, 3600) / 60)
    sec = np.round(np.remainder(sec, 60))
    return hour, mn, sec


@match_args_return
def julian(y, m=0, d=0, h=0, mi=0, s=0):
    """
    RPStuff compat version
    Converts Gregorian calendar dates to Julian dates

    USAGE: [j]=julian(y,m,d,h)

    DESCRIPTION:  Converts Gregorian dates to decimal Julian days using the
                  astronomical convension, but with time zero starting at
                  midnight instead of noon.  In this convention, Julian day
                  2440000 begins at 0000 hours, May 23, 1968. The decimal
                  Julian day, with Matlab's double precision, yeilds an
                  accuracy of decimal days of about 0.1 milliseconds.

    INPUT:
        y =  year (e.g., 1979) component
        m =  month (1-12) component
        d =  day (1-31) component of Gregorian date

        hour = hours (0-23)
        min =  minutes (0-59)
        sec =  decimal seconds
            or
        h =  decimal hours (assumed 0 if absent)

    OUTPUT:
        j =  decimal Julian day number

    last revised 1/3/96 by Rich Signell (rsignell@usgs.gov)
    """
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

    """
    If you want julian days to start and end at noon,
    replace the following line with:
    j=j+(h-12)/24
    """
    j = j + h / 24.

    return j


@match_args_return
def jdrps2jdmat(jd):
    """Convert Signell's Julian days to Matlab's Serial day
    matlab's serial date = 1 at 0000 UTC, 1-Jan-0000
    """
    jdmat = jd - julian(0000, 1, 1, 0, 0, 0) + 1
    return jdmat


@match_args_return
def jdmat2jdrps(jdmat):
    """Convert Matlab's Serial Day to Signell's Julian days
    matlab's serial date = 1 at 0000 UTC, 1-Jan-0000
    """

    jd = jdmat + julian(0000, 1, 1, 0, 0, 0) - 1

    return jd


@match_args_return
def gregorian(jd):
    """
    GREGORIAN:  Converts Julian day numbers to Gregorian calendar.

    USAGE:      [gtime]=gregorian(jd)

    DESCRIPTION:  Converts decimal Julian days to Gregorian dates using the
                  astronomical convension, but with time zero starting
                  at midnight instead of noon.  In this convention,
                  Julian day 2440000 begins at 0000 hours, May 23, 1968.
                  The Julian day does not have to be an integer, and with
                  Matlab's double precision, the accuracy of decimal days
                  is about 0.1 milliseconds.


    INPUT:   jd  = decimal Julian days

    OUTPUT:  gtime = six column Gregorian time matrix, where each row is
                     [yyyy mo da hr mi sec].
                      yyyy = year (e.g., 1979)
                        mo = month (1-12)
                        da = day (1-31)
                        hr = hour (0-23)
                        mi = minute (0-59)
                       sec = decimal seconds
                   example: [1990 12 12 0 0 0] is midnight on Dec 12, 1990.

    AUTHOR: Rich Signell  (rsignell@usgs.gov)


    Add 0.2 milliseconds before Gregorian calculation to prevent
    roundoff error resulting from math operations on time
    from occasionally representing midnight as
    (for example) [1990 11 30 23 59 59.99...] instead of [1990 12 1 0 0 0]);
    If adding a 0.2 ms to time (each time you go back and forth between
    Julian and Gregorian) bothers you more than the inconvenient representation
    of Gregorian time at midnight you can comment this line out...
    """
    jd = jd + 2.e-9

    #if you want Julian Days to start at noon...
    #h = np.remainder(jd,1) * 24 + 12
    #i = (h >= 24)
    #jd[i] = jd[i] + 1
    #h[i] = h[i] - 24

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
    gtime = np.c_[yr, mo, d, hr, mi, sc]

    return gtime


@match_args_return
def s2hms(secs):
    """Converts seconds to integer hour,minute,seconds
    Usage: hour, min, sec = s2hms(secs)"""

    hr = np.floor(secs / 3600)
    mi = np.floor(np.remainder(secs, 3600) / 60)
    sc = np.round(np.remainder(secs, 60))

    return hr, mi, sc


@match_args_return
def ss2(jd):
    """Return Gregorian start and stop dates of Julian day variable
    Usage:  start, stop =ss2(jd)"""

    start = gregorian(jd[0])
    stop = gregorian(jd[-1])

    return start, stop


@match_args_return
def angled(h):
    r"""
    ANGLED: Returns the phase angles in degrees of a matrix with complex
            elements.

    Usage:
        deg = angled(h)
        h = complex matrix
        deg = angle in math convention (degrees counterclockwise from "east")
    """

    pd = np.angle(h, deg=True)

    return pd


@match_args_return
def ij2ind(a, i, j):
    m, n = a.shape
    return m * i - j + 1  # TODO: Check this +1


@match_args_return
def ind2ij(a, ind):
    """ind2ij returns i, j indices of array."""

    m, n = a.shape
    j = np.ceil(ind / m)
    i = np.remainder(ind, m)
    i[i == 0] = m

    return i, j


@match_args_return
def rms(u):
    """Compute root mean square for each column of matrix u."""
    # TODO: use an axis arg.
    if u.ndim > 1:
        m, n = u.shape
    else:
        m = u.size

    return np.sqrt(np.sum(u ** 2) / m)


@match_args_return
def z0toCn(z0, H):
    r"""
    Convert roughness height z0 to Chezy "C" and Manning's "n" which is a
    function of the water depth

    Inputs:
        z0 = roughness height (meters)
        H = water depth (meters) (can be vector)
    Outputs:
        C = Chezy "C" (non-dimensional)
        n = Manning's "n" (non-dimensional)

    Example:
        C, n = z0toCn(0.003, np.arange(2, 200))
        finds vectors C and n corresponding to a z0=0.003 and
        a range of water depths from 2--200 meters
    """

    k_s = 30 * z0
    C = 18 * np.log10(12 * H / k_s)
    n = (H ** (1.0 / 6.0)) / C

    return C, n


@match_args_return
def z0tocd(z0, zr):
    """Calculates CD at a given ZR corresponding to Z0."""

    cd = (0.4 * np.ones(z0.size) / np.log(zr / z0)) ** 2

    return cd


@match_args_return
def short_calc(amin, amax):
    rang = 32767 - (-32767)
    add_offset = (amax + amin) * 0.5
    scale_factor = (amax - amin) / rang

    return add_offset, scale_factor


@match_args_return
def gsum(x, **kw):
    r"""Just like sum, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.sum(xnew, **kw)


@match_args_return
def gmean(x, **kw):
    r"""Just like mean, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.mean(xnew, **kw)


@match_args_return
def gmedian(x, **kw):
    r"""Just like median, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.median(xnew, **kw)


@match_args_return
def gmin(x, **kw):
    r"""Just like min, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.min(xnew, **kw)


@match_args_return
def gmax(x, **kw):
    r"""Just like max, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.max(xnew, **kw)


@match_args_return
def gstd(x, **kw):
    r"""Just like std, except that it skips over bad points."""
    xnew = ma.masked_invalid(x)
    return np.std(xnew, **kw)

# TODO
#/home/filipe/00-NOBKP/matlab_2010b/toolbox/map/map/distance.m
#def near(x, x0, n=1):
    #r"""Finds the indices of x that are closest to the point x0.  x is an
    #array, x0 is a point, n is the number of closest points to get (in order
    #of increasing distance).  Distance is the abs(x-x0).
    #"""

    #distance, index = sort(abs(x-x0))
    #distance = distance(0:n)
    #index = index(0:n)

    #return index, distance

#@match_args_return
#def nearxy(x, y, x0, y0, dist=None):
    #r"""Finds the indices of (x,y) that are closest to the point (x0,y0).

    #Finds the closest point and the distance:
        #index, distance = nearxy(x, y, x0, y0)

    #Finds all points closer than the value of dist:
        #index, distance = nearxy(x, y, x0, y0, dist)
    #"""

    #distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    #if not dist:
        #index = np.where(distance <= dist)  # Finds points closer than dist.
    #else:
        #index = np.where(distance == distance.min())  # Finds closest point.
        #index = index[0]

    #distance = distance[index]
    #return index, distance


def swantime(a):
    r"""Converts SWAN default time format to datetime object."""
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


@match_args_return
def shift(a, b, n):
    r"""a and b are vectors
    n is number of points of a to cut off
    anew and bnew will be the same length.
    """

    #la, lb = a.size, lb = b.size

    anew = a[range(0 + n, len(a)), :]

    if len(anew) > len(b):
        anew = anew[range(0, len(b)), :]
        bnew = b
    else:
        bnew = b[range(0, len(anew)), :]

    return anew, bnew


@match_args_return
def lagcor(a, b, n):
    r"""Finds lagged correlations between two series.
    a and b are two column vectors
    n is range of lags
    cor is correlation as fn of lag
    """
    cor = []
    for i in range(0, n + 1):
        d1, d2 = shift(a, b, i)
        ind = ~np.isnan(d1 + d2)
        c = np.corrcoef(d1[ind], d2[ind])
        if len(c) > 1:
            cor.append(c[0, 1])

    return cor

# TODO: Check basemap
#def ll2utm(lon, lat, zone):
    """LL2UTM convert lat,lon to UTM."""
    # m_proj('UTM','ellipsoid','wgs84','zone',zone)
    # x, y = m_ll2xy(lon,lat,'clip','off')
    #return x, y

#def ll2merc(lon, lat):
    #m_proj('mercator')
    #x, y = m_ll2xy(lon, lat)

    ## Convert mercator to meters
    #x = x * earth_radius
    #y = y * earth_radius
    #return x, y

#def merc2ll(x, y):
    #"""LL2MERC Converts lon,lat to Mercator."""
    #m_proj('mercator')
    #lon, lat = m_xy2ll(x / earth_radius, y / earth_radius)
    #return lon, lat

#def utm2ll(x, y, zone):
    #"""Convert UTM to lat, lon."""
    #m_proj('UTM','ellipsoid','wgs84','zone',zone)
    #lon, lat = m_xy2ll(x,y)
    #return lon,lat


def coast2bln(coast, bln_file):
    r"""Converts a matlab coast (two column array w/ nan for line breaks) into
    a Surfer blanking file.

    Where coast is a two column vector and bln_file is the output file name.

    Needs `fixcoast`."""

    c2 = fixcoast(coast)
    ind = np.where(np.isnan(c2[:, 0]))[0]
    n = len(ind) - 1
    bln = c2.copy()

    for i in range(0, n - 1):
        ii = range(ind[i] + 1, ind[i + 1])
        NP = int(len(ii))
        bln[ind[i], 0] = NP
        bln[ind[i], 1] = int(1)

    bln = bln[:-1]
    np.savetxt(bln_file, bln, fmt='%g')


def fixcoast(coast):
    r"""FIXCOAST  Makes sure coastlines meet Signell's conventions.

    Fixes coastline is in the format we want.  Assumes that lon/lat are in the
    first two columns of the matrix coast, and that coastline segments are
    separated by rows of NaNs (or -99999s).  This routine ensures that only 1
    row of NaNs separates each segment, and makes sure that the first and last
    rows contain NaNs."""

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
