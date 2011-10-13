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
# modified: Thu 13 Oct 2011 02:43:24 PM EDT
#
# obs: This "legacy" package is intended for compatibility only.
#      Most of function should be re-written in a more pythonic way.
#

import numpy as np
from ff_tools.library import match_args_return

__all__ = ['hms2h',
           'julian',
           'gregorian',
           's2hms',
           'angled'
          ]


@match_args_return
def hms2h(h, m=None, s=None):
    """
    Converts hours, minutes, and seconds to hours.

    Usage:
        hours = hms2h(h,m,s
        hours = hms2h(hhmmss)
    """
    if (m == None) and (s == None):
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

    """
    if you want Julian Days to start at noon...
    h = np.remainder(jd,1)*24+12
    i = (h >= 24)
    jd[i] = jd[i]+1
    h[i] = h[i]-24
    """

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
    """
    S2HMS:  converts seconds to integer hour,minute,seconds

    Usage: hour, min, sec = s2hms(secs)

    Rich Signell rsignell@usgs.gov
    """
    hr = np.floor(secs / 3600)
    mi = np.floor(np.remainder(secs, 3600) / 60)
    sc = np.round(np.remainder(secs, 60))

    return hr, mi, sc


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
