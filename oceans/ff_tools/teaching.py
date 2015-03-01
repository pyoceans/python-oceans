# -*- coding: utf-8 -*-
#
#
# teaching.py
#
# purpose:  Teaching module of ff_tools.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Fri 27 Feb 2015 05:45:06 PM BRT
#
# obs: Just some basic example function.
#

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

__all__ = ['TimeSeries',
           'cov',
           'rms',
           'rmsd',
           'allstats',
           'lsqfity',
           'lsqfitx',
           'lsqfitgm',
           'lsqfitma',
           'lsqbisec',
           'lsqcubic',
           'lsqfityw',
           'lsqfityz',
           'gmregress',
           'r_earth']


class TimeSeries(object):
    """Time-series object to store data and time information.
    Contains some handy methods... Still a work in progress.
    """
    def __init__(self, data, time):
        """data : array_like
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


def cov(x, y):
    """Compute covariance for `x`, `y`

    input:  `x`, `y` -> data sets `x` and `y`
    `c` -> covariance of `x` and `y`
    """

    x, y = map(np.asanyarray, (x, y))

    x = x - x.mean()
    y = y - y.mean()

    # Compute covariance.
    c = np.dot(x, y) / (x.size - 1)

    return c


def rms(x):
    """Compute root mean square."""

    x = np.asanyarray(x)
    rms = np.sqrt(np.sum(x ** 2) / x.size)

    return rms


def rmsd(x, y, normalize=False):
    """Compute root mean square difference (or distance).

    The normalized root-mean-square deviation or error (NRMSD or NRMSE) is the
    RMSD divided by the range of observed values.  The value is often expressed
    as a percentage, where lower values indicate less residual variance.
    """

    x, y = map(np.asanyarray, (x, y))

    rmsd = np.sqrt(np.sum((x - y) ** 2) / x.size)

    if normalize:
        rmsd = rmsd / x.ptp()

    return rmsd


def allstats(Cr, Cf):
    """Compute statistics from 2 series.

    statm = allstats(Cr, Cf)

    Compute statistics from 2 series considering Cr as the reference.

    Inputs:
        Cr and Cf are of same length and uni-dimensional.

    Outputs:
        statm[0, :] => Mean
        statm[1, :] => Standard Deviation (scaled by N)
        statm[2, :] => Centered Root Mean Square Difference (scaled by N)
        statm[3, :] => Correlation

    Notes:
        - N is the number of points where BOTH Cr and Cf are defined
        - NaN are handled in the following way: because this function
            aims to compair 2 series, statistics are computed with indices
            where both Cr and Cf are defined.

        - statm[:, 0] are from Cr (i.e. with C=Cr hereafter)
          statm[:, 1] are from Cf versus Cr (i.e. with C=Cf hereafter)

        - The MEAN is computed using the mean function.

        - The STANDARD DEVIATION is computed as:

                                 /  sum[ {C-mean(C)} .^2]  \
                       STD = sqrt|  ---------------------  |
                                 \          N              /

       - The CENTERED ROOT MEAN SQUARE DIFFERENCE is computed as:
                             /  sum[  { [C-mean(C)] - [Cr-mean(Cr)] }.^2  ]  \
                  RMSD = sqrt|  -------------------------------------------  |
                             \                      N                        /

       - The CORRELATION is computed as:
                             sum( [C-mean(C)].*[Cr-mean(Cr)] )
                       COR = ---------------------------------
                                     N*STD(C)*STD(Cr)

       - statm[2, 0] = 0 and statm[3, 0] = 1 by definition !
    """
    Cr, Cf = map(np.asanyarray, (Cr, Cf))

    # Check NaNs.
    # iok = find(isnan(Cr)==0 & isnan(Cf)==0);
    # if length(iok) ~= length(Cr)
    #    warning('Found NaNs in inputs, removed them to compute statistics');
    #    Cr  = Cr(iok);
    #    Cf  = Cf(iok);

    N = len(Cr)

    # STD:
    st0 = np.sqrt(np.sum((Cr - Cr.mean()) ** 2) / N)
    st1 = np.sqrt(np.sum((Cf - Cf.mean()) ** 2) / N)
    st = np.c_[st0, st1]

    # MEAN:
    me0 = Cr.mean()
    me1 = Cf.mean()
    me = np.c_[me0, me1]

    # RMSD:
    rms0 = np.sqrt(np.sum(((Cr - Cr.mean()) - (Cr - Cr.mean())) ** 2) / N)
    rms1 = np.sqrt(np.sum(((Cf - Cf.mean()) - (Cr - Cr.mean())) ** 2) / N)
    np.c_[rms0, rms1]

    # CORRELATIONS:
    co0 = np.sum(((Cr - Cr.mean()) * (Cr - Cr.mean()))) / N / st[0] / st[0]
    co1 = np.sum(((Cf - Cf.mean()) * (Cr - Cr.mean()))) / N / st[1] / st[0]
    co = np.c_[co0, co1]

    # OUTPUT
    statm = np.r_[me, st, rms, co]

    return statm

# lsqfit_regression.py
#
# purpose:  Several snippets for least regression.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  14-Sep-2011
# modified: Wed 14 Sep 2011 02:16:53 PM EDT
#
# obs: Modified from Edward T Peltzer
#      http://www.mbari.org/staff/etp3/regress/rules.htm

"""
For Model II regressions, neither X nor Y is an INDEPENDENT variable but both
are assumed to be DEPENDENT on some other parameter which is often unknown.
Neither are "controlled", both are measured, and both include some error. We do
not seek an equation of how Y varies in response to a change in X, but rather
we look for how they both co-vary in time or space in response to some other
variable or process. There are several possible Model II regressions. Which one
is used depends upon the specifics of the case. See Ricker (1973) or Sokal and
Rohlf (1995, pp. 541-549) for a discussion of which may apply. For convenience,
I have also compiled some rules of thumb.
"""


def lsqfity(X, Y):
    """
    Calculate a "MODEL-1" least squares fit.

    The line is fit by MINIMIZING the residuals in Y only.

    The equation of the line is:     Y = my * X + by.

    Equations are from Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    pp: 104, 108-109, 199.

    Data are input and output as follows:

    my, by, ry, smy, sby = lsqfity(X,Y)
    X     =    x data (vector)
    Y     =    y data (vector)
    my    =    slope
    by    =    y-intercept
    ry    =    correlation coefficient
    smy   =    standard deviation of the slope
    sby   =    standard deviation of the y-intercept

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the sums.

    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2

    # Calculate my, by, ry, s2, smy and sby.
    my = num / den
    by = (Sx2 * Sy - Sx * Sxy) / den
    ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

    diff = Y - by - my * X

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)
    sby = np.sqrt(Sx2 * s2 / den)

    return my, by, ry, smy, sby


def lsqfitx(X, Y):
    """Calculate a "MODEL-1" least squares fit.

    The line is fit by MINIMIZING the residuals in X only.

    The equation of the line is:     Y = mx * X + bx.

    Equations are modified from those in Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    pp: 104, 108-109, 199.

    Data are input and output as follows:

    mx, bx, rx, smx, sbx = lsqfitx(X, Y)
    X      =    x data (vector)
    Y      =    y data (vector)
    mx     =    slope
    bx     =    y-intercept
    rx     =    correlation coefficient
    smx    =    standard deviation of the slope
    sbx    =    standard deviation of the y-intercept

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the sums.
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sy * Sx
    den = n * Sy2 - Sy ** 2

    # Calculate m, a, rx, s2, sm, and sb.
    mxi = num / den
    a = (Sy2 * Sx - Sy * Sxy) / den
    rx = num / (np.sqrt(den) * np.sqrt(n * Sx2 - Sx ** 2))

    diff = X - a - mxi * Y

    s2 = np.sum(diff * diff) / (n - 2)
    sm = np.sqrt(n * s2 / den)
    sa = np.sqrt(Sy2 * s2 / den)

    # Transpose coefficients
    mx = 1 / mxi
    bx = -a / mxi

    smx = mx * sm / mxi
    sbx = np.abs(sa / mxi)

    return mx, bx, rx, smx, sbx


def lsqfitgm(X, Y):
    """Calculate a "MODEL-2" least squares fit.

    The SLOPE of the line is determined by calculating the GEOMETRIC MEAN
    of the slopes from the regression of Y-on-X and X-on-Y.

    The equation of the line is:     y = mx + b.

    This line is called the GEOMETRIC MEAN or the REDUCED MAJOR AXIS.

    See Ricker (1973) Linear regressions in Fishery Research, J. Fish.
    Res. Board Can. 30: 409-434, for the derivation of the geometric
    mean regression.

    Since no statistical treatment exists for the estimation of the
    asymmetrical uncertainty limits for the geometric mean slope,
    I have used the symmetrical limits for a model I regression
    following Ricker's (1973) treatment.  For ease of computation,
    equations from Bevington and Robinson (1992) "Data Reduction and
    Error Analysis for the Physical Sciences, 2nd Ed."  pp: 104, and
    108-109, were used to calculate the symmetrical limits: sm and sb.

    Data are input and output as follows:

    m, b, r, sm, sb = lsqfitgm(X,Y)
    X    =    x data (vector)
    Y    =    y data (vector)
    m    =    slope
    b    =    y-intercept
    r    =    correlation coefficient
    sm   =    standard deviation of the slope
    sb   =    standard deviation of the y-intercept

    Note that the equation passes through the centroid:  (x-mean, y-mean)

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine slope of Y-on-X regression.
    my = lsqfity(X, Y)[0]

    # Determine slope of X-on-Y regression.
    mx = lsqfitx(X, Y)[0]

    # Calculate geometric mean slope.
    m = np.sqrt(my * mx)

    if (my < 0) and (mx < 0):
        m = -m

    # Determine the size of the vector.
    n = len(X)

    # Calculate sums and means.
    Sx = np.sum(X)
    Sy = np.sum(Y)
    xbar = Sx / n
    ybar = Sy / n

    # Calculate geometric mean intercept.
    b = ybar - m * xbar

    # Calculate more sums.
    # Sxy = np.sum(X * Y)  # FIXME: Assigned but never used.
    Sx2 = np.sum(X ** 2)
    # Sy2 = np.sum(Y ** 2)  # FIXME: Assigned but never used.

    # Calculate re-used expressions.
    # num = n * Sxy - Sx * Sy  # FIXME: Assigned but never used.
    den = n * Sx2 - Sx ** 2

    # Calculate r, sm, sb and s2.

    r = np.sqrt(my / mx)

    if (my < 0) and (mx < 0):
        r = -r

    diff = Y - b - m * X

    s2 = np.sum(diff * diff) / (n - 2)
    sm = np.sqrt(n * s2 / den)
    sb = np.sqrt(Sx2 * s2 / den)

    return m, b, r, sm, sb


def lsqfitma(X, Y):
    """
    Calculate a "MODEL-2" least squares fit.

    The line is fit by MINIMIZING the NORMAL deviates.

    The equation of the line is:     y = mx + b.

    This line is called the MAJOR AXIS.  All points are given EQUAL
    weight.  The units and range for X and Y must be the same.
    Equations are from York (1966) Canad. J. Phys. 44: 1079-1086;
    re-written from Kermack & Haldane (1950) Biometrika 37: 30-41;
    after a derivation by Pearson (1901) Phil. Mag. V2(6): 559-572.

    Data are input and output as follows:

    m, b, r, sm, sb = lsqfitma(X, Y)
    X    =    x data (vector)
    Y    =    y data (vector)
    m    =    slope
    b    =    y-intercept
    r    =    correlation coefficient
    sm   =    standard deviation of the slope
    sb   =    standard deviation of the y-intercept

    Note that the equation passes through the centroid:  (x-mean, y-mean)

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate sums and other re-used expressions.
    Sx = np.sum(X)
    Sy = np.sum(Y)
    xbar = Sx / n
    ybar = Sy / n
    U = X - xbar
    V = Y - ybar

    Suv = np.sum(U * V)
    Su2 = np.sum(U ** 2)
    Sv2 = np.sum(V ** 2)

    sigx = np.sqrt(Su2 / (n - 1))
    sigy = np.sqrt(Sv2 / (n - 1))

    # Calculate m, b, r, sm, and sb.
    m = (Sv2 - Su2 + np.sqrt(((Sv2 - Su2) ** 2) + (4 * Suv ** 2))) / (2 * Suv)
    b = ybar - m * xbar
    r = Suv / np.sqrt(Su2 * Sv2)

    sm = (m / r) * np.sqrt((1 - r ** 2) / n)
    sb1 = (sigy - sigx * m) ** 2
    sb2 = (2 * sigx * sigy) + ((xbar ** 2 * m * (1 + r)) / r ** 2)
    sb = np.sqrt((sb1 + ((1 - r) * m * sb2)) / n)

    return m, b, r, sm, sb


def lsqbisec(X, Y):
    """
    Calculate a "MODEL-2" least squares fit.

    The SLOPE of the line is determined by calculating the slope of the line
    that bisects the minor angle between the regression of Y-on-X and X-on-Y.

    The equation of the line is:     y = mx + b.

    This line is called the LEAST SQUARES BISECTOR.

    See: Sprent and Dolby (1980). The Geometric Mean Functional Relationship.
    Biometrics 36: 547-550, for the rationale behind this regression.

    Sprent and Dolby (1980) did not present a statistical treatment for the
    estimation of the uncertainty limits for the least squares bisector
    slope, or intercept.

    I have used the symmetrical limits for a model I regression following
    Ricker's (1973) treatment.  For ease of computation, equations from
    Bevington and Robinson (1992) "Data Reduction and Error Analysis for
    the Physical Sciences, 2nd Ed."  pp: 104, and 108-109, were used to
    calculate the symmetrical limits: sm and sb.

    Data are input and output as follows:

    m, b, r, sm, sb = lsqbisec(X,Y)
    X    =    x data (vector)
    Y    =    y data (vector)
    m    =    slope
    b    =    y-intercept
    r    =    correlation coefficient
    sm   =    standard deviation of the slope
    sb   =    standard deviation of the y-intercept

    Note that the equation passes through the centroid:  (x-mean, y-mean)

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine slope of Y-on-X regression.
    my = lsqfity(X, Y)[0]

    # Determine slope of X-on-Y regression.
    mx = lsqfitx(X, Y)[0]

    # Calculate the least squares bisector slope.
    theta = (np.arctan(my) + np.arctan(mx)) / 2
    m = np.tan(theta)

    # Determine the size of the vector
    n = len(X)

    # Calculate sums and means
    Sx = np.sum(X)
    Sy = np.sum(Y)
    xbar = Sx / n
    ybar = Sy / n

    # Calculate the least squares bisector intercept.
    b = ybar - m * xbar

    # Calculate more sums.
    # Sxy = np.sum(X * Y)  # FIXME: Assigned but never used.
    Sx2 = np.sum(X ** 2)
    # Sy2 = np.sum(Y ** 2)  # FIXME: Assigned but never used.

    # Calculate re-used expressions.
    # num = n * Sxy - Sx * Sy  # FIXME: Assigned but never used.
    den = n * Sx2 - Sx ** 2

    # Calculate r, sm, sb and s2.
    r = np.sqrt(my / mx)

    if (my < 0) and (mx < 0):
        r = -r

    diff = Y - b - m * X

    s2 = np.sum(diff * diff) / (n - 2)
    sm = np.sqrt(n * s2 / den)
    sb = np.sqrt(Sx2 * s2 / den)

    return m, b, r, sm, sb


def lsqcubic(X, Y, sX, sY, tl=1e-6):
    """
    Calculate a MODEL-2 least squares fit from weighted data.

    The line is fit by MINIMIZING the weighted residuals in both x & y.
    The equation of the line is:     y = mx + b,
    where m is determined by finding the roots to the cubic equation:

    m^3 + P * m^2 + Q * m + R = 0.

    Eqs for P, Q and R are from York (1966) Canad. J. Phys. 44: 1079-1086.

    Data are input and output as follows:
    m, b, r, sm, sb, xc, yc, ct = lsqcubic(X, Y, sX, sY, tl)
    X    =    x data (vector)
    Y    =    y data (vector)
    sX   =    uncertainty of x data (vector)
    sY   =    uncertainty of y data (vector)
    tl   =    test limit for difference between slope iterations

    m    =    slope
    b    =    y-intercept
    r    =    weighted correlation coefficient
    sm   =    standard deviation of the slope
    sb   =    standard deviation of the y-intercept
    xc   =    WEIGHTED mean of x values
    yc   =    WEIGHTED mean of y values
    ct   =    count: number of iterations

    Notes:  1.  (xc,yc) is the WEIGHTED centroid.
            2.  Iteration of slope continues until successive differences
                are less than the user-set limit "tl".  Smaller values of
                tl require more iterations to find the slope.
            3.  Suggested values of tl = 1e-4 to 1e-6.

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Find the number of data points and make one time calculations:
    n = len(X)
    wX = 1 / (sX ** 2)
    wY = 1 / (sY ** 2)

    # Set-up a few initial conditions:
    ct, ML = 0, 1

    # ESTIMATE the slope by calculating the major axis according
    # to Pearson's (1901) derivation, see: lsqfitma.

    MC = lsqfitma(X, Y)[0]

    test = np.abs((ML - MC) / ML)

    # Calculate the least-squares-cubic. Make iterative calculations until the
    # relative difference is less than the test conditions

    while test > tl:
        # Calculate sums and other re-used expressions:
        MC2 = MC ** 2
        W = (wX * wY) / ((MC2 * wY) + wX)
        W2 = W ** 2

        SW = np.sum(W)
        xc = (np.sum(W * X)) / SW
        yc = (np.sum(W * Y)) / SW

        U = X - xc
        V = Y - yc

        U2 = U ** 2
        V2 = V ** 2

        SW2U2wX = np.sum(W2 * U2 / wX)

        # Calculate coefficients for least-squares cubic:
        P = -2 * np.sum(W2 * U * V / wX) / SW2U2wX
        Q = (np.sum(W2 * V2 / wX) - np.sum(W * U2)) / SW2U2wX
        R = np.sum(W * U * V) / SW2U2wX
        # Find the roots to the least-squares cubic:
        LSC = [1, P, Q, R]
        MR = np.roots(LSC)

        # Find the root closest to the slope:
        DIF = np.abs(MR - MC)
        Index = DIF.argmin()

        ML = MC
        MC = MR[Index]
        test = np.abs((ML - MC) / ML)
        ct = ct + 1

    # Calculate m, b, r, sm, and sb.
    m = MC
    b = yc - m * xc
    r = np.sum(U * V) / np.sqrt(np.sum(U2) * np.sum(V2))
    sm2 = (1 / (n - 2)) * (np.sum(W * (((m * U) - V) ** 2)) / np.sum(W * U2))
    sm = np.sqrt(sm2)
    sb = np.sqrt(sm2 * (np.sum(W * (X ** 2)) / SW))

    return m, b, r, sm, sb, xc, yc, ct


def lsqfityw(X, Y, sY):
    """
    Calculate a "MODEL-1" least squares fit to WEIGHTED x,y-data pairs:

    The line is fit by MINIMIZING the WEIGHTED residuals in Y only.

    The equation of the line is:     Y = mw * X + bw.

    Equations are from Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    for mw, bw, smw and sbw, see p. 98, example calculation in Table 6.2;
    for rw, see p. 199, and modify eqn 11.17 for a weighted regression by
    substituting Sw for n, Swx for Sx, Swy for Sy, Swxy for Sxy, etc.

    Data are input and output as follows:
    mw, bw, rw, smw, sbw, xw, yw = lsqfityw(X, Y, sY)
    X     =    x data (vector)
    Y     =    y data (vector)
    sY    =    estimated uncertainty in y data (vector)
    sy may be measured or calculated:
        sY = sqrt(Y), 2% of y, etc.
    data points are then weighted by:
        w = 1 / sY-squared.
    mw    =    slope
    bw    =    y-intercept
    rw    =    weighted correlation coefficient
    smw   =    standard deviation of the slope
    sbw   =    standard deviation of the y-intercept

    NOTE: that the line passes through the weighted centroid: (xw,yw).

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    # n = len(X)  # FIXME: Assigned but never used.

    # Calculate the weighting factors.
    W = 1 / (sY ** 2)

    # Calculate the sums.
    Sw = np.sum(W)
    Swx = np.sum(W * X)
    Swy = np.sum(W * Y)
    Swx2 = np.sum(W * X ** 2)
    Swxy = np.sum(W * X * Y)
    Swy2 = np.sum(W * Y ** 2)

    # Determine the weighted centroid.
    xw = Swx / Sw
    yw = Swy / Sw

    # Calculate re-used expressions.
    num = Sw * Swxy - Swx * Swy
    del1 = Sw * Swx2 - Swx ** 2
    del2 = Sw * Swy2 - Swy ** 2

    # Calculate mw, bw, rw, smw, and sbw.
    mw = num / del1
    bw = (Swx2 * Swy - Swx * Swxy) / del1
    rw = num / (np.sqrt(del1 * del2))

    smw = np.sqrt(Sw / del1)
    sbw = np.sqrt(Swx2 / del1)

    return mw, bw, rw, smw, sbw, xw, yw


def lsqfityz(X, Y, sY):
    """
    Calculate a "MODEL-1" least squares fit to WEIGHTED x,y-data pairs:
    The line is fit by MINIMIZING the WEIGHTED residuals in Y only.

    The equation of the line is:     Y = mz * X + bz.

    Equations are from Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    for mz and bz, see p. 98, example calculation in Table 6.2;
    for rz, see p. 199, and modify eqn 11.17 for a weighted regression by
    substituting Sw for n, Swx for Sx, Swy for Sy, Swxy for Sxy, etc.
    smz, sbz are adapted from: York (1966) Canad. J. Phys. 44: 1079-1086.

    Data are input and output as follows:
    mz, bz, rz, smz, sbz, xz, yz = lsqfityz(X, Y, sY)
    X     =    x data (vector)
    Y     =    y data (vector)
    sY    =    estimated uncertainty in y data (vector)

    sy may be measured or calculated:
        sY = sqrt(Y), 2% of y, etc.
    data points are then weighted by:
        w = 1 / sY-squared.

    mz    =    slope
    bz    =    y-intercept
    rz    =    weighted correlation coefficient
    smz   =    standard deviation of the slope
    sbz   =    standard deviation of the y-intercept

    NOTE: that the line passes through the weighted centroid: (xz,yz).

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the weighting factors.
    W = 1 / (sY ** 2)

    # Calculate the sums.
    Sw = np.sum(W)
    Swx = np.sum(W * X)
    Swy = np.sum(W * Y)
    Swx2 = np.sum(W * X ** 2)
    Swxy = np.sum(W * X * Y)
    Swy2 = np.sum(W * Y ** 2)

    # Determine the weighted centroid.
    xz = Swx / Sw
    yz = Swy / Sw

    # Calculate re-used expressions.
    num = Sw * Swxy - Swx * Swy
    del1 = Sw * Swx2 - Swx ** 2
    del2 = Sw * Swy2 - Swy ** 2

    U = X - xz
    V = Y - yz
    U2 = U ** 2
    # V2 = V ** 2  # FIXME: Assigned but never used.

    # Calculate mw, bw, rw, smw, and sbw.
    mz = num / del1
    bz = (Swx2 * Swy - Swx * Swxy) / del1

    rz = num / (np.sqrt(del1 * del2))

    sm2 = (1 / (n - 2)) * (np.sum(W * (((mz * U) - V) ** 2)) / sum(W * U2))
    smz = np.sqrt(sm2)
    sbz = np.sqrt(sm2 * (np.sum(W * (X ** 2)) / Sw))

    return mz, bz, rz, smz, sbz, xz, yz


def gmregress(X, Y, alpha=0.05):
    """
    GMREGRESS Geometric Mean Regression (Reduced Major Axis Regression).
    Model II regression should be used when the two variables in the
    regression equation are random and subject to error, i.e. not
    controlled by the researcher. Model I regression using ordinary least
    squares underestimates the slope of the linear relationship between the
    variables when they both contain error. According to Sokal and Rohlf
    (1995), the subject of Model II regression is one on which research and
    controversy are continuing and definitive recommendations are difficult
    to make.

    GMREGRESS is a Model II procedure. It standardize variables before the
    slope is computed. Each of the two variables is transformed to have a
    mean of zero and a standard deviation of one. The resulting slope is
    the geometric mean of the linear regression coefficient of Y on X.
    Ricker (1973) coined this term and gives an extensive review of Model
    II regression. It is also known as Standard Major Axis.

    b, bintr, bintjm = gmregress(X,Y,alpha)
    returns the vector B of regression coefficients in the linear Model II and
    a matrix BINT of the given confidence intervals for B by the Ricker (1973)
    and Jolicoeur and Mosimann (1968)-McArdle (1988) procedure.

    gmregress treats NaNs in X or Y as missing values, and removes them.

    Syntax: function b, bintr, bintjm = gmregress(X, Y, alpha)

    Example. From the Box 14.12 (California fish cabezon [Scorpaenichthys
    marmoratus]) of Sokal and Rohlf (1995). The data are:

    x = [14, 17, 24, 25, 27, 33, 34, 37, 40, 41, 42]
    y = [61, 37, 65, 69, 54, 93, 87, 89, 100, 90, 97]

    Calling on Matlab the function:
    b, bintr, bintjm = gmregress(x,y)

    Answer is:
    b = 12.1938    2.1194

    bintr = -10.6445   35.0320
            1.3672    2.8715

    bintjm = -14.5769   31.0996
            1.4967    3.0010

    http://www.mathworks.com/matlabcentral/fileexchange/27918-gmregress

    References:
        Jolicoeur, P. and Mosimann, J. E. (1968), Intervalles de confiance pour
            la pente de l'axe majeur d'une distribution normale
            bidimensionnelle. Biométrie-Praximétrie, 9:121-140.
        McArdle, B. (1988), The structural relationship: regression in biology.
            Can. Jour. Zool. 66:2329-2339.
        Ricker, W. E. (1973), Linear regression in fishery research. J. Fish.
            Res. Board Can., 30:409-434.
        Sokal, R. R. and Rohlf, F. J. (1995), Biometry. The principles and
            practice of the statistics in biologicalreserach. 3rd. ed.
            New-York:W.H.,Freeman. [Sections 14.13 and 15.7]
    """

    X, Y = map(np.asanyarray, (X, Y))

    n = len(Y)
    S = np.cov(X, Y)
    SCX = S[0, 0] * (n - 1)
    SCY = S[1, 1] * (n - 1)
    SCP = S[0, 1] * (n - 1)
    v = np.sqrt(SCY / SCX)  # Slope.
    u = Y.mean() - X.mean() * v  # Intercept.
    b = np.r_[u, v]

    SCv = SCY - (SCP ** 2) / SCX
    N = SCv / (n - 2)
    sv = np.sqrt(N / SCX)
    t = stats.t.isf(alpha / 2, n - 2)

    vi = v - t * sv  # Confidence lower limit of slope.
    vs = v + t * sv  # Confidence upper limit of slope.
    ui = Y.mean() - X.mean() * vs  # Confidence lower limit of intercept.
    us = Y.mean() - X.mean() * vi  # Confidence upper limit of intercept.
    bintr = np.r_[np.c_[ui, us], np.c_[vi, vs]]

    R = np.corrcoef(X, Y)
    r = R[0, 1]

    F = stats.f.isf(alpha, 1, n - 2)
    B = F * (1 - r ** 2) / (n - 2)

    a = np.sqrt(B + 1)
    c = np.sqrt(B)
    qi = v * (a - c)  # Confidence lower limit of slope.
    qs = v * (a + c)  # Confidence upper limit of slope.
    pi = Y.mean() - X.mean() * qs  # Confidence lower limit of intercept.
    ps = Y.mean() - X.mean() * qi  # Confidence upper limit of intercept.
    bintjm = np.r_[np.c_[pi, ps], np.c_[qi, qs]]

    return b, bintr, bintjm


def r_earth(lon=None, lat=None):
    """Radius of the earth as a function of latitude and longitude using the
    WGS-84 earth ellipsoid.

    Parameters
    ----------
    lon : float
          longitude [Degrees East]
    lat : float
          latitude [Degrees North]
        Input 1D data

    Returns
    -------
    r : float
    radius of earth [m] at corresponding point on ellipsoid

    Examples
    --------
    >>> import numpy as np
    >>> from oceans import r_earth
    >>> a, b = 6378137.0, 6356752.314245  # In meters.
    >>> r_earth()
    6371000.7900090935
    >>> north_r = r_earth(lat=90,lon=0)
    >>> assert np.allclose(b, north_r)
    >>> south_r = r_earth(lat=-90, lon=0)
    >>> assert np.allclose(b, south_r)
    >>> east_r = r_earth(lat=0,lon=90)
    >>> assert np.allclose(a, east_r)
    >>> west_r = r_earth(lat=0,lon=-90)
    >>> assert np.allclose(a, west_r)
    >>> dateline_r = r_earth(lat=0,lon=180)
    >>> assert np.allclose(a, dateline_r)
    >>> # Original definition of meter occurs at this latitude (48.276841).
    >>> original_m = 2.0 * np.pi * r_earth(lat=48.276841, lon=0) / 4.0
    >>> assert np.allclose(1e7, original_m)

    Notes
    -----
    Based on http://staff.washington.edu/bdjwww/earth_radius.py
    """

    # WGS-84 semi-major and semi-minor axes,
    a, b = 6378137.0, 6356752.314245  # In meters.

    if (lon is None) or (lat is None):
        # Best known single estimate (this is R3; could use R1 or R2 as well)
        # Geometric mean radius, sphere of equivalent volume.
        return (a ** 2 * b) ** (1. / 3.)

    """Convert to physicist's spherical coordinates (e.g. Arfken, 1985)
    phi = azimuthal angle, 0 <= phi < 2 pi
    theta = polar angle, 0 <= theta <= pi
    Conversion notes:
    phi is longitude converted to radians.
    theta is co-latitude: theta = pi/2 - lat (after lat converted to radians).
    """

    # Spherical coordinates
    phi = lon * np.pi / 180.
    theta = np.pi / 2. - lat * np.pi / 180.

    """The equation of an ellipsoid is $x^2 /a^2 + y^2/a^2 + z^2/b^2 = 1$,
    with two "a" axes because the earth is fat around the equator.  Now use
    x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta),
    and we easily obtain the next equation."""

    inv_r_squared = ((np.sin(theta) * np.cos(phi) / a) ** 2 +
                     (np.sin(theta) * np.sin(phi) / a) ** 2 +
                     (np.cos(theta) / b) ** 2)

    return 1.0 / np.sqrt(inv_r_squared)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
