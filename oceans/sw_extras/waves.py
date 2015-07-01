from __future__ import absolute_import, division


import gsw
import numpy as np


class Waves(object):
    """
    Solves the wave dispersion relationship via Newton-raphson.

    .. math::
        \omega^2 = gk\tanh kh

    Parameters
    ----------
    h : array_like, str
        Water depth [m] or 'deep', 'shallow' as keywords
    T : array_like
        Wave period [s]
    L : array_like
        Wave length [m]
    thetao : array_like
        TODO
    Ho : array_like
        TODO

    Returns
    -------
    omega : array_like
            Wave frequency
    TODO: hoLo, hoL, Lo, L, k, T, Co, C, Cg, G, Ks, Kr, theta, H

    Notes
    -----
    Compare values with:
    http://www.coastal.udel.edu/faculty/rad/wavetheory.html

    Examples
    --------
    >>> from oceans.sw_extras import Waves
    >>> wav = Waves(h=10, T=5, L=None)
    >>> print("ho/Lo = %s" % wav.hoLo)
    ho/Lo = 0.256195119559
    >>> print("ho/L  = %s" % wav.hoL)
    ho/L  = 0.273273564378
    >>> print("Lo    = %s" % wav.Lo)
    Lo    = 39.0327497933
    >>> print("L     = %s" % wav.L)
    L     = 36.5933676122
    >>> print("k     = %s" % wav.k)
    k     = 0.171702844454
    >>> print("omega = %s" % wav.omega)
    omega = 1.25663706144
    >>> print("T     = %s" % wav.T)
    T     = 5.0
    >>> print("C     = %s" % wav.C)
    C     = 7.31867352244
    >>> print("Cg    = %s" % wav.Cg)
    Cg    = 4.47085819307
    >>> print("G     = %s" % wav.G)
    G     = 0.22176735425
    >>> wav = Waves(h=10, T=None, L=100)
    >>> print("ho/Lo = %s" % wav.hoLo)
    ho/Lo = 0.05568933069
    >>> print("ho/L  = %s" % wav.hoL)
    ho/L  = 0.1
    >>> print("Lo    = %s" % wav.Lo)
    Lo    = 179.56760974
    >>> print("L     = %s" % wav.L)
    L     = 100.0
    >>> print("k     = %s" % wav.k)
    k     = 0.0628318530718
    >>> print("omega = %s" % wav.omega)
    omega = 0.585882379881
    >>> print("T     = %s" % wav.T)
    T     = 10.7243117782
    >>> print("C     = %s" % wav.C)
    C     = 9.32460768286
    >>> print("Cg    = %s" % wav.Cg)
    Cg    = 8.29120888868
    >>> print("G     = %s" % wav.G)
    G     = 0.778350182802
    >>> # print("  = %s" % wav.Ks)
    >>> # print("  = %s" % wav.Kr)
    >>> # print("  = %s" % wav.theta)
    >>> # print("  = %s" % wav.H)
    >>> # print("  = %s" % wav.Ks)
    >>> # print("  = %s" % wav.Kr)
    >>> # print("  = %s" % wav.theta)
    >>> # print("  = %s" % wav.H)

    """
    def __init__(self, h, T=None, L=None, thetao=None, Ho=None, lat=None):
        self.T = np.asarray(T, dtype=np.float)
        self.L = np.asarray(L, dtype=np.float)
        self.Ho = np.asarray(Ho, dtype=np.float)
        self.lat = np.asarray(lat, dtype=np.float)
        self.thetao = np.asarray(thetao, dtype=np.float)

        if isinstance(h, str):
            if L is not None:
                if h == 'deep':
                    self.h = self.L / 2.
                elif h == 'shallow':
                    self.h = self.L * 0.05
        else:
            self.h = np.asarray(h, dtype=np.float)

        if lat is None:
            g = 9.81  # Default gravity.
        else:
            g = gsw.grav(lat, p=0)

        if L is None:
            self.omega = 2 * np.pi / self.T
            self.Lo = (g * self.T ** 2) / 2 / np.pi
            # Returns wavenumber of the gravity wave dispersion relation using
            # newtons method. The initial guess is shallow water wavenumber.
            self.k = self.omega / np.sqrt(g)
            # TODO: May change to,
            # self.k = self.w ** 2 / (g * np.sqrt(self.w ** 2 * self.h / g))
            f = g * self.k * np.tanh(self.k * self.h) - self.omega ** 2

            while np.abs(f.max()) > 1e-10:
                dfdk = (g * self.k * self.h *
                        (1 / (np.cosh(self.k * self.h))) ** 2 +
                        g * np.tanh(self.k * self.h))
                self.k = self.k - f / dfdk
                # FIXME:
                f = g * self.k * np.tanh(self.k * self.h) - self.omega ** 2

            self.L = 2 * np.pi / self.k
            if isinstance(h, str):
                if h == 'deep':
                    self.h = self.L / 2.
                elif h == 'shallow':
                    self.h = self.L * 0.05
        else:
            self.Lo = self.L / np.tanh(2 * np.pi * self.h / self.L)
            self.k = 2 * np.pi / self.L
            self.T = np.sqrt(2 * np.pi * self.Lo / g)
            self.omega = 2 * np.pi / self.T

        self.hoL = self.h / self.L
        self.hoLo = self.h / self.Lo
        self.C = self.omega / self.k  # or L / T
        self.Co = self.Lo / self.T
        self.G = 2 * self.k * self.h / np.sinh(2 * self.k * self.h)
        self.n = (1 + self.G) / 2
        self.Cg = self.n * self.C
        self.Ks = np.sqrt(1 / (1 + self.G) / np.tanh(self.k * self.h))

        if thetao is None:
            self.theta = np.NaN
            self.Kr = np.NaN
        if thetao is not None:
            self.theta = np.rad2deg(np.asin(self.C / self.Co *
                                            np.sin(np.deg2rad(self.thetao))))
            self.Kr = np.sqrt(np.cos(np.deg2rad(self.thetao)) /
                              np.cos(np.deg2rad(self.theta)))

        if Ho is None:
            self.H = np.NaN
        if Ho is not None:
            self.H = self.Ho * self.Ks * self.Kr

if __name__ == '__main__':
    import doctest
    doctest.testmod()
