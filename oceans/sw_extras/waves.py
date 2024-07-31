import gsw
import numpy as np


class Waves:
    r"""
    Solves the wave dispersion relationship via Newton-Raphson.

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
    >>> from oceans.sw_extras.waves import Waves
    >>> wav = Waves(h=10, T=5, L=None)
    >>> print(f"ho/Lo = {wav.hoLo:.3f}")
    ho/Lo = 0.256
    >>> print(f"ho/L  = {wav.hoL:.3f}")
    ho/L  = 0.273
    >>> print(f"Lo    = {wav.Lo:.3f}")
    Lo    = 39.033
    >>> print(f"L     = {wav.L:.3f}")
    L     = 36.593
    >>> print(f"k     = {wav.k:.3f}")
    k     = 0.172
    >>> print(f"omega = {wav.omega:.3f}")
    omega = 1.257
    >>> print(f"T     = {wav.T:.3f}")
    T     = 5.000
    >>> print(f"C     = {wav.C:.3f}")
    C     = 7.319
    >>> print(f"Cg    = {wav.Cg:.3f}")
    Cg    = 4.471
    >>> print(f"G     = {wav.G:.3f}")
    G     = 0.222
    >>> wav = Waves(h=10, T=None, L=100)
    >>> print(f"ho/Lo = {wav.hoLo:.3f}")
    ho/Lo = 0.056
    >>> print(f"ho/L  = {wav.hoL:.3f}")
    ho/L  = 0.100
    >>> print(f"Lo    = {wav.Lo:.3f}")
    Lo    = 179.568
    >>> print(f"L     = {wav.L:.3f}")
    L     = 100.000
    >>> print(f"k     = {wav.k:.3f}")
    k     = 0.063
    >>> print(f"omega = {wav.omega:.3f}")
    omega = 0.586
    >>> print(f"T     = {wav.T:.3f}")
    T     = 10.724
    >>> print(f"C     = {wav.C:.3f}")
    C     = 9.325
    >>> print(f"Cg    = {wav.Cg:.3f}")
    Cg    = 8.291
    >>> print(f"G     = {wav.G:.3f}")
    G     = 0.778
    >>> print(f"Ks  = {wav.Ks:.3f}")
    Ks  = 1.005

    """

    def __init__(self, h, T=None, L=None, thetao=None, Ho=None, lat=None):
        self.T = np.asarray(T, dtype=np.float64)
        self.L = np.asarray(L, dtype=np.float64)
        self.Ho = np.asarray(Ho, dtype=np.float64)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.thetao = np.asarray(thetao, dtype=np.float64)

        if isinstance(h, str):
            if L is not None:
                if h == "deep":
                    self.h = self.L / 2.0
                elif h == "shallow":
                    self.h = self.L * 0.05
        else:
            self.h = np.asarray(h, dtype=np.float64)

        if lat is None:
            g = 9.81  # Default gravity.
        else:
            g = gsw.grav(lat, p=0)

        if L is None:
            self.omega = 2 * np.pi / self.T
            self.Lo = (g * self.T**2) / 2 / np.pi
            # Returns wavenumber of the gravity wave dispersion relation using
            # newtons method. The initial guess is shallow water wavenumber.
            self.k = self.omega / np.sqrt(g)
            # TODO: May change to,
            # self.k = self.w ** 2 / (g * np.sqrt(self.w ** 2 * self.h / g))
            f = g * self.k * np.tanh(self.k * self.h) - self.omega**2

            while np.abs(f.max()) > 1e-10:
                dfdk = g * self.k * self.h * (
                    1 / (np.cosh(self.k * self.h))
                ) ** 2 + g * np.tanh(self.k * self.h)
                self.k = self.k - f / dfdk
                # FIXME:
                f = g * self.k * np.tanh(self.k * self.h) - self.omega**2

            self.L = 2 * np.pi / self.k
            if isinstance(h, str):
                if h == "deep":
                    self.h = self.L / 2.0
                elif h == "shallow":
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
            self.theta = np.nan
            self.Kr = np.nan
        if thetao is not None:
            self.theta = np.rad2deg(
                np.asin(self.C / self.Co * np.sin(np.deg2rad(self.thetao))),
            )
            self.Kr = np.sqrt(
                np.cos(np.deg2rad(self.thetao)) / np.cos(np.deg2rad(self.theta)),
            )

        if Ho is None:
            self.H = np.nan
        if Ho is not None:
            self.H = self.Ho * self.Ks * self.Kr
