# -*- coding: utf-8 -*-

"""
Extra seawater functions
========================
"""

from __future__ import division

import numpy as np
from pandas import rolling_mean
import seawater as sw
from seawater.eos80 import T68conv
from seawater.constants import OMEGA, Kelvin, earth_radius


__all__ = ['o2sat',
           'sigma_t',
           'sigmatheta',
           'N',
           'cph',
           'shear',
           'richnumb',
           'cor_beta',
           'inertial_period',
           'strat_period',
           'visc',
           'tcond',
           'spice',
           'psu2ppt',
           'visc',
           'soundspeed']


def o2sat(s, pt):
    r"""Calculate oxygen concentration at saturation.  Molar volume of oxygen
    at STP obtained from NIST website on the thermophysical properties of fluid
    systems (http://webbook.nist.gov/chemistry/fluid/).

    Parameters
    ----------
    s : array_like
        Salinity [pss-78]
    pt : array_like
         Potential Temperature [degC ITS-90]

    Returns
    -------
    osat : array_like
          Oxygen concentration at saturation [umol/kg]

    Examples
    --------
    >>> import os
    >>> from pandas import read_csv
    >>> import oceans.seawater.sw_extras as swe
    >>> path = os.path.split(os.path.realpath(swe.__file__))[0]
    # Table 9 pg. 732. Values in ml / kg
    >>> pt = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22,
    ...                24, 26, 28, 30, 32, 34, 36, 38, 40]) / 1.00024
    >>> s = np.array([0, 10, 20, 30, 34, 35, 36, 38, 40])
    >>> s, pt = np.meshgrid(s, pt)
    >>> osat = swe.o2sat(s, pt) * 22.392 / 1000  # um/kg to ml/kg.
    >>> weiss_1979 = read_csv('%s/test/o2.csv' % path, index_col=0).values
    >>> np.testing.assert_almost_equal(osat.ravel()[2:],
    ...                                weiss_1979.ravel()[2:], decimal=3)


    References
    -----
    .. [1] The solubility of nitrogen, oxygen and argon in water and seawater -
    Weiss (1970) Deep Sea Research V17(4): 721-735.
    """

    t = T68conv(pt) + Kelvin
    # Eqn (4) of Weiss 1970 (the constants are used for units of ml O2/kg).
    a = (-177.7888, 255.5907, 146.4813, -22.2040)
    b = (-0.037362, 0.016504, -0.0020564)
    lnC = (a[0] + a[1] * (100. / t) + a[2] * np.log(t / 100.) + a[3] *
           (t / 100.) +
           s * (b[0] + b[1] * (t / 100.) + b[2] * (t / 100.) ** 2))
    osat = np.exp(lnC) * 1000. / 22.392  # Convert from ml/kg to um/kg.

    """The Apparent Oxygen Utilization (AOU) value was obtained by subtracting
    the measured value from the saturation value computed at the potential
    temperature of water and 1 atm total pressure using the following
    expression based on the data of Murray and Riley (1969):

    ln(O2 in µmol/kg) = - 173.9894 + 255.5907(100/TK) + 146.4813 ln(TK/100) -
    22.2040(TK/100) + Sal [-0.037362 + 0.016504(TK/100) - 0.0020564(TK/100)2],
    where TK is temperature in °K and Sal in the Practical Salinity (SP) scale.
    """
    return osat


def sigma_t(s, t, p):
    r""":math:`\\sigma_{t}` is the remainder of subtracting 1000 kg m :sup:`-3`
    from the density of a sea water sample at atmospheric pressure.

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:`^\\circ` C (ITS-90)]
    p : array_like
        pressure [db]

    Returns
    -------
    sgmt : array_like
           density  [kg m :sup:`3`]

    Notes
    -----
    Density of Sea Water using UNESCO 1983 (EOS 80) polynomial.

    Examples
    --------
    Data from Unesco Tech. Paper in Marine Sci. No. 44, p22

    >>> from oceans import sw_extras as swe
    >>> s = [0, 0, 0, 0, 35, 35, 35, 35]
    >>> t = T90conv([0, 0, 30, 30, 0, 0, 30, 30])
    >>> p = [0, 10000, 0, 10000, 0, 10000, 0, 10000]
    >>> swe.sigma_t(s, t, p)
    array([ -0.157406  ,  45.33710972,  -4.34886626,  36.03148891,
            28.10633141,  70.95838408,  21.72863949,  60.55058771])

    References
    ----------
    .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
    computation of fundamental properties of seawater. UNESCO Tech. Pap. in
    Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
    http://www.scor-int.org/Publications.htm

    .. [2] Millero, F.J., Chen, C.T., Bradshaw, A., and Schleicher, K. A new
    high pressure equation of state for seawater. Deap-Sea Research., 1980,
    Vol27A, pp255-264. doi:10.1016/0198-0149(80)90016-3

    Modifications: Filipe Fernandes, 2010
                   10-01-26. Filipe Fernandes, first version.
    """
    s, t, p = list(map(np.asanyarray, (s, t, p)))
    return sw.dens(s, t, p) - 1000.0


def sigmatheta(s, t, p, pr=0):
    r""":math:`\\sigma_{\\theta}` is a measure of the density of ocean water
    where the quantity :math:`\\sigma_{t}` is calculated using the potential
    temperature (:math:`\\theta`) rather than the in situ temperature and
    potential density of water mass relative to the specified reference
    pressure.

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:`^\\circ` C (ITS-90)]
    p : array_like
        pressure [db]
    pr : number
         reference pressure [db], default = 0

    Returns
    -------
    sgmte : array_like
           density  [kg m :sup:`3`]

    Examples
    --------
    Data from Unesco Tech. Paper in Marine Sci. No. 44, p22

    >>> from oceans import sw_extras as swe
    >>> s = [0, 0, 0, 0, 35, 35, 35, 35]
    >>> t = T90conv([0, 0, 30, 30, 0, 0, 30, 30])
    >>> p = [0, 10000, 0, 10000, 0, 10000, 0, 10000]
    >>> swe.sigmatheta(s, t, p)
    array([ -0.157406  ,  -0.20476006,  -4.34886626,  -3.63884068,
            28.10633141,  28.15738545,  21.72863949,  22.59634627])

    References
    ----------
    .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
    computation of fundamental properties of seawater. UNESCO Tech. Pap. in
    Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
    http://www.scor-int.org/Publications.htm

    .. [2] Millero, F.J., Chen, C.T., Bradshaw, A., and Schleicher, K. A new
    high pressure equation of state for seawater. Deap-Sea Research., 1980,
    Vol27A, pp255-264. doi:10.1016/0198-0149(80)90016-3
    """
    s, t, p, pr = list(map(np.asanyarray, (s, t, p, pr)))
    return sw.pden(s, t, p, pr) - 1000.0


def N(bvfr2):
    r"""Buoyancy frequency is the frequency with which a parcel or particle of
    fluid displaced a small vertical distance from its equilibrium position in
    a stable environment will oscillate. It will oscillate in simple harmonic
    motion with an angular frequency defined by

    .. math:: N = \\left(\\frac{-g}{\\sigma_{\\theta}}
              \\frac{d\\sigma_{\\theta}}{dz}\\right)^{2}

    Parameters
    ----------
    n2 : array_like
         Brünt-Väisälä Frequency squared [s :sup:`-2`]

    Returns
    -------
    n : array_like
        Brünt-Väisälä Frequency not-squared [s :sup:`-1`]

    Examples
    --------
    >>> import numpy as np
    >>> from oceans import sw_extras as swe
    >>> s = np.array([[0, 0, 0], [15, 15, 15], [30, 30, 30],[35,35,35]])
    >>> t = np.repeat(15, s.size).reshape(s.shape)
    >>> p = [[0], [250], [500], [1000]]
    >>> lat = [30,32,35]
    >>> swe.N(sw.bfrq(s, t, p, lat)[0])
    array([[ 0.02124956,  0.02125302,  0.02125843],
           [ 0.02110919,  0.02111263,  0.02111801],
           [ 0.00860812,  0.00860952,  0.00861171]])

    References
    ----------
    .. [1] A.E. Gill 1982. p.54  eqn 3.7.15 "Atmosphere-Ocean Dynamics"
    Academic Press: New York. ISBN: 0-12-283522-0

    .. [2] Jackett, David R., Trevor J. Mcdougall, 1995: Minimal Adjustment of
    Hydrographic Profiles to Achieve Static Stability. J. Atmos. Oceanic
    Technol., 12, 381-389. doi: 10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """

    bvfr2 = np.asanyarray(bvfr2)
    return np.sqrt(np.abs(bvfr2)) * np.sign(bvfr2)


def cph(bvfr2):
    r"""Buoyancy frequency in Cycles Per Hour.

    Parameters
    ----------
    n2 : array_like
         Brünt-Väisälä Frequency squared [s :sup:`-2`]

    Returns
    -------
    cph : array_like
          Brünt-Väisälä Frequency [ cylcles hour :sup:`-1`]

    Examples
    --------
    >>> import numpy as np
    >>> from oceans import sw_extras as swe
    >>> s = np.array([[0, 0, 0], [15, 15, 15], [30, 30, 30],[35,35,35]])
    >>> t = np.repeat(15, s.size).reshape(s.shape)
    >>> p = [[0], [250], [500], [1000]]
    >>> lat = [30,32,35]
    >>> swe.cph(sw.bfrq(s, t, p, lat)[0])
    array([[ 12.17509899,  12.17708145,  12.18018192],
           [ 12.09467754,  12.09664676,  12.09972655],
           [  4.93208775,   4.9328907 ,   4.93414649]])

    References
    ----------
    .. [1] A.E. Gill 1982. p.54  eqn 3.7.15 "Atmosphere-Ocean Dynamics"
    Academic Press: New York. ISBN: 0-12-283522-0
    """
    bvfr2 = np.asanyarray(bvfr2)

    # Root squared preserving the sign.
    bvfr = np.sqrt(np.abs(bvfr2)) * np.sign(bvfr2)
    return bvfr * 60. * 60. / (2. * np.pi)


def shear(z, u, v=0):
    r"""Calculates the vertical shear for u, v velocity section.

    .. math::
        \\textrm{shear} = \\frac{\\partial (u^2 + v^2)^{0.5}}{\partial z}

    Parameters
    ----------
    z : array_like
        depth [m]
    u(z) : array_like
           Eastward velocity [m s :sup:`-1`]
    v(z) : array_like
           Northward velocity [m s :sup:`-1`]

    Returns
    -------
    shr : array_like
          frequency [s :sup:`-1`]
    z_ave : array_like
            depth between z grid (M-1xN)  [m]

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> z = [[0], [250], [500], [1000]]
    >>> u = [[0.5, 0.5, 0.5], [0.15, 0.15, 0.15],
    ...      [0.03, 0.03, .03], [0.,0.,0.]]
    >>> swe.shear(z, u)[0]
    array([[ -1.40000000e-03,  -1.40000000e-03,  -1.40000000e-03],
           [ -4.80000000e-04,  -4.80000000e-04,  -4.80000000e-04],
           [ -6.00000000e-05,  -6.00000000e-05,  -6.00000000e-05]])
    """
    z, u, v = list(map(np.asanyarray, (z, u, v)))
    z, u, v = np.broadcast_arrays(z, u, v)

    m, n = z.shape
    iup = np.arange(0, m - 1)
    ilo = np.arange(1, m)
    z_ave = (z[iup, :] + z[ilo, :]) / 2.
    vel = np.sqrt(u ** 2 + v ** 2)
    diff_vel = np.diff(vel, axis=0)
    diff_z = np.diff(z, axis=0)
    shr = diff_vel / diff_z
    return shr, z_ave


def richnumb(bvfr2, S2):
    r"""Calculates  the ratio of buoyancy to inertial forces which measures the
    stability of a fluid layer. this functions computes the gradient Richardson
    number in the form of:

    .. math::
        Ri = \\frac{N^2}{S^2}

    Representing a dimensionless number that expresses the ratio of the energy
    extracted by buoyancy forces to the energy gained from the shear of the
    large-scale velocity field.

    Parameters
    ----------
    bvfr2 : array_like
    Brünt-Väisälä Frequency squared (M-1xN)  [rad\ :sup:`-2` s\ :sup:`-2`]
    S2 : array_like
         shear squared [s :sup:`-2`]

    Returns
    -------
    ri : array_like
         non-dimensional

    Examples
    --------
    TODO: check the example and add real values
    >>> import numpy as np
    >>> import seawater as sw
    >>> from oceans import sw_extras as swe
    >>> s = np.array([[0, 0, 0], [15, 15, 15], [30, 30, 30],[ 35, 35, 35]])
    >>> t = np.repeat(15, s.size).reshape(s.shape)
    >>> p = [[0], [250], [500], [1000]]
    >>> lat = [30, 32, 35]
    >>> bvfr2 = sw.bfrq(s, t, p, lat)[0]
    >>> vel = [[0.5, 0.5, 0.5], [0.15, 0.15, 0.15],
    ...        [0.03, 0.03, .03], [0.,0.,0.]]
    >>> S2 = swe.shear(p, vel)[0] ** 2
    >>> swe.richnumb(bvfr2, S2)
    array([[   230.37941215,    230.45444299,    230.57181258],
           [  1934.01949759,   1934.64933431,   1935.63457818],
           [ 20583.24410868,  20589.94661835,  20600.43125069]])
    """
    bvfr2, S2 = list(map(np.asanyarray, (bvfr2, S2)))
    # FIXME: check this.
    return bvfr2 / S2


def cor_beta(lat):
    r"""Calculates the Coriolis :math:`\\beta` factor defined by:

    .. math::
        beta = 2 \\Omega \\cos(lat)

    where:

    .. math::
        \\Omega = \\frac{2 \\pi}{\\textrm{sidereal day}} = 7.292e^{-5}
        \\textrm{ radians sec}^{-1}

    Parameters
    ----------
    lat : array_like
          latitude in decimal degrees north [-90..+90].

    Returns
    -------
    beta : array_like
        Beta Coriolis [s :sup:`-1`]

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> swe.cor_beta(0)
    2.2891586878041123e-11

    References
    ----------
    .. [1] S. Pond & G.Pickard 2nd Edition 1986 Introductory Dynamical
    Oceanogrpahy Pergamon Press Sydney. ISBN 0-08-028728-X

    .. [2] A.E. Gill 1982. p.54  eqn 3.7.15 "Atmosphere-Ocean Dynamics"
    Academic Press: New York. ISBN: 0-12-283522-0
    """
    lat = np.asanyarray(lat)
    return 2 * OMEGA * np.cos(lat) / earth_radius


def inertial_period(lat):
    r"""Calculate the inertial period as:

    .. math::
        Ti = \\frac{2\\pi}{f} = \\frac{T_{sd}}{2\\sin\\phi}

    Parameters
    ----------
    lat : array_like
          latitude in decimal degress north [-90..+90]

    Returns
    -------
    Ti : array_like
         period in seconds

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> lat = 30.
    >>> swe.inertial_period(lat)/3600
    23.934472399219292
    """
    lat = np.asanyarray(lat)
    return 2 * np.pi / sw.f(lat)


def strat_period(N):
    r"""Stratification period is the inverse of the Buoyancy frequency and it
    is defined by:

    .. math:: Tn = \\frac{2\\pi}{N}

    Parameters
    ----------
    N : array_like
        Brünt-Väisälä Frequency [s :sup:`-1`]

    Returns
    -------
    Tn : array_like
         Brünt-Väisälä Period [s]

    Examples
    --------
    >>> import numpy as np
    >>> import seawater as sw
    >>> from oceans import sw_extras as swe
    >>> s = np.array([[0, 0, 0], [15, 15, 15], [30, 30, 30],[35,35,35]])
    >>> t = np.repeat(15, s.size).reshape(s.shape)
    >>> p = [[0], [250], [500], [1000]]
    >>> lat = [30,32,35]
    >>> swe.strat_period(swe.N( sw.bfrq(s, t, p, lat)[0]))
    array([[ 295.68548089,  295.63734267,  295.56208791],
           [ 297.6515901 ,  297.60313502,  297.52738493],
           [ 729.91402019,  729.79520847,  729.60946944]])

    References
    ----------
    .. [1] TODO: Pickard
    """
    N = np.asanyarray(N)
    return 2 * np.pi / N


def visc(s, t, p):
    """Calculates kinematic viscosity of sea-water.  Based on Dan Kelley's fit
    to Knauss's TABLE II-8.

    Parameters
    ----------
    s : array_like
        salinity [psu (PSS-78)]
    t : array_like
        temperature [℃ (ITS-90)]  # FIXME: [degree C (IPTS-68)]
    p : array_like
        pressure [db]
    Returns
    -------
    visc : kinematic viscosity of sea-water [m^2/s]

    Notes
    -----
    From matlab airsea

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> swe.visc(40., 40., 1000.)
    8.2001924966338036e-07

    Modifications: Original 1998/01/19 - Ayal Anis 1998
                   2010/11/25. Filipe Fernandes, python translation.
    """
    s, t, p = np.broadcast_arrays(s, t, p)

    visc = 1e-4 * (17.91 - 0.5381 * t + 0.00694 * t**2 + 0.02305 * s)
    visc /= sw.dens(s, t, p)

    return visc


def tcond(s, t, p):
    r"""Calculates thermal conductivity of sea-water.

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:`^\\circ` C (ITS-90)]
    p : array_like
        pressure [db]

    Returns
    -------
    therm : array_like
           thermal conductivity [W m :sup: `-1` K :sup: `-1`]

    Notes
    -----
    From matlab airsea

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> swe.tcond(35, 20, 0)
    0.5972445569999999

    References
    ----------
    .. [1] Caldwell's DSR 21:131-137 (1974)  eq. 9
    .. [2] Catelli et al.'s DSR 21:311-3179(1974)  eq. 5

    Modifications: Original 1998/01/19 - Ayal Anis 1998
                   2010/11/25. Filipe Fernandes, python translation.
    """
    s, t, p = list(map(np.asanyarray, (s, t, p)))

    if False:  # Castelli's option.
        therm = 100. * (5.5286e-3 + 3.4025e-8 * p + 1.8364e-5 *
                        t - 3.3058e-9 * t ** 3)  # [W/m/K]

    # 1) Caldwell's option # 2 - simplified formula, accurate to 0.5% (eqn. 9)
    # in [cal/cm/C/sec]
    therm = 0.001365 * (1. + 0.003 * t - 1.025e-5 * t ** 2 + 0.0653 *
                        (1e-4 * p) - 0.00029 * s)
    return therm * 418.4  # [cal/cm/C/sec] ->[ W/m/K]


def spice(s, t, p):
    r"""Compute sea spiciness as defined by Flament (2002).

    .. math:: \pi(\theta,s) = \sum^5_{i=0} \sum^4_{j=0} b_{ij}\theta^i(s-35)^i

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:`^\\circ` C (ITS-90)]
    p : array_like
        pressure [db]

    Returns
    -------
    sp : array_like
         :math:`\pi` [kg m :sup:`3`]

    See Also
    --------
    pressure is not used... should the input be theta instead of t?
    Go read the paper!

    Notes
    -----
    Spiciness, just like potential density, is only useful over limited
    vertical excursions near the pressure to which they are referenced; for
    large vertical ranges, the slope of the isopycnals and spiciness isopleths
    vary signiﬁcantly with pressure, and generalization of the polynomial
    expansion to include a reference pressure dependence is needed.

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> swe.spice(33, 15, 0)
    array(0.5445864137500002)

    References
    ----------
    .. [1] A state variable for characterizing water masses and their
    diffusive stability: spiciness. Prog. in Oceanography Volume 54, 2002,
    Pages 493-501.

    http://www.satlab.hawaii.edu/spice/spice.m

    Modifications: 2011/03/15. Filipe Fernandes, python translation.
    """
    s, t, p = list(map(np.asanyarray, (s, t, p)))

    pt = sw.ptmp(s, t, p)  # FIXME: I'm not sure about this.

    B = np.zeros((6, 5))
    B[0, 0] = 0.
    B[0, 1] = 7.7442e-001
    B[0, 2] = -5.85e-003
    B[0, 3] = -9.84e-004
    B[0, 4] = -2.06e-004

    B[1, 0] = 5.1655e-002
    B[1, 1] = 2.034e-003
    B[1, 2] = -2.742e-004
    B[1, 3] = -8.5e-006
    B[1, 4] = 1.36e-005

    B[2, 0] = 6.64783e-003
    B[2, 1] = -2.4681e-004
    B[2, 2] = -1.428e-005
    B[2, 3] = 3.337e-005
    B[2, 4] = 7.894e-006

    B[3, 0] = -5.4023e-005
    B[3, 1] = 7.326e-006
    B[3, 2] = 7.0036e-006
    B[3, 3] = -3.0412e-006
    B[3, 4] = -1.0853e-006

    B[4, 0] = 3.949e-007
    B[4, 1] = -3.029e-008
    B[4, 2] = -3.8209e-007
    B[4, 3] = 1.0012e-007
    B[4, 4] = 4.7133e-008

    B[5, 0] = -6.36e-010
    B[5, 1] = -1.309e-009
    B[5, 2] = 6.048e-009
    B[5, 3] = -1.1409e-009
    B[5, 4] = -6.676e-010

    sp = np.zeros_like(pt)
    T = np.ones_like(pt)
    s -= 35.
    r, c = B.shape
    for i in range(r):
        S = np.ones_like(pt)
        for j in range(c):
            sp += B[i, j] * T * S
            S *= s
        T *= pt

    return sp


def psu2ppt(psu):
    r"""Converts salinity from PSU units to PPT
    http://stommel.tamu.edu/~baum/paleo/ocean/node31.html
    #PracticalSalinityScale
    """

    a = [0.008, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081]
    return (a[1] + a[2] * psu ** 0.5 + a[3] * psu + a[4] * psu ** 1.5 + a[5] *
            psu ** 2 + a[6] * psu ** 2.5)


def soundspeed(S, T, D, equation='mackenzie'):
    """Various sound-speed equations.
    1)  soundspeed(s, t, d) returns the sound speed (m/sec) given vectors
       of salinity (ppt), temperature (deg C) and DEPTH (m) using
       the formula of Mackenzie:  Mackenzie, K.V. "Nine-term Equation for
       Sound Speed in the Oceans", J. Acoust. Soc. Am. 70 (1981), 807-812.

    2) soundspeed(s, t, p, 'del_grosso') returns the sound speed (m/sec)given
       vectors of salinity (ppt), temperature (deg C), and  PRESSURE (dbar)
       using the Del Grosso equation:  Del Grosso, "A New Equation for the
       speed of sound in Natural Waters", J. Acoust. Soc. Am. 56#4 (1974).

    3) soundspeed(s, t, p, 'chen') returns the sound speed (m/sec) given
       vectors of salinity (ppt), temperature (deg C), and PRESSURE (dbar)
       using the Chen and Millero equation:  Chen and Millero, "The Sound
       Speed in Seawater", J. Acoust. Soc. Am. 62 (1977), 1129-1135.

    4) soundspeed(s, t, p, 'state') returns the sound speed (m/sec) given
       vectors of salinity (ppt), temperature (deg C), and PRESSURE (dbar) by
       using derivatives of the EOS80 equation of state for seawater and the
       adiabatic lapse rate.

    Notes: RP (WHOI) 3/dec/91
            Added state equation ss
    """
    if equation == 'mackenzie':
        c = 1.44896e3
        t = 4.591e0
        t2 = -5.304e-2
        t3 = 2.374e-4
        s = 1.340e0
        d = 1.630e-2
        d2 = 1.675e-7
        ts = -1.025e-2
        td3 = -7.139e-13
        ssp = (c + t * T + t2 * T * T + t3 * T * T * T + s * (S-35.0) + d *
               D + d2 * D * D + ts * T * (S-35.0) + td3 * T * D * D * D)
    elif equation == 'del_grosso':
        # Del grosso uses pressure in kg/cm^2.  To get to this from dbars
        # we  must divide by "g".  From the UNESCO algorithms (referring to
        # ANON (1970) BULLETIN GEODESIQUE) we have this formula for g as a
        # function of latitude and pressure.  We set latitude to 45 degrees
        # for convenience!
        XX = np.sin(45 * np.pi/180)
        GR = 9.780318 * (1.0 + (5.2788E-3 + 2.36E-5 * XX) * XX) + 1.092E-6 * D
        P = D / GR
        # This is from VSOUND.f.
        C000 = 1402.392
        DCT = (0.501109398873e1 - (0.550946843172e-1 - 0.221535969240e-3 * T) *
               T) * T
        DCS = (0.132952290781e1 + 0.128955756844e-3 * S) * S
        DCP = (0.156059257041e0 + (0.244998688441e-4 - 0.883392332513e-8 * P) *
               P) * P
        DCSTP = ((-0.127562783426e-1 * T * S + 0.635191613389e-2 * T * P +
                  0.265484716608e-7 * T * T * P * P - 0.159349479045e-5 * T *
                  P * P + 0.522116437235e-9 * T * P * P * P -
                  0.438031096213e-6 * T * T * T * P) - 0.161674495909e-8 * S *
                 S * P * P + 0.968403156410e-4 * T * T * S +
                 0.485639620015e-5 * T * S * S * P - 0.340597039004e-3 * T *
                 S * P)
        ssp = C000 + DCT + DCS + DCP + DCSTP
    elif equation == 'chen':
        P0 = D
        # This is copied directly from the UNESCO algorithms.
        # CHECKVALUE: SVEL=1731.995 M/S, S=40 (IPSS-78),T=40 DEG C,P=10000 DBAR
        # SCALE PRESSURE TO BARS
        P = P0 / 10.
        SR = np.sqrt(np.abs(S))
        # S**2 TERM.
        D = 1.727E-3 - 7.9836E-6 * P
        # S**3/2 TERM.
        B1 = 7.3637E-5 + 1.7945E-7 * T
        B0 = -1.922E-2 - 4.42E-5 * T
        B = B0 + B1 * P
        # S**1 TERM.
        A3 = (-3.389E-13 * T + 6.649E-12) * T + 1.100E-10
        A2 = ((7.988E-12 * T - 1.6002E-10) * T + 9.1041E-9) * T - 3.9064E-7
        A1 = ((((-2.0122E-10 * T + 1.0507E-8) * T - 6.4885E-8) * T -
               1.2580E-5) * T + 9.4742E-5)
        A0 = ((((-3.21E-8 * T + 2.006E-6) * T + 7.164E-5) * T - 1.262E-2) *
              T + 1.389)
        A = ((A3 * P + A2) * P + A1) * P + A0
        # S**0 TERM.
        C3 = (-2.3643E-12 * T + 3.8504E-10) * T - 9.7729E-9
        C2 = (((1.0405E-12 * T - 2.5335E-10) * T + 2.5974E-8) * T -
              1.7107E-6) * T + 3.1260E-5
        C1 = (((-6.1185E-10 * T + 1.3621E-7) * T - 8.1788E-6) * T +
              6.8982E-4) * T + 0.153563
        C0 = ((((3.1464E-9 * T - 1.47800E-6) * T + 3.3420E-4) * T -
               5.80852E-2) * T + 5.03711) * T + 1402.388
        C = ((C3 * P + C2) * P + C1) * P + C0
        # SOUND SPEED RETURN.
        ssp = C + (A + B * SR + D * S) * S
    elif equation == 'state':
        raise TypeError("Not implemented!")
        P = D
        # Copied somewhat from program EOSSPEED.F
        svan, sigma = swstate(S, T, P)
        VOL = (1.) / (1000. + sigma)
        # DV/DP|ADIA = (DV/DP) AT CONSTANT T + ADIA.LAPSE RATE *
        # (DV/DT) AT CONSTANT P
        # Note: factor of 10 is convert pressure from dB to Bars.
        dVdP = swstate(S, T, P, 'dP')
        dVdT = swstate(S, T, P, 'dT')
        dVdPad = (dVdP + adiabattempgrad(S, T, P) * dVdT) * 10
        # C = V * SQRT ( 1/DV/DP| ADIA)
        ssp = VOL * np.sqrt(np.abs((1.e5) / dVdPad))
    else:
        raise TypeError('Unrecognizable equation specified: %s' % equation)
    return ssp


def photic_depth(z, par):
    """
    Computes photic depth, based on 1% of surface PAR (Photosynthetically
    Available Radiation).

    Parameters
    ----------
    z : array_like
        depth in meters.
    par : array_like
        float values of PAR

    Returns
    -------
    photic_depth : array_like
        Array of depth in which light is available.
    photic_ix : array_like
        Index of available `par` data from surface to critical depth
    """
    photic_ix = np.where(par >= par[0] / 100.)[0]
    photic_depth = z[photic_ix]

    return photic_depth, photic_ix


def cr_depth(z, par):
    """
    Computes Critical depth. Depth where 1% of surface PAR (Photosynthetically
    Available Radiation).

    Parameters
    ----------
    z : array_like
        depth in meters.
    par : array_like
        float values of PAR

    Returns
    -------
    crdepth : int
        Critical depth. Depth where 1% of surface PAR is available.
    """
    ix = photic_depth(z, par)[1]
    crdepth = z[ix][-1]

    return crdepth


def kdpar(z, par, boundary):
    """
    Compute Kd value, since light extiontion coefficient can be computed
    from depth and Photossintentecally Available Radiation (PAR).
    It will compute a linear regression through out following depths from
    boundary and them will be regressed to the upper depth to boundary
    limits.

    Parameters
    ----------
    z : array_like
        depth in meters of respective PAR values
    par : array_like
        PAR values
    boundary : np.float
        First good upper limit of downcast, when PAR data has stabilized

    Return
    ------
    kd : float
        Light extintion coefficient.
    cr_depth : int
        Critical depth. Depth where 1% of surface PAR is available.
    par_surface : float
        Surface PAR, modeled from first meters data.
    """
    # First linear regression. Returns fit parameters to be used on
    # reconstruction of surface PAR.
    b = np.int32(boundary)
    i_b = np.where(z >= b)[0]
    par_b = par[i_b]
    z_b = z[i_b]
    z_light = photic_depth(z_b, par_b)[1]
    par_z = par_b[z_light]
    z_z = z_b[z_light]
    xp = np.polyfit(z_z, np.log(par_z), 1)

    # Linear regression based on surface PAR, obtained from linear fitting.
    # z = 0
    # PAR_surface = a(z) + b
    par_surface = np.exp(xp[1])
    par = np.r_[par_surface, par]
    z = np.r_[0, z]
    z_par = photic_depth(z, par)[1]
    kd, linear = np.polyfit(z[z_par], np.log(par[z_par]), 1)

    return kd, par_surface


def zmld(s, t, p, smooth=none):
    """
    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [℃ (ITS-90)]
    p : array_like
        pressure [db].
    smooth : int
        size of running mean to smooth data

    Notes
    -----
    Based on density with fixed threshold criteria
    de Boyer Montégut et al., 2004. Mixed layer depth over the global ocean:
        An examination of profile data and a profile-based climatology. doi:10.1029/2004JC002378

    dataset for test and more explanation can be found at:
    http://www.ifremer.fr/cerweb/deboyer/mld/Surface_Mixed_Layer_Depth.php

    Codes based on : http://mixedlayer.ucsd.edu/
    """
    sigma_t = sw.dens0(s, t) - 1000.
    depth = p
    if smooth is not None:
        sigma_t = rolling_mean(sigma_t, smooth, min_periods=1)

    sublayer = np.where(depth[(depth>= 5) & (depth <= 10)])[0]
    sigma_x = np.nanmean(sigma_t[sublayer])
    nan_sigma = np.where(sigma_t > sigma_x + 0.02)[0]
    sigma_t[nan_sigma] = np.nan
    der = np.divide(np.diff(sigma_t), np.diff(depth))
    mld = np.where(der == np.nanmax(der))[0]
    zmld = depth[mld]

    return zmld


if __name__ == '__main__':
    import doctest
    doctest.testmod()
