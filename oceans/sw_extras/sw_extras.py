# -*- coding: utf-8 -*-

"""
Extra seawater functions
========================
"""

from __future__ import division

import numpy as np
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
           'psu2ppt']


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

    >>> import seawater as sw
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
    s, t, p = map(np.asanyarray, (s, t, p))
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

    >>> import seawater as sw
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
    s, t, p, pr = map(np.asanyarray, (s, t, p, pr))
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
    z, u, v = map(np.asanyarray, (z, u, v))
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
    bvfr2, S2 = map(np.asanyarray, (bvfr2, S2))
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
    >>> swe.strat_period( swe.N( sw.bfrq(s, t, p, lat)[0] ) )
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
    r"""Calculates kinematic viscosity of sea-water.

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
    visw : array_like
           [m :sup: `2` s :sup: `-1`]

    See Also
    --------
    visc_air from airsea toolbox

    Notes
    -----
    From matlab airsea

    Examples
    --------
    >>> from oceans import sw_extras as swe
    >>> swe.visc(40, 40, 1000)
    8.2001924966338036e-07

    References
    ----------
    .. [1] Dan Kelley's fit to Knauss's TABLE II-8.

    Modifications: Original 1998/01/19 - Ayal Anis 1998
                   2010/11/25. Filipe Fernandes, python translation.
    """
    s, t, p = map(np.asanyarray, (s, t, p))
    return (1e-4 * (17.91 - 0.5381 * t + 0.00694 * t ** 2 + 0.02305 * s) /
            sw.dens(s, t, p))


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
    s, t, p = map(np.asanyarray, (s, t, p))

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
    s, t, p = map(np.asanyarray, (s, t, p))

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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
