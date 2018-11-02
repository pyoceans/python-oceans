import re
import warnings

import gsw

import numpy as np
import numpy.ma as ma


def spdir2uv(spd, ang, deg=False):
    """
    Computes u, v components from speed and direction.

    Parameters
    ----------
    spd : array_like
          speed [m s :sup:`-1`]
    ang : array_like
          direction [deg]
    deg : bool
          option, True if data is in degrees. Default is False

    Returns
    -------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]

    """

    if deg:
        ang = np.deg2rad(ang)

    # Calculate U (E-W) and V (N-S) components
    u = spd * np.sin(ang)
    v = spd * np.cos(ang)
    return u, v


def uv2spdir(u, v, mag=0, rot=0):
    """
    Computes speed and direction from u, v components.
    Converts rectangular to polar coordinate, geographic convention
    Allows for rotation and magnetic declination correction.

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]
    mag : float, array_like, optional
          Magnetic correction [deg]
    rot : float, array_like
          Angle for rotation [deg]

    Returns
    -------
    ang : array_like
          direction [deg]
    spd : array_like
          speed [m s :sup:`-1`]

    Notes
    -----
    Zero degrees is north.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from oceans.ocfis import uv2spdir
    >>> u = [+0, -0.5, -0.50, +0.90]
    >>> v = [+1, +0.5, -0.45, -0.85]
    >>> wd, ws = uv2spdir(u,v)
    >>> fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    >>> kw = dict(arrowstyle="->")
    >>> wd = np.deg2rad(wd)
    >>> lines = [ax.annotate("", xy=(d, s), xytext=(0, 0), arrowprops=kw)
    ...  for d, s in zip(wd, ws)]
    >>> _ = ax.set_ylim(0, np.max(ws))

    """

    u, v, mag, rot = list(map(np.asarray, (u, v, mag, rot)))

    vec = u + 1j * v
    spd = np.abs(vec)
    ang = np.angle(vec, deg=True)
    ang = ang - mag + rot
    ang = np.mod(90. - ang, 360.)  # Zero is North.

    return ang, spd


def del_eta_del_x(U, f, g, balance='geostrophic', R=None):
    r"""
    Calculate :mat: `\frac{\partial \eta} {\partial x}` for different
    force balances

    Parameters
    ----------
    U : array_like
        velocity magnitude [m/s]
    f : float
        Coriolis parameter
    d : float
        Acceleration of gravity
    balance : str, optional
              geostrophic [default], gradient or max_gradient
    R : float, optional
        Radius

    """

    if balance == 'geostrophic':
        detadx = f * U / g

    elif balance == 'gradient':
        detadx = (U ** 2 / R + f * U) / g

    elif balance == 'max_gradient':
        detadx = (R * f ** 2) / (4 * g)

    return detadx


def mld(SA, CT, p, criterion='pdvar'):
    r"""
    Compute the mixed layer depth.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]
    criterion : str, optional
               MLD Criteria

    Mixed layer depth criteria are:

    'temperature' : Computed based on constant temperature difference
    criterion, CT(0) - T[mld] = 0.5 degree C.

    'density' : computed based on the constant potential density difference
    criterion, pd[0] - pd[mld] = 0.125 in sigma units.

    'pdvar' : computed based on variable potential density criterion
    pd[0] - pd[mld] = var(T[0], S[0]), where var is a variable potential
    density difference which corresponds to constant temperature difference of
    0.5 degree C.

    Returns
    -------
    MLD : array_like
          Mixed layer depth
    idx_mld : bool array
              Boolean array in the shape of p with MLD index.

    References
    ----------
    Monterey, G., and S. Levitus, 1997: Seasonal variability of mixed
    layer depth for the World Ocean. NOAA Atlas, NESDIS 14, 100 pp.
    Washington, D.C.

    """

    SA, CT, p = list(map(np.asanyarray, (SA, CT, p)))
    SA, CT, p = np.broadcast_arrays(SA, CT, p)
    SA, CT, p = list(map(ma.masked_invalid, (SA, CT, p)))

    p_min, idx = p.min(), p.argmin()

    sigma = gsw.rho(SA, CT, p_min) - 1000.

    # Temperature and Salinity at the surface,
    T0, S0, Sig0 = CT[idx], SA[idx], sigma[idx]

    # NOTE: The temperature difference criterion for MLD
    Tdiff = T0 - 0.5  # 0.8 on the matlab original

    if criterion == 'temperature':
        idx_mld = (CT > Tdiff)
    elif criterion == 'pdvar':
        pdvar_diff = gsw.rho(S0, Tdiff, p_min) - 1000.
        idx_mld = (sigma <= pdvar_diff)
    elif criterion == 'density':
        sig_diff = Sig0 + 0.125
        idx_mld = (sigma <= sig_diff)
    else:
        raise NameError('Unknown criteria {}'.format(criterion))

    MLD = ma.masked_all_like(p)
    MLD[idx_mld] = p[idx_mld]

    return MLD.max(axis=0), idx_mld


def pcaben(u, v):
    """
    Principal components of 2-d (e.g. current meter) data
    calculates ellipse parameters for currents.

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]

    Returns
    -------
    major axis (majrax)
    minor axis (minrax)
    major azimuth (majaz)
    minor azimuth (minaz)
    elipticity (elptcty)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from oceans.ocfis import pcaben, uv2spdir
    >>> u = np.r_[(0., 1., -2., -1., 1.), np.random.randn(10)]
    >>> v = np.r_[(3., 1., 0., -1., -1.), np.random.randn(10)]
    >>> (mjrax, mjaz, mirax, miaz, el), (x1, x2, y1, y2) = pcaben(u, v)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot(x1, y1,'r-', x2, y2, 'r-')
    >>> ax.set_aspect('equal')
    >>> _ = ax.set_xlabel('U component')
    >>> _ = ax.set_ylabel('V component')
    >>> _ = ax.plot(u, v, 'bo')
    >>> _ = ax.axis([-3.2, 3.2, -3.2, 3.2])
    >>> mdir, mspd = uv2spdir(u.mean(), v.mean())
    >>> _ = ax.plot([0, u.mean()],[0, v.mean()], 'k-')
    >>> flatness = 1 - mirax / mjrax
    >>> if False:
    ...     print('Mean u = {}'.format(u.mean()))
    ...     print('Mean v = {}'.format(v.mean()))
    ...     print('Mean magnitude: {}'.format(mspd))
    ...     print('Direction: {}'.format(mdir))
    ...     print('Axis 1 Length: {}'.format(mjrax))
    ...     print('Axis 1 Azimuth: {}'.format(mjaz))
    ...     print('Axis 2 Length: {}'.format.format(mirax))
    ...     print('Axis 2 Azimuth: {}'.format(miaz))
    ...     print('elipticity is : {}'.format(el))
    ...     print('Negative (positive) means clockwise (anti-clockwise)')
    ...     print('Flatness: {}'.format(flatness))

    Notes:
    https://pubs.usgs.gov/of/2002/of02-217/m-files/pcaben.m

    """

    u, v = np.broadcast_arrays(u, v)

    C = np.cov(u, v)
    D, V = np.linalg.eig(C)

    x1 = np.r_[0.5 * np.sqrt(D[0]) * V[0, 0], -0.5 * np.sqrt(D[0]) * V[0, 0]]
    y1 = np.r_[0.5 * np.sqrt(D[0]) * V[1, 0], -0.5 * np.sqrt(D[0]) * V[1, 0]]
    x2 = np.r_[0.5 * np.sqrt(D[1]) * V[0, 1], -0.5 * np.sqrt(D[1]) * V[0, 1]]
    y2 = np.r_[0.5 * np.sqrt(D[1]) * V[1, 1], -0.5 * np.sqrt(D[1]) * V[1, 1]]

    # Length and direction.
    az, leng = np.c_[uv2spdir(x1[0], y1[0]), uv2spdir(x2[1], y2[1])]

    if (leng[0] >= leng[1]):
        majrax, majaz = leng[0], az[0]
        minrax, minaz = leng[1], az[1]
    else:
        majrax, majaz = leng[1], az[1]
        minrax, minaz = leng[0], az[0]

    elptcty = minrax / majrax

    # Radius to diameter.
    majrax *= 2
    minrax *= 2

    return (majrax, majaz, minrax, minaz, elptcty), (x1, x2, y1, y2)


def spec_rot(u, v):
    """
    Compute the rotary spectra from u,v velocity components

    Parameters
    ----------
    u : array_like
        zonal wind velocity [m s :sup:`-1`]
    v : array_like
        meridional wind velocity [m s :sup:`-1`]

    Returns
    -------
    cw : array_like
         Clockwise spectrum [TODO]
    ccw : array_like
          Counter-clockwise spectrum [TODO]
    puv : array_like
          Cross spectra [TODO]
    quv : array_like
          Quadrature spectra [ TODO]

    Notes
    -----
    The spectral energy at some frequency can be decomposed into two circularly
    polarized constituents, one rotating clockwise and other anti-clockwise.

    Examples
    --------
    TODO: puv, quv, cw, ccw = spec_rot(u, v)

    References
    ----------
    J. Gonella Deep Sea Res., 833-846, 1972.

    """

    # Individual components Fourier series.
    fu, fv = list(map(np.fft.fft, (u, v)))

    # Auto-spectra of the scalar components.
    pu = fu * np.conj(fu)
    pv = fv * np.conj(fv)

    # Cross spectra.
    puv = fu.real * fv.real + fu.imag * fv.imag

    # Quadrature spectra.
    quv = -fu.real * fv.imag + fv.real * fu.imag

    # Rotatory components
    # TODO: Check the division, 4 (original code) or 8 (paper)?
    cw = (pu + pv - 2 * quv) / 4.
    ccw = (pu + pv + 2 * quv) / 4.
    N = len(u)
    F = np.arange(0, N) / N
    return puv, quv, cw, ccw, F


def lagcorr(x, y, M=None):
    """
    Compute lagged correlation between two series.
    Follow emery and Thomson book "summation" notation.

    Parameters
    ----------
    y : array
        time-series
    y : array
        time-series
    M : integer
        number of lags

    Returns
    -------
    Cxy : array
          normalized cross-correlation function

    Examples
    --------
    TODO: Implement Emery and Thomson example.

    """

    x, y = list(map(np.asanyarray, (x, y)))

    if not M:
        M = x.size

    N = x.size
    Cxy = np.zeros(M)
    x_bar, y_bar = x.mean(), y.mean()

    for k in range(0, M, 1):
        a = 0.
        for i in range(N - k):
            a = a + (y[i] - y_bar) * (x[i + k] - x_bar)

        Cxy[k] = 1. / (N - k) * a

    return Cxy / (np.std(y) * np.std(x))


def complex_demodulation(series, f, fc, axis=-1):
    """
    Perform a Complex Demodulation
    It acts as a bandpass filter for `f`.

    series => Time-Series object with data and datetime
    f => inertial frequency in rad/sec
    fc => normalized cutoff [0.2]

    math : series.data * np.exp(2 * np.pi * 1j * (1 / T) * time_in_seconds)

    """
    import scipy.signal as signal

    # Time period ie freq = 1 / T
    T = 2 * np.pi / f
    # fc = fc * 1 / T # Filipe

    # De-mean data.
    d = series.data - series.data.mean(axis=axis)
    dfs = d * np.exp(2 * np.pi * 1j * (1 / T) * series.time_in_seconds)

    # Make a 5th order butter filter.
    # FIXME: Why 5th order? Try signal.buttord!?
    Wn = fc / series.Nyq

    [b, a] = signal.butter(5, Wn, btype='low')

    # FIXME: These are a factor of a thousand different from Matlab, why?
    cc = signal.filtfilt(b, a, dfs)  # FIXME: * 1e3
    amplitude = 2 * np.abs(cc)

    # TODO: Fix the outputs.
    # phase = np.arctan2(np.imag(cc), np.real(cc))

    filtered_series = amplitude * np.exp(-2 * np.pi * 1j *
                                         (1 / T) * series.time_in_seconds)
    new_series = filtered_series.real, series.time
    # Return cc, amplitude, phase, dfs, filtered_series
    return new_series


def binave(datain, r):
    """
    Averages vector data in bins of length r. The last bin may be the
    average of less than r elements. Useful for computing daily average time
    series (with r=24 for hourly data).

    Parameters
    ----------
    data : array_like
    r : int
        bin length

    Returns
    -------
    bindata : array_like
              bin-averaged vector

    Notes
    -----
    Original from MATLAB AIRSEA TOOLBOX.

    Examples
    --------
    >>> from oceans.ocfis import binave
    >>> data = [10., 11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123.,
    ...         3.2, 0.52, 18.2, 10., 11., 13., 2., 34., 21.5, 6.46, 6.27,
    ...         7.0867, 15., 123., 3.2, 0.52, 18.2, 10., 11., 13., 2., 34.,
    ...         21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2, 0.52, 18.2, 10.,
    ...         11., 13., 2., 34., 21.5, 6.46, 6.27, 7.0867, 15., 123., 3.2,
    ...         0.52, 18.2]
    >>> binave(data, 24)
    array([16.564725 , 21.1523625, 22.4670875])

    References
    ----------
    03/08/1997: version 1.0
    09/19/1998: version 1.1 (vectorized by RP)
    08/05/1999: version 2.0

    """

    datain, rows = np.asarray(datain), np.asarray(r, dtype=np.int)

    if datain.ndim != 1:
        raise ValueError('Must be a 1D array.')

    if rows <= 0:
        raise ValueError('Bin size R must be a positive integer.')

    # Compute bin averaged series.
    lines = datain.size // r
    z = datain[0:(lines * rows)].reshape(rows, lines, order='F')
    bindata = np.r_[np.mean(z, axis=0), np.mean(datain[(lines * r):])]

    return bindata


def binavg(x, y, db):
    """
    Bins y(x) into db spacing.  The spacing is given in `x` units.
    y = np.random.random(20)
    x = np.arange(len(y))
    xb, yb = binavg(x, y, 2)
    plt.figure()
    plt.plot(x, y, 'k.-')
    plt.plot(xb, yb, 'r.-')

    """
    # Cut the corners.
    x_min, x_max = np.ceil(x.min()), np.floor(x.max())
    x = x.clip(x_min, x_max)

    # This is used to get the `inds`.
    xbin = np.arange(x_min, x_max, db)
    inds = np.digitize(x, xbin)

    # But this is the center of the bins.
    xbin = xbin - (db / 2.)

    # FIXME there is an IndexError if I want to show this.
    # for n in range(x.size):
    #    print xbin[inds[n]-1], "<=", x[n], "<", xbin[inds[n]]

    ybin = np.array([y[inds == i].mean() for i in range(0, len(xbin))])
    # xbin = np.array([x[inds == i].mean() for i in range(0, len(xbin))])

    return xbin, ybin


def bin_dates(self, freq, tz=None):
    """
    Take a pandas time Series and return a new Series on the specified
    frequency.

    Examples
    --------
    >>> import numpy as np
    >>> from pandas import Series, date_range
    >>> n = 24*30
    >>> sig = np.random.rand(n) + 2 * np.cos(2 * np.pi * np.arange(n))
    >>> dates = date_range(start='1/1/2000', periods=n, freq='H')
    >>> series = Series(data=sig, index=dates)
    >>> new_series = bin_dates(series, freq='D', tz=None)

    """
    from pandas import date_range

    new_index = date_range(start=self.index[0], end=self.index[-1],
                           freq=freq, tz=tz)
    new_series = self.groupby(new_index.asof).mean()
    # Averages at the center.
    secs = new_index.freq.delta.total_seconds()
    new_series.index = new_series.index.values + int(secs // 2)
    return new_series


def series_spline(self):
    """
    Fill NaNs using a spline interpolation.

    """
    from pandas import Series, isnull
    from scipy.interpolate import InterpolatedUnivariateSpline

    inds, values = np.arange(len(self)), self.values

    invalid = isnull(values)
    valid = -invalid

    firstIndex = valid.argmax()
    valid = valid[firstIndex:]
    invalid = invalid[firstIndex:]
    inds = inds[firstIndex:]

    result = values.copy()
    s = InterpolatedUnivariateSpline(inds[valid], values[firstIndex:][valid])
    result[firstIndex:][invalid] = s(inds[invalid])

    return Series(result, index=self.index, name=self.name)


def despike(self, n=3, recursive=False):
    """
    Replace spikes with np.NaN.
    Removing spikes that are >= n * std.
    default n = 3.

    """
    from pandas import Series

    result = self.values.copy()
    outliers = (np.abs(self.values - np.nanmean(self.values)) >= n *
                np.nanstd(self.values))

    removed = np.count_nonzero(outliers)
    result[outliers] = np.NaN

    counter = 0
    if recursive:
        while outliers.any():
            result[outliers] = np.NaN
            base = np.abs(result - np.nanmean(result))
            outliers = base >= n * np.nanstd(result)
            counter += 1
            removed += np.count_nonzero(outliers)
    return Series(result, index=self.index, name=self.name)


def pol2cart(theta, radius, units='deg'):
    """
    Convert from polar to Cartesian coordinates

    Examples
    --------
    >>> pol2cart(0, 1, units='deg')
    (1.0, 0.0)

    """
    if units in ['deg', 'degs']:
        theta = theta * np.pi / 180.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def cart2pol(x, y):
    """
    Convert from Cartesian to polar coordinates.

    Examples
    --------
    >>> x = [+0, -0.5]
    >>> y = [+1, +0.5]
    >>> cart2pol(x, y)
    (array([1.57079633, 2.35619449]), array([1.        , 0.70710678]))

    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius


def wrap_lon180(lon):
    lon = np.atleast_1d(lon).copy()
    angles = np.logical_or((lon < -180), (180 < lon))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon


def wrap_lon360(lon):
    lon = np.atleast_1d(lon).copy()
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon


def alphanum_key(s):
    key = re.split(r'(\d+)', s)
    key[1::2] = list(map(int, key[1::2]))
    return key


def get_profile(x, y, f, xi, yi, mode='nearest', order=3):
    """
    Interpolate regular data.

    Parameters
    ----------
    x : two dimensional np.ndarray
        an array for the :math:`x` coordinates

    y : two dimensional np.ndarray
        an array for the :math:`y` coordinates

    f : two dimensional np.ndarray
        an array with the value of the function to be interpolated
        at :math:`x,y` coordinates.

    xi : one dimension np.ndarray
        the :math:`x` coordinates of the point where we want
        the function to be interpolated.

    yi : one dimension np.ndarray
        the :math:`y` coordinates of the point where we want
        the function to be interpolated.

    order : int
        the order of the bivariate spline interpolation


    Returns
    -------
    fi : one dimension np.ndarray
        the value of the interpolating spline at :math:`xi,yi`


    Examples
    --------
    >>> import numpy as np
    >>> from oceans.ocfis import get_profile
    >>> x, y = np.meshgrid(range(360), range(91))
    >>> f = np.array(range(91 * 360)).reshape((91, 360))
    >>> Paris = [2.4, 48.9]
    >>> Rome = [12.5, 41.9]
    >>> Greenwich = [0, 51.5]
    >>> xi = Paris[0], Rome[0], Greenwich[0]
    >>> yi = Paris[1], Rome[1], Greenwich[1]
    >>> get_profile(x, y, f, xi, yi, order=3)
    array([17606, 15096, 18540])

    """
    from scipy.ndimage import map_coordinates

    x, y, f, xi, yi = list(map(np.asanyarray, (x, y, f, xi, yi)))
    conditions = np.array([xi.min() < x.min(),
                           xi.max() > x.max(),
                           yi.min() < y.min(),
                           yi.max() > y.max()])

    if conditions.any():
        warnings.warn('Warning! Extrapolation!!')

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    jvals = (xi - x[0, 0]) / dx
    ivals = (yi - y[0, 0]) / dy

    coords = np.array([ivals, jvals])

    return map_coordinates(f, coords, mode=mode, order=order)


def strip_mask(arr, fill_value=np.NaN):
    """
    Take a masked array and return its data(filled) + mask.

    """
    if ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        arr = np.ma.filled(arr, fill_value)
        return mask, arr
    else:
        return arr


def shiftdim(x, n=None):
    """
    Matlab-like shiftdim in python.

    Examples
    --------
    >>> import oceans.ocfis as ff
    >>> a = np.random.rand(1,1,3,1,2)
    >>> print("a shape and dimension: %s, %s" % (a.shape, a.ndim))
    a shape and dimension: (1, 1, 3, 1, 2), 5
    >>> # print(range(a.ndim))
    >>> # print(np.roll(range(a.ndim), -2))
    >>> # print(a.transpose(np.roll(range(a.ndim), -2)))
    >>> b = ff.shiftdim(a)
    >>> print("b shape and dimension: %s, %s" % (b.shape, b.ndim))
    b shape and dimension: (3, 1, 2), 3
    >>> c = ff.shiftdim(b, -2)
    >>> c.shape == a.shape
    True

    """

    def no_leading_ones(shape):
        shape = np.atleast_1d(shape)
        if shape[0] == 1:
            shape = shape[1:]
            return no_leading_ones(shape)
        else:
            return shape

    if n is None:
        # returns the array B with the same number of
        # elements as X but with any leading singleton
        # dimensions removed.
        return x.reshape(no_leading_ones(x.shape))
    elif n >= 0:
        # When n is positive, shiftdim shifts the dimensions
        # to the left and wraps the n leading dimensions to the end.
        return x.transpose(np.roll(list(range(x.ndim)), -n))
    else:
        # When n is negative, shiftdim shifts the dimensions
        # to the right and pads with singletons.
        return x.reshape((1,) * -n + x.shape)
