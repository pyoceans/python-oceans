# -*- coding: utf-8 -*-
#
# ctd.py
#
# purpose:  Some classes and functions to work with CTD data.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2012
# modified: Thu 18 Apr 2013 12:23:17 PM BRT
#
# obs: Should I sub-classing instead of a "Monkey Patching"?
#


from __future__ import division

import bz2
import gzip
import zipfile
import cStringIO

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA

from scipy import signal
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot

from pandas import Panel, DataFrame, Series, Index, read_table

import gsw
from oceans.utilities import basename


degree = u"\u00b0"


# Utilities.
def load_bl(blfile):
    names = ['bottles', 'datetime', 'start', 'finish']
    bl = read_table(blfile, skiprows=2, sep=',', parse_dates=False,
                    index_col=1, names=names)

    if (bl.index == bl['bottles']).all():
        del bl['bottles']
    else:
        raise ValueError("First column is not identical to the second.")

    return bl


def extrap1d(interpolator):
    r"""http://stackoverflow.com/questions/2745329/"""
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return (ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] -
                                                                 xs[-2]))
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def normalize_names(name):
    name = name.strip()
    name = name.strip('*')  # Seabird only
    name = name.lower()
    #name = name.replace(' ', '_')
    return name


def read_file(fname, compression=None):
    if compression == 'gzip':
        f = gzip.open(fname)
    elif compression == 'bz2':
        f = bz2.BZ2File(fname)
    elif compression == 'zip':
        zfile = zipfile.ZipFile(fname)
        name = zfile.namelist()[0]
        f = cStringIO.StringIO(zfile.read(name))
    else:
        f = open(fname)
    return f


def movingaverage(series, window_size=48):
    window = np.ones(int(window_size)) / float(window_size)
    return Series(np.convolve(series, window, 'same'), index=series.index)


def rolling_window(data, block):
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


# TEOS-10 and other functions.
def check_index(ix0, ix1):
    if isinstance(ix0, Series) and isinstance(ix1, Series):
        ix0, ix1 = ix0.index.float_(), ix1.index.float_()
        if (ix0 == ix1).all():
            return ix0, ix1
    elif isinstance(ix0, float) and isinstance(ix1, float):
        if ix0 == ix1:
            return ix0, ix1
    else:
        raise ValueError("Series index must be the same.")


def mixed_layer_depth(t, verbose=True):
    mask = t[0] - t < 0.5
    if verbose:
        print("MLD: %3.2f " % t.index[mask][-1])
    return Series(mask, index=t.index, name='MLD')


def barrier_layer_depth(SA, CT, verbose=True):
    sigma_theta = gsw.sigma0_CT_exact(SA, CT)
    # Density difference from the surface to the "expected" using the
    # temperature at the base of the mixed layer.
    mask = mixed_layer_depth(CT)
    mld = np.where(mask)[0][-1]
    sig_surface = sigma_theta[0]
    sig_bottom_mld = gsw.sigma0_CT_exact(SA[0], CT[mld])
    d_sig_t = sig_surface - sig_bottom_mld
    d_sig = sigma_theta - sig_bottom_mld
    mask = d_sig < d_sig_t  # Barrier layer.
    if verbose:
        if mask.any():
            print("Barrier layer: %3.2f." % SA.index[mask][-1])
        else:
            print("No barrier layer present.")
    return Series(mask, index=SA.index, name='Barrier_layer')


def extrap_sec(data, dist, depth, w1=1., w2=0):
    r"""Extrapolates `data` to zones where the shallow stations are shadowed by
    the deep stations.  The shadow region usually cannot be extrapolates via
    linear interpolation.

    The extrapolation is applied using the gradients of the `data` at a certain
    level.

    Parameters
    ----------
    data : array_like
          Data to be extrapolated
    dist : array_like
           Stations distance
    fd : float
         Decay factor [0-1]


    Returns
    -------
    Sec_extrap : array_like
                 Extrapolated variable

    Examples
    --------
    Sec_extrap = extrap_sec(data, dist, z, fd=1.)
    """
    new_data1 = []
    for row in data:
        mask = ~np.isnan(row)
        if mask.any():
            y = row[mask]
            if y.size == 1:
                row = np.repeat(y, len(mask))
            else:
                x = dist[mask]
                f_i = interp1d(x, y)
                f_x = extrap1d(f_i)
                row = f_x(dist)
        new_data1.append(row)

    new_data2 = []
    for col in data.T:
        mask = ~np.isnan(col)
        if mask.any():
            y = col[mask]
            if y.size == 1:
                col = np.repeat(y, len(mask))
            else:
                z = depth[mask]
                f_i = interp1d(z, y)
                f_z = extrap1d(f_i)
                col = f_z(depth)
        new_data2.append(col)

    new_data = np.array(new_data1) * w1 + np.array(new_data2).T * w2
    return new_data


def gen_topomask(h, lon, lat, dx=1., kind='linear', plot=False):
    r"""Generates a topography mask from an oceanographic transect taking the
    deepest CTD scan as the depth of each station.

    Inputs
    ------
    h : array
        Pressure of the deepest CTD scan for each station [dbar].
    lons : array
           Longitude of each station [decimal degrees east].
    lat : Latitude of each station. [decimal degrees north].
    dx : float
         Horizontal resolution of the output arrays [km].
    kind : string, optional
           Type of the interpolation to be performed.
           See scipy.interpolate.interp1d documentation for details.
    plot : bool
           Whether to plot mask for visualization.

    Outputs
    -------
    xm : array
         Horizontal distances [km].
    hm : array
         Local depth [m].

    Examples
    --------
    >>> xm, hm = gen_topomask(h, lon, lat, dx=1., kind='linear', plot=False')
    >>> fig, ax = plt.subplots()
    >>> ax.plot(xm, hm, 'k', linewidth=1.5)
    >>> ax.plot(x, h, 'ro')
    >>> ax.set_xlabel('Distance [km]')
    >>> ax.set_ylabel('Depth [m]')
    >>> ax.grid(True)
    >>> plt.show()

    Author
    ------
    André Palóczy Filho (paloczy@gmail.com) --  October/2012
    """

    h, lon, lat = map(np.asanyarray, (h, lon, lat))
    # Distance in km.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    h = -gsw.z_from_p(h, lat.mean())
    Ih = interp1d(x, h, kind=kind, bounds_error=False, fill_value=h[-1])
    xm = np.arange(0, x.max() + dx, dx)
    hm = Ih(xm)

    return xm, hm


def pmel_inversion_check():
    r"""Additional clean-up and flagging of data after the SBE Processing.
    Look for inversions in the processed, binned via computing the centered
    square of the buoyancy frequency, N2, for each bin and linearly
    interpolating temperature, conductivity, and oxygen over those records
    where N2 ≤ -1 x 10-5 s-2, where there appear to be density inversions.

    NOTE: While these could be actual inversions in the CTD records, it is much
    more likely that shed wakes cause these anomalies.  Records that fail the
    density inversion criteria in the top 20 meters are retained, but flagged
    as questionable.

    FIXME: The codes also manually remove spikes or glitches from profiles as
    necessary, and linearly interpolate over them."""

    # TODO
    pass


# Index methods.
def asof(self, label):
    if label not in self:
        loc = self.searchsorted(label, side='left')
        if loc > 0:
            return self[loc - 1]
        else:
            return np.nan
    return label


def float_(self):
    return np.float_(self.values)


# Series methods.
def split(self):
    r"""Returns a tupple with down- and up-cast."""
    down = self.ix[:self.index.argmax()]
    up = self.ix[self.index.argmax():][::-1]  # Reverse up index.
    return down, up


def press_check(self, column='index'):
    r"""Remove pressure reversal.
    NOTE: Must be applied after split."""
    data = self.copy()
    if column is not 'index':
        press = data[column]
    else:
        press = data.index.float_()

    ref = press[0]
    inversions = np.diff(np.r_[press, press[-1]]) < 0
    mask = np.zeros_like(inversions)
    for k, p in enumerate(inversions):
        if p:
            ref = press[k]
            cut = press[k + 1:] < ref
            mask[k + 1:][cut] = True
    data[mask] = np.NaN
    return data


def rosette_summary(self, bl):
    r"""Make a BTL (bottle) file from a BL (bottle log) file.

    Seabird produce their own BTL file, but here we have more control for the
    averaging process and on which step we want to perform this.  Therefore
    eliminating the need to read the data into SB Software again after some
    pre-processing.  NOTE: Run after LoopEdit.

    FIXME: Write to a file like this:
    AMB09_101_CTD_rad5.btl
    Bottle        Date     Scan       PrDM      T090C     C0S/m
  Position        Time
      1    Oct 16 2012   109892   3729.447     1.2721  3.148013  (avg)
              18:52:19       14      0.439     0.0038  0.000340  (sdev)
                         109868   3728.720     1.2645  3.147310  (min)
                         109916   3730.234     1.2772  3.148478  (max)
      2    Oct 16 2012   117087   3825.047     1.1905  3.143622  (avg)
              18:57:19       14      0.316     0.0068  0.000602  (sdev)
                         117063   3824.464     1.1792  3.142608  (min)
                         117111   3825.501     1.1985  3.144288  (max)
    """

    rossum = dict()
    for bottle, k0, k1 in zip(bl.index, bl['start'], bl['finish']):
        mask = np.logical_and(self['scan'] >= k0, self['scan'] <= k1).values
        rossum[bottle] = swap_index(self, 'scan')[mask].describe()

    return Panel.fromDict(rossum, orient='items')


def seabird_filter(data, sample_rate=24.0, time_constant=0.15):
    r"""Filter a series with `time_constant` (use 0.15 s for pressure), and for
    a signal of `sample_rate` in Hertz (24 Hz for 911+).
    NOTE: Seabird actually uses a cosine filter.  I use a kaiser filter
          instead.
    NOTE: 911+ does note require filter for temperature nor salinity."""

    nyq_rate = sample_rate / 2.0
    width = 5.0 / nyq_rate  # 5 Hz transition rate.
    ripple_db = 60.0  # Attenuation at the stop band.
    N, beta = signal.kaiserord(ripple_db, width)

    cutoff_hz = (1. / time_constant)  # Cutoff frequency at 0.15 s.
    taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
    data = signal.filtfilt(taps, [1.0], data)
    return data


def despike(self, n1=2, n2=20, block=100, keep=0):
    r"""Wild Edit Seabird-like function.  Passes with Standard deviation
    `n1` and `n2` with window size `block`."""
    """ TODO: Keep data within this distance of mean: Do not flag data within
    this distance of mean, even if it falls outside specified standard
    deviation.  Set to a value where difference between data and mean would
    indicate a wild point.  May need to use if data is very quiet (for example,
    a single bit change in voltage may cause data to fall outside specified
    standard deviation and be marked bad).  A typical sequence for using
    parameter follows:
        Run Wild Edit for all desired variables, with parameter set to 0.
        Compare output to input data.  If a variable’s data points that are
        very close to mean were set to badflag:

        Rerun Wild Edit for all other variables, leaving parameter at 0 and
        overwriting output file from Step 1.  Rerun Wild Edit for quiet
        variable only, setting parameter to desired value to prevent flagging
        of data close to mean."""

    data = self.values.copy()
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(data - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    data[mask] = np.NaN

    # Pass two recompute the mean and std without the flagged values from pass
    # one and removed the flagged data.
    roll = rolling_window(data, block)
    roll = ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(self.values - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    self[mask] = np.NaN
    return self


def bindata(self, delta=1.):
    r"""Bin average the index (usually pressure) to a given interval (default
    delta = 1).

    Note that this method does not drop NA automatically.  Therefore, one can
    check the quality of the binned data."""
    # TODO: Check scipy.stats.binned_statistic for bindata.

    # TODO: save number of points used in each the bin.
    start = np.floor(self.index[0])
    end = np.ceil(self.index[-1])
    shift = delta / 2.  # To get centered bins.
    new_index = np.arange(start, end, delta) - shift

    new_index = Index(new_index)
    newdf = self.groupby(new_index.asof).mean()
    newdf.index += shift  # Not shifted.

    return newdf


def smooth(self, window_len=11, window='hanning'):
    r"""Smooth the data using a window with requested size."""
    # TODO: variable window_len (5 <100, 11 100--500, 21 > 500)
    # TODO: Compare with rolling_window.

    windows = dict(flat=np.ones, hanning=np.hanning, hamming=np.hamming,
                   bartlett=np.bartlett, blackman=np.blackman)
    data = self.values.copy()

    if window_len < 3:
        return Series(data, index=self.index, name=self.name)

    if not window in windows.keys():
        raise ValueError("""Window is on of 'flat', 'hanning', 'hamming',
                         'bartlett', 'blackman'""")

    s = np.r_[2 * data[0] - data[window_len:1:-1], data, 2 *
              data[-1] - data[-1:-window_len:-1]]

    w = windows[window](window_len)

    data = np.convolve(w / w.sum(), s, mode='same')
    data = data[window_len - 1:-window_len + 1]
    return Series(data, index=self.index, name=self.name)


# DataFrame methods.
def get_maxdepth(self):
    valid_last_depth = self.apply(Series.notnull).values.T
    return np.float_(self.index * valid_last_depth).max(axis=1)


def plot_section(self, inverse=False, filled=False, **kw):
    if inverse:
        lon = self.lon[::-1].copy()
        lat = self.lat[::-1].copy()
        data = self.T[::-1].T.copy()
    else:
        lon = self.lon.copy()
        lat = self.lat.copy()
        data = self.copy()
    # Contour key words.
    fmt = kw.pop('fmt', '%1.0f')
    extend = kw.pop('extend', 'both')
    fontsize = kw.pop('fontsize', 12)
    labelsize = kw.pop('labelsize', 11)
    cmap = kw.pop('cmap', plt.cm.rainbow)
    levels = kw.pop('levels', np.arange(np.floor(data.min().min()),
                    np.ceil(data.max().max()) + 0.5, 0.5))

    # Colorbar key words.
    pad = kw.pop('pad', 0.04)
    aspect = kw.pop('aspect', 40)
    shrink = kw.pop('shrink', 0.9)
    fraction = kw.pop('fraction', 0.05)

    # Topography mask key words.
    dx = kw.pop('dx', 1.)
    kind = kw.pop('kind', 'linear')

    # Station symbols key words.
    color = kw.pop('color', 'k')
    offset = kw.pop('offset', -5)
    linewidth = kw.pop('linewidth', 1.5)

    # Get data for plotting.
    x = np.append(0, np.cumsum(gsw.distance(lon, lat)[0] / 1e3))
    z = np.float_(data.index.values)
    h = data.get_maxdepth()
    data = ma.masked_invalid(data.values)
    if filled:
        # FIXME:  This cause discontinuities.
        data = data.filled(fill_value=np.nan)
        data = extrap_sec(data, x, z, w1=0.97, w2=0.03)

    xm, hm = gen_topomask(h, lon, lat, dx=dx, kind=kind)

    # Figure.
    fig, ax = plt.subplots()
    ax.plot(xm, hm, color='black', linewidth=linewidth, zorder=3)
    ax.fill_between(xm, hm, y2=hm.max(), color='0.9', zorder=3)

    ax.plot(x, [offset] * len(h), color=color, marker='v',
            alpha=0.5, zorder=5)
    ax.set_xlabel('Cross-shore distance [km]', fontsize=fontsize)
    ax.set_ylabel('Depth [m]', fontsize=fontsize)
    ax.set_ylim(offset, hm.max())
    ax.invert_yaxis()

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.xaxis.set_tick_params(tickdir='out', labelsize=labelsize, pad=1)
    ax.yaxis.set_tick_params(tickdir='out', labelsize=labelsize, pad=1)

    if False:  # TODO: +/- Black-and-White version.
        cs = ax.contour(x, z, data, colors='grey', levels=levels,
                        extend=extend, linewidths=1., alpha=1., zorder=2)
        ax.clabel(cs, fontsize=8, colors='grey', fmt=fmt, zorder=1)
        cb = None
    if 1:  # Color version.
        cs = ax.contourf(x, z, data, cmap=cmap, levels=levels, alpha=1.,
                         extend=extend, zorder=2)  # manual=True
        # Colorbar.
        cb = fig.colorbar(mappable=cs, ax=ax, orientation='vertical',
                          aspect=aspect, shrink=shrink, fraction=fraction,
                          pad=pad)
    return fig, ax, cb


# DataFrame classmethods.
@classmethod
def from_fsi(cls, fname, compression=None, skiprows=9):
    r"""Read FSI CTD ASCII columns."""
    cast = read_table(fname, header='infer', index_col=None, dtype=np.float_,
                      compression=compression, skiprows=skiprows,
                      delim_whitespace=True)

    cast.set_index('PRES', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'
    cast.name = basename(fname)[0]
    return cast


@classmethod
def from_edf(cls, fname, compression=None):
    r"""Parse XBT EDF ASCII file."""
    f = read_file(fname, compression=compression)
    header, names = [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if line.startswith('Serial Number'):
            serial = line.strip().split(':')[1].strip()
        elif line.startswith('Latitude'):
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split(':')[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == 'S':
                lat = -(lat[0] + lat[1] / 60.)
            elif hemisphere == 'N':
                lat = lat[0] + lat[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        elif line.startswith('Longitude'):
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split(':')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError("Longitude not recognized.")
        else:
            header.append(line)
            if line.startswith('Field'):
                col, unit = [l.strip().lower() for l in line.split(':')]
                names.append(unit.split()[0])
        if line == '// Data':
            skiprows = k + 1
            break

    f.seek(0)
    cast = read_table(f, header=None, index_col=None, names=names,
                      skiprows=skiprows, dtype=np.float_,
                      delim_whitespace=True)
    f.close()

    cast.set_index('depth', drop=True, inplace=True)
    cast.index.name = 'Depth [m]'
    # FIXME: Try metadata class.
    cast.header = header
    cast.lon = lon
    cast.lat = lat
    cast.serial = serial
    cast.name = basename(fname)[0]

    return cast


@classmethod
def from_cnv(cls, fname, compression=None, blfile=None):
    r"""Read ASCII CTD file from Seabird model cnv file.
    Return two DataFrames with up/down casts."""

    f = read_file(fname, compression=compression)
    header, config, names = [], [], []
    for k, line in enumerate(f.readlines()):
        line = line.strip()
        if '# name' in line:  # Get columns names.
            name, unit = line.split('=')[1].split(':')
            name, unit = map(normalize_names, (name, unit))
            names.append(name)
        if line.startswith('*'):  # Get header.
            header.append(line)
        if line.startswith('#'):  # Get configuration file.
            config.append(line)
        if 'NMEA Latitude' in line:
            hemisphere = line[-1]
            lat = line.strip(hemisphere).split('=')[1].strip()
            lat = np.float_(lat.split())
            if hemisphere == 'S':
                lat = -(lat[0] + lat[1] / 60.)
            elif hemisphere == 'N':
                lat = lat[0] + lat[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        if 'NMEA Longitude' in line:
            hemisphere = line[-1]
            lon = line.strip(hemisphere).split('=')[1].strip()
            lon = np.float_(lon.split())
            if hemisphere == 'W':
                lon = -(lon[0] + lon[1] / 60.)
            elif hemisphere == 'E':
                lon = lon[0] + lon[1] / 60.
            else:
                raise ValueError("Latitude not recognized.")
        if line == '*END*':  # Get end of header.
            skiprows = k + 1
            break

    f.seek(0)
    cast = read_table(f, header=None, index_col=None, names=names,
                      skiprows=skiprows, dtype=np.float_,
                      delim_whitespace=True)
    f.close()

    cast.set_index('prdm', drop=True, inplace=True)
    cast.index.name = 'Pressure [dbar]'

    # FIXME: Use metadata class here!
    if blfile:
        bl = load_bl(blfile)
    else:
        bl = None

    cast.bl = bl
    cast.lon = lon
    cast.lat = lat
    cast.header = header
    cast.config = config
    cast.name = basename(fname)[0]
    if 'pumps' in cast.columns:
        cast['pumps'] = np.bool_(cast['pumps'])
    if 'flag' in cast.columns:
        cast['flag'] = np.bool_(cast['flag'])
    return cast


def plot(self, **kwds):
    r"""Plot a CTD variable against the index (pressure or depth)."""
    fig, ax = plt.subplots()
    ax.plot(self.values, self.index, **kwds)
    ax.set_ylabel(self.index.name)
    ax.set_xlabel(self.name)
    ax.invert_yaxis()
    offset = 0.01
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
    return fig, ax


def swap_index(self, keys):
    r"""Swap index with a columns in the DataFrame and return the index as a
    new column."""
    data = self.copy()
    if not data.index.name:
        data.index.name = "index"
    data[data.index.name] = data.index.float_()
    return data.set_index(keys, drop=True, append=False, verify_integrity=True)


def plot_vars(self, variables=None, **kwds):
    r"""Plot CTD temperature and salinity."""
    # TODO: pop kw like in `section`.

    fig = plt.figure(figsize=(8, 10))
    ax0 = host_subplot(111, axes_class=AA.Axes)
    ax1 = ax0.twiny()

    # Axis location.
    host_new_axis = ax0.get_grid_helper().new_fixed_axis
    ax0.axis["bottom"] = host_new_axis(loc="top", axes=ax0, offset=(0, 0))
    par_new_axis = ax1.get_grid_helper().new_fixed_axis
    ax1.axis["top"] = par_new_axis(loc="bottom", axes=ax1, offset=(0, 0))

    ax0.plot(self[variables[0]], self.index, 'r.', label='Temperature')
    ax1.plot(self[variables[1]], self.index, 'b.', label='Salinity')

    ax0.set_ylabel("Pressure [dbar]")
    ax0.set_xlabel("Temperature [%sC]" % degree)
    ax1.set_xlabel("Salinity [kg g$^{-1}$]")
    ax1.invert_yaxis()

    try:  # FIXME with metadata.
        fig.suptitle(r"Station %s profile" % self.name)
    except AttributeError:
        pass

    ax0.legend(shadow=True, fancybox=True,
               numpoints=1, loc='lower right')

    offset = 0.01
    x1, x2 = ax0.get_xlim()[0] - offset, ax0.get_xlim()[1] + offset
    ax0.set_xlim(x1, x2)

    offset = 0.01
    x1, x2 = ax1.get_xlim()[0] - offset, ax1.get_xlim()[1] + offset
    ax1.set_xlim(x1, x2)

    return fig, (ax0, ax1)

Index.asof = asof
Index.float_ = float_

Series.plot = plot
Series.split = split
Series.smooth = smooth
Series.despike = despike
Series.bindata = bindata
Series.press_check = press_check

DataFrame.split = split
DataFrame.from_cnv = from_cnv
DataFrame.from_edf = from_edf
DataFrame.from_fsi = from_fsi
DataFrame.plot_vars = plot_vars
DataFrame.swap_index = swap_index
DataFrame.press_check = press_check
DataFrame.get_maxdepth = get_maxdepth
DataFrame.plot_section = plot_section

if __name__ == '__main__':
    import doctest
    doctest.testmod()