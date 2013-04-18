# -*- coding: utf-8 -*-
#
# cruise.py
#
# purpose:  Calculate cruise time and other handy functions.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  05-Sep-2012
# modified: Thu 18 Apr 2013 12:22:21 PM BRT
#
# obs:
#

from __future__ import division

import gsw
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from oceans.datasets import get_depth
from oceans.ff_tools import cart2pol, pol2cart


__all__ = [
           'DataFrame',  # A DataFrame "patched" for radials.
           'make_line',
           'make_transect',
           'filter_station',
           'split_stations',
           'get_cruise_time',
           'transect2dataframe',
           ]


"""Utilities."""


def transect2dataframe(lon, lat, depth, sort='lat'):
    """Creates a sorted Transect-DataFrame from lon, lat and depth columns of
    data.

    Example
    -------
    >>> lon = [-40.223, -40.267, -40.304, -40.538, -40.787]
    >>> lat = [-21.499, -21.484, -21.471, -21.397, -21.318]
    >>> depth = [135, 61, 50, 25, 20]
    >>> transect2dataframe(lon, lat, depth)
    """
    transect = DataFrame(np.c_[lon, lat, depth],
                         columns=['lon', 'lat', 'depth'])

    # Sort by longitude first then by latitude.
    transect.sort(columns=['lon', 'lat'], ascending=[False, True],
                  inplace=True)
    transect.index = np.arange(len(lon)) + 1
    return transect


def make_line(start, end, spacing=20):
    r"""Create a stations line with stations separated by a
    fixed `spacing` [km] from a `start` and `end` points.  Returns two
    vectors (lon, lat).  The distance is usually a fraction of the Rossby
    radius)."""
    degree2km = 111.  # Degree to km.
    spacing = spacing / degree2km

    dx, dy = np.diff(np.c_[start, end]).squeeze()
    th, dist = cart2pol(dx, dy, units='deg')

    radii = np.arange(spacing, dist, spacing)
    x, y = [], []
    for r in radii:
        dx, dy = pol2cart(th, r)
        x.append(start[0] + dx)
        y.append(start[1] + dy)
    lon, lat = map(np.array, (x, y))
    return lon, lat


def make_transect(start, end, tfile='dap', rossby=25.):
    r"""Enter first and last point of a radial and the Rossby Radius [km].
    Returns a transect with stations space 0.5*rossby at the slope and rossby
    at after.  NOTE: This is should not be used for coastal planning!"""

    # First guess is every 1 (0.17) minute (degree) (etopo1 resolution).
    x_len = len(np.arange(start[0], end[0], 0.017))
    y_len = len(np.arange(start[1], end[1], 0.017))
    length = x_len if x_len > y_len else y_len
    if length <= 0:
        raise ValueError("Could not get a valid length (lon = %s, lat = %s)" %
                         (x_len, y_len))

    lon = np.linspace(start[0], end[0], length)
    lat = np.linspace(start[1], end[1], length)
    depth = -np.fix(get_depth(lon, lat, tfile=tfile))

    mask = depth > 0  # Eliminate land points.
    lon, lat, depth = lon[mask], lat[mask], depth[mask]

    mask = np.logical_and(depth > 100., depth <= 1000.)  # Slope.
    first, last = np.where(mask)[0][0], np.where(mask)[0][-1]
    start, end = (lon[first], lat[first]), (lon[last], lat[last])
    # Half Rossby radius at the Slope.
    lon_slope, lat_slope = make_line(start, end, spacing=rossby / 2.)

    mask = depth > 1000.  # Deep ocean.
    first, last = np.where(mask)[0][0], np.where(mask)[0][-1]
    start, end = (lon[first], lat[first]), (lon[last], lat[last])
    # One Rossby radius after Slope.
    lon_deep, lat_deep = make_line(start, end, spacing=rossby)

    lon = np.r_[lon_slope, lon_deep]
    lat = np.r_[lat_slope, lat_deep]
    depth = -np.fix(get_depth(lon, lat, tfile=tfile))
    depth[depth < 0] = 0

    # NOTE: Eliminate station that are closer than 0.5 * Rossby radius.
    idx = lon.argsort()
    lon, lat = lon[idx], lat[idx]
    dist = gsw.distance(lon, lat, p=0).squeeze() / 1e3 >= rossby / 2.5
    mask = np.r_[True, dist]
    return transect2dataframe(lon[mask], lat[mask], depth[mask])


def split_stations(df):
    r"""Enter a transect DataFrame.  Return a transect with stations split
    between CTDs and XBTs."""

    df_new = df.copy()
    df_new['Type'] = ['CTD'] * len(df_new)  # Start type column as CTDs.

    # Water samples are collected at all CTD stations after 500 meters depth.
    mask = np.logical_and(df_new.depth >= 500., df_new['Type'] == 'CTD')
    df_new['Type'][mask] = 'CTD/Water'

    # Add a XBT stations every other CTD station after 1000 meters depth.
    mask = df_new.depth > 1000.
    idx = df_new.index[mask][1::2]
    df_new['Type'].ix[idx] = 'XBT'

    # Last point is always 'everything'.
    df_new['Type'][df_new.irow(-1).name] = 'CTD/XBT/Water'
    return df_new


def filter_station(df, st_type='CTD'):
    mask = [st_type in st for st in df['Type']]
    return df[mask]


"""Transect DataFrame."""


def ctd_cast_time(self, method="RuleOfThumb", ctdvel=1., preparation=1800.):
    r"""Time it takes for each oceanographic station in the transect.

    method: RuleOfThumb, Simple rule based on the depth of the water column.
            CTDTime, time it actually takes to lower and retrieve the CTD.

    If one chooses RuleOfThumb the following times are used:
        30 min before slope [< 100 m]
        1 h at the slope [> 100 m and < 1000 m]
        2 h ocean floor [> 1000 m and < 2000 m]
        4 h for depths > 2000 m

    If one chooses the CTDTime you can tweak the following keywords:
    `ctdvel`: The ctd velocity.
    `preparation`: A "buffer" time.
    Default velocity is 1 meters per second.
    NOTE: Use 30 min preparations if using LADCP."""

    depth = np.abs(self.depth)
    if method == "depth":
        # Time in seconds times two (up-/downcast).
        depth[depth < 200] = 200  # Avoid etopo bogus depths.
        buffer_ctd = len(depth) * preparation
        time = ((depth * 2) / ctdvel) + buffer_ctd
    elif method == "RuleOfThumb":
        coast = depth < 100
        slope = np.logical_and(depth >= 100, depth < 1000)
        floor = np.logical_and(depth >= 1000, depth < 2000)
        deep = depth >= 2000

        coast = 1800 * coast
        slope = 3600 * slope
        floor = 7200 * floor
        deep = 14400 * deep
        time = coast + slope + floor + deep
    else:
        raise TypeError("Unrecognized method option %s" % method)

    return time


def distances(self):
    r"""Compute distances between stations."""
    dist = np.r_[0, gsw.distance(self.lon, self.lat, p=0).squeeze()]
    return Series(np.fix(dist), index=self.index)


def navigation_time(self, vel=7):
    r"""Compute the time it takes to navigate all the stations.
    Assumes cruise velocity even though it is a bad assumption!
    Enter the velocity in knots."""
    dist = self.distances()
    vel *= 1852 / (60 * 60)  # Convert knots to meters per seconds.
    return np.sum(dist / vel)


def mid_point(self):
    r"""Returns the mid-point between an array of positions [lon, lat]."""
    lonc = (self.lon[1:].values + self.lon[0:-1]).values / 2.
    latc = (self.lat[1:].values + self.lat[0:-1]).values / 2.
    return lonc, latc


def deg2degmin(self):
    r"""Convert Degrees to Degrees and minutes."""
    df = self.copy()
    conv = lambda x: '%i %.6f' % (np.fix(x), 60 * np.remainder(x, 1))
    try:
        df.lon = [conv(x) for x in df.lon]
        df.lat = [conv(y) for y in df.lat]
    except TypeError:
        warnings.warn("Already in deg2degmin format.")
    return df


def degmin2deg(self):
    r"""Convert Degrees and minutes to Degrees."""
    df = self.copy()
    conv = lambda x: float(x.split()[0]) + float(x.split()[1]) / 60
    try:
        df.lon = [conv(x) for x in df.lon]
        df.lat = [conv(y) for y in df.lat]
    except AttributeError:
        warnings.warn("Already in degmin2deg format.")
    return df


"""Figure tools."""


def ginput(fig, m, **kw):
    points = np.array(fig.ginput(**kw))
    x, y = np.array(points)[:, 0], np.array(points)[:, 1]
    return m(x, y, inverse=True)


def _draw_arrow(m, points, **kw):
    color = kw.pop('color', 'k')
    zorder = kw.pop('zorder', 10)
    alpha = kw.pop('alpha', 0.85)
    shape = kw.pop('shape', 'full')
    width = kw.pop('width', 2500.)
    overhang = kw.pop('overhang', 0)
    x1, y1 = points[:, 0][0], points[:, 1][0]
    x2, y2 = points[:, 0][1], points[:, 1][1]
    dx, dy = x2 - x1, y2 - y1
    arrow = m.ax.arrow(x1, y1, dx, dy, color=color, zorder=zorder, alpha=alpha,
                       shape=shape, width=width, overhang=overhang)
    plt.draw()
    return arrow


def get_cruise_time(fig, m, vel=7, times=1, **kw):
    r"""Click on two points of the Basemap object `m` to compute the
    cruise time at the velocity `vel` in knots (default=7 knots)."""

    vel *= 0.514444  # Convert to meters per seconds.
    print("Click on the first and last point of navigation for %s sections." %
          times)

    total = []
    while times:
        points = np.array(fig.ginput(n=2))
        _draw_arrow(m, points, **kw)
        lon, lat = m(points[:, 0], points[:, 1], inverse=True)
        dist = gsw.distance(lon, lat, p=0)
        time = np.sum(dist / vel)
        total.append(time)
        times -= 1

    return np.sum(total)


DataFrame.distances = distances
DataFrame.mid_point = mid_point
DataFrame.deg2degmin = deg2degmin
DataFrame.degmin2deg = degmin2deg
DataFrame.ctd_cast_time = ctd_cast_time
DataFrame.navigation_time = navigation_time

if __name__ == '__main__':
    from mpl_toolkits.basemap import Basemap

    start, end = (-40.84, -22.18), (-38.54, -22.20)  # "Normal."
    #start, end = (-39., -19.), (-36., -20.5)  # Several bumps.
    #start, end = (-37.467, -17.8819), (-35.9307, -17.9133)  # Hot spur.

    if 1:
        lon, lat = make_line(start, end, spacing=20)
        depth = -np.fix(get_depth(lon, lat, tfile='dap'))
        dist = gsw.distance(lon, lat, p=0).squeeze() / 1e3
        transect = transect2dataframe(lon, lat, depth)

        lonc, latc = transect.mid_point()
        # NOTE: Figure with stations and depth profile.
        fig, (ax0, ax1) = plt.subplots(nrows=2)
        ax0.plot(lon, lat, 'ko', alpha=0.6)
        # NOTE: Rounding distance just to get a cleaner graph.
        [ax0.text(x, y, str(np.int_(text))) for
         x, y, text in zip(lonc, latc, dist)]
        ax1.plot(np.r_[0, dist].cumsum(), depth)
        ax1.invert_yaxis()

    # Test Transect.
    transect = make_transect(start, end, tfile='dap', rossby=25.)
    radial = split_stations(transect)
    print(transect.deg2degmin().degmin2deg())

    secs2hours = 60. * 60.
    secs2days = 60. * 60. * 24.
    print("\nTrack distances:\n%s" % transect.distances())
    print("\nCTD time (rule of thumb):\n%s" % transect.ctd_cast_time())
    print("\nCTD time (depth):\n%s" % transect.ctd_cast_time(method="depth"))
    print("\nTotal CTD time %s hours" %
          (transect.ctd_cast_time().sum() / secs2hours))
    print("\nNavigation time: %s days" %
          (transect.navigation_time() / secs2days))
    print("\nTotal transect time: %s days" %
          ((transect.ctd_cast_time().sum() +
            transect.navigation_time()) / secs2days))

    # Test Figure tools.
    def make_map(lonStart=-43., lonEnd=-34., latStart=-22.5, latEnd=-17.):
        m = Basemap(projection='merc', llcrnrlon=-59.0, urcrnrlon=-25.0,
                    llcrnrlat=-38.0, urcrnrlat=9.0, lat_ts=20, resolution='c')

        # Create the figure and attach the axis.
        fig, ax = plt.subplots(figsize=(20, 20), facecolor='w')
        m.ax = ax

        m.imshow(plt.imread('chart_brazil.png'), origin='upper', alpha=0.5)

        lon_lim, lat_lim = m([lonStart, lonEnd], [latStart, latEnd])
        m.ax.axis([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])

        parallels = np.arange(latStart,  latEnd, 2)
        meridians = np.arange(lonStart, lonEnd, 2)
        xoffset = -lon_lim[0] + lonStart + 1e4
        yoffset = -lat_lim[0] + latStart + 1e4
        kw = dict(linewidth=0, fontsize=14, fontweight='demibold')
        m.drawparallels(parallels, xoffset=xoffset, labels=[1, 0, 0, 0], **kw)
        m.drawmeridians(meridians, yoffset=yoffset, labels=[0, 0, 0, 1], **kw)
        plt.draw()
        return fig, ax, m

    if 0:
        fig, ax, m = make_map()
        ctd = filter_station(radial, st_type='CTD')
        xbt = filter_station(radial, st_type='XBT')
        water = filter_station(radial, st_type='Water')
        ctd_xbt = filter_station(radial, st_type='CTD/XBT')

        kw = dict(linestyle='none', markersize=8, marker='d')
        m.plot(*m(ctd.lon, ctd.lat), label='%2i CTD' % len(ctd),
               markerfacecolor='r', markeredgecolor='w', zorder=2, **kw)

        m.plot(*m(xbt.lon, xbt.lat), label='%2i XBT' % len(xbt),
               markerfacecolor='g', markeredgecolor='w', zorder=2, **kw)

        m.plot(*m(water.lon, water.lat), label='%2i Water' % len(water),
               markerfacecolor='b', markeredgecolor='w', zorder=2, **kw)

        m.ax.legend(numpoints=1, fancybox=True, shadow=True, loc='upper left')

        m.plot(*m(ctd_xbt.lon, ctd_xbt.lat), markerfacecolor='g',
               markerfacecoloralt='r', fillstyle='top', markeredgecolor='w',
               zorder=3, **kw)

        m.plot(*m(water.lon, water.lat), markerfacecolor='b',
               markerfacecoloralt='r', fillstyle='top', markeredgecolor='w',
               zorder=3, **kw)

        raw_input("""\nClick at the first and the last station to compare with
        the method transect.navigation_time().  Press Enter when ready.""")
        cruise = get_cruise_time(fig, m, vel=7, times=1, alpha=0.5)
        print("\nCruise time: %s days\nNavigation time: %s" %
            (cruise / secs2days, transect.navigation_time() / secs2days))

        raw_input("""\nNow do an actual cruise plan by clicking in a route
        port-first station-last station-port.  Press Enter when ready.""")
        cruise = get_cruise_time(fig, m, vel=7, times=2, color='r')
        print("Cruise time: %s days" % (cruise / secs2days))
