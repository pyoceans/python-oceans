# -*- coding: utf-8 -*-
#
# cruise.py
#
# purpose:  Calculate cruise time and other handy functions.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  05-Sep-2012
# modified: Thu 13 Sep 2012 11:05:32 AM BRT
#
# obs:
#

import gsw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from oceans.datasets import get_depth
from matplotlib.ticker import MultipleLocator


def draw_arrow(m, points, **kwargs):
    x1, y1 = points[:, 0][0], points[:, 1][0]
    x2, y2 = points[:, 0][1], points[:, 1][1]
    dx, dy = x2 - x1, y2 - y1
    return m.ax.arrow(x1, y1, dx, dy, **kwargs)

def get_cruise_time(fig, m, vel=8, times=1):
    r"""Click on two points of the Basemap object `m` to compute the
    cruise time at the velocity `vel` in nots (default=8 nots)."""

    vel *= 0.514444  # Convert to meters per seconds.
    print("Click on the first and last point of navigation for %s sections." %
          times)

    total = []
    while times:
        points = np.array(fig.ginput(n=2))
        draw_arrow(m, points, width=2500., color='k', shape='full', overhang=0,
                   alpha=0.85, zorder=10)
        lon, lat = m(points[:, 0], points[:, 1], inverse=True)
        dist = gsw.distance(lon, lat, p=0)
        time = np.sum(dist / vel)
        total.append(time)
        times -= 1

    return np.sum(total)


class Transect(object):
    r"""Container class to store oceanographic transect.
    Info (`lon`, `lat`, `depth`)."""

    def __init__(self, lon=None, lat=None, depth=None):
        lon, lat, depth = map(np.asanyarray, (lon, lat, depth))

        if not depth.all():
            print("Depth not provided, getting depth from etopo2.")
            depth = get_depth(lon, lat)

        # Sort by longitude
        # FIXME: Maybe that is not the best option.
        # The ideal should be sorted by distance from the coast.
        sort = lon.argsort()
        self.lon = lon[sort]
        self.lat = lat[sort]
        self.depth = depth[sort]

    def station_time(self, ctdvel=1.):
        r"""Time it takes for each oceanographic station in
        the transect.  `ctdvel` is the ctd velocity.
        Default velocity is 1 meters per second."""

        # Time in seconds times two (up-/downcast).
        return np.sum(np.abs(self.depth) / ctdvel * 2)

    def cruise_stations_time(self, vel=8):
        r"""Compute the time it takes to navigate all the stations.
        Assumes cruise velocity even though it is a bad assumption!
        Enter the velocity in nots."""
        dist = self.distances()
        vel *= 0.514444  # Convert to meters per seconds.
        return np.sum(dist / vel)

    def transect_time(self):
        r"""Compute the time it takes to complete the transect in days."""
        return (self.station_time() + self.cruise_stations_time())

    def distances(self):
        r"""Compute distances between stations."""
        return gsw.distance(self.lon, self.lat, p=0)


class Chart(object):
    r"""Geo-reference a raster nautical chart."""
    def __init__(self, image='cadeia_vitoria_trindade.png',
                 window=[-47., -14., -24., -3.],  # Chart 20
                 lon_tick_interval=2.0 / 60.0,
                 lat_tick_interval=2.0 / 60.0,
                 **kw):
        r"""Enter a the window corners as:
        window=[lower left lon, upper right lon,
                lower left lat, upper right lat]
        And the lon_tick_interval, lat_tick_interval tick intervals.

        Example
        -------
        >>> chart = Chart(image='cadeia_vitoria_trindade.png')
        >>> fig, ax = chart.plot()
        >>> ax.axis([-43., -31., -22., -16.5])
        >>> chart.update_ticks(ax)
        """

        self.kw = kw
        self.image = image
        self.window = window
        self.lon_tick_interval = lon_tick_interval
        self.lat_tick_interval = lat_tick_interval
        if self.image is not None:
            if isinstance(self.image, str):
                self.image = plt.imread(self.image)

    def deg2str(self, deg, ref='lon', fmt="%3.1f", usetex=True):
        r"""Enter number in degree and decimal degree `deg, a `ref` either lat
        or lon."""
        min = 60 * (deg - np.floor(deg))
        deg = np.floor(deg)
        if min != 0.0:
            deg += 1.0
            min -= 60.0
        if ref == 'lon':
            if deg < 0.0:
                dir = 'W'
            elif deg > 0.0:
                dir = 'E'
            else:
                dir = ''
        elif ref == 'lat':
            if deg < 0.0:
                dir = 'S'
            elif deg > 0.0:
                dir = 'N'
            else:
                dir = ''
        if rcParams['text.usetex'] and usetex:
            return (r"%d$^\circ$" + fmt + "'%s ") % (abs(deg), abs(min), dir)
        else:
            return ((u"%d\N{DEGREE SIGN}" + fmt + "'%s ") %
                    (abs(deg), abs(min), dir))

    def update_ticks(self):
        xlocator = MultipleLocator(self.lon_tick_interval)
        ylocator = MultipleLocator(self.lat_tick_interval)
        self.ax.xaxis.set_major_locator(xlocator)
        self.ax.yaxis.set_major_locator(ylocator)
        xlab = []
        for xtick in self.ax.get_xticks():
            xlab.append(self.deg2str(xtick, ref='lon'))
        self.ax.set_xticklabels(xlab)
        ylab = []
        for ytick in self.ax.get_yticks():
            ylab.append(self.deg2str(ytick, ref='lat'))
        self.ax.set_yticklabels(ylab)
        self.ax.fmt_xdata = lambda x: self.deg2str(x, ref='lon', fmt="%5.3f",
                                                   usetex=False)
        self.ax.fmt_ydata = lambda y: self.deg2str(y, ref='lat', fmt="%5.3f",
                                                   usetex=False)
        plt.draw()

    def update_aspect(self):
        aspect = 1.0 / np.cos(np.mean(self.ax.get_ylim()) * np.pi / 180.)
        self.ax.set_aspect(aspect, adjustable='box', anchor='C')
        plt.draw()

    def plot(self):
        self.fig, self.ax = plt.subplots(**self.kw)
        self.ax.imshow(self.image, extent=self.window)
        self.update_aspect()
        self.update_ticks()

        return self.fig, self.ax

if __name__ == '__main__':
    import doctest
    doctest.testmod()
