# -*- coding: utf-8 -*-
#
# cruise.py
#
# purpose:  Calculate cruise time and other handy functions.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  05-Sep-2012
# modified: Wed 30 Jan 2013 02:10:50 PM BRST
#
# obs:
#

import gsw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from oceans.datasets import get_depth
from matplotlib.ticker import MultipleLocator
from oceans.ff_tools import cart2pol, pol2cart

__all__ = [
           'Transect',
           'Chart',
           'draw_arrow',
           'get_cruise_time',
           'make_radial',
           'create_transect',
           ]


#TODO: Monkey patch a DataFrame for this.
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

    #FIXME:
    def station_time_ctd(self, ctdvel=1., prep=1800.):
        r"""Time it takes for each oceanographic station in
        the transect.  `ctdvel` is the ctd velocity.
        Default velocity is 1 meters per second.
        NOTE: 30 min preparations if using LADCP."""

        # Time in seconds times two (up-/downcast).
        depth = np.abs(self.depth)
        depth[depth < 200.] = 200.  # Avoid etopo bogus depths.
        buffer_ctd = len(depth) * prep
        return np.sum((depth * 2) / ctdvel) + buffer_ctd

    def station_time(self):
        r"""
        30 min before slope.
        1 h at the slope.
        2 h ocean floor."""

        depth = np.abs(self.depth)

        coast = depth < 100
        slope = np.logical_and(depth >= 100, depth < 1000)
        basin = np.logical_and(depth >= 1000, depth < 2000)
        deep = depth >= 2000

        coast = 1800 * len(coast.nonzero()[0])
        slope = 3600 * len(slope.nonzero()[0])
        basin = 7200 * len(basin.nonzero()[0])
        deep = 14400 * len(deep.nonzero()[0])
        return coast + slope + basin + deep

    def cruise_stations_time(self, vel=7):
        r"""Compute the time it takes to navigate all the stations.
        Assumes cruise velocity even though it is a bad assumption!
        Enter the velocity in nots."""
        dist = self.distances()
        vel *= 0.514444  # Convert to meters per seconds.
        return np.sum(dist / vel)

    def distances(self):
        r"""Compute distances between stations."""
        return gsw.distance(self.lon, self.lat, p=0)

    def save_csv(self, fname):
        r"""Save the radial as a Comma Separated Value file."""
        np.savetxt(fname, np.c_[self.lon, self.lat, np.abs(self.depth)],
                   header='longitude,latitude,depth', comments='',
                   fmt='%3.8f,%3.8f,%i')


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
        self.ax.imshow(self.image, extent=self.window, origin='upper')
        self.update_aspect()
        self.update_ticks()

        return self.fig, self.ax


def draw_arrow(m, points, **kwargs):
    x1, y1 = points[:, 0][0], points[:, 1][0]
    x2, y2 = points[:, 0][1], points[:, 1][1]
    dx, dy = x2 - x1, y2 - y1
    arrow = m.ax.arrow(x1, y1, dx, dy, **kwargs)
    plt.draw()
    return arrow


def get_cruise_time(fig, m, vel=7, times=1):
    r"""Click on two points of the Basemap object `m` to compute the
    cruise time at the velocity `vel` in nots (default=7 nots)."""

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


def _mid_point(lon, lat):
    r"""Returns the mid-point between an array of positions [lon, lat]."""
    lonc = (lon[1:] + lon[0:-1]) / 2.
    latc = (lat[1:] + lat[0:-1]) / 2.
    return lonc, latc


def _make_line(lon, lat, rossby, tfile, fraction=0.5):
    r"""Create a lon, lat station line with stations separated by a
    `fraction` of the `rossby` radius."""
    degree2km = 111.  # Degree to km.
    rossby_rd = rossby / degree2km
    rossby_rd *= fraction

    dx, dy = lon[-1] - lon[0], lat[-1] - lat[0]
    th, rd = cart2pol(dx, dy, units='deg')

    radii = np.arange(rossby_rd, rd, rossby_rd)
    lon_l, lat_l = [], []
    for r in radii:
        dx, dy = pol2cart(th, r)
        lon_l.append(lon[0] + dx)
        lat_l.append(lat[0] + dy)
    lon, lat = map(np.array, (lon_l, lat_l))
    depth = np.int_(-np.fix(get_depth(lon, lat, tfile=tfile)))
    return lon, lat, depth


def make_radial(lonStart, lonEnd, latStart, latEnd, tfile=None, rossby=25.):
    r"""Enter first and last point of a radial and the Rossby Radius."""
    # First guess is every 1 (0.17) minute (degree) (etopo1 resolution).
    lon = len(np.arange(lonStart, lonEnd, 0.017))
    lat = len(np.arange(latStart, latEnd, 0.017))
    if lon > lat:
        length = lon
    elif lat > lon:
        length = lat
    else:
        raise ValueError("Could not get a valid length (lon = %s, lat = %s)" %
                         (lon, lat))

    lon = np.linspace(lonStart, lonEnd, length)
    lat = np.linspace(latStart, latEnd, length)
    depth = -np.fix(get_depth(lon, lat, tfile=tfile))

    # Eliminate spurious depths.
    mask = depth > 0
    lon, lat, depth = lon[mask], lat[mask], depth[mask]

    # Shelf.
    if 0:  # FIXME: etopo1 sucks at the continental shelf.
        mask = depth <= 100.
        lonc, latc, depthc = lon[mask], lat[mask], depth[mask]
        coast = np.abs(np.diff(depthc)) >= 20.
        indices = np.zeros_like(coast)
        idx = np.where(coast)[0]
        for k in idx:
            indices[k] = True
            indices[k + 1] = True
        # Last one is always True.
        indices[-1] = True
        lon_c, lat_c, depth_c = lon[indices], lat[indices], depth[indices]

    # Slope.
    mask = np.logical_and(depth > 100., depth <= 1000.)
    first = np.where(mask)[0][0]
    last = np.where(mask)[0][-1]
    lon_t, lat_t = (lon[first], lon[last]), (lat[first], lat[last])

    # Half Rossby radius at the Slope.
    lon_t, lat_t, depth_t = _make_line(lon_t, lat_t, tfile=tfile, rossby=25.,
                                      fraction=0.5)

    # Deep ocean.
    mask = depth > 1000.
    first = np.where(mask)[0][0]
    last = np.where(mask)[0][-1]
    lon_d, lat_d = (lon[first], lon[last]), (lat[first], lat[last])
    # One Rossby radius after Slope.
    lon_d, lat_d, depth_d = _make_line(lon_d, lat_d, tfile=tfile, rossby=25.,
                                      fraction=1)

    lon = np.r_[lon_t, lon_d]
    lat = np.r_[lat_t, lat_d]
    depth = np.r_[depth_t, depth_d]

    return lon, lat, depth


def create_transect(ax, points, tfile='dap'):
    lonStart, lonEnd = points[0][0], points[1][0]
    latStart, latEnd = points[0][1], points[1][1]
    lon, lat, depth = make_radial(lonStart, lonEnd, latStart, latEnd,
                                  tfile=tfile, rossby=25.)

    # Split CTDs and XBTs.
    mask = depth < 1000.
    lon_ctd, lat_ctd, depth_ctd = lon[mask], lat[mask], depth[mask]

    mask = depth >= 1000.
    # Last point is always CTD+XBT.
    if len(lon) % 2:
        lon_xbt = np.r_[lon[mask][1::2], lon[-1]]
        lat_xbt = np.r_[lat[mask][1::2], lat[-1]]
        #depth_xbt = np.r_[depth[mask][1::2], depth[-1]]
        lon_ctd = np.r_[lon_ctd, lon[mask][0::2]]
        lat_ctd = np.r_[lat_ctd, lat[mask][0::2]]
        depth_ctd = np.r_[depth_ctd, depth[mask][0::2]]
    else:
        lon_xbt = np.r_[lon[mask][1::2]]
        lat_xbt = np.r_[lat[mask][1::2]]
        #depth_xbt = np.r_[depth[mask][1::2]]
        lon_ctd = np.r_[lon_ctd, lon[mask][0::2], lon[-1]]
        lat_ctd = np.r_[lat_ctd, lat[mask][0::2], lat[-1]]
        depth_ctd = np.r_[depth_ctd, depth[mask][0::2], depth[-1]]

    ax.plot(lon_xbt, lat_xbt, 'rd', alpha=0.4)
    ax.plot(lon_ctd, lat_ctd, 'k.')

    dist = np.int_(gsw.distance(lon, lat))[0] / 1e3
    lonc, latc = _mid_point(lon, lat)
    for k, text in enumerate(dist):
        ax.text(lonc[k], latc[k], str(np.int(text)))

    plt.draw()
    return Transect(lon, lat, depth)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
