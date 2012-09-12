# -*- coding: utf-8 -*-
#
# cruise.py
#
# purpose:  Calculate cruise time and other handy functions.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  05-Sep-2012
# modified: Wed 12 Sep 2012 11:58:05 AM BRT
#
# obs:
#

import gsw
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from oceans.datasets import get_depth


def cruise_time(lon, lat, vel=8):
    r"""Compute the time it takes to navigate all the stations.
    Assumes cruise velocity even though it is a bad assumption!"""
    dist = gsw.distance(lon, lat, p=0)
    vel *= 0.514444  # Convert to meters per seconds.
    return np.sum(dist / vel)


def get_cruise_time(m, vel=8, times=1):
    r"""Click on two points of the Basemap object `m` to compute the
    cruise time at the velocity `vel` in nots (default=8 nots)."""

    print("Click on the first and last point of navigation for %s sections." %
          times)

    total = []
    while times:
        points = np.array(fig.ginput(n=2))
        lon, lat = m(points[:, 0], points[:, 1], inverse=True)
        total.append(cruise_time(lon, lat, vel=vel))
        times -= 1

    return np.sum(total)


class Transec(object):
    r"""Container class to store oceanographic transect.
    Info (`lon`, `lat`, `depth`)."""

    def __init__(self, lon=None, lat=None, depth=None):
        lon, lat, depth = map(np.asanyarray, (lon, lat, depth))

        if not depth.all():
            print("Depth not provided, getting depth from etopo2.")
            depth = get_depth(lon, lat)
        self.lon = lon
        self.lat = lat
        self.depth = depth

    def station_time(self, ctdvel=1.):
        r"""Compute the time that it takes for each oceanographic station in
        the transect.  `ctdvel` is the ctd velocity.
        Default velocity is 1 meters per second."""

        # Time in seconds times two (up-/downcast).
        return np.sum(np.abs(self.depth) / ctdvel * 2)

    def navigation_time(self, vel=8):
        r"""Compute the time it takes to navigate between stations."""
        return cruise_time(self.lon, self.lat, vel=vel)

    def transect_time(self):
        r"""Compute the time it takes to complete the transect in days."""
        return (self.station_time() + self.navigation_time())


if __name__ == '__main__':
    # Load stations positions from Abrolhos II.
    lond, lonm, latd, latm, prof = np.loadtxt('grade_abrolhosII.dat',
                                              unpack=True)
    lon, lat = lond + lonm / 60., latd + latm / 60.

    # Load new positions.
    lon_02, lat_02 = np.loadtxt('ambes_extra_transect.dat', unpack=True)
    depths_02 = get_depth(lon_02, lat_02)

    lon_06, lat_06 = np.loadtxt('ambes_transec4_complement.dat', unpack=True)
    depths_06 = get_depth(lon_06, lat_06)

    # Create the figure.
    lonStart, lonEnd, latStart, latEnd = -47, -30.0, -23.0, -17.0
    m = Basemap(projection='merc', llcrnrlon=-59.0, urcrnrlon=-25.0,
                llcrnrlat=-38.0, urcrnrlat=9.0, lat_ts=20, resolution='i')
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='w')
    m.ax = ax
    image = plt.imread('../../../Nautical_chart_01/chart_brazil_small.png')
    m.imshow(image, origin='upper')
    lon_lim, lat_lim = m([lonStart, lonEnd], [latStart, latEnd])
    m.ax.axis([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])

    # Plotting the cruise stations.
    kw = dict(marker='o', linestyle='none', markersize=10,
              markeredgecolor='k', latlon=True)

    m.plot(lon, lat, markerfacecolor='r', label='Abrolhos II', **kw)
    m.plot(lon_02, lat_02, markerfacecolor='g', label='Transec 02', **kw)
    m.plot(lon_06, lat_06, markerfacecolor='b', label='New positions', **kw)

    plt.show()

    # Create transect objects.
    tran_00 = Transec(lon[:11], lat[:11], prof[:11])
    tran_01 = Transec(lon[11:25], lat[11:25], prof[11:25])
    tran_02 = Transec(lon_02, lat_02, depths_02)
    tran_03 = Transec(lon[25:51], lat[25:51], prof[25:51])
    tran_04 = Transec(lon[51:62], lat[51:62], prof[51:62])
    tran_05 = Transec(lon[62:78], lat[62:78], prof[62:78])
    tran_06 = Transec(np.r_[lon[78:], lon_06],
                      np.r_[lat[78:], lat_06],
                      np.r_[prof[78:], depths_06])

    # Compute the time.
    total_transect_time = (tran_00.transect_time() + tran_01.transect_time() +
                           tran_02.transect_time() + tran_03.transect_time() +
                           tran_04.transect_time() + tran_05.transect_time() +
                           tran_06.transect_time())

    total_cruise_time = get_cruise_time(m, vel=8, times=6)
    total_time = (total_transect_time + total_cruise_time) / 60. / 60. / 24.
    total_time += total_time * 0.2  # Add 20 % "buffer" time.
    print("Total cruise time is %s" % (total_time))
