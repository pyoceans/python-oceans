# -*- coding: utf-8 -*-
#
# datasets.py
#
# purpose:  Functions to handle datasets.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Fri 05 Jul 2013 02:26:49 PM BRT
#
# obs: some Functions were based on:
# http://www.trondkristiansen.com/?page_id=1071


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from pandas import Panel4D, Panel
from netCDF4 import Dataset, num2date
from oceans.ff_tools import get_profile

# TODO get_woa profile.

__all__ = ['woa_subset',
           'etopo_subset',
           'laplace_X',
           'laplace_Y',
           'laplace_filter',
           'get_depth',
           'get_isobath']


def wrap_lon180(lon):
    lon = np.atleast_1d(lon)
    angles = np.logical_or((lon < -180), (180 < lon))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon


def wrap_lon360(lon):
    lon = np.atleast_1d(lon)
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon


def map_limits(m):
    lons, lats = wrap_lon360(m.boundarylons), m.boundarylats
    boundary = dict(llcrnrlon=min(lons),
                    urcrnrlon=max(lons),
                    llcrnrlat=min(lats),
                    urcrnrlat=max(lats))
    return boundary


def woa_subset(llcrnrlon=2.5, urcrnrlon=357.5, llcrnrlat=-87.5, urcrnrlat=87.5,
               var='temperature', clim_type='monthly', resolution='1deg',
               levels=slice(0, 40)):
    """Get World Ocean Atlas variables at a given lon, lat bounding box.
    Choose data `var` from:
        `dissolved_oxygen`, `salinity`, `temperature`, `oxygen_saturation`,
        `apparent_oxygen_utilization`, `phosphate`, `silicate`, or `nitrate`.
    Choose `clim_type` averages from:
        `monthly`, `seasonal`, or `annual`.
    Choose `resolution` from:
        `1deg` or `5deg`
    Choose `levels` slice:
        all slice(0, 40, 1) , surface slice(0, 1)

    Returns
    -------
    Nested dictionary with with climatology (first level), variables
    (second level) and the data as a pandas 3D Panel.

    Example
    -------
    Extract a 2D surface -- Annual temperature climatology:
    >>> import numpy as np
    >>> import numpy.ma as ma
    >>> import matplotlib.pyplot as plt
    >>> from oceans.colormaps import cm
    >>> from mpl_toolkits.basemap import Basemap
    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> def make_map(llcrnrlon=2.5, urcrnrlon=360, llcrnrlat=-80, urcrnrlat=80,
    ...              projection='robin', lon_0=-55, lat_0=-10, resolution='c'):
    ...    m = Basemap(llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
    ...                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
    ...                projection=projection, resolution=resolution,
    ...                lon_0=lon_0, ax=ax)
    ...    m.drawcoastlines()
    ...    m.fillcontinents(color='0.85')
    ...    dx, dy = 60, 20
    ...    meridians = np.arange(llcrnrlon, urcrnrlon + dx, dx)
    ...    parallels = np.arange(llcrnrlat, urcrnrlat + dy, dy)
    ...    m.drawparallels(parallels, linewidth=0, labels=[1, 0, 0, 0])
    ...    m.drawmeridians(meridians, linewidth=0, labels=[0, 0, 0, 1])
    ...    return m
    >>> m = make_map()
    >>> boundary = map_limits(m)
    >>> dataset = woa_subset(var='temperature', clim_type='annual',
    ...                      resolution='1deg', levels=slice(0, 1), **boundary)
    >>> dataset = dataset['OA Climatology']
    >>> lon, lat = dataset.minor_axis.values, dataset.major_axis.values
    >>> lon, lat = np.meshgrid(lon,lat)
    >>> surface_temp = ma.masked_invalid(dataset['annual'].ix[0].values)
    >>> cs = m.pcolormesh(lon, lat, surface_temp, latlon=True, cmap=cm.avhrr)
    >>> _ = fig.colorbar(cs)
    Extract a square around (averaged into a profile) the Mariana Trench:
    >>> dataset = woa_subset(var='temperature', clim_type='monthly',
    ...                      resolution='1deg', levels=slice(0, 40),
    ...                      llcrnrlon=-143, urcrnrlon=-141, llcrnrlat=10,
    ...                      urcrnrlat=12)
    >>> dataset = dataset['OA Climatology']
    >>> fig, ax = plt.subplots()
    >>> z = dataset['Jan'].items.values.astype(float)
    >>> colors = get_color(12)
    >>> for month in dataset:
    ...     profile = dataset[month].mean().mean()
    ...     ax.plot(profile, z, label=month, color=next(colors))
    >>> ax.grid(True)
    >>> ax.invert_yaxis()
    >>> ax.legend(loc='lower right')
    >>> plt.show()
    """

    uri = "http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA09/NetCDFdata"
    fname = "%s_%s_%s.nc" % (var, clim_type, resolution)
    url = '%s/%s' % (uri, fname)
    nc = Dataset(url)

    v = dict(temperature='t', dissolved_oxygen='o', salinity='s',
             oxygen_saturation='O', apparent_oxygen_utilization='A',
             phosphate='p', silicate='p', nitrate='n')

    d = dict({'%s_an' % v[var]: 'OA Climatology',
              '%s_mn' % v[var]: 'Statistical Mean',
              '%s_dd' % v[var]: 'N. of Observations',
              '%s_se' % v[var]: 'Std Error of the Statistical Mean',
              '%s_sd' % v[var]: 'Std Deviation from Statistical Mean',
              '%s_oa' % v[var]: 'Statistical Mean minus OA Climatology',
              '%s_ma' % v[var]: 'Seasonal/Monthly minus Annual Climatology',
              '%s_gp' % v[var]: 'N. of Mean Values within Influence Radius'})

    depths = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
              600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750,
              2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000,
              7500, 8000, 8500, 9000][levels]

    llcrnrlon, urcrnrlon = map(wrap_lon360, (llcrnrlon, urcrnrlon))
    lon = wrap_lon360(nc.variables.pop('lon')[:])
    lat = nc.variables.pop('lat')[:]
    depth = nc.variables.pop('depth')[:]
    times = nc.variables.pop('time')
    times = num2date(times[:], times.units, calendar='365_day')
    times = [time.strftime('%b') for time in times]

    if clim_type == 'annual':
        times = clim_type

    # Select data subset.
    maskx = np.logical_and(lon >= llcrnrlon, lon <= urcrnrlon)
    masky = np.logical_and(lat >= llcrnrlat, lat <= urcrnrlat)
    maskz = np.array([z in depths for z in depth])

    lon, lat, depth = lon[maskx], lat[masky], depth[maskz]

    start = '%s_' % v[var]
    variables = dict()
    for variable in nc.variables.keys():
        if variable.startswith(start):
            subset = nc.variables[variable][..., maskz, masky, maskx]
            data = Panel4D(subset, major_axis=lat, minor_axis=lon,
                           labels=np.atleast_1d(times),
                           items=np.atleast_1d(depth))
            variables.update({d[variable]: data})
    return variables



def etopo_subset(llcrnrlon=None, urcrnrlon=None, llcrnrlat=None,
                 urcrnrlat=None, tfile='dap', smoo=False, subsample=False):
    """Get a etopo subset.
    Should work on any netCDF with x, y, data
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/contourICEMaps.py

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> offset = 5
    >>> tfile = '/home/filipe/00-NOBKP/OcFisData/ETOPO1_Bed_g_gmt4.grd'
    >>> llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = -43, -30, -22, -17
    >>> lons, lats, bathy = etopo_subset(llcrnrlon - offset,
    ...                                  urcrnrlon + offset,
    ...                                  llcrnrlat - offset,
    ...                                  urcrnrlat + offset,
    ...                                  smoo=True, tfile=tfile)
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lons, lats, bathy)
    >>> _ = ax.axis([-42, -28, -23, -15])
    >>> _ = ax.set_title(tfile)
    >>> plt.show()
    """

    if tfile == 'dap':
        tfile = 'http://opendap.ccst.inpe.br/Misc/etopo2/ETOPO2v2c_f4.nc'

    etopo = Dataset(tfile, 'r')

    lons = etopo.variables["x"][:]
    lats = etopo.variables["y"][:]

    res = get_indices(llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, lons, lats)

    lon, lat = np.meshgrid(lons[res[0]:res[1]], lats[res[2]:res[3]])

    bathy = etopo.variables["z"][int(res[2]):int(res[3]),
                                 int(res[0]):int(res[1])]

    if smoo:
        bathy = laplace_filter(bathy, M=None)

    if subsample:
        lon, lat, bathy = lon[::subsample], lat[::subsample], bathy[::subsample]
    return lon, lat, bathy


def get_depth(lon, lat, tfile='dap'):
    """Find the depths for each station on the etopo2 database."""
    lon, lat = map(np.atleast_1d, (lon, lat))

    lons, lats, bathy = etopo_subset(lat.min() - 5, lat.max() + 5,
                                     lon.min() - 5, lon.max() + 5,
                                     tfile=tfile, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode='nearest', order=3)


def get_isobath(lon, lat, iso=-200., tfile='dap'):
    """Find isobath."""
    plt.ioff()
    topo = get_depth(lon, lat, tfile=tfile)

    fig, ax = plt.subplots()
    cs = ax.contour(lon, lat, topo, [iso])
    path = cs.collections[0].get_paths()[0]
    del(fig, ax, cs)
    plt.ion()
    return path.vertices[:, 0], path.vertices[:, 1]


def get_indices(min_lat, max_lat, min_lon, max_lon, lons, lats):
    """Return the data indices for a lon, lat square."""

    distances1, distances2, indices = [], [], []
    index = 1
    for point in lats:
        s1 = max_lat - point
        s2 = min_lat - point
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index - 1))
        index = index + 1

    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    distances1, distances2 = [], []
    index = 1
    for point in lons:
        s1 = max_lon - point
        s2 = min_lon - point
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index - 1))
        index = index + 1

    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    # max_lat_indices, min_lat_indices, max_lon_indices, min_lon_indices.
    res = np.zeros((4), dtype=np.float64)
    res[0] = indices[3][2]
    res[1] = indices[2][2]
    res[2] = indices[1][2]
    res[3] = indices[0][2]
    return res


def laplace_X(F, M):
    """1D Laplace Filter in X-direction.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    jmax, imax = F.shape

    # Add strips of land.
    F2 = np.zeros((jmax, imax + 2), dtype=F.dtype)
    F2[:, 1:-1] = F
    M2 = np.zeros((jmax, imax + 2), dtype=M.dtype)
    M2[:, 1:-1] = M

    MS = M2[:, 2:] + M2[:, :-2]
    FS = F2[:, 2:] * M2[:, 2:] + F2[:, :-2] * M2[:, :-2]

    return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)


def laplace_Y(F, M):
    """1D Laplace Filter in Y-direction.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    jmax, imax = F.shape

    # Add strips of land.
    F2 = np.zeros((jmax + 2, imax), dtype=F.dtype)
    F2[1:-1, :] = F
    M2 = np.zeros((jmax + 2, imax), dtype=M.dtype)
    M2[1:-1, :] = M

    MS = M2[2:, :] + M2[:-2, :]
    FS = F2[2:, :] * M2[2:, :] + F2[:-2, :] * M2[:-2, :]

    return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)


def laplace_filter(F, M=None):
    """Laplace filter a 2D field with mask.  The mask may cause laplace_X and
    laplace_Y to not commute.  Take average of both directions.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    if not M:
        M = np.ones_like(F)

    return 0.5 * (laplace_X(laplace_Y(F, M), M) +
                  laplace_Y(laplace_X(F, M), M))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
