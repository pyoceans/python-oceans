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

from pandas import Panel
from netCDF4 import Dataset, num2date
from oceans.ff_tools import get_profile

__all__ = ['woa_subset',
           'etopo_subset',
           'laplace_X',
           'laplace_Y',
           'laplace_filter',
           'get_depth',
           'get_isobath']


def woa_subset(bbox=[-43, -29.5, -22.5, -17], var='temperature',
               clim_type='monthly', resolution='1deg'):
    r"""Get World Ocean Atlas variables at a given lon, lat bounding box.
    Choose variable from:
        `dissolved_oxygen`, `salinity`, `temperature`, `oxygen_saturation`,
        `apparent_oxygen_utilization`, `phosphate`, `silicate`, or `nitrate`.
    Choose clim_type averages from:
        `monthly`, `seasonal`, or `annual`.
    Choose resolution from:
        `1deg` or `5deg`

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> bbox = [-59, -25, -38, 9]
    >>> dataset = woa_subset(bbox=bbox, var='temperature', clim_type='annual',
    ...                      resolution='5deg')
    >>> dataset = dataset['annual']['Statistical Mean']
    >>> lon, lat = dataset.minor_axis, dataset.major_axis
    >>> surface_temp = dataset.ix[0]
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lon, lat, ma.masked_invalid(surface_temp.values))
    >>> fig.colorbar(cs)
    """

    uri = "http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA09/NetCDFdata/"
    dataset = "%s_%s_%s.nc" % (var, clim_type, resolution)

    nc = Dataset(uri + dataset)
    latStart, latEnd = bbox[2], bbox[3]
    lonStart, lonEnd = bbox[0], bbox[1]

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

    lon = nc.variables.pop('lon')[:] - 360
    lat = nc.variables.pop('lat')[:]
    depth = nc.variables.pop('depth')[:]
    time = nc.variables.pop('time')
    time = num2date(time[:], time.units, calendar='365_day')
    months = [t.strftime('%b') for t in time]

    # Select data subset.
    maskx = np.logical_and(lon >= lonStart, lon <= lonEnd)
    masky = np.logical_and(lat >= latStart, lat <= latEnd)
    lon, lat = lon[maskx], lat[masky]

    start = '%s_' % v[var]
    dataset, clim = dict(), dict()
    for k, month in enumerate(months):
        for data in nc.variables.keys():
            if data.startswith(start):
                subset = nc.variables[data][..., masky, maskx].squeeze()
                if clim_type == 'annual':
                    panel = Panel(subset[...], items=depth, major_axis=lat,
                                  minor_axis=lon)
                    month = clim_type
                else:
                    panel = Panel(subset[k, ...], items=depth, major_axis=lat,
                                  minor_axis=lon)
                clim.update({d[data]: panel})
        dataset.update({month: clim})
    return dataset


def get_indices(min_lat, max_lat, min_lon, max_lon, lons, lats):
    r"""Return the data indices for a lon, lat square."""

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


def etopo_subset(llcrnrlon=None, urcrnrlon=None, llcrnrlat=None,
                 urcrnrlat=None, tfile='dap', smoo=False):
    r"""Get a etopo subset.
    Should work on any netCDF with x, y, data
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/contourICEMaps.py

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> offset = 5
    >>> tfile = '/home/filipe/00-NOBKP/OcFisData/ETOPO1_Bed_g_gmt4.grd'
    >>> toponame = basename(tfile)[0]
    >>> llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat = -43, -30, -22, -17
    >>> lons, lats, bathy = etopo_subset(llcrnrlon - offset,
    ...                                  urcrnrlon + offset,
    ...                                  llcrnrlat - offset,
    ...                                  urcrnrlat + offset,
    ...                                  smoo=True, tfile=tfile)
    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(lons, lats, bathy)
    >>> ax.axis([-42, -28, -23, -15])
    >>> ax.set_title(toponame)
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

    return lon, lat, bathy


def laplace_X(F, M):
    r"""1D Laplace Filter in X-direction (axis=1)
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    jmax, imax = F.shape

    # Add strips of land
    F2 = np.zeros((jmax, imax + 2), dtype=F.dtype)
    F2[:, 1:-1] = F
    M2 = np.zeros((jmax, imax + 2), dtype=M.dtype)
    M2[:, 1:-1] = M

    MS = M2[:, 2:] + M2[:, :-2]
    FS = F2[:, 2:] * M2[:, 2:] + F2[:, :-2] * M2[:, :-2]

    return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)


def laplace_Y(F, M):
    r"""1D Laplace Filter in Y-direction (axis=1)
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    jmax, imax = F.shape

    # Add strips of land
    F2 = np.zeros((jmax + 2, imax), dtype=F.dtype)
    F2[1:-1, :] = F
    M2 = np.zeros((jmax + 2, imax), dtype=M.dtype)
    M2[1:-1, :] = M

    MS = M2[2:, :] + M2[:-2, :]
    FS = F2[2:, :] * M2[2:, :] + F2[:-2, :] * M2[:-2, :]

    return np.where(M > 0.5, (1 - 0.25 * MS) * F + 0.25 * FS, F)


def laplace_filter(F, M=None):
    r"""Laplace filter a 2D field with mask.  The mask may cause laplace_X and
    laplace_Y to not commute. Take average of both directions.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2010/09/laplaceFilter.py"""

    if not M:
        M = np.ones_like(F)

    return 0.5 * (laplace_X(laplace_Y(F, M), M) +
                  laplace_Y(laplace_X(F, M), M))


def get_depth(lon, lat, tfile='dap'):
    r"""Find the depths for each station on the etopo2 database."""
    lon, lat = map(np.atleast_1d, (lon, lat))

    lons, lats, bathy = etopo_subset(lat.min() - 5, lat.max() + 5,
                                     lon.min() - 5, lon.max() + 5,
                                     tfile=tfile, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode='nearest', order=3)


def get_isobath(lon, lat, iso=-200., tfile='dap'):
    r"""Find isobath."""
    plt.ioff()
    topo = get_depth(lon, lat, tfile=tfile)

    fig, ax = plt.subplots()
    cs = ax.contour(lon, lat, topo, [iso])
    path = cs.collections[0].get_paths()[0]
    del fig
    del ax
    del cs
    plt.ion()
    return path.vertices[:, 0], path.vertices[:, 1]
