# -*- coding: utf-8 -*-
#
# datasets.py
#
# purpose:  Functions to handle datasets.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Mon 04 May 2015 05:44:40 PM BRT
#
# obs: some Functions were based on:
# http://www.trondkristiansen.com/?page_id=1071


from __future__ import division

import warnings

import iris
import numpy as np
import matplotlib.pyplot as plt


from netCDF4 import Dataset
from ..ff_tools import get_profile, wrap_lon360
from iris.analysis.interpolate import extract_nearest_neighbour


__all__ = ['map_limits',
           'woa_subset',
           'woa_profile',
           'etopo_subset',
           'map_limits',
           'get_depth',
           'get_isobath',
           'laplace_filter']


def map_limits(m):
    lons, lats = wrap_lon360(m.boundarylons), m.boundarylats
    boundary = dict(llcrnrlon=min(lons),
                    urcrnrlon=max(lons),
                    llcrnrlat=min(lats),
                    urcrnrlat=max(lats))
    return boundary


def woa_subset(bbox=[2.5, 357.5, -87.5, 87.5], variable='temperature',
               clim_type='00', resolution='1.00', full=False):
    """Return an iris.cube instance from a World Ocean Atlas 2013 variable at a
    given lon, lat bounding box.

    Parameters
    ----------
    bbox: list, tuple
          minx, maxx, miny, maxy positions to extract.
    Choose data `variable` from:
        `dissolved_oxygen`, `salinity`, `temperature`, `oxygen_saturation`,
        `apparent_oxygen_utilization`, `phosphate`, `silicate`, or `nitrate`.
    Choose `clim_type` averages from:
        01-12 :: monthly
        13-16 :: seasonal (North Hemisphere Winter, Spring, Summer,
                           and Autumn respectively)
        00 :: annual
    Choose `resolution` from:
        1 (1 degree), or 4 (0.25 degrees)

    Returns
    -------
    Iris.cube instance with the climatology.

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.feature as cfeature
    >>> from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
    ...                                    LATITUDE_FORMATTER)
    >>> LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
    ...                                     edgecolor='face',
    ...                                     facecolor=cfeature.COLORS['land'])
    >>> def make_map(bbox, projection=ccrs.PlateCarree()):
    ...     fig, ax = plt.subplots(figsize=(8, 6),
    ...                            subplot_kw=dict(projection=projection))
    ...     ax.set_extent(bbox)
    ...     ax.add_feature(LAND, facecolor='0.75')
    ...     ax.coastlines(resolution='50m')
    ...     gl = ax.gridlines(draw_labels=True)
    ...     gl.xlabels_top = gl.ylabels_right = False
    ...     gl.xformatter = LONGITUDE_FORMATTER
    ...     gl.yformatter = LATITUDE_FORMATTER
    ...     return fig, ax
    >>> # Extract a 2D surface -- Annual temperature climatology:
    >>> import matplotlib.pyplot as plt
    >>> from oceans.ff_tools import wrap_lon180
    >>> from oceans.colormaps import cm, get_color
    >>> import iris.plot as iplt
    >>> from oceans.datasets import woa_subset
    >>> bbox = [2.5, 357.5, -87.5, 87.5]
    >>> kw = dict(bbox=bbox, variable='temperature', clim_type='00',
    ...           resolution='0.25')
    >>> cube = woa_subset(**kw)
    >>> c = cube[0, 0, ...]  # Slice singleton time and first level.
    >>> cs = iplt.pcolormesh(c, cmap=cm.avhrr)
    >>> cbar = plt.colorbar(cs)
    >>> # Extract a square around the Mariana Trench averaging into a profile.
    >>> bbox = [-143, -141, 10, 12]
    >>> kw = dict(bbox=bbox, variable='temperature', resolution='0.25',
    ...           clim_type=None)
    >>> fig, ax = plt.subplots(figsize=(5, 5))
    >>> colors = get_color(12)
    >>> months = 'Jan Feb Apr Mar May Jun Jul Aug Sep Oct Nov Dec'.split()
    >>> months = dict(zip(months, range(12)))
    >>> for month, clim_type in months.items():
    ...     clim_type = '{0:02d}'.format(clim_type+1)
    ...     kw.update(clim_type=clim_type)
    ...     cube = woa_subset(**kw)
    ...     grid_areas = iris.analysis.cartography.area_weights(cube)
    ...     c = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN,
    ...                         weights=grid_areas)
    ...     z = c.coord(axis='Z').points
    ...     l = ax.plot(c[0, :].data, z, label=month, color=next(colors))
    >>> ax.grid(True)
    >>> ax.invert_yaxis()
    >>> leg = ax.legend(loc='lower left')
    >>> _ = ax.set_ylim(200, 0)
    """

    if variable not in ['salinity', 'temperature']:
        resolution = '1.00'
        decav = 'all'
        msg = '{} is only available at 1 degree resolution'.format
        warnings.warn(msg(variable))
    else:
        decav = 'decav'

    v = dict(temperature='t', silicate='i', salinity='s', phosphate='p',
             oxygen='o', o2sat='O', nitrate='n', AOU='A')

    r = dict({'1.00': '1', '0.25': '4'})

    var = v[variable]
    res = r[resolution]

    uri = ("http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/"
           "{variable}/netcdf/{decav}/{resolution}/woa13_{decav}_{var}"
           "{clim_type}_0{res}.nc").format
    url = uri(**dict(variable=variable, decav=decav, resolution=resolution,
                     var=var, clim_type=clim_type, res=res))

    cubes = iris.load_raw(url)
    cubes = [cube.intersection(longitude=(bbox[0], bbox[1]),
                               latitude=(bbox[2], bbox[3])) for cube in cubes]
    cubes = iris.cube.CubeList(cubes)
    if full:
        return cubes
    else:
        cubes = [c for c in cubes if c.var_name == '{}_an'.format(var)]
        return cubes[0]


def woa_profile(lon, lat, variable='temperature', clim_type='00',
                resolution='1.00', full=False):
    """Return an iris.cube instance from a World Ocean Atlas 2013 variable at a
    given lon, lat point.

    Parameters
    ----------
    lon, lat: float
          point positions to extract the profile.
    Choose data `variable` from:
          'temperature', 'silicate', 'salinity', 'phosphate',
          'oxygen', 'o2sat', 'nitrate', and 'AOU'.
    Choose `clim_type` averages from:
        01-12 :: monthly
        13-16 :: seasonal (North Hemisphere Winter, Spring, Summer,
                           and Autumn respectively)
        00 :: annual
    Choose `resolution` from:
        1 (1 degree), or 4 (0.25 degrees)

    Returns
    -------
    Iris.cube instance with the climatology.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from oceans.datasets import woa_profile
    >>> cube = woa_profile(-143, 10, variable='temperature',
    ...                    clim_type='00', resolution='1.00', full=False)
    >>> fig, ax = plt.subplots(figsize=(2.25, 5))
    >>> z = cube.coord(axis='Z').points
    >>> l = ax.plot(cube[0, :].data, z)
    >>> ax.grid(True)
    >>> ax.invert_yaxis()
    """

    if variable not in ['salinity', 'temperature']:
        resolution = '1.00'
        decav = 'all'
        msg = '{} is only available at 1 degree resolution'.format
        warnings.warn(msg(variable))
    else:
        decav = 'decav'

    v = dict(temperature='t', silicate='i', salinity='s', phosphate='p',
             oxygen='o', o2sat='O', nitrate='n', AOU='A')

    r = dict({'1.00': '1', '0.25': '4'})

    var = v[variable]
    res = r[resolution]

    uri = ("http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/"
           "{variable}/netcdf/{decav}/{resolution}/woa13_{decav}_{var}"
           "{clim_type}_0{res}.nc").format
    url = uri(**dict(variable=variable, decav=decav, resolution=resolution,
                     var=var, clim_type=clim_type, res=res))

    cubes = iris.load_raw(url)
    cubes = [extract_nearest_neighbour(cube, [('longitude', lon),
                                              ('latitude', lat)])
             for cube in cubes]
    cubes = iris.cube.CubeList(cubes)
    if full:
        return cubes
    else:
        cubes = [c for c in cubes if c.var_name == '{}_an'.format(var)]
        return cubes[0]


def etopo_subset(llcrnrlon=None, urcrnrlon=None, llcrnrlat=None,
                 urcrnrlat=None, tfile='dap', smoo=False, subsample=False):
    """Get a etopo subset.
    Should work on any netCDF with x, y, data
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/contourICEMaps.py

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> offset = 5
    >>> #tfile = './ETOPO1_Bed_g_gmt4.grd'
    >>> tfile = 'dap'
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
        bathy = bathy[::subsample]
        lon, lat = lon[::subsample], lat[::subsample]
    return lon, lat, bathy


def get_depth(lon, lat, tfile='dap'):
    """Find the depths for each station on the etopo2 database."""
    lon, lat = map(np.atleast_1d, (lon, lat))

    lons, lats, bathy = etopo_subset(lat.min() - 5, lat.max() + 5,
                                     lon.min() - 5, lon.max() + 5,
                                     tfile=tfile, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode='nearest', order=3)


def get_isobath(llcrnrlon=None, urcrnrlon=None, llcrnrlat=None,
                urcrnrlat=None, iso=-200., tfile='dap'):
    """Finds an isobath on the etopo2 database and returns
    its lon,lat coordinates for plotting."""
    plt.ioff()
    lon, lat, topo = etopo_subset(llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                                  tfile=tfile)

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
