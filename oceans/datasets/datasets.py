import warnings

from netCDF4 import Dataset

import numpy as np

from ..RPSstuff import near
from ..ocfis import get_profile, wrap_lon180


def woa_subset(bbox=[2.5, 357.5, -87.5, 87.5], variable='temperature', clim_type='00', resolution='1.00', full=False):  # noqa
    """
    Return an iris.cube instance from a World Ocean Atlas 2013 variable at a
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
    >>> import iris
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
    ...                            subplot_kw={'projection': projection})
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
    >>> from oceans.ocfis import wrap_lon180
    >>> from oceans.colormaps import cm, get_color
    >>> import iris.plot as iplt
    >>> from oceans.datasets import woa_subset
    >>> bbox = [2.5, 357.5, -87.5, 87.5]
    >>> kw = {'bbox': bbox, 'variable': 'temperature', 'clim_type': '00',
    ...        'resolution': '0.25'}
    >>> cube = woa_subset(**kw)
    >>> c = cube[0, 0, ...]  # Slice singleton time and first level.
    >>> cs = iplt.pcolormesh(c, cmap=cm.avhrr)
    >>> cbar = plt.colorbar(cs)
    >>> # Extract a square around the Mariana Trench averaging into a profile.
    >>> bbox = [-143, -141, 10, 12]
    >>> kw = {'bbox': bbox, 'variable': 'temperature', 'resolution': '0.25',
    ...       'clim_type': None}
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
    import iris

    if variable not in ['salinity', 'temperature']:
        resolution = '1.00'
        decav = 'all'
        msg = '{} is only available at 1 degree resolution'.format
        warnings.warn(msg(variable))
    else:
        decav = 'decav'

    v = {
        'temperature': 't',
        'silicate': 'i',
        'salinity': 's',
        'phosphate': 'p',
        'oxygen': 'o',
        'o2sat': 'O',
        'nitrate': 'n',
        'AOU': 'A'
    }

    r = {'1.00': '1', '0.25': '4'}

    var = v[variable]
    res = r[resolution]

    url = (
        f'https://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/'
        f'{variable}/netcdf/{decav}/{resolution}/woa13_{decav}_{var}'
        f'{clim_type}_0{res}.nc')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cubes = iris.load_raw(url)
    cubes = [cube.intersection(longitude=(bbox[0], bbox[1]),
                               latitude=(bbox[2], bbox[3])) for cube in cubes]
    cubes = iris.cube.CubeList(cubes)
    if full:
        return cubes
    else:
        cubes = [c for c in cubes if c.var_name == '{}_an'.format(var)]
        return cubes[0]


def woa_profile(lon, lat, variable='temperature', clim_type='00', resolution='1.00'):
    """
    Return an iris.cube instance from a World Ocean Atlas 2013 variable at a
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
    ...                    clim_type='00', resolution='1.00')
    >>> fig, ax = plt.subplots(figsize=(2.25, 5))
    >>> z = cube.coord(axis='Z').points
    >>> l = ax.plot(cube[0, :].data, z)
    >>> ax.grid(True)
    >>> ax.invert_yaxis()

    """
    import iris

    if variable not in ['salinity', 'temperature']:
        resolution = '1.00'
        decav = 'all'
        msg = '{} is only available at 1 degree resolution'.format
        warnings.warn(msg(variable))
    else:
        decav = 'decav'

    v = {
        'temperature': 't',
        'silicate': 'i',
        'salinity': 's',
        'phosphate': 'p',
        'oxygen': 'o',
        'o2sat': 'O',
        'nitrate': 'n',
        'AOU': 'A'
    }

    r = {'1.00': '1', '0.25': '4'}

    var = v[variable]
    res = r[resolution]

    url = (
        f'https://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATAv2/'
        f'{variable}/netcdf/{decav}/{resolution}/woa13_{decav}_{var}'
        f'{clim_type}_0{res}v2.nc')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cubes = iris.load_raw(url)

    cube = [c for c in cubes if c.var_name == '{}_an'.format(var)][0]
    scheme = iris.analysis.Nearest()
    sample_points = [('longitude', lon), ('latitude', lat)]
    kw = {
        'sample_points': sample_points,
        'scheme': scheme,
        'collapse_scalar': True
    }

    return cube.interpolate(**kw)


def etopo_subset(bbox=[-43, -30, -22, -17], tfile=None, smoo=False):
    """
    Get a etopo subset.
    Should work on any netCDF with x, y, data
    http://www.trondkristiansen.com/wp-content/uploads/downloads/2011/07/contourICEMaps.py

    Examples
    --------
    >>> from oceans.datasets import etopo_subset
    >>> import matplotlib.pyplot as plt
    >>> bbox = [-43, -30, -22, -17]
    >>> lon, lat, bathy = etopo_subset(bbox=bbox, smoo=True)
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lon, lat, bathy)

    """
    if tfile is None:
        tfile = 'http://opendap.ccst.inpe.br/Misc/etopo2/ETOPO2v2c_f4.nc'

    with Dataset(tfile, 'r') as etopo:
        lons = etopo.variables['x'][:]
        lats = etopo.variables['y'][:]

        imin, imax, jmin, jmax = _get_indices(bbox, lons, lats)
        lon, lat = np.meshgrid(lons[imin:imax], lats[jmin:jmax])

        # FIXME: This assumes j, i order.
        bathy = etopo.variables['z'][jmin:jmax, imin:imax]

    if smoo:
        from scipy.ndimage.filters import gaussian_filter
        bathy = gaussian_filter(bathy, sigma=1)

    return lon, lat, bathy


def get_depth(lon, lat, tfile=None):
    """
    Find the depths for each station on the etopo2 database.

    Examples
    --------
    >>> from oceans.datasets import get_depth
    >>> station_lon = [-40, -32]
    >>> station_lat = [-20, -20]
    >>> get_depth(station_lon, station_lat)
    array([  -32.988163, -4275.634   ], dtype=float32)

    """
    lon, lat = list(map(np.atleast_1d, (lon, lat)))

    offset = 5
    bbox = [lon.min() - offset, lon.max() + offset,
            lat.min() - offset, lat.max() + offset]
    lons, lats, bathy = etopo_subset(bbox, tfile=tfile, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode='nearest', order=3)


def get_isobath(bbox, iso=-200, tfile=None, smoo=False):
    """
    Finds an isobath on the etopo2 database and returns
    its lon, lat segments for plotting.

    Examples
    --------
    >>> from oceans.datasets import etopo_subset, get_isobath
    >>> import matplotlib.pyplot as plt
    >>> bbox = [-43, -30, -22, -17]
    >>> segments = get_isobath(bbox=bbox, iso=-200, smoo=True)
    >>> lon, lat, bathy = etopo_subset(bbox=bbox, smoo=True)
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lon, lat, bathy)
    >>> for segment in segments:
    ...     lines = ax.plot(segment[:, 0], segment[:, -1], 'k', linewidth=2)

    """
    import matplotlib._contour as contour
    lon, lat, topo = etopo_subset(bbox, tfile=tfile, smoo=smoo)

    # Required args for QuadContourGenerator.
    mask, corner_mask, nchunk = None, True, 0
    c = contour.QuadContourGenerator(lon, lat, topo, mask, corner_mask, nchunk)
    res = c.create_contour(iso)
    nseg = len(res) // 2
    segments = res[:nseg]
    return segments


def _minmax(v):
    return np.min(v), np.max(v)


def _get_indices(bbox, lons, lats):
    """Return the data indices for a lon, lat square."""
    lons = wrap_lon180(lons)

    idx_x = np.logical_and(lons >= bbox[0], lons <= bbox[1])
    idx_y = np.logical_and(lats >= bbox[2], lats <= bbox[3])
    if lons.ndim == 2 and lats.ndim == 2:
        inregion = np.logical_and(idx_x, idx_y)
        region_inds = np.where(inregion)
        imin, imax = _minmax(region_inds[0])
        jmin, jmax = _minmax(region_inds[1])
    elif lons.ndim == 1 and lats.ndim == 1:
        imin, imax = _minmax(np.where(idx_x))
        jmin, jmax = _minmax(np.where(idx_y))
    else:
        msg = 'Cannot understand input shapes lons {!r} and lats {!r}'.format
        raise ValueError(msg(lons.shape, lats.shape))
    return imin, imax+1, jmin, jmax+1


def get_gebco15(x, y, topofile='gebco15-40s_30-52w_30seg.nc'):
    """
    Usage
    -----
    H, D, Xo, Yo = get_gebco15(x, y, topofile='gebco/gebco_08_30seg.nc')

    Description
    -----------
    Finds the depth of points of coordinates 'x','y' using GEBCO data set.

    Parameters
    ----------
    x         : 1D array
                Array containing longitudes of the points of unknown depth.

    y         : 1D array
                Array containing latitudes of the points of unknown depth.

    topofile  : string, optional
                String containing path to the GEBCO netCDF file.

    Returns
    -------
    H         : 1D array
                Array containing depths of points closest to the input X, Y
                coordinates.

    X         : 1D array
                Array of horizontal distance associated with array 'H'.

    D         : 1D array
                Array containing distances (in km) from the input X, Y
                coordinates to the data points.

    Xo        : 1D array
                Array containing longitudes of the data points.

    Yo        : 1D array
                Array containing latitudes of the data points.

    NOTES
    -------
    This function reads the entire netCDF file before extracting the wanted
    data.  Therefore, it does not handle the full GEBCO dataset (1.8 GB)
    efficiently.

    TODO
    -------
    Make it possible to work with the full gebco dataset, by extracting only
    the wanted indexes.

    Code History
    ---------------------------------------
    Author of the original Matlab code (ftopo.m, ETOPO2 dataset):
    Marcelo Andrioni <marceloandrioni@yahoo.com.br>
    December 2008: Modification performed by Cesar Rocha <cesar.rocha@usp.br>
    to handle ETOPO1 dataset.
    July 2012: Python Translation and modifications performed by André Palóczy
    Filho <paloczy@gmail.com>
    to handle GEBCO dataset (30 arc seconds resolution).

    """
    from seawater import dist

    x, y = list(map(np.asanyarray, (x, y)))

    # Opening netCDF file and extracting data.
    grid = Dataset(topofile)
    yyr = grid.variables['y_range'][:]
    xxr = grid.variables['x_range'][:]
    spacing = grid.variables['spacing'][:]
    dx, dy = spacing[0], spacing[1]

    # Creating lon and lat 1D arrays.
    xx = np.arange(xxr[0], xxr[1], dx)
    xx = xx + dx / 2
    yy = np.arange(yyr[0], yyr[1], dy)
    yy = yy + dy / 2
    h = grid.variables['z'][:]
    grid.close()

    # Retrieving nearest point for each input coordinate.
    A = np.asanyarray([])
    xx, yy = np.meshgrid(xx, yy)
    ni, nj = xx.shape[0], yy.shape[1]
    h = np.reshape(h, (ni, nj))
    h = np.flipud(h)
    Xo = A.copy()
    Yo = A.copy()
    H = A.copy()
    D = A.copy()
    for I in range(x.size):
        ix = near(xx[0, :], x[I])
        iy = near(yy[:, 0], y[I])
        H = np.append(H, h[iy, ix])
        # Calculating distance between input and GEBCO points.
        D = np.append(D, dist([x[I], xx[0, ix]], [y[I], yy[iy, 0]],
                              units='km')[0])
        Xo = np.append(Xo, xx[0, ix])
        Yo = np.append(Yo, yy[iy, 0])
        # Calculating distance axis.
        X = np.append(0, np.cumsum(dist(Xo, Yo, units='km')[0]))

    return H, X, D, Xo, Yo


if __name__ == '__main__':
    import doctest
    doctest.testmod()
