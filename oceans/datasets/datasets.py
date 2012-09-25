# -*- coding: utf-8 -*-
#
# datasets.py
#
# purpose:  Functions to handle datasets.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Wed 12 Sep 2012 12:00:36 PM BRT
#
# obs: some Functions were based on:
# http://www.trondkristiansen.com/?page_id=1071


from __future__ import division

import numpy as np

from netCDF4 import Dataset

from oceans.ff_tools import get_profile

__all__ = [
    'woa_subset',
    'etopo_subset',
    'laplace_X',
    'laplace_Y',
    'laplace_filter',
    'get_depth'
]


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


def woa_subset(min_lat, max_lat, min_lon, max_lon, woa_file=None):
    r"""Get a World Ocean Atlas subset.
    Should work on any netCDF with x, y, data.
    `woa_file` can be an OpenDap url."""

    if not woa_file:
        # TODO: Check for a online dap version of this file.
        woa_file = None

    woa = Dataset(woa_file, 'r')

    lons = woa.variables["x"][:]
    lats = woa.variables["y"][:]

    res = get_indices(min_lat, max_lat, min_lon, max_lon, lons, lats)

    lon, lat = np.meshgrid(lons[res[0]:res[1]], lats[res[2]:res[3]])

    bathy = woa.variables["temperature"][int(res[2]):int(res[3]),
                                         int(res[0]):int(res[1])]

    return lon, lat, temperature


def etopo_subset(min_lat, max_lat, min_lon, max_lon, tfile=None, smoo=False):
    r"""Get a etopo subset.
    Should work on any netCDF with x, y, data
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/contourICEMaps.py

    Example
    -------
    >>> offset = 5
    >>> lonStart, lonEnd, latStart, latEnd = -43, -30.0, -22.0, -17.0
    >>> lons, lats, bathy = etopo_subset(latStart - offset, latEnd + offset,
    ...                                  lonStart - offset, lonEnd + offset, smoo=True)
    """

    if not tfile:
        # TODO: Check for a online dap version of this file.
        tfile = 'http://opendap.ccst.inpe.br/Misc/etopo2/ETOPO2v2c_f4.nc'

    etopo = Dataset(tfile, 'r')

    lons = etopo.variables["x"][:]
    lats = etopo.variables["y"][:]

    res = get_indices(min_lat, max_lat, min_lon, max_lon, lons, lats)

    lon, lat = np.meshgrid(lons[res[0]:res[1]], lats[res[2]:res[3]])

    bathy = etopo.variables["z"][int(res[2]):int(res[3]),
                                 int(res[0]):int(res[1])]

    if smoo == 'True':
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


def get_depth(lon, lat):
    r"""Find the depths for each station on the etopo2 database."""
    lons, lats, bathy = etopo_subset(lat.min() - 5, lat.max() + 5,
                                     lon.min() - 5, lon.max() + 5, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode='nearest', order=3)
