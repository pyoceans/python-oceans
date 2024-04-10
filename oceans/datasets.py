import functools
import warnings

import numpy as np
import pooch
from netCDF4 import Dataset

from oceans.ocfis import get_profile, wrap_lon180


def _woa_variable(variable):
    _VAR = {
        "temperature": "t",
        "salinity": "s",
        "silicate": "i",
        "phosphate": "p",
        "nitrate": "n",
        "oxygen_saturation": "O",
        "dissolved_oxygen": "o",
        "apparent_oxygen_utilization": "A",
    }
    v = _VAR.get(variable)
    if not v:
        raise ValueError(
            f'Unrecognizable variable. Expected one of {list(_VAR.keys())}, got "{variable}".',
        )
    return v


def _woa_url(variable, time_period, resolution):
    base = "https://data.nodc.noaa.gov/thredds/dodsC"

    v = _woa_variable(variable)

    if variable not in ["salinity", "temperature"]:
        pref = "woa09"
        warnings.warn(
            f'The variable "{variable}" is only available at 1 degree resolution, '
            f'annual time period, and "{pref}".',
            stacklevel=2,
        )
        return f"{base}/" f"{pref}/" f"{variable}_annual_1deg.nc"
    else:
        dddd = "decav"
        pref = "woa18"

    grids = {
        "5": ("5deg", "5d"),
        "1": ("1.00", "01"),
        "1/4": ("0.25", "04"),
    }
    grid = grids.get(resolution)
    if not grid:
        raise ValueError(
            f'Unrecognizable resolution. Expected one of {list(grids.keys())}, got "{resolution}".',
        )
    res = grid[0]
    gg = grid[1]

    time_periods = {
        "annual": "00",
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
        "winter": "13",
        "spring": "14",
        "summer": "15",
        "autumn": "16",
    }

    time_period = time_period.lower()
    if len(time_period) == 3:
        tt = [
            time_periods.get(k)
            for k in time_periods.keys()
            if k.startswith(time_period)
        ][0]
    elif len(time_period) == 2 and time_period in time_periods.values():
        tt = time_period
    else:
        tt = time_periods.get(time_period)

    if not tt:
        raise ValueError(
            f"Unrecognizable time_period. "
            f'Expected one of {list(time_periods.keys())}, got "{time_period}".',
        )

    url = (
        f"{base}/"
        "/ncei/woa/"
        f"{variable}/decav/{res}/"
        f"{pref}_{dddd}_{v}{tt}_{gg}.nc"  # '[PREF]_[DDDD]_[V][TT][FF][GG]' Is [FF] used?
    )
    return url


@functools.lru_cache(maxsize=256)
def woa_profile(lon, lat, variable="temperature", time_period="annual", resolution="1"):
    """
    Return a xarray DAtaset instance from a World Ocean Atlas variable at a
    given lon, lat point.

    Parameters
    ----------
    lon, lat: float
        point positions to extract the interpolated profile.
    Choose data `variable` from:
        'temperature', 'salinity', 'silicate', 'phosphate',
        'nitrate', 'oxygen_saturation', 'dissolved_oxygen', or
        'apparent_oxygen_utilization'.
    Choose `time_period` from:
        01-12: January to December
        13-16: seasonal (North Hemisphere `Winter`, `Spring`, `Summer`, and `Autumn` respectively)
        00: Annual
    Choose `resolution` from:
        '5', '1', or '1/4' degrees (str)

    Returns
    -------
    xr.Dataset instance with the climatology.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from oceans.datasets import woa_profile
    >>> woa = woa_profile(
    ...     -143, 10, variable="temperature", time_period="annual", resolution="5"
    ... )
    >>> fig, ax = plt.subplots(figsize=(2.25, 5))
    >>> l = woa.plot(ax=ax, y="depth")
    >>> ax.grid(True)
    >>> ax.invert_yaxis()

    """
    import cf_xarray  # noqa
    import xarray as xr

    url = _woa_url(variable=variable, time_period=time_period, resolution=resolution)
    v = _woa_variable(variable)

    ds = xr.open_dataset(url, decode_times=False)
    ds = ds[f"{v}_mn"]
    return ds.cf.sel({"X": lon, "Y": lat}, method="nearest")


@functools.lru_cache(maxsize=256)
def woa_subset(
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    variable="temperature",
    time_period="annual",
    resolution="5",
    full=False,
):
    """
    Return an xarray Dataset instance from a World Ocean Atlas variable at a
    given lon, lat bounding box.

    Parameters
    ----------
    min_lon, max_lon, min_lat, max_lat: positions to extract.
    See `woa_profile` for the other options.

    Returns
    -------
    `xr.Dataset` instance with the climatology.

    Examples
    --------
    >>> # Extract a 2D surface -- Annual temperature climatology:
    >>> import matplotlib.pyplot as plt
    >>> from cmcrameri import cm
    >>> from oceans.datasets import woa_subset
    >>> bbox = [-177.5, 177.5, -87.5, 87.5]
    >>> woa = woa_subset(
    ...     *bbox, variable="temperature", time_period="annual", resolution="5"
    ... )
    >>> cs = woa["t_mn"].sel(depth=0).plot(cmap=cm.lajolla)

    >>> # Extract a square around the Mariana Trench averaging into a profile.
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from oceans.colormaps import get_color
    >>> colors = get_color(12)
    >>> months = "Jan Feb Apr Mar May Jun Jul Aug Sep Oct Nov Dec".split()
    >>> def area_weights_avg(woa):
    ...     woa = woa["t_mn"].squeeze()
    ...     weights = np.cos(np.deg2rad(woa["lat"])).where(~woa.isnull())
    ...     weights /= weights.mean()
    ...     return (woa * weights).mean(dim=["lon", "lat"])
    ...
    >>> bbox = [-143, -141, 10, 12]
    >>> fig, ax = plt.subplots(figsize=(5, 5))
    >>> for month in months:
    ...     woa = woa_subset(
    ...         *bbox, time_period=month, variable="temperature", resolution="1"
    ...     )
    ...     profile = area_weights_avg(woa)
    ...     l = profile.plot(ax=ax, y="depth", label=month, color=next(colors))
    ...
    >>> ax.grid(True)
    >>> ax.invert_yaxis()
    >>> leg = ax.legend(loc="lower left")
    >>> _ = ax.set_ylim(200, 0)

    """
    import cf_xarray  # noqa
    import xarray as xr

    url = _woa_url(variable, time_period, resolution)
    ds = xr.open_dataset(url, decode_times=False)
    ds = ds.cf.sel({"X": slice(min_lon, max_lon), "Y": slice(min_lat, max_lat)})
    v = _woa_variable(variable)
    if full:
        return ds
    return ds[[f"{v}_mn"]]  # always return a dataset


def _download_etopo2():
    url = "https://github.com/pyoceans/python-oceans/releases/download"
    version = "v2024.04"

    return pooch.retrieve(
        url=f"{url}/{version}/ETOPO2v2c_f4.nc",
        known_hash="sha256:30159a3f15a06398db3cae4ec75986bedc3317dda8e89d049ddc92ba1c352ff1",
    )


@functools.lru_cache(maxsize=256)
def etopo_subset(min_lon, max_lon, min_lat, max_lat, tfile=None, smoo=False):
    """
    Get a etopo subset.
    Should work on any netCDF with x, y, data

    Examples
    --------
    >>> from oceans.datasets import etopo_subset
    >>> import matplotlib.pyplot as plt
    >>> bbox = [-43, -30, -22, -17]
    >>> lon, lat, bathy = etopo_subset(*bbox, smoo=True)
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lon, lat, bathy)

    Based on trondkristiansen contourICEMaps.py
    """
    if tfile is None:
        tfile = _download_etopo2()

    with Dataset(tfile, "r") as etopo:
        lons = etopo.variables["x"][:]
        lats = etopo.variables["y"][:]

        bbox = min_lon, max_lon, min_lat, max_lat
        imin, imax, jmin, jmax = _get_indices(bbox, lons, lats)
        lon, lat = np.meshgrid(lons[imin:imax], lats[jmin:jmax])

        # FIXME: This assumes j, i order.
        bathy = etopo.variables["z"][jmin:jmax, imin:imax]

    if smoo:
        from scipy.ndimage import gaussian_filter

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
    array([  -32.98816423, -4275.63374601])

    """
    lon, lat = list(map(np.atleast_1d, (lon, lat)))

    offset = 5
    bbox = [
        lon.min() - offset,
        lon.max() + offset,
        lat.min() - offset,
        lat.max() + offset,
    ]
    lons, lats, bathy = etopo_subset(*bbox, tfile=tfile, smoo=False)

    return get_profile(lons, lats, bathy, lon, lat, mode="nearest", order=3)


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
    >>> lon, lat, bathy = etopo_subset(*bbox, smoo=True)
    >>> fig, ax = plt.subplots()
    >>> cs = ax.pcolormesh(lon, lat, bathy)
    >>> for segment in segments:
    ...     lines = ax.plot(segment[:, 0], segment[:, -1], "k", linewidth=2)
    ...

    """
    import contourpy

    lon, lat, topo = etopo_subset(*bbox, tfile=tfile, smoo=smoo)

    c = contourpy.contour_generator(
        lon,
        lat,
        topo,
        name="mpl2014",
        line_type=contourpy.LineType.SeparateCode,
        fill_type=contourpy.FillType.OuterCode,
        corner_mask=True,
        chunk_size=0,
    )
    res = c.create_contour(iso)
    nseg = len(res) // 2
    segments = res[:nseg]
    if len(segments) == 1:
        segments = segments[0]
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
        msg = "Cannot understand input shapes lons {!r} and lats {!r}".format
        raise ValueError(msg(lons.shape, lats.shape))
    return imin, imax + 1, jmin, jmax + 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
