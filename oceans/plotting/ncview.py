import sys
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import AxesGrid

__version__ = '0.1.0'

try:
    from netCDF4 import Dataset
except ImportError:
    warnings.warn("Loading scipy's netcdf instead.  No netcdf4 support.")
    from scipy.io.netcdf import NetCDFFile as Dataset


def parse_args(arglist):
    r"""Parse options with argparse."""

    usage = "usage: %(prog)s [options] ncfile variable --levs"

    description = "ncview is a simple python netcdf viewer for 2D data."

    parser = argparse.ArgumentParser(usage=usage,
                                     description=description)
    parser.add_argument('ncfile',
                        metavar='ncfile',
                        help="input nc file")

    parser.add_argument('-v', '--variable',
                        metavar='var')

    parser.add_argument('-l', '--levs',
                        nargs='+',
                        type=float,
                        default=0,
                        help="Choose level to plot, default=0")

    parser.add_argument('--version',
                        action='version',
                        version="%(prog)s version " + __version__)

    args = parser.parse_args()

    return args


def ncview(args):
    colors = ('blue', 'cyan', 'orange', 'lightgreen', 'gray', 'yellow', 'red')

    # TODO: Improve to accept a list and make multiple figures
    level = args.levs
    ncfile = args.ncfile
    variable = args.variable

    # Read the ncfile.
    try:
        nc = Dataset(ncfile, 'r')
    except Exception, err:
        raise Exception("Problem while reading ncfile %s.\nError: %s" %
                              (ncfile, err.args[0]))

    # Check variable.
    try:
        print('%s' % nc.variables[variable])
        print('%s' % nc.dimensions.keys())
        variable = nc.variables[variable]
        nc_var = variable[:]
    except KeyError:
        variables = ', '.join(nc.variables.keys())
        raise KeyError("Variable %s not found.  Choose from: %s" %
              (variable, variables))

    # NOTE: I assume 3D model output (time, lon, lat).
    ndim = len(variable.dimensions)
    dimensions = variable.dimensions
    print("ncfile has %s dimensions:\n\t%s" % (ndim, ', '.join(dimensions)))
    time = nc.variables[dimensions[0]][:]
    lat = nc.variables[dimensions[1]][:]
    lon = nc.variables[dimensions[2]][:]

    if hasattr(nc_var, '_FillValue') and not hasattr(var, 'mask'):
        print nc_var._FillValue
        var = np.ma.masked_array(var, mask=(var == nc_var._FillValue))
        flatdat = var[var != nc_var._FillValue]
        vmin = flatdat.min()
        vmax = flatdat.max()
    #else:
        #vmin = var.min()
        #vmax = var.max()

    #print(vmin, vmax)

    #nc.close()


def plot_map(lon, lat):
    r"""If the data has geographical coordinates use Basemap."""
    # Corners.
    llcrnrlon, urcrnrlon = lon.min(), lon.max()
    llcrnrlat, urcrnrlat = lat.min(), lat.max()

    m = Basemap(projection='merc', llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                                   llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                                   resolution='i')


def plot():
    fig = plt.figure()

    shape = (1, 2) if args.levs is not None else (1, 1)
    grid = AxesGrid(fig, 111, nrows_ncols=shape, share_all=True)

    if args.levs is not None:
        for lev in args.levs:
            var = np.where(var == lev, lev + 0.00001, var)

        ax = grid[0]
        res = ax.contourf(x, y, var, args.levs, edgecolor='r')
        for i, col in enumerate(res.collections):
            print(type(col))
            col.set_edgecolor('r')
            col.set_facecolor(colors[i % len(colors)])

    ax = grid[-1]
    im = ax.imshow(var, interpolation='nearest', origin='lower',
                extent=(x[0], x[-1], y[0], y[-1]),
                vmin=vmin, vmax=vmax)

    plt.show()


def main(argv=None):
    r"""Run ncview."""
    if argv is None:
        argv = sys.argv

    args = parse_args(argv[1:])
    print(args)

    ncview(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
