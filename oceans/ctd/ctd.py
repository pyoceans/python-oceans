#
# ctd.py
#
# purpose:  Some class and function to work with ctd data.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2012
# modified: Wed 12 Sep 2012 11:38:55 AM BRT
#
# obs: Instead of sub-classing I opted for a "Monkey Patch" approach
#      following Wes own suggestion.
#

import pandas
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

# CTDProfile is a just Monkey Patched pandas DataFrame!
from pandas import DataFrame as CTDProfile

from oceans.ff_tools.time_series import despike

# Degree symbol
deg = u"\u00b0"

""" TODO:
Methods:
  * Series or DataFrame
    - smooth
    - interpolate, find MLD
    - find barrier
    - plot profile
  * Methods (Panel):
    - geostrophic velocity
    - plot section,
    - station map (lon, lat)
"""


def read_ctd(self, fname):
    r"""Read ascii ctd file from Seabird model xx."""

    # Predefined columns.
    # NOTE: This might break.
    # TODO: Read the raw file.
    dtype = [('SCANS', 'f4'),
             ('PRES', 'f4'),
             ('TEMP', 'f4'),
             ('COND', 'f4'),
             ('OXCU', 'f4'),
             ('OXTM', 'f4'),
             ('FLUO', 'f4'),
             ('OBS', 'f4'),
             ('SAL', 'f4'),
             ('DEN', 'f4'),
             ('SV', 'f4'),
             ('DEPTH', 'f4')]

    # Read as a record array and load it as a DataFrame.
    # TODO: Subclass DataFrame.
    header = 10
    recarray = np.loadtxt(fname, dtype=dtype, skiprows=header)
    #exclude* = ['SCANS', 'OBS', 'SV', 'DEPTH', 'SAL']
    cast = pandas.DataFrame.from_records(recarray, index='PRES',
                                         exclude=None,  # exclude*
                                         coerce_float=True)

    # Sort the index in ascending order.
    # TODO: Add a flag to identify up-/down-cast.
    #FIXME to sort or not to sort?:
    #cast = cast.sort_index(ascending=True, inplace=True)
    return cast


def bindata(self, db=1., plot=False):
    r"""Bin average the index (pressure) to a given interval in decibars [db].
    default db = 1.
    This method assumes that the profile start at the surface (0) and "ceils"
    the last depth.

    Note that this method does not drop NA automatically.  Therefore, one can
    check the quality of the binned data."""

    # NOTE: This is an alternative to use the current index as a guide.
    # new_index = np.arange(self.index[0], self.index[-1], db)

    start = 0.
    end = np.ceil(self.index[-1])
    shift = db / 2  # To get centered bins.
    new_index = np.arange(start, end, db)
    shifted = new_index - shift

    shifted = pandas.Index(shifted)
    newdf = self.groupby(shifted.asof).mean()
    newdf.index = new_index  # Not shifted.

    if plot:  # TODO: Add salinity at the same plot.
        fig, ax = plt.subplots()
        ax.plot(self.TEMP, self.index, 'k-.', label='Original')
        ax.plot(newdf.TEMP, newdf.index, 'ro', label='Binned')
        ax.set_title("Binned vs Original data")
        ax.set_ylabel("Pressure [db]")
        ax.set_xlabel("Temperature [%sC]" % deg)
        ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
        ax.invert_yaxis()
        plt.show()

    return newdf


def plot_ctd(self, station=None, title=None, subplots=False, **kwds):
    r"""Plot the CTD profile."""

    # FIXME: At the moment it just plots temperature and salinity.
    if subplots:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
    else:
        fig = plt.figure()
        ax0 = host_subplot(111, axes_class=AA.Axes)  # Host.
        ax1 = ax0.twiny()  # Parasite.

        # Axis location.
        host_new_axis = ax0.get_grid_helper().new_fixed_axis
        ax0.axis["bottom"] = host_new_axis(loc="top", axes=ax0, offset=(0, 0))
        par_new_axis = ax1.get_grid_helper().new_fixed_axis
        ax1.axis["top"] = par_new_axis(loc="bottom", axes=ax1, offset=(0, 0))

    ax0.plot(self.TEMP, self.index, 'r-.', label='Temperature')
    ax1.plot(self.SAL, self.index, 'b-.', label='Salinity')

    ax0.set_ylabel("Pressure [db]")
    ax0.set_xlabel("Temperature [%sC]" % deg)
    ax1.set_xlabel("Salinity [Kg g$^{-1}$]")
    ax1.invert_yaxis()

    if not title:
        title = r"Station %s profile" % station

    ax0.text(0.5, 0.99, title,
             horizontalalignment='center',
             verticalalignment='center',
             transform=fig.transFigure,
             rotation='horizontal')

    ax0.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
    if subplots:
        ax1.legend(shadow=True, fancybox=True, numpoints=1, loc='best')

    plt.show()

    return fig, (ax0, ax1)


# Utilities.
def normalize_names(name):
    r"""Convert arbitrary column names into a consistent format.
    Te result is a lower case name with spaces replaced with underscores and no
    leading or trailing whitespace."""
    name = name.strip()
    name = name.strip('*')
    name = name.lower()
    name = name.replace(' ', '_')
    return name


def despike_ctd(self, n=3, recursive=False, verbose=False):
    r"""Apply despike from time_series to the DataFrame."""
    # TODO: Need special attention to the index!
    # NOTE: No interpolation is done at this point.
    # NOTE: A better "Series" method is recommend to we can use different `n`.
    return self.apply(despike, n=3, recursive=recursive, verbose=verbose)

# Bind methods.
CTDProfile.bindata = bindata
CTDProfile.read_ctd = read_ctd
CTDProfile.despike = despike_ctd
CTDProfile.plot_ctd = plot_ctd  # FIXME: Override plot?

if __name__ == '__main__':
    # Load.
    fname = 'data/txt/d3_7101.txt'
    cast = CTDProfile().read_ctd(fname)

    # Plot
    if 0:
        fig, ax = cast.plot_ctd(subplots=False, station='Test')
        fig, ax = cast.plot_ctd(subplots=True, title='Subplot test',
                                station='Test')

    # Bin average.
    if 1:
        binned = cast.bindata(db=0.5, plot=True)

    # Remove spikes.
    if 0:
        # Add a spike.
        spiked = cast.bindata(db=1, plot=False).copy()
        # At the limit. FIXME.
        spiked.TEMP.ix[5] = spiked.TEMP.mean() + spiked.TEMP.std() * 3
        # Large negative. NOTE: Negative could be could be treated separately.
        spiked.TEMP.ix[10] = -50.
        # Large positive. NOTE: Need recursive to removed after -50.
        spiked.TEMP.ix[15] = 50.

        # Using the apply method.
        despiked = spiked.despike(n=3, recursive=True, verbose=True)
