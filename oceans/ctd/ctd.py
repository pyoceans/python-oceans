#
# ctd.py
#
# purpose:  Some class and function to work with ctd data.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2012
# modified: Tue 26 Jun 2012 11:35:35 AM BRT
#
# obs: Instead of sub-classing I use a monkey patch approach following Wes
#      suggestion.
#

import pandas
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame as CTDProfile

# Degree symbol
deg = u"\u00b0"

""" TODO:
Methods:
  * Series or DataFrame
    - remove spikes
    - smooth
    - interpolate, find MLD
    - find barrier
    - plot profile
  * Methods (Panel):
    - geostrophic velocityq
    - plot section,
    - station map
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
    return cast.sort_index(ascending=True, inplace=True)


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
    new_series = self.groupby(shifted.asof).mean()
    new_series.index = new_index  # Not shifted.

    if plot:
        # TODO: Add salinity at the same plot.
        plt.figure()
        ax = self.TEMP.plot(style='k-.', label='Original')
        ax.plot(new_series.index, new_series.TEMP, 'ro', label='Binned')
        ax.set_title("Binned vs Original data")
        ax.set_ylabel("Pressure [db]")
        ax.set_xlabel("Temperature [%sC]" % deg)
        ax.legend(shadow=True, fancybox=True, numpoints=1)

    return new_series

# Bind methods.
CTDProfile.read_ctd = read_ctd
CTDProfile.bindata = bindata


if __name__ == '__main__':
    fname = 'data/txt/d3_7101.txt'
    cast = CTDProfile().read_ctd(fname)
    binned = cast.bindata(db=0.5, plot=True)
