# -*- coding: utf-8 -*-
#
# test_ctd.py
#
# purpose:  Test ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Thu 07 Mar 2013 06:09:29 PM BRT
#
# obs:
#


import os
import unittest

from glob import glob
from collections import OrderedDict

import matplotlib.pyplot as plt

from pandas import HDFStore
from oceans.ff_tools import apoxu, alphanum_key

from oceans.ctd import DataFrame, Series
from oceans.ctd import SP_from_C, seabird_filter, movingaverage


# Load tests.
def test_fsi(fname='data/d3_7165.txt.gz'):
    kw = dict(names=('SCANS', 'PRES', 'TEMP', 'COND', 'OXCU', 'OXTM', 'FLUO',
                     'OBS', 'SAL', 'DEN', 'SV', 'DEPTH'), compression='gzip',
              skiprows=10)
    return DataFrame.from_fsi(fname, **kw)


def test_xbt(fname='data/AMB09_037_XBT_rad3.EDF.zip', compression='zip'):
    return DataFrame.from_edf(fname, compression='zip')


def test_cnv(fname='data/AMB09_055_CTD_rad3.cnv.gz', compression='gzip'):
    return DataFrame.from_cnv(fname, compression=compression)


"""
%timeit DataFrame.from_cnv(fname='data/AMB09_059_CTD_rad3.cnv')
1 loops, best of 3: 9.84 s per loop
%timeit DataFrame.from_cnv(fname='data/AMB09_059_CTD_rad3.cnv.gz',
                            compression='gzip')
1 loops, best of 3: 9.97 s per loop
%timeit DataFrame.from_cnv(fname='data/AMB09_059_CTD_rad3.cnv.zip',
                            compression='zip')
1 loops, best of 3: 11.9 s per loop
%timeit DataFrame.from_cnv(fname='data/AMB09_059_CTD_rad3.cnv.bz2',
                            compression='bz2')
1 loops, best of 3: 31.9 s per loop
"""


def proc_ctd(fname, plot=True):
    # 00-Split, clean 'bad pump' data, and apply flag.
    cast = DataFrame.from_cnv(fname, compression='gzip').split()[0]
    cols = ('v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'pla',
            'sbeox0mg/l', 'spar', 'par', 'flsp', 'xmiss', 'sbeox0v')
    _ = map(cast.pop, cols)
    cast = cast[cast['pumps']]
    cast = cast[~cast['flag']]  # True for bad values.
    _ = map(cast.pop, ('pumps', 'flag'))
    # Smooth velocity.
    cast['dz/dtm'] = movingaverage(cast['dz/dtm'], window_size=48)

    fname = os.path.basename(fname).split('.')[0]
    print(fname)
    if plot:  # Original
        plotkw = dict(color='k', marker='.', linestyle='-', alpha=0.5)
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, sharex=True,
                                                      sharey=True)
        fig.suptitle("Processing steps -- %s" % fname)
        ax0.set_ylabel("Pressure [dbar]")
        ax0.invert_yaxis()
        ax0.plot(cast['t090c'], cast['t090c'].index, color='k', alpha=0.5)

    # 01-Filter pressure.
    kw = dict(sample_rate=24.0, time_constat=0.15)
    cast.index = seabird_filter(cast.index, **kw)
    if plot:  # Use the filtered pressure for a cleaner comparison.
        kw = dict(color='grey', alpha=0.5)
        ax1.plot(cast['t090c'], cast['t090c'].index, **kw)
        ax2.plot(cast['t090c'], cast['t090c'].index, **kw)
        ax3.plot(cast['t090c'], cast['t090c'].index, **kw)

    # 02-Remove pressure reversals.
    cast = cast.press_check()
    cast = cast.dropna()

    # 03-Loop Edit.
    cast = cast[cast['dz/dtm'] >= 0.25]  # Threshold velocity.
    if plot:
        ax0.plot(cast['t090c'], cast['t090c'].index, **plotkw)
        ax0.set_title("01-Loop Edit (Filter)")

    # 04-Remove spikes.
    kw = dict(n1=2, n2=20, block=15)
    cast = cast.apply(Series.despike, **kw)
    if plot:
        ax1.plot(cast['t090c'], cast['t090c'].index, **plotkw)
        ax1.set_title("02-Wild Edit")

    # 05-Bin-average.
    cast = cast.apply(Series.bindata, **dict(db=1.))
    if plot:
        ax2.plot(cast['t090c'], cast['t090c'].index, **plotkw)
        ax2.set_title("03-Bin Average")

    # 06-interpolate.
    cast = cast.apply(Series.interpolate)

    # 07-Smooth.
    pmax = max(cast.index)
    if pmax >= 500.:
        window_len = 21
    elif pmax >= 100.:
        window_len = 11
    else:
        window_len = 5
    kw = dict(window_len=window_len, window='hanning')
    cast = cast.apply(Series.smooth, **kw)
    if plot:
        ax3.plot(cast['t090c'], cast['t090c'].index, **plotkw)
        ax3.set_title("04-Smoothed")
        ax0.grid(True)
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        plt.show()

    # 08-Derive.
    cast['sal0'] = SP_from_C(cast['c0s/m'] * 10., cast['t090c'])
    cast['aou'] = apoxu(cast['t090c'], cast['sal0'],
                        cast['sbeox0mm/kg'])
    cast.name = fname
    return cast


if __name__ == '__main__':
    # Common.
    kw = dict(linestyle='-', marker='.', alpha=0.5)

    if False:  # TEST 00: Load FSI processed downcast.
        fsi = test_fsi()
        fig, ax = fsi.plot_vars(['TEMP', 'SAL'])

    if False:  # TEST 00: Load XBT-T6 EDF.
        xbt = test_xbt()
        fig, ax = xbt['temperature'].plot()

    if False:  # TEST 00: Load Seabird v7 ascii.
        cnv = test_cnv()
        fig, ax = cnv.t090c.plot(color='k')

    if False:  # TEST 01: Load, split, and plot.
        cnv = test_cnv()
        down, up = cnv.split()  # FIXME: Lose Metadata in the split() method.
        # Compute salinity.
        down['sal00'] = SP_from_C(down['c0s/m'] * 10., down['t090c'])
        fig, ax = down.plot_vars(['t090c', 'sal00'])

    if False:  # TEST 02: Filter pressure.
        down = test_cnv().split()[0]
        unfilt = down.index  # Store unfiltered pressure for comparison.
        filt = seabird_filter(down.index, sample_rate=24.0, time_constat=0.15)
        down.index = filt  # Re-attach filtered for easy plots.
        fig, ax = down.t190c.plot(color='k', label='Filtered', **kw)
        ax.plot(down.t190c, unfilt, color='r', label='Unfiltered', **kw)
        ax.axis([19.20, 19.55, 149, 148])  # 1 meter zoom to check the filter.
        ax.legend(numpoints=1, loc='best')

    if False:  # TEST 02: Check pressure reversals.
        down = test_cnv().split()[0]
        # Need to filter first to avoid false positives.
        down.index = seabird_filter(down.index, sample_rate=24.0,
                                    time_constat=0.15)

        fig, ax = down.t090c.plot(color='r', label='With reversals', **kw)
        down = down.press_check()  # NOTE: Masking applies to all observations.
        ax.plot(down.t090c, down.index, color='k', label='No reversals', **kw)
        ax.legend(numpoints=1, loc='best')
        ax.axis([5.4, 6.4, 730, 640])  # Within 30 meters 8 big reversals!
        ax.set_ylabel("Pressure [dbar]")
        ax.set_xlabel(u"Temperature [\xb0C]")

    if False:  # TEST 02: Remove spikes.
        down = test_cnv().split()[0]
        down.index = seabird_filter(down.index, sample_rate=24.0,
                                    time_constat=0.15)
        down = down.press_check()
        down['sal00'] = SP_from_C(down['c0s/m'] * 10., down['t090c'])
        down = down.dropna()  # Remove pressure reversals mask.

        fig, ax = down.sal00.plot(color='k', label='Original', **kw)
        mask = down.sal00.despike(n1=2, n2=20, block=100).isnull()
        ax.plot(down.sal00[mask], down.index[mask], 'r.', label='Spikes')
        ax.set_title("Original vs De-spiked data")
        ax.legend(numpoints=1, loc='best')
        ax.axis([37.27, 37.30, 88.51, 0])
        ax.set_ylabel("Pressure [dbar]")
        ax.set_xlabel(r"Salinity [g kg$^{-1}$]")

    if False:  # TEST 02: Loop Edit.
        cast = test_cnv(fname='data/AMB09_059_CTD_rad3.cnv.gz',
                        compression='gzip').split()[0]
        # Smooth velocity.
        cast['dz/dtm'] = movingaverage(cast['dz/dtm'], window_size=48)

        fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True)
        ax0.plot(cast['t090c'], cast['t090c'].index, label='Original',
                 markersize=8., alpha=0.65)

        # Filter pressure.
        kw = dict(sample_rate=24.0, time_constat=0.15)
        cast.index = seabird_filter(cast.index, **kw)

        ax1.plot(cast['t090c'], cast['t090c'].index, 'k.-', markersize=8.,
                 alpha=0.5)

        # Check inversion.
        t090c = cast['t090c'].copy()  # Store original values.
        cast = cast.press_check()

        # Threshold velocity from SeaBird manual and get press_check mask.
        mask_v = cast['dz/dtm'] < 0.25
        mask_p = cast['t090c'].isnull()

        kw = dict(marker='o', linestyle='none', markeredgecolor='w', zorder=2,
                  markersize=8., alpha=0.65)
        ax1.plot(cast['t090c'][mask_v], cast.index[mask_v],
                 markerfacecolor='g', label='Velocity', **kw)

        ax1.plot(t090c[mask_p], t090c.index[mask_p], markerfacecolor='r',
                 label='Pressure', **kw)
        ax0.invert_yaxis()
        ax0.grid(True)
        ax1.grid(True)
        kw = dict(numpoints=1, loc='lower right')
        ax0.legend(**kw)
        ax1.legend(**kw)
        ax0.axis([0.9, 1.2, 3850, 3760])
        fig.suptitle(u"Temperature [\xb0C]")
        ax0.set_ylabel("Pressure [dbar]")

    if False:  # TEST 02: Bin average.
        down = test_cnv().split()[0]
        down.index = seabird_filter(down.index, sample_rate=24.0,
                                    time_constat=0.15)
        down = down.press_check()
        down['sal00'] = SP_from_C(down['c0s/m'] * 10., down['t090c'])
        dsp = dict(n1=2, n2=20, block=100)
        down = down.apply(Series.despike, **dsp)

        fig, ax = down.t090c.plot(color='k', marker='.', label='Original')
        down = down.apply(Series.bindata, db=1.)

        ax.plot(down.t090c, down.index, 'r.', label='Binned')
        ax.set_title("Binned vs De-spiked data")
        ax.set_ylabel("Pressure [db]")
        ax.set_xlabel(u"Temperature [\xb0C]")
        ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
        ax.axis([15.4, 15.75, 206.5, 192.5])
        ax.grid()

    if False:  # TEST 02: Smooth.
        down = test_cnv().split()[0]
        temp = down.t090c.copy()
        temp.index = seabird_filter(temp.index, sample_rate=24.0,
                                    time_constat=0.15)
        temp = temp.press_check()
        temp = temp.despike(n1=2, n2=20, block=15)
        temp = temp.bindata(db=1.)
        fig, ax = temp.plot(color='k', marker='.', label='Original')
        temp = temp.smooth(window_len=21, window='hanning')
        ax.plot(temp, temp.index, 'r.', label='Smoothed')
        ax.set_title("Smoothed vs Binned data")
        ax.set_ylabel("Pressure [db]")
        ax.set_xlabel(u"Temperature [\xb0C]")
        ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
        ax.axis([15.3, 17.3, 207, 155])

    if False:  # TEST 03: Read a radial.
        lon, lat = [], []
        rad = 4
        pattern = 'data/AMB09_*_CTD_rad%s.cnv.gz' % rad
        fnames = sorted(glob(pattern), key=alphanum_key)
        Temp, Sal, Oxy = OrderedDict(), OrderedDict(), OrderedDict()

        for fname in fnames:
            cast = proc_ctd(fname)
            lon.append(cast.longitude.mean())
            lat.append(cast.latitude.mean())
            Temp.update({cast.name: cast['t090c']})
            Sal.update({cast.name: cast['sal0']})
            Oxy.update({cast.name: cast['aou']})

        Temp, Sal, Oxy = map(DataFrame.from_dict, (Temp, Sal, Oxy))
        Oxy.lon = lon
        Oxy.lat = lat
        store = HDFStore("data/radial_%s.h5" % rad, 'w')
        store['Temp'] = Temp
        store['Oxy'] = Oxy
        store['Sal'] = Sal
        store['lon'] = Series(lon, index=Temp.columns)
        store['lat'] = Series(lat, index=Temp.columns)
        store.close()

    if True:  # Plot sections.
        from pandas import HDFStore
        from oceans.ctd import plot_section
        # FIXME: If I can find a way to pass metadata I can skip this step.

        def attach_position(df, pos):
            df.lon = pos[0]
            df.lat = pos[1]
            return df

        rad = 4
        store = HDFStore("data/radial_%s.h5" % rad, 'r')
        pos = store['lon'].values, store['lat'].values
        Temp = attach_position(store['Temp'], pos)
        Sal = attach_position(store['Sal'], pos)
        Oxy = attach_position(store['Oxy'], pos)
        store.close()

        fig, ax, cb = plot_section(Temp, inverse=True)
        fig, ax, cb = plot_section(Temp, inverse=True, filled=True)

        fig, ax, cb = plot_section(Sal, inverse=True)
        fig, ax, cb = plot_section(Sal, filled=True)

        fig, ax, cb = plot_section(Oxy, inverse=True)
        fig, ax, cb = plot_section(Oxy, filled=True)

    plt.show()
