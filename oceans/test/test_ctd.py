# -*- coding: utf-8 -*-
#
# test_ctd.py
#
# purpose:  Test ctd.py
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  01-Mar-2013
# modified: Fri 01 Mar 2013 07:03:48 PM BRT
#
# obs:
#


import re
import unittest
import matplotlib.pyplot as plt

from glob import glob
from pandas import HDFStore
from collections import OrderedDict
from oceans.ctd import DataFrame, SP_from_C, despike_old, seabird_filter


# Load test.
def test_fsi(fname='data/d3_7165.txt.gz'):
    kw = dict(names=('SCANS', 'PRES', 'TEMP', 'COND', 'OXCU', 'OXTM', 'FLUO',
                     'OBS', 'SAL', 'DEN', 'SV', 'DEPTH'), compression='gzip',
              skiprows=10)
    return DataFrame.from_fsi(fname, **kw)


def test_xbt(fname='data/AMB09_037_XBT_rad3.EDF.zip', compression='zip'):
    return DataFrame.from_edf(fname, compression='zip')


def test_cnv(fname='data/AMB09_055_CTD_rad3.cnv.gz', compression='gzip'):
    return DataFrame.from_cnv(fname, compression=compression)
    return cast

if __name__ == '__main__':
    # TEST 00: Load.
    #fsi = test_fsi()  # FSI processed downcast.
    #fig, ax = fsi.plot_vars(['TEMP', 'SAL'])

    #xbt = test_xbt()  # XBT-T6 EDF.
    #fig, ax = xbt['temperature'].plot()

    cnv = test_cnv()  # Seabird v7 ascii.

    # TEST 01: plot and plot_vars.
    down, up = cnv.split()  # FIXME: Metadata disapear in the split() method.
    fig, ax = up.t090c.plot(color='k', marker='.')

    down['sal00'] = SP_from_C(down['c0s/m'] * 10., down['t090c'])
    fig, ax = down.plot_vars(['t090c', 'sal00'])

    # TEST 02: Process data.

    # Filter pressure.
    kw = dict(linestyle='-', marker='.', alpha=0.5)
    unfiltered = down.index
    filtered = seabird_filter(unfiltered, sample_rate=24.0, time_constat=0.15)
    down.index = filtered  # Re-attach filtered for easy plots.
    fig, ax = down.t190c.plot(color='k', label='Filtered', **kw)
    ax.plot(down.t190c, unfiltered, color='r', label='Unfiltered', **kw)
    ax.axis([19.20, 19.55, 149, 148])  # 1 meter zoom to check the filter.
    ax.legend(numpoints=1, loc='best')

    # Check pressure reversals.
    fixed = down.press_check()  # Mask is applied to all observations.
    fig, ax = down.t090c.plot(color='r', label='No reversals check', **kw)
    ax.plot(fixed.t090c, fixed.index, color='k', label='Reversals check', **kw)
    ax.legend(numpoints=1, loc='best')
    ax.axis([5.4, 6.4, 730, 640])  # 30 meters 8 big reversals!

    # Remove spikes.

    plt.show()

if 0:
    despiked = down.sal00.despike(std1=2., std2=20., window=100)
    despiked_old = despike_old(down.sal00, n1=2., n2=20., block=100)
    mask = despiked.isnull()
    fig, ax = down.sal00.plot(color='k', marker='.', alpha=0.5,
                                label='Original')
    ax.plot(down.sal00[mask], down.sal00.index[mask], 'rx', alpha=0.5,
            label='Spikes removed with Wild Edit -- pandas.rolling')
    ax.set_title("De-spiked vs Original data")
    ax.set_ylabel("Pressure [db]")
    ax.set_xlabel(r"Salinity [kg g^{-1}]")
    ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
    offset = 0.05
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
    mask = despiked_old.isnull()
    ax.plot(down.sal00[mask], down.sal00.index[mask], 'g.', alpha=0.5,
            label='Spikes removed with Wild Edit -- old loop')
if 0:  # Bin average.
    binned = cast.apply(Series.bindata, db=1.)
    # TODO: Add salinity at the same plot.
    fig, ax = plt.subplots()
    ax.plot(cast.t090c, cast.index, 'k-.', label='Original')
    ax.plot(binned.t090c, binned.index, 'r.', label='Binned')
    ax.set_title("Binned vs Original data")
    ax.set_ylabel("Pressure [db]")
    ax.set_xlabel("Temperature [%sC]" % degree)
    ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
    ax.invert_yaxis()
    ax.grid(True)
    offset = 0.05
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)
if 0:  # Smooth.
    t090c_smoo = cast.t090c.smooth(window_len=111, window='hanning')
    fig, ax = plt.subplots()
    ax.plot(cast.t090c, cast.index, 'r', linewidth=2.0,
            label='Original')
    ax.plot(t090c_smoo, t090c_smoo.index, 'k', alpha=0.5,
            label='Smoothed')
    ax.set_title("Smoothed vs Original data")
    ax.set_ylabel("Pressure [db]")
    ax.set_xlabel("Temperature [%sC]" % degree)
    ax.legend(shadow=True, fancybox=True, numpoints=1, loc='best')
    ax.invert_yaxis()
    offset = 0.05
    x1, x2 = ax.get_xlim()[0] - offset, ax.get_xlim()[1] + offset
    ax.set_xlim(x1, x2)

if 0:  # Read a radial.
    digits = re.compile(r'(\d+)')

    def tokenize(fname):
        return tuple(int(token) if match else token for token, match in
                    ((frag, digits.search(frag)) for frag in
                    digits.split(fname)))

    pattern = base + '/DATA_SJ/CTD/AMB09_*_CTD_rad1.zip'
    fnames = sorted(glob(pattern), key=tokenize)

    Temp = OrderedDict()
    lon, lat = [], []
    for fname in fnames:
        cast = DataFrame.from_cnv(fname, verbose=False)
        print(fname)
        temp = cast['t090c'].bindata()
        lon.append(cast.longitude.values[0])
        lat.append(cast.latitude.values[0])
        Temp.update({cast.name: temp})

    Temp = DataFrame.from_dict(Temp)
    Temp.lon = lon
    Temp.lat = lat

if 0:  # Save radial
    store = HDFStore("radial_01.h5", 'w')
    store['temperature'] = Temp
    store.close()
    # FIXME: Need to find a way to serialize the metadata!
    np.savez("radial_01.npz", lon=lon, lat=lat)

if 0:  # Plot sections.
    store = HDFStore("radial_01.h5", 'r')
    Temp = store['temperature']
    store.close()
    npz = np.load("radial_01.npz")
    Temp.lon = npz['lon']
    Temp.lat = npz['lat']

    fig, ax, cb = Temp.plot_section()
    fig, ax, cb = Temp.plot_section(filled=True)

if 0:  # Loop Edit test.
    fname = base + 'CTD/Translate_After_Align/AMB09_051_CTD_rad3.cnv.zip'
    #fname = base + 'CTD/Translate_After_Align/AMB09_004_CTD_rad1.cnv.zip'

    cast = DataFrame.from_cnv(fname, verbose=False)
    cast['sal'] = SP_from_C(cast['c0s/m'] * 10., cast['t090c'])
    # Smooth velocity.
    cast['dz/dtm'] = movingaverage(cast['dz/dtm'], window_size=48)

    downcast, upcast = cast.split()

    # Filter pressure.
    kw = dict(sample_rate=24.0, time_constat=0.15)
    downcast = downcast.seabird_filter(**kw)

    # Threshold velocity from SeaBird manual.
    mask_v = downcast['dz/dtm'] < 0.25

    # Check inversion.
    checked = downcast.press_check()
    mask_p = checked['t090c'].isnull()

    mask = ~mask_v * ~mask_p

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharex=True, sharey=True)
    kw = dict(markersize=8., alpha=0.65)
    ax0.plot(downcast['t090c'], downcast['Pressure [db]'], 'k.-', zorder=1,
            label='Original', **kw)  # Original.
    ax1.plot(downcast['t090c'], downcast['t090c'].index, 'k.-', zorder=1,
            label='Filtered', **kw)  # Filtered + masks.

    kw = dict(marker='o', linestyle='none', markerfacecolor='none',
            markeredgecolor='g', zorder=3, markersize=7.)
    ax1.plot(downcast['t090c'][mask_v], downcast.index[mask_v], **kw)

    kw = dict(marker='o', linestyle='none', markerfacecolor='r',
            markeredgecolor='none', zorder=2, markersize=6., alpha=0.65)
    ax1.plot(downcast['t090c'][mask_p], downcast.index[mask_p], **kw)
    ax2.plot(downcast['t090c'][mask], downcast.index[mask], 'k.-',
            label='Keep', markersize=8., alpha=0.65)  # Final.
    ax0.invert_yaxis()
    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax0.legend(numpoints=1)
    ax1.legend(numpoints=1)
    ax2.legend(numpoints=1)
    plt.show()
