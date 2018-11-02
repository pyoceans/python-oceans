import os
from colorsys import hsv_to_rgb
from glob import glob

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

cmap_path = os.path.join(os.path.dirname(__file__), 'cmap_data')


class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self


def get_color(color):
    """
    https://stackoverflow.com/questions/10254207/color-and-line-writing-using-matplotlib

    """
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in hsv_to_rgb(hue, 1.0, 230)]
        yield '#{0:02x}{1:02x}{2:02x}'.format(*col)


def cmat2cmpl(rgb, reverse=False):
    """
    Convert RGB matplotlib colormap.

    """
    rgb = np.asanyarray(rgb)
    if reverse:
        rgb = np.flipud(rgb)
    return colors.ListedColormap(rgb)


def phasemap_cm(m=256):
    """
    Colormap periodic/circular data (phase).

    """

    theta = 2 * np.pi * np.arange(0, m) / m
    circ = np.exp(1j * theta)

    # Vertices of colour triangle.
    vred, vgreen, vblue = -2, 1 - np.sqrt(3) * 1j, 1 + np.sqrt(3) * 1j

    vredc = vred - circ
    vgreenc = vgreen - circ
    vbluec = vblue - circ

    red = np.abs(np.imag(vgreenc * np.conj(vbluec)))
    green = np.abs(np.imag(vbluec * np.conj(vredc)))
    blue = np.abs(np.imag(vredc * np.conj(vgreenc)))

    return (1.5 * np.c_[red, green, blue] /
            np.abs(np.imag((vred - vgreen) * np.conj(vred - vblue))))


def zebra_cm(a=4, m=0.5, n=256):
    """
    Zebra palette colormap with NBANDS broad bands and NENTRIES rows in
    the color map.

    The default is 4 broad bands

    cmap = zebra(nbands, nentries)

    References
    ----------
    Hooker, S. B. et al, Detecting Dipole Ring Separatrices with Zebra
    Palettes, IEEE Transactions on Geosciences and Remote Sensing, vol. 33,
    1306-1312, 1995

    Notes
    -----
    Saturation and value go from m to 1 don't use m = 0.
    a = 4 -> there are this many large bands in the palette.

    """
    from scipy.signal import sawtooth

    x = np.arange(0, n)
    hue = np.exp(-3. * x / n)
    sat = m + (1. - m) * (0.5 * (1. + sawtooth(2. * np.pi * x / (n / a))))
    val = m + (1. - m) * 0.5 * (1. + np.cos(2. * np.pi * x / (n / a / 2.)))
    return np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, sat, val)])


def ctopo_pos_neg_cm(m=256):
    """
    Colormap for positive/negative data with gray scale only
    original from cushman-roisin book cd-rom.

    """
    dx = 1. / (m - 1)
    values = np.arange(0., 1., dx)
    return np.c_[values, values, values]


def avhrr_cm(m=256):
    """
    AHVRR colormap used by NOAA Coastwatch.

    """

    x = np.arange(0.0, m) / (m - 1)

    xr = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    rr = [0.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.5]

    xg = [0.0, 0.4, 0.6, 1.0]
    gg = [0.0, 1.0, 1.0, 0.0]

    xb = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    bb = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.5]

    r = np.interp(x, xr, rr)
    g = np.interp(x, xg, gg)
    b = np.interp(x, xb, bb)

    return np.flipud(np.c_[r, g, b])


def load_cmap(fname):
    return np.loadtxt(fname, delimiter=',') / 255


# Functions colormaps.
arrays = {
    'zebra': zebra_cm(),
    'avhrr': avhrr_cm(),
    'phasemap': phasemap_cm(),
    'ctopo_pos_neg': ctopo_pos_neg_cm()
}

# Data colormaps.
for fname in glob('%s/*.dat' % cmap_path):
    cmap = os.path.basename(fname).split('.')[0]
    data = load_cmap(fname)
    arrays.update({cmap: data})

cm = Bunch()
for key, value in arrays.items():
    cm.update({key: cmat2cmpl(value)})
    cm.update({'%s_r' % key: cmat2cmpl(value, reverse=True)})


def demo():
    data = np.outer(np.arange(0, 1, 0.01), np.ones(10))
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)
    cmaps = sorted((m for m in cm.keys() if not m.endswith('_r')))
    length = len(cmaps)
    for k, cmap in enumerate(cmaps):
        plt.subplot(1, length + 1, k + 1)
        plt.axis('off')
        plt.imshow(data, aspect='auto', cmap=cm.get(cmap), origin='lower')
        plt.title(cmap, rotation=90, fontsize=10)
