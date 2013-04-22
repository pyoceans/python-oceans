# -*- coding: utf-8 -*-
#
# plotting.py
#
# purpose:  Some plotting helper functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Thu 18 Apr 2013 12:25:08 PM BRT
#
# obs:  rstyle, rhist and rbox are from:
# http://messymind.net/2012/07/making-matplotlib-look-like-ggplot/
#


from __future__ import division

from textwrap import dedent

import matplotlib
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.pyplot import MultipleLocator, rcParams, Polygon

__all__ = [
           'simpleaxis',
           'rstyle',
           'rhist',
           'rbox',
           'landmask',
           'LevelColormap',
           'get_pointsxy',
           'EditPoints'
          ]


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def rstyle(ax):
    """Styles an axes to appear like ggplot2.  Must be called after all plot
    and axis manipulation operations have been carried out (needs to know final
    tick spacing).

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.stats
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(0.0, 100.0, 0.1)
    >>> s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot(t, s, label="Original")
    >>> _ = ax.plot(t, s * 2, label="Doubled")
    >>> _ = ax.legend()
    >>> rstyle(ax)
    >>> plt.show()
    """

    # Set the style of the major and minor grid lines, filled blocks.
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)

    # Set minor tick spacing to 1/2 of the major ticks.
    ax.xaxis.set_minor_locator(MultipleLocator((plt.xticks()[0][1] -
                                                plt.xticks()[0][0]) / 2.0))
    ax.yaxis.set_minor_locator(MultipleLocator((plt.yticks()[0][1] -
                                                plt.yticks()[0][0]) / 2.0))

    # Remove axis border.
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    # Restyle the tick lines.
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    # Remove the minor tick lines.
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    # Only show bottom left ticks, pointing out of axis.
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if ax.legend_:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)


def rhist(ax, data, **keywords):
    """Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.stats
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(0.0, 100.0, 0.1)
    >>> s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)
    >>> fig, ax = plt.subplots()
    >>> data = scipy.stats.norm.rvs(size=1000)
    >>> _ = rhist(ax, data, label="Histogram")
    >>> _ = ax.legend()
    >>> rstyle(ax)
    >>> plt.show()
    """

    defaults = {
                'facecolor': '0.3',
                'edgecolor': '0.28',
                'linewidth': '1',
                'bins': 100
                }

    for k, v in defaults.items():
        if k not in keywords:
            keywords[k] = v

    return ax.hist(data, **keywords)


def rbox(ax, data, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with
    the following additions:
        Keyword arguments:
        colors -- array-like collection of colors for box fills.
        names -- array-like collection of box names which are passed on as tick
        labels.

    Examples
    --------
    >>> import scipy.stats
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> data = [scipy.stats.norm.rvs(size=100), scipy.stats.norm.rvs(size=100),
    ...         scipy.stats.norm.rvs(size=100)]
    >>> _ = ax.legend()
    >>> _ = rbox(ax, data, names=("One", "Two", "Three"),
    ...          colors=('white', 'cyan'))
    >>> rstyle(ax)
    >>> plt.show()
    """

    hasColors = 'colors' in keywords
    if hasColors:
        colors = keywords['colors']
        keywords.pop('colors')

    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'])
        keywords.pop('names')

    bp = ax.boxplot(data, **keywords)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle='solid')
    plt.setp(bp['fliers'], color='black', alpha=0.9, marker='o',
               markersize=3)
    plt.setp(bp['medians'], color='black')

    numBoxes = len(data)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX, boxY)

        if hasColors:
            boxPolygon = Polygon(boxCoords, facecolor=colors[i % len(colors)])
        else:
            boxPolygon = Polygon(boxCoords, facecolor='0.95')

        ax.add_patch(boxPolygon)
    return bp


def landmask(M, color='0.8'):
    r"""Plot land mask.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/mpl_util.py."""
    # Make a constant colormap, default = grey
    constmap = np.matplotlib.colors.ListedColormap([color])

    jmax, imax = M.shape
    # X and Y give the grid cell boundaries,
    # one more than number of grid cells + 1
    # half integers (grid cell centers are integers)
    X = -0.5 + np.arange(imax + 1)
    Y = -0.5 + np.arange(jmax + 1)

    # Draw the mask by pcolor.
    M = ma.masked_where(M > 0, M)
    plt.pcolor(X, Y, M, shading='flat', cmap=constmap)


def LevelColormap(levels, cmap=None):
    r"""Make a colormap based on an increasing sequence of levels.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/mpl_util.py."""

    # Start with an existing colormap.
    if not cmap:
        cmap = plt.get_cmap()

    # Spread the colors maximally.
    nlev = len(levels)
    S = np.arange(nlev, dtype='float') / (nlev - 1)
    A = cmap(S)

    # Normalize the levels to interval [0, 1]
    levels = np.array(levels, dtype='float')
    L = (levels - levels[0]) / (levels[-1] - levels[0])

    # Make the colour dictionary
    R = [(L[i], A[i, 0], A[i, 0]) for i in xrange(nlev)]
    G = [(L[i], A[i, 1], A[i, 1]) for i in xrange(nlev)]
    B = [(L[i], A[i, 2], A[i, 2]) for i in xrange(nlev)]
    cdict = dict(red=tuple(R), green=tuple(G), blue=tuple(B))

    return matplotlib.colors.LinearSegmentedColormap('%s_levels' %
                                                     cmap.name, cdict, 256)


def get_pointsxy(points):
    r"""Return x, y of the given point object."""
    return points.get_xdata(), points.get_ydata()


class EditPoints(object):
    r"""Edit points on a graph with the mouse.  Handles only one set of points.

    Key-bindings:
      't' toggle on and off.  (When on, you can move, delete, or add points.)
      'd' delete the point.
      'i' insert a point.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> theta = np.arange(0, 2 * np.pi, 0.1)
    >>> r = 1.5
    >>> xs = r * np.cos(theta)
    >>> ys = r * np.sin(theta)
    >>> points = ax.plot(xs, ys, 'ko')
    >>> p = EditPoints(fig, ax, points[0], verbose=True)
    >>> _ = ax.set_title('Click and drag a point to move it')
    >>> _ = ax.axis([-2, 2, -2, 2])
    >>> plt.show()

    Based on http://matplotlib.org/examples/event_handling/poly_editor.html
    """

    epsilon = 5  # Maximum pixel distance to count as a point hit.
    showpoint = True

    def __init__(self, fig, ax, points, verbose=False):
        matplotlib.interactive(True)
        if points is None:
            raise RuntimeError("""First add points to a figure or canvas.""")
        canvas = fig.canvas
        self.ax = ax
        self.dragged = None
        self.points = points
        self.verbose = verbose
        x, y = get_pointsxy(points)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r',
                           linestyle='none', animated=True)
        self.ax.add_line(self.line)

        if False:  # FIXME:  Not really sure how to use this.
            cid = self.points.add_callback(self.points_changed)
        self._ind = None  # The active point.

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event',
                           self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        if self.verbose:
            print("\nDrawing...")

    def points_changed(self, points):
        r"""This method is called whenever the points object is called."""
        # Only copy the artist props to the line (except visibility).
        vis = self.line.get_visible()
        Artist.update_from(self.line, points)
        # Don't use the points visibility state.
        self.line.set_visible(vis)

        if self.verbose:
            print("\nPoints modified.")

    def get_ind_under_point(self, event):
        r"""Get the index of the point under mouse if within epsilon
            tolerance."""

        # Display coordinates.
        arr = self.ax.transData.transform(self.points.get_xydata())
        x, y = arr[:, 0], arr[:, 1]
        d = np.sqrt((x - event.x) ** 2 + (y - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if self.verbose:
            print("d[ind] %s epsilon %s" % (d[ind], self.epsilon))
        if d[ind] >= self.epsilon:
            ind = None

        if self.verbose:
            print("\nClicked at (%s, %s)" % (event.xdata, event.ydata))
        return ind

    def button_press_callback(self, event):
        r"""Whenever a mouse button is pressed."""
        if not self.showpoint:
            return
        if not event.inaxes:
            return
        if not event.button:
            return
        self._ind = self.get_ind_under_point(event)

        # Get point position.
        x, y = get_pointsxy(self.points)
        self.pick_pos = (x[self._ind], y[self._ind])

        if self.verbose:
            print("\nGot point: (%s), ind: %s" % (self.pick_pos, self._ind))

    def button_release_callback(self, event):
        r"""Whenever a mouse button is released."""
        if not self.showpoint:
            return
        if not event.button:
            return
        self._ind = None
        if self.verbose:
            print("\nButton released.")

    def key_press_callback(self, event):
        r"""Whenever a key is pressed."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showpoint = not self.showpoint
            self.line.set_visible(self.showpoint)
            if not self.showpoint:
                self._ind = None

            if self.verbose:
                print("\nToggle %d" % self.showpoint)
            return get_pointsxy(self.points)
        elif event.key == 'd':
            x, y = get_pointsxy(self.points)
            ind = self.get_ind_under_point(event)
            if ind is not None:
                if self.verbose:
                    print("\nDeleted (%s, %s) ind: %s" % (x[ind], y[ind], ind))
                x = np.delete(x, ind)
                y = np.delete(y, ind)
                self.points.set_xdata(x)
                self.points.set_ydata(y)
                self.line.set_data(self.points.get_data())
        elif event.key == 'i':
            if self.verbose:
                print("Insert point")
            xs, ys = self.points.get_xdata(), self.points.get_ydata()
            ex, ey = event.xdata, event.ydata
            for i in range(len(xs) - 1):
                d = np.sqrt((xs[i] - event.xdata) ** 2 +
                            (ys[i] - event.ydata) ** 2)
                self.points.set_xdata(np.r_[self.points.get_xdata(), ex])
                self.points.set_ydata(np.r_[self.points.get_ydata(), ey])
                self.line.set_data(self.points.get_data())
                if self.verbose:
                    print("\nInserting: (%s, %s)" % (ex, ey))
                break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        r"""On mouse movement."""
        if not self.showpoint:
            return
        if not self._ind:
            return
        if not event.inaxes:
            return
        if not event.button:
            return
        x, y = get_pointsxy(self.points)
        dx = event.xdata - self.pick_pos[0]
        dy = event.ydata - self.pick_pos[1]
        x[self._ind] = self.pick_pos[0] + dx
        y[self._ind] = self.pick_pos[1] + dy
        if self.verbose:
            print("\nevent.xdata %s" % event.xdata)
            print("\nevent.ydata %s" % event.ydata)
        self.points.set_xdata(x)
        self.points.set_ydata(y)
        self.line.set_data(zip(self.points.get_data()))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        if self.verbose:
            print("\nMoving")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
