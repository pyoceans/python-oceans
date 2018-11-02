import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.dates import date2num
from matplotlib.lines import Line2D

import numpy as np
import numpy.ma as ma


from oceans.ocfis import cart2pol


def stick_plot(time, u, v, **kw):
    """
    Parameters
    ----------
    time: list/arrays of datetime objects
    u, v: list/arrays of 2D vector components.

    Returns
    -------
    q: matplotlib's quiver handle for quiverkey.

    Examples
    --------
    >>> from pandas import date_range
    >>> time = date_range(start='1990-11-01 00:00', end='1991-2-1 00:00')
    >>> u = np.sin(0.1 * time.to_julian_date().values) ** 2 -0.5
    >>> v = np.cos(0.1 * time.to_julian_date().values)
    >>> fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 6),
    ...                                     sharex=True)
    >>> q = stick_plot(time, u, v, ax=ax0)
    >>> qk = ax0.quiverkey(q, 0.2, 0.65, 1, "1 m s$^{-1}$",
    ...                    labelpos='N', coordinates='axes')
    >>> l = ax1.plot(time.to_pydatetime(), np.sqrt(u**2 + v**2), label='speed')
    >>> l0 = ax2.plot(time.to_pydatetime(), u, label='u')
    >>> l1 = ax2.plot(time.to_pydatetime(), v, label='v')

    Based on Stephane Raynaud's example from:
    https://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg18051.html

    """
    from pandas import DatetimeIndex

    width = kw.pop('width', 0.002)
    headwidth = kw.pop('headwidth', 0)
    headlength = kw.pop('headlength', 0)
    headaxislength = kw.pop('headaxislength', 0)
    angles = kw.pop('angles', 'uv')
    ax = kw.pop('ax', None)

    if angles != 'uv':
        raise AssertionError('Stickplot angles must be `uv` so that'
                             'if *U*==*V* the angle of the arrow on'
                             'the plot is 45 degrees CCW from the *x*-axis.')

    if isinstance(time, DatetimeIndex):
        time = time.to_pydatetime()

    time, u, v = list(map(np.asanyarray, (time, u, v)))
    if not ax:
        fig, ax = plt.subplots()

    q = ax.quiver(date2num(time), [[0]*len(time)], u, v,
                  angles='uv', width=width, headwidth=headwidth,
                  headlength=headlength, headaxislength=headaxislength, **kw)

    ax.axes.get_yaxis().set_visible(False)
    ax.xaxis_date()
    return q


def landmask(M, color='0.8'):
    """
    Plot land mask.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/2011/07/mpl_util.py.

    """
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


def level_colormap(levels, cmap=None):
    """
    Make a colormap based on an increasing sequence of levels.
    http://www.trondkristiansen.com/wp-content/uploads/downloads/
    2011/07/mpl_util.py.

    """
    # Start with an existing colormap.
    if not cmap:
        cmap = plt.get_cmap()

    # Spread the colors maximally.
    nlev = len(levels)
    S = np.arange(nlev, dtype='float') / (nlev - 1)
    A = cmap(S)

    # Normalize the levels to interval [0, 1].
    levels = np.array(levels, dtype='float')
    L = (levels - levels[0]) / (levels[-1] - levels[0])

    # Make the color dictionary.
    R = [(L[i], A[i, 0], A[i, 0]) for i in range(nlev)]
    G = [(L[i], A[i, 1], A[i, 1]) for i in range(nlev)]
    B = [(L[i], A[i, 2], A[i, 2]) for i in range(nlev)]
    cdict = {
        'red': tuple(R),
        'green': tuple(G),
        'blue': tuple(B)
    }

    return matplotlib.colors.LinearSegmentedColormap('%s_levels' % cmap.name, cdict, 256)


def get_pointsxy(points):
    """
    Return x, y of the given point object.

    """
    return points.get_xdata(), points.get_ydata()


def compass(u, v, **arrowprops):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, -0.5, -0.50, +0.90]
    >>> v = [+1, +0.5, -0.45, -0.85]
    >>> fig, ax = compass(u, v)

    """

    # Create plot.
    fig, ax = plt.subplots(subplot_kw={'polar': True})

    angles, radii = cart2pol(u, v)

    # Arrows or sticks?
    kw = {'arrowstyle': '->'}
    kw.update(arrowprops)
    [ax.annotate('', xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax


def plot_spectrum(data, fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t).

    """
    n = len(data)  # Length of the signal.
    k = np.arange(n)
    T = n / fs
    frq = k / T  # Two sides frequency range.
    N = list(range(n // 2))
    frq = frq[N]  # One side frequency range

    # FFT computing and normalization.
    Y = np.fft.fft(data) / n
    Y = Y[N]

    # Plotting the spectrum.
    plt.semilogx(frq, np.abs(Y), 'r')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()


class EditPoints(object):
    """
    Edit points on a graph with the mouse.  Handles only one set of points.

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

    Based on https://matplotlib.org/examples/event_handling/poly_editor.html

    """
    epsilon = 5  # Maximum pixel distance to count as a point hit.
    showpoint = True

    def __init__(self, fig, ax, points, verbose=False):
        matplotlib.interactive(True)
        if points is None:
            raise RuntimeError('First add points to a figure or canvas.')
        canvas = fig.canvas
        self.ax = ax
        self.dragged = None
        self.points = points
        self.verbose = verbose
        x, y = get_pointsxy(points)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r',
                           linestyle='none', animated=True)
        self.ax.add_line(self.line)

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
            print('\nDrawing...')  # noqa

    def points_changed(self, points):
        """
        This method is called whenever the points object is called.

        """
        # Only copy the artist props to the line (except visibility).
        vis = self.line.get_visible()
        Artist.update_from(self.line, points)
        # Don't use the points visibility state.
        self.line.set_visible(vis)

        if self.verbose:
            print('\nPoints modified.')  # noqa

    def get_ind_under_point(self, event):
        """
        Get the index of the point under mouse if within epsilon tolerance.

        """
        # Display coordinates.
        arr = self.ax.transData.transform(self.points.get_xydata())
        x, y = arr[:, 0], arr[:, 1]
        d = np.sqrt((x - event.x) ** 2 + (y - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if self.verbose:
            print('d[ind] {} epsilon {}'.format(d[ind], self.epsilon))  # noqa
        if d[ind] >= self.epsilon:
            ind = None

        if self.verbose:
            print('\nClicked at ({}, {})'.format(event.xdata, event.ydata))  # noqa
        return ind

    def button_press_callback(self, event):
        """
        Whenever a mouse button is pressed.

        """
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
            print('\nGot point: ({}), ind: {}'.format(self.pick_pos, self._ind))  # noqa

    def button_release_callback(self, event):
        """
        Whenever a mouse button is released.

        """
        if not self.showpoint:
            return
        if not event.button:
            return
        self._ind = None
        if self.verbose:
            print('\nButton released.')  # noqa

    def key_press_callback(self, event):
        """
        Whenever a key is pressed.

        """
        if not event.inaxes:
            return
        if event.key == 't':
            self.showpoint = not self.showpoint
            self.line.set_visible(self.showpoint)
            if not self.showpoint:
                self._ind = None

            if self.verbose:
                print('\nToggle {:d}'.format(self.showpoint))  # noqa
            return get_pointsxy(self.points)
        elif event.key == 'd':
            x, y = get_pointsxy(self.points)
            ind = self.get_ind_under_point(event)
            if ind is not None:
                if self.verbose:
                    print('\nDeleted ({}, {}) ind: {}'.format(x[ind], y[ind], ind))  # noqa
                x = np.delete(x, ind)
                y = np.delete(y, ind)
                self.points.set_xdata(x)
                self.points.set_ydata(y)
                self.line.set_data(self.points.get_data())
        elif event.key == 'i':
            if self.verbose:
                print('Insert point')  # noqa
            xs = self.points.get_xdata()
            ex, ey = event.xdata, event.ydata
            for i in range(len(xs) - 1):
                self.points.set_xdata(np.r_[self.points.get_xdata(), ex])
                self.points.set_ydata(np.r_[self.points.get_ydata(), ey])
                self.line.set_data(self.points.get_data())
                if self.verbose:
                    print('\nInserting: ({}, {})'.format(ex, ey))  # noqa
                break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        """
        On mouse movement.

        """
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
            print('\nevent.xdata {}'.format(event.xdata))  # noqa
            print('\nevent.ydata {}'.format(event.ydata))  # noqa
        self.points.set_xdata(x)
        self.points.set_ydata(y)
        self.line.set_data(list(zip(self.points.get_data())))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        if self.verbose:
            print('\nMoving')  # noqa
