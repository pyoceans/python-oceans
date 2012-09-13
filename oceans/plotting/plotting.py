# -*- coding: utf-8 -*-
#
# plotting.py
#
# purpose:  Some plotting helper functions
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  09-Sep-2011
# modified: Wed 12 Sep 2012 11:56:24 AM BRT
#
# obs:
#


from __future__ import division

import numpy as np

from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment

# Based on http://matplotlib.org/examples/event_handling/poly_editor.html

def get_pointsxy(points):
    r"""Return x, y of the given point object."""
    return points.get_xdata(), points.get_ydata()


class EditPoints(object):
    r"""Edit points on a graph with the mouse.

    Key-bindings

      't' toggle on and off.  When on, you can move, delete, or add points.

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
    #  Handles only one set of points.
    >>> points = ax.plot(xs, ys, 'ko')[0]
    >>> p = EditPoints(fig, ax, points)
    >>> ax.set_title('Click and drag a point to move it')
    >>> ax.set_xlim((-2, 2))
    >>> ax.set_ylim((-2, 2))
    >>> plt.show()
    """

    showpoint = True
    epsilon = 5  # max pixel distance to count as a point hit.

    def __init__(self, fig, ax, points):
        if points is None:
            raise RuntimeError("""You must first add the points to a figure or
            canvas before defining the interactor""")
        canvas = fig.canvas
        self.dragged = None
        self.ax = ax
        self.points = points
        x, y = get_pointsxy(points)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r',
                           linestyle='none', animated=True)
        self.ax.add_line(self.line)

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

        print("\nDrawing...")

    def points_changed(self, points):
        r"""This method is called whenever the points object is called."""
        # Only copy the artist props to the line (except visibility).
        vis = self.line.get_visible()
        Artist.update_from(self.line, points)
        # Don't use the points visibility state.
        self.line.set_visible(vis)

        print("\nPoints modified.")

    def get_ind_under_point(self, event):
        r"""Get the index of the point under mouse if within epsilon
            tolerance."""

        # Display coordinates.
        x, y = get_pointsxy(self.points)
        d = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        print("\nGot point (%s, %s) ind: %s" % (event.xdata, event.ydata, ind))
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

        # Testing:
        x, y = get_pointsxy(self.points)
        self.pick_pos = (x[self._ind], y[self._ind])
        print("\nPick: (%s, %s)" % self.pick_pos)

        print("\nself._ind: %s" % self._ind)

    def button_release_callback(self, event):
        r"""Whenever a mouse button is released."""
        if not self.showpoint:
            return
        if not event.button:
            return
        self._ind = None
        print("\nButton released %s" % self._ind)


    def key_press_callback(self, event):
        r"""Whenever a key is pressed."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showpoint = not self.showpoint
            self.line.set_visible(self.showpoint)
            if not self.showpoint:
                self._ind = None
            print("\nToggle")
            return get_pointsxy(self.points)
        elif event.key == 'd':
            x, y = get_pointsxy(self.points)
            ind = self.get_ind_under_point(event)
            if ind is not None:
                x = np.delete(x, ind)
                y = np.delete(y, ind)
                self.points.set_xdata(x)
                self.points.set_ydata(y)
                self.line.set_data(self.points.get_data())
            print("\nDeleted (%s, %s) ind: %s" % (x[ind], y[ind], ind))
        elif event.key == 'i':
            print("Insert point")
            xs, ys = self.points.get_xdata(), self.points.get_ydata()
            ex, ey = event.xdata, event.ydata
            for i in range(len(xs) - 1):
                d = np.sqrt((xs[i] - event.xdata) ** 2 +
                            (ys[i] - event.ydata) ** 2)
                if d <= self.epsilon:
                    self.points.set_xdata(np.r_[self.points.get_xdata(), ex])
                    self.points.set_ydata(np.r_[self.points.get_ydata(), ey])
                    self.line.set_data(self.points.get_data())
                    break
            print("\nInserting: (%s, %s)" % (event.xdata, event.ydata))

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
        print("\nevent.xdata %s" % event.xdata)
        print("\nevent.ydata %s" % event.ydata)
        self.points.set_xdata(x)
        self.points.set_ydata(y)
        self.line.set_data(zip(self.points.get_data()))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        print("\nMoving")


if __name__ == '__main__':
    import doctest
    doctest.testmod()

