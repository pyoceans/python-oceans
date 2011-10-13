#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# brownian-2D.py
#
# purpose:  run 2D brownian
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  27-Oct-2010
# modified: Wed 27 Oct 2010 12:10:33 PM EDT
#
# obs: http://www.scipy.org/Cookbook/BrownianMotion
#

import numpy
from pylab import plot, show, grid, axis, xlabel, ylabel, title

from brownian import brownian


def main():

    # The Wiener process parameter.
    delta = 0.25
    # Total time.
    T = 10.0
    # Number of steps.
    N = 500
    # Time step size
    dt = T/N
    # Initial values of x.
    x = numpy.empty((2,N+1))
    x[:, 0] = 0.0

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    # Plot the 2D trajectory.
    plot(x[0],x[1])

    # Mark the start and end points.
    plot(x[0,0],x[1,0], 'go')
    plot(x[0,-1], x[1,-1], 'ro')

    # More plot decorations.
    title('2D Brownian Motion')
    xlabel('x', fontsize=16)
    ylabel('y', fontsize=16)
    axis('equal')
    grid(True)
    show()


if __name__ == "__main__":
    main()

