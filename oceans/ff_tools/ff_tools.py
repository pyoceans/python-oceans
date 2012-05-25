# -*- coding: utf-8 -*-
#
# ff_tools.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  12-Feb-2012
# modified: Fri 25 May 2012 03:59:47 PM EDT
#
# obs:
#

from __future__ import division

import warnings

import numpy as np
import numpy.ma as ma
from scipy.ndimage import map_coordinates

from dateutil import rrule, parser

__all__ = [
           'get_profile',
           'gen_dates',
           'princomp',
           'strip_mask',
           'shiftdim'
           ]


def get_profile(x, y, f, xi, yi, order=3):
    r"""Interpolate regular data.

    Parameters
    ----------
    x : two dimensional np.ndarray
        an array for the :math:`x` coordinates

    y : two dimensional np.ndarray
        an array for the :math:`y` coordinates

    f : two dimensional np.ndarray
        an array with the value of the function to be interpolated
        at :math:`x,y` coordinates.

    xi : one dimension np.ndarray
        the :math:`x` coordinates of the point where we want
        the function to be interpolated.

    yi : one dimension np.ndarray
        the :math:`y` coordinates of the point where we want
        the function to be interpolated.

    order : int
        the order of the bivariate spline interpolation


    Returns
    -------
    fi : one dimension np.ndarray
        the value of the interpolating spline at :math:`xi,yi`


    Examples
    --------
    >>> import numpy as np
    >>> import oceans.ff_tools as ff
    >>> x, y = np.meshgrid(range(360), range(91))
    >>> f = np.array(range(91 * 360)).reshape((91, 360))
    >>> Paris = [2.4, 48.9]
    >>> Rome = [12.5, 41.9]
    >>> Greenwich = [0, 51.5]
    >>> xi = Paris[0], Rome[0], Greenwich[0]
    >>> yi = Paris[1], Rome[1], Greenwich[1]
    >>> ff.get_profile(x, y, f, xi, yi, order=3)

    Notes
    -----
    http://mail.scipy.org/pipermail/scipy-user/2011-June/029857.html
    """

    x, y, f, xi, yi = map(np.asanyarray, (x, y, f, xi, yi))
    conditions = np.array([xi.min() < x.min(),
                           xi.max() > x.max(),
                           yi.min() < y.min(),
                           yi.max() > y.max()])

    if conditions.any():
        warnings.warn("Warning, extrapolation in being done!!")

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    jvals = (xi - x[0, 0]) / dx
    ivals = (yi - y[0, 0]) / dy

    coords = np.array([ivals, jvals])

    return  map_coordinates(f, coords, mode='nearest', order=order)


def gen_dates(start, end, dt=None):
    r"""Date range from `start` to `end` at `dt` intervals.

    Examples
    --------
    >>> import datetime
    >>> from dateutil import rrule
    >>> start = '1980-01-19'
    >>> end = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    list(gen_dates(start, end, dt=rrule.YEARLY))
    """
    dates = (rrule.rrule(dt, dtstart=parser.parse(start),
                             until=parser.parse(end)))
    return dates


def princomp(A, numpc=None):
    r"""Performs principal components analysis (PCA) on the n-by-p data matrix
    `A`.  Rows of A correspond to observations, columns to variables.

    Returns :
        coeff :
            is a p-by-p matrix, each column containing coefficients
            for one principal component.
        score :
            the principal component scores; that is, the representation
            of A in the principal component space. Rows of SCORE
            correspond to observations, columns to components.
        latent :
        a vector containing the eigenvalues
        of the covariance matrix of A.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import oceans.ff_tools as ff
    >>> # 2D dataset.
    >>> A = np.array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
                     [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])
    >>> coeff, score, latent = ff.princomp(A.T)
    >>> fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    >>> # Every eigenvector describe the direction
    >>> # of a principal component.
    >>> m = np.mean(A, axis=1)
    >>> ax1.plot([0, -coeff[0,0] * 2] + m[0], [0, -coeff[0,1] * 2] +
    ...                                                           m[1], '--k')
    >>> ax1.plot([0, coeff[1,0] * 2] + m[0], [0, coeff[1,1] * 2] +
    ...                                                           m[1], '--k')
    >>> ax1.plot(A[0,:], A[1,:], 'ob')  # The data.
    >>> ax1.axis('equal')
    >>> # New data.
    >>> ax2.plot(score[0,:], score[1,:], '*g')
    >>> ax2.axis('equal')
    >>> plt.show()
    >>> # 4D dataset.
    >>> A = np.array([[-1, 1, 2, 2],
    ...               [-2, 3, 1, 0],
    ...               [ 4, 0, 3,-1]], dtype=np.double)
    >>> coeff, score, latent = ff.princomp(A)
    >>> perc = np.cumsum(latent) / np.sum(latent)
    >>> fig, ax = plt.subplots()
    >>> # The following plot show that first two components account for
    >>> # 100% of the variance.
    >>> ax.stem(range(len(perc)), perc, '--b')
    >>> ax.axis([-0.3, 4.3, 0, 1.3])
    >>> plt.show()
    >>> # Image example:
    >>> import matplotlib.cbook as cbook
    >>> from matplotlib.ticker import NullLocator
    >>> img = cbook.get_sample_data('lena.jpg', asfileobj=False)
    >>> A = plt.imread(img) # load an image
    >>> A = np.mean(A, 2)  # to get a 2-D array
    >>> full_pc = np.size(A, axis=1)  # numbers of all the principal components
    >>> i, dist = 1, []
    >>> for numpc in range(0, full_pc+20, 20):
    >>>     coeff, score, latent = ff.princomp(A, numpc)
    >>>     # Reconstruction.difference in Frobenius norm
    >>>     Ar = np.dot(coeff, score).T + np.mean(A, axis=0)
    >>>     dist.append(np.linalg.norm(A - Ar, 'fro'))
    >>>     # Showing the pics reconstructed with less than 50 PCs
    >>>     if numpc <= 50:
    >>>         i += 1
    >>>         print(i)
    >>>         ax = plt.subplot(2, 4, i, frame_on=False)
    >>>         ax.xaxis.set_major_locator(NullLocator())  # remove ticks
    >>>         ax.yaxis.set_major_locator(NullLocator())
    >>>         i += 1
    >>>         plt.imshow(np.flipud(Ar))
    >>>         plt.title('PCs # ' + str(numpc))
    >>>         plt.gray()
    >>> plt.figure()
    >>> plt.imshow(flipud(A))
    >>> plt.title('numpc FULL')
    >>> plt.gray()
    >>> plt.show()
    # At the end of this experiment, we can plot the distance of the
    # reconstructed images from the original image in Frobenius norm
    # (red curve) and the cumulative sum of the eigenvalues (blue curve).
    # Recall that the cumulative sum of the eigenvalues shows the level of
    # variance accounted by each of the corresponding eigenvectors. On the x
    # axis there is the number of eigenvalues/eigenvectors used.
    >>> plt.figure()
    >>> perc = np.cumsum(latent) / np.sum(latent)
    >>> dist = dist / np.max(dist)
    >>> plt.plot(range(len(perc)), perc, 'b',
    >>>          range(0, full_pc + 10, 10), dist, 'r')
    >>> plt.axis([0, full_pc, 0, 1.1])
    >>> plt.show()

    Notes
    -----
    http://glowingpython.blogspot.com/
    2011/07/principal-component-analysis-with-numpy.html?spref=tw
    """

    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # Subtract the mean along columns.
    [latent, coeff] = np.linalg.eig(np.cov(M))

    if numpc:
        p = np.size(coeff, axis=1)
        idx = np.argsort(latent)  # sorting the eigenvalues
        idx = idx[::-1]  # in ascending order
        # Sorting eigenvectors according to the sorted eigenvalues.
        coeff = coeff[:, idx]
        latent = latent[idx]  # sorting eigenvalues

        if numpc < p or numpc >= 0:
            coeff = coeff[:, range(numpc)]  # Cutting some PCs.

    score = np.dot(coeff.T, M)  # Projection of the data in the new space.
    return coeff, score, latent


def strip_mask(arr, fill_value=np.NaN):
    r"""Take a masked array and return its data(filled) + mask."""
    if ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        arr = np.ma.filled(arr, fill_value)
        return mask, arr
    else:
        return arr


def shiftdim(x, n=None):
    r""" Matlab's shiftdim in python.

    Examples
    --------
    >>> import oceans.ff_tools as ff
    >>> a = np.random.rand(1,1,3,1,2)
    >>> print("a shape and dimension: %s, %s" % (a.shape, a.ndim))
    a shape and dimension: (1, 1, 3, 1, 2), 5
    >>> # print(range(a.ndim))
    >>> # print(np.roll(range(a.ndim), -2))
    >>> # print(a.transpose(np.roll(range(a.ndim), -2)))
    >>> b = ff.shiftdim(a)
    >>> print("b shape and dimension: %s, %s" % (b.shape, b.ndim))
    b shape and dimension: (3, 1, 2), 3
    >>> c = ff.shiftdim(b, -2)
    >>> c.shape == a.shape
    True

    Notes
    -----
    http://www.python-it.org/forum/index.php?topic=4688.0
    """

    def no_leading_ones(shape):
        shape = np.atleast_1d(shape)
        if shape[0] == 1:
            shape = shape[1:]
            return no_leading_ones(shape)
        else:
            return shape

    if n is None:
        # returns the array B with the same number of
        # elements as X but with any leading singleton
        # dimensions removed.
        return x.reshape(no_leading_ones(x.shape))
    elif n >= 0:
        # When n is positive, shiftdim shifts the dimensions
        # to the left and wraps the n leading dimensions to the end.
        return x.transpose(np.roll(range(x.ndim), -n))
    else:
        # When n is negative, shiftdim shifts the dimensions
        # to the right and pads with singletons.
        return x.reshape((1,) * -n + x.shape)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
