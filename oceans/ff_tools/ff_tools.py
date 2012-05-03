# -*- coding: utf-8 -*-
#
# ff_tools.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  12-Feb-2012
# modified: Thu 03 May 2012 10:04:04 AM EDT
#
# obs:
#

from __future__ import division

import numpy as np
import numpy.ma as ma


def sub2ind(shape, I, J, row_major=True):
    r"""
    Quick-and-dirty matlab sub2ind substitute
    TODO: enter integers-math-with-float-output-integer
    """
    if row_major:
        ind = (I % shape[0]) * shape[1] + (J % shape[1])
    else:
        ind = (J % shape[1]) * shape[0] + (I % shape[0])

    ind = np.int64(ind)

    #inds = sub2ind(shape, z, y, x)
    # or
    #inds = z + y * shape[0] + x * shape[0] * shape[1]

    return ind


def del_eta_del_x(U, f, g, balance, R=None):
    r"""
    Calculate :mat: `\frac{\partial \eta} {\partial x}`
    for different force balances

    Parameters:
    ----------
    U : array_like
        velocity magnitude [m/s]
    balance : str
              geostrophic, gradient or max_gradient
    """

    if balance == 'geostrophic':
        detadx = f * U / g

    elif balance == 'gradient':
        detadx = (U ** 2 / R + f * U) / g

    elif balance == 'max_gradient':
        detadx = (R * f ** 2) / (4 * g)

    return detadx


def get_profile(x, y, f, xi, yi, order=3):
    import scipy.ndimage.map_coordinates
    r"""
    Interpolate regular data.

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


    """
    conditions = [xi.min() < x.min(),
                  xi.max() > x.max(),
                  yi.min() < y.min(),
                  yi.max() > y.max()]

    if True in conditions:
        print "Warning, extrapolation in being done!!"

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    jvals = (xi - x[0, 0]) / dx
    ivals = (yi - y[0, 0]) / dy

    coords = np.array([ivals, jvals])

    return  scipy.ndimage.map_coordinates(f, coords, mode='nearest',
                                                        order=order)


def gen_dates(start, end, dt=None):
    r"""
    start = '1980-01-19'
    end = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    list(gen_dates(start, end, dt=rrule.YEARLY))
    """
    from dateutil import rrule, parser
    dates = (rrule.rrule(dt,
                         dtstart=parser.parse(start),
                         until=parser.parse(end)))
    return dates


def princomp(A, numpc=None):
    r"""
    Performs principal components analysis (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables.

    http://glowingpython.blogspot.com/
    2011/07/principal-component-analysis-with-numpy.html?spref=tw

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
    # 2D dataset.
    >>> A = np.array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
                     [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])
    >>> coeff, score, latent = princomp(A.T)
    >>> plt.figure()
    >>> plt.subplot(121)
    # every eigenvector describe the direction
    # of a principal component.
    >>> m = np.mean(A, axis=1)
    >>> plt.plot([0, -coeff[0,0] * 2] + m[0], [0, -coeff[0,1] * 2] +
    ...                                                           m[1], '--k')
    >>> plt.plot([0, coeff[1,0] * 2] + m[0], [0, coeff[1,1] * 2] +
    ...                                                           m[1], '--k')
    >>> plt.plot(A[0,:], A[1,:], 'ob')  # The data.
    >>> plt.axis('equal')
    >>> plt.subplot(122)
    # New data.
    >>> plt.plot(score[0,:], score[1,:], '*g')
    >>> plt.axis('equal')
    >>> plt.show()

    # 4D dataset.
    >>> A = np.array([[-1, 1, 2, 2],
                      [-2, 3, 1, 0],
                      [ 4, 0, 3,-1]], dtype=np.double)
    >>> coeff, score, latent = princomp(A)
    >>> perc = np.cumsum(latent) / np.sum(latent)
    >>> plt.figure()
    # The following plot show that first two components
    # account for 100% of the variance.
    >>> plt.stem(range(len(perc)), perc, '--b')
    >>> plt.axis([-0.3,4.3,0,1.3])
    >>> plt.show()
    >>> print('the principal component scores')
    >>> print(score.T)  # Only the first two columns are nonzero.
    >>> print('The rank of A is')
    >>> print(np.rank(A))  # Indeed, the rank of A is 2.

    # Image example:
    >>> A = plt.imread('image.jpg') # load an image
    >>> A = np.mean(A, 2)  # to get a 2-D array
    >>> full_pc = np.size(A, axis=1)  # numbers of all the principal components
    >>> i, dist = 1, []
    >>> for numpc in range(0, full_pc+10, 10):  # 0 10 20 ... full_pc
    >>>     coeff, score, latent = princomp(A, numpc)
    >>>     Ar = np.dot(coeff, score).T + np.mean(A, axis=0)  # Reconstruction.
            # difference in Frobenius norm
    >>>     dist.append(linalg.norm(A - Ar, 'fro'))
            # showing the pics reconstructed with less than 50 PCs
    >>>     if numpc <= 50:
    >>>         ax = plt.subplot(2, 3, i, frame_on=False)
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
    r"""
    Take a masked array and return its data(filled) + mask

    """
    if ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        arr = np.ma.filled(arr, fill_value)
        return mask, arr
    else:
        return arr


def shiftdim(x, n=None):
    """
    #http://www.python-it.org/forum/index.php?topic=4688.0
    a = np.rand(1,1,3,1,2)
    print(a.shape)
    print(a.ndim)
    print(range(a.ndim))
    print(np.roll(range(a.ndim), -2))
    print(a.transpose(np.roll(range(a.ndim), -2)))

    b = shiftdim(a)
    print(b.shape)

    c = shiftdim(b, -2)
    print(c.shape)

    print(c==a)
    """

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


def no_leading_ones(x):
    """Used in shiftdim."""
    x = np.atleast_1d(x)
    if x[0] == 1:
        x = x[1:]
        return no_leading_ones(x)
    else:
        return x
