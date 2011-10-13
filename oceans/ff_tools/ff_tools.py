from __future__ import division

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

def smoo1(datain, window_len=11, window='hanning'):
    r"""
    Smooth the data using a window with requested size.

    Parameters
    ----------
    datain : array_like
             input series
    window_len : int
                 size of the smoothing window; should be an odd integer
    window : str
             window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
             flat window will produce a moving average smoothing.

    Returns
    -------
    data_out : array_like
            smoothed signal

    See Also
    --------
    binave, binavg
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve scipy.signal.lfilter

    Notes
    -----
    original from: http://www.scipy.org/Cookbook/SignalSmooth
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with
    the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output signal.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import ff_tools as ff
    >>> time = np.linspace( -4, 4, 100 )
    >>> series = np.sin(time)
    >>> noise_series = series + np.random.randn( len(time) ) * 0.1
    >>> data_out = ff.smoo1(series)
    >>> ws = 31
    >>> plt.subplot(211)
    >>> plt.plot( np.ones(ws) )
    >>> windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    >>> plt.hold(True)
    >>> for w in windows[1:]:
    >>>     eval('plt.plot(np.'+w+'(ws) )')
    >>> plt.axis([0,30,0,1.1])
    >>> plt.legend(windows)
    >>> plt.title("The smoothing windows")
    >>> plt.subplot(212)
    >>> plt.plot(series)
    >>> plt.plot(noise_series)
    >>> for w in windows:
    >>>     plt.plot( ff.smoo1(noise_series, 10, w) )
    >>> l = ['original signal', 'signal with noise']
    >>> l.extend(windows)
    >>> plt.legend(l)
    >>> plt.title("Smoothing a noisy signal")
    >>> plt.show()

    TODO: window parameter can be the window itself (i.e. an array)
    instead of a string.
    """

    datain = np.asarray(datain)

    if datain.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if datain.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[ 2 * datain[0] - datain[window_len:1:-1], datain, 2 * datain[-1]
                                                  - datain[-1:-window_len:-1] ]

    if window == 'flat': # moving average
        w = np.ones(window_len,'d')
    else:
        w = eval( 'np.' + window + '(window_len)' )

    data_out = np.convolve( w / w.sum(), s , mode='same' )
    return data_out[window_len-1:-window_len+1]


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
        detadx = (U**2 / R + f*U ) / g #?

    elif balance == 'max_gradient':
        detadx = (R*f**2) / (4*g)

    return detadx

def get_profile( x, y, f, xi, yi, order=3):
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
    conditions = [ xi.min() < x.min(),
                   xi.max() > x.max(),
                   yi.min() < y.min(),
                   yi.max() > y.max() ]

    if True in conditions:
        print "Warning, extrapolation in being done!!"

    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]

    jvals = (xi - x[0,0]) / dx
    ivals = (yi - y[0,0]) / dy

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

    http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html?spref=tw

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
    >>> plt.plot([0, -coeff[0,0] * 2] + m[0], [0, -coeff[0,1] * 2] + m[1], '--k')
    >>> plt.plot([0, coeff[1,0] * 2] + m[0], [0, coeff[1,1] * 2] + m[1], '--k')
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
    >>>     Ar = np.dot(coeff, score).T + np.mean(A, axis=0)  # image reconstruction
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
    M = ( A - np.mean( A.T, axis=1 ) ).T  # Subtract the mean along columns.
    [latent, coeff] = np.linalg.eig(np.cov(M))

    if numpc:
        p = np.size(coeff, axis=1)
        idx = np.argsort(latent)  # sorting the eigenvalues
        idx = idx[::-1]  # in ascending order
        # Sorting eigenvectors according to the sorted eigenvalues.
        coeff = coeff[:,idx]
        latent = latent[idx]  # sorting eigenvalues

        if numpc < p or numpc >= 0:
            coeff = coeff[:,range(numpc)] # cutting some PCs

    score = np.dot(coeff.T, M) # projection of the data in the new space.
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