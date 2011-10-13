#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# filtfilt.py
#
# purpose:  Matlab filtfilt
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  27-Oct-2010
# modified: Wed 27 Oct 2010 11:48:08 AM EDT
#
# obs: http://www.scipy.org/Cookbook/FiltFilt
#

from numpy import vstack, hstack, eye, ones, zeros, linalg, \
newaxis, r_, flipud, convolve, matrix, array
from scipy.signal import lfilter

def lfilter_zi(b,a):
    #compute the zi state from the filter parameters. see [Gust96].

    #Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996,
    # Volume 44, Issue 4

    n=max(len(a),len(b))

    zin = (  eye(n-1) - hstack( (-a[1:n,newaxis],
                                 vstack((eye(n-2), zeros(n-2))))))

    zid=  b[1:n] - a[1:n]*b[0]

    zi_matrix=linalg.inv(zin)*(matrix(zid).transpose())
    zi_return=[]

    #convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
      zi_return.append(float(zi_matrix[i][0]))

    return array(zi_return)




def filtfilt(b,a,x):
    #For now only accepting 1d arrays
    ntaps=max(len(a),len(b))
    edge=ntaps*3

    if x.ndim != 1:
        raise ValueError, "Filiflit is only accepting 1 dimension arrays."

    #x must be bigger than edge
    if x.size < edge:
        raise ValueError, "Input vector needs to be bigger than 3 * max(len(a),len(b)."

    if len(a) < ntaps:
        a=r_[a,zeros(len(b)-len(a))]

    if len(b) < ntaps:
        b=r_[b,zeros(len(a)-len(b))]

    zi=lfilter_zi(b,a)

    #Grow the signal to have edges for stabilizing
    #the filter with inverted replicas of the signal
    s=r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]
    #in the case of one go we only need one of the extrems
    # both are needed for filtfilt

    (y,zf)=lfilter(b,a,s,-1,zi*s[0])

    (y,zf)=lfilter(b,a,flipud(y),-1,zi*y[-1])

    return flipud(y[edge-1:-edge+1])



if __name__=='__main__':

    from scipy.signal import butter
    from scipy import sin, arange, pi, randn

    from pylab import plot, legend, show, hold

    t=arange(-1,1,.01)
    x=sin(2*pi*t*.5+2)
    #xn=x + sin(2*pi*t*10)*.1
    xn=x+randn(len(t))*0.05

    [b,a]=butter(3,0.05)

    z=lfilter(b,a,xn)
    y=filtfilt(b,a,xn)



    plot(x,'c')
    hold(True)
    plot(xn,'k')
    plot(z,'r')
    plot(y,'g')

    legend(('original','noisy signal','lfilter - butter 3 order','filtfilt - butter 3 order'))
    hold(False)
    show()