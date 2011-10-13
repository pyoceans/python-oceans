#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# smooth-2d.py
#
# purpose:  Smooth 2D data (horizontal/vertical sections)
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  27-Oct-2010
# modified: Wed 27 Oct 2010 11:43:39 AM EDT
#
# obs: http://www.scipy.org/Cookbook/SignalSmooth
#  check also cookb_signalsmooth.py

from scipy import signal, mgrid, cos, random, exp

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)



X, Y = mgrid[-70:70, -70:70]
Z = cos((X**2+Y**2)/200.)+ random.normal(size=X.shape)
blur_image(Z, 3)