#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# deco.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  22-Jun-2011
# modified: Wed 22 Jun 2011 02:40:29 PM EDT
#
# obs:
#

import numpy as np

class match_args_return(object):
    """
    Function decorator to homogenize input arguments and to
    make the output match the original input with respect to
    scalar versus array, and masked versus ndarray.
    """
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __call__(self, *args, **kw):
        # get pressure from keywords
        # and throw it in the list of args
        p = kw.get('p', None)
        if p is not None:
            args = list(args)
            args.append(p)

        # check if is array
        self.array = np.any([hasattr(a, '__iter__') for a in args])
        # check if is masked
        self.masked = np.any([np.ma.isMaskedArray(a) for a in args])
        newargs = [np.ma.atleast_1d(np.ma.masked_invalid(a)) for a in args]
        newargs = [a.astype(np.float) for a in newargs]
        # return p to keywords
        if p is not None:
            kw['p'] = newargs.pop()
        ret = self.func(*newargs, **kw)
        if not self.masked: # return a filled array if not masked
            ret = np.ma.filled(ret, np.nan)
        if not self.array: # return scalar if not array
            ret = ret[0]

        return ret