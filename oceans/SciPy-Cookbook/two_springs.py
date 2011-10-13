#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# two_springs.py
#
# purpose:  solve a system of differential equations
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  27-Oct-2010
# modified: Wed 27 Oct 2010 11:57:00 AM EDT
#
# obs: http://www.scipy.org/Cookbook/CoupledSpringMassSystem
#

#
# two_springs.py
#
"""
This module defines the vector field for a spring-mass system
consisting of two masses and two springs.
"""

"""
The vector field

x1' = y1
y1' = (-b1 y1 - k1 (x1 - L1) + k2 (x2 - x1 - L2))/m1
x2' = y2
y2' = (-b2 y2 - k2 (x2 - x1 - L2))/m2
"""

def vectorfield(w,t,p):
    """
    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    m1, m2, k1, k2, L1, L2, b1, b2 = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1,
         (-b1*y1 - k1*(x1-L1) + k2*(x2-x1-L2))/m1,
         y2,
         (-b2*y2 - k2*(x2-x1-L2))]
    return f