# -*- coding: utf-8 -*-
#
#
# exemplo_matlabpipe.py
#
# purpose:  Examplo on how to wrap a matlab function for python.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  17-Jul-2012
# modified: Wed 12 Sep 2012 02:12:06 PM BRT
#
# obs: Requires mlabwrap from oceans.
#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from oceans.mlabwrap import MatlabPipe


def LineCurvature2Dpy(Vertices, Lines=None, close_matlab=True):
    r"""Call matlab and run LineCurvature2D."""
    matlab = MatlabPipe(matlab_process_path='guess', matlab_version='2011a')
    matlab.open()

    matlab.put({'Vertices': Vertices})

    if isinstance(Lines, np.ndarray):
        matlab.put({'Lines': Lines})
        cmd1 = "k = LineCurvature2D(Vertices, Lines)"
        cmd2 = "N = LineNormals2D(Vertices, Lines)"
    elif not Lines:
        cmd1 = "k = LineCurvature2D(Vertices)"
        cmd2 = "N = LineNormals2D(Vertices)"
    else:
        print("Lines is passed but not recognized.")

    _, _ = matlab.eval(cmd1), matlab.eval(cmd2)
    k, N = matlab.get('k'), matlab.get('N')

    if close_matlab:
        matlab.close()

    return k, N

if __name__ == '__main__':
    data = sio.loadmat('testdata.mat', squeeze_me=True)

    Lines, Vertices = data['Lines'], data['Vertices']
    Lines = np.int_(Lines)

    k, N = LineCurvature2Dpy(Vertices, Lines)
    k = k * 100

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.c_[Vertices[:, 0], Vertices[:, 0] + k * N[:, 0]].T,
            np.c_[Vertices[:, 1], Vertices[:, 1] + k * N[:, 1]].T, 'g')
    ax.plot(np.c_[Vertices[Lines[:, 0] - 1, 0],
                  Vertices[Lines[:, 1] - 1, 0]].T,
            np.c_[Vertices[Lines[:, 0] - 1, 1],
                  Vertices[Lines[:, 1] - 1, 1]].T, 'b')
    ax.plot(Vertices[:, 0], Vertices[:, 1], 'r.')

    plt.show()
