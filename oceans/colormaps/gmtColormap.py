# -*- coding: utf-8 -*-

# http://www.scipy.org/Cookbook/Matplotlib/Loading_a_colormap_dynamically

from __future__ import division

import colorsys

import numpy as np
import matplotlib as mpl


def gmtColormap(fileName, GMTPath=None):
    if not GMTPath:
        filePath = "/usr/local/cmaps/" + fileName + ".cpt"
    else:
        filePath = GMTPath + "/" + fileName + ".cpt"
    try:
        f = open(filePath)
    except:
        print("file ", filePath, "not found")
        return None

    lines = f.readlines()
    f.close()

    colorModel = "RGB"
    x, r, g, b = [], [], [], []

    for l in lines:
        ls = l.split()
        if l[0] == "#":
            if ls[-1] == "HSV":
                colorModel = "HSV"
                continue
            else:
                continue
        if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
            pass
        else:
            x.append(float(ls[0]))
            r.append(float(ls[1]))
            g.append(float(ls[2]))
            b.append(float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

    x.append(xtemp)
    r.append(rtemp)
    g.append(gtemp)
    b.append(btemp)

    #nTable = len(r)
    x = np.asanyarray(x, dtype=np.float)
    r = np.asanyarray(r, dtype=np.float)
    g = np.asanyarray(g, dtype=np.float)
    b = np.asanyarray(b, dtype=np.float)
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
            r[i], g[i], b[i] = rr, gg, bb
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
            r[i], g[i], b[i] = rr, gg, bb
    if colorModel == "RGB":
        r, g, b = r / 255., g / 255., b / 255.
    xNorm = (x - x[0]) / (x[-1] - x[0])

    red = []
    blue = []
    green = []
    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])
    colorDict = {"red": red, "green": green, "blue": blue}
    cmap = mpl.colors.LinearSegmentedColormap(fileName, colorDict, N=256,
                                              gamma=1.0)
    return cmap
