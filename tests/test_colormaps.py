# -*- coding: utf-8 -*-
#
# test_colormaps.py
#
# purpose:
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  08-Mar-2013
# modified: Sat 09 Mar 2013 03:38:03 PM BRT
#
# obs:
#

# TODO: Show all colormaps
import numpy as np
import matplotlib.pyplot as plt

data = np.outer(np.arange(0, 1, 0.01), np.ones(10))
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)
cmaps = sorted([m for m in plt.cm.datad if not m.endswith("_r")])
length = len(cmaps) + 1
for k, cm in enumerate(cmaps):
    plt.subplot(1, length, k + 1)
    plt.axis("off")
    plt.imshow(data, aspect='auto', cmap=plt.get_cmap(cm), origin="lower")
    plt.title(cm, rotation=90, fontsize=10)

#savefig("colormaps.png", dpi=100, facecolor='gray')
