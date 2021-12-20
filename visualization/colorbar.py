# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def col_bar(image, fontsize=None, fontname=None, ticks=None):
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.clim(0, 1)
    if fontsize is None or fontname is None:
        plt.colorbar(image, cax=cax, orientation="vertical")
    else:
        cbar = plt.colorbar(image, cax=cax, orientation="vertical", ticks=ticks)

        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family(fontname)
        cbar.ax.tick_params(labelsize=fontsize)
