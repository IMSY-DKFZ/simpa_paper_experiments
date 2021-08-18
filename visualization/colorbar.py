import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def col_bar(image):
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.clim(0, 1)
    plt.colorbar(image, cax=cax, orientation="vertical")
