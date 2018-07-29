import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from functools import wraps

DEFAULT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:cyan', 'tab:red', 'tab:gray', 'tab:olive']

ELEMENT_COLORS = {
    'H' : 'white',
    'C' : 'k',
    'N' : 'b',
    'O' : 'r',
    'F' : 'g',
    'Cl' : 'g',
    'Br' : 'darkred',
    'I' : 'purple',
    'P' : 'orange',
    'Li' : 'violet'
}
OTHER_ELEMENT_COLOR = 'gray'

def color_for_species(species):
    return ELEMENT_COLORS.get(species, OTHER_ELEMENT_COLOR)

def plotter(is3D = True, **outer):
    def plotter_wrapper(func):
        @wraps(func)
        def plotter_wraped(*args, **kwargs):
            fig = None
            ax = None
            if not ('ax' in kwargs and 'fig' in kwargs):
                # No existing axis/figure
                fig = plt.figure(**outer)
                if is3D:
                    ax = fig.add_subplot(111, projection = '3d')
                else:
                    ax = fig.add_subplot(111)
            else:
                fig = kwargs['fig']
                ax = kwargs['ax']
                del kwargs['fig']
                del kwargs['ax']

                ax_is_3D = ax.name == '3d'
                if not ax_is_3D == is3D:
                    raise TypeError("Tryed to compose 2D and 3D plotting functions.")

            if not ('i' in kwargs):
                kwargs['i'] = 0

            func(*args, fig = fig, ax = ax, **kwargs)
            return fig, ax
        return plotter_wraped
    return plotter_wrapper

@plotter(is3D = True)
def layers(*args, **fax):
    i = fax['i']
    print i
    for p, kwargs in args:
        p(fig = fax['fig'], ax = fax['ax'], i = i, **kwargs)
        i += 1

def grid(*args, **kwargs):
    defaults = {
        'is3D' : True,
        'figsize' : (8, 6)
    }

    defaults.update(kwargs)

    is3D = defaults['is3D']
    del defaults['is3D']

    fig = plt.figure(**defaults)
    if is3D:
        axargs = {'projection' : '3d', 'aspect' : 'equal'}
    else:
        axargs = {}

    nrows = len(args)
    ncols = max(len(row) for row in args)

    for i, row in enumerate(args):
        for j, p in enumerate(row):
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1, **axargs)
            if isinstance(p, tuple):
                p[0](fig = fig, ax = ax, i = 0, **p[1])
            else:
                p(fig = fig, ax = ax, i = 0)

    fig.tight_layout()

    return fig

# From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
