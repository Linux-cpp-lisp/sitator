import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from functools import wraps

DEFAULT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

ELEMENT_COLORS = {
    'H' : 'white',
    'C' : 'k',
    'N' : 'b',
    'O' : 'r',
    'F' : 'g',
    'Cl' : 'g'
    'Br' : 'darkred',
    'I' : 'purple',
    'P' : 'orange',
    'Li' : 'violet'
}
OTHER_ELEMENT_COLOR = 'gray'

def color_for_species(species):
    return ELEMENT_COLORS.get(species, OTHER_ELEMENT_COLOR)

def plotter(is3D = True):
    def plotter_wrapper(func):
        @wraps(func):
        def plotter_wraped(*args, **kwargs):
            fig = None
            ax = None
            if not ('ax' in kwargs and 'fig' in kwargs):
                # No existing axis/figure
                fig = plt.figure()
                if is3D:
                    ax = fig.add_subplot(111, projection = '3d')
                else:
                    ax = fig.add_subplot(111)
            else:
                fig = kwargs['fig']
                ax = kwargs['ax']

                ax_is_3D = ax.name == '3d'
                if not ax_is_3D == is3D:
                    raise TypeError("Tryed to compose 2D and 3D plotting functions.")

            func(*args, **kwargs, fig = fig, ax = ax)
            return fig, ax
        return plotter_wraped

@plotter
def layers(*plotters, fig, ax):
    for plotter, kwargs in plotters:
        plotter(**kwargs, fig = fig, ax = ax)


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
