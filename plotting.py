import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import ase

DEFAULT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] * 2

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

def plot_atoms(atms, species, pts=None, pts_cs = None, cell = None, hide_species = (), wrap = False, pts_marker = 'x'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cs = {
        'Li' : "blue", 'O' : "red", 'Ta' : "gray", 'Ge' : "darkgray", 'P' : "orange",
        'point' : 'black'
    }

    if wrap and not cell is None:
        atms = np.asarray([collapse_into_unit_cell(pt, cell) for pt in atms])
        if not pts is None:
            pts = np.asarray([collapse_into_unit_cell(pt, cell) for pt in pts])

    for s in hide_species:
        cs[s] = 'none'

    ax.scatter(atms[:,0],
               atms[:,1],
               atms[:,2],
               c = [cs[e] for e in species],
               s = [10.0*(ase.data.atomic_numbers[s])**0.5 for s in species])


    if not pts is None:
        c = None
        if pts_cs is None:
            c = cs['point']
        else:
            c = pts_cs
        ax.scatter(pts[:,0],
                   pts[:,1],
                   pts[:,2],
                   marker = pts_marker,
                   c = c,
                   cmap=matplotlib.cm.Spectral)

    if not cell is None:
        for cvec in cell:
            cvec = np.array([[0, 0, 0], cvec])
            ax.plot(cvec[:,0],
                   cvec[:,1],
                   cvec[:,2],
                   color = "gray",
                    alpha=0.5,
                   linestyle="--")

    set_axes_equal(ax)

    return ax
