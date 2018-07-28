import numpy as np

import itertools

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import ase
import ase.data

from analysis.util import PBCCalculator

from analysis.visualization.common import plotter, DEFAULT_COLORS, set_axes_equal, color_for_species

@plotter(is3D = True)
def plot_atoms(atoms, hide_species = (), wrap = False, fig = None, ax = None, i = None):

    mask = [not (e in hide_species) for e in atoms.get_chemical_symbols()]

    pts = atoms.get_positions()[mask]
    species = [s for i, s in enumerate(atoms.get_chemical_symbols()) if mask[i]]

    if wrap:
        pbcc = PBCCalculator(atoms.cell)
        pts = atoms.get_positions().copy()
        pbcc.wrap_points(pts)

    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c = [color_for_species(s) for s in species],
               s = [20.0 * ase.data.covalent_radii[ase.data.atomic_numbers[s]] for s in species])

    for cvec in atoms.cell:
        cvec = np.array([[0, 0, 0], cvec])
        ax.plot(cvec[:,0],
                cvec[:,1],
                cvec[:,2],
                color = "gray",
                alpha=0.5,
                linestyle="--")

    set_axes_equal(ax)

@plotter(is3D = True)
def plot_points(points, marker = 'x', fig = None, ax = None, i = None, **kwargs):
    if (type(points) is np.ndarray) and (points.ndim == 2):
        points = [points]

    for j, pts in enumerate(points):
        if 'c' in kwargs or 'color' in kwargs:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       marker = marker, **kwargs)
        else:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       color = matplotlib.cm.get_cmap('Dark2')(j + i), marker = marker, **kwargs)
