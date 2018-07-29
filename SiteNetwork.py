from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

import matplotlib
from analysis.visualization import plotter, plot_atoms, plot_points, layers, DEFAULT_COLORS

class SiteNetwork(object):
    """A network of sites for some diffusive/mobile particle in a static lattice.

    Stores the locations of sites, their defining static atoms, and their "types".
    """

    def __init__(self,
                 structure,
                 static_mask,
                 mobile_mask):
        """
        :param Atoms structure: an ASE/Quippy :class:Atoms object containing the structure simulated in the trajectory
          Should be a representative/ideal/thermal-average structure.
        :param ndarray(bool) static_mask: Boolean mask indicating which atoms to consider immobile
        :param ndarray(bool) mobile_mask: Boolean mask indicating which atoms to track
        """

        assert static_mask.ndim == mobile_mask.ndim == 1, "The masks must be one-dimensional"
        assert len(structure) == len(static_mask) == len(mobile_mask), "The masks must have the same length as the # of atoms in the strucutre."

        # No overlap
        assert not np.any(static_mask & mobile_mask), "static_mask and mobile_mask cannot overlap."

        self.structure = structure
        self.static_mask = static_mask
        self.n_static = np.sum(static_mask)
        self.mobile_mask = mobile_mask
        self.n_mobile = np.sum(mobile_mask)

        # Create static structure
        self.static_structure = structure.copy()
        del self.static_structure[(~static_mask) | mobile_mask]
        assert len(self.static_structure) == self.n_static

        # Set variables
        self._centers = None
        self._vertices = None
        self._types = None

    def copy(self):
        """Returns a (shallowish) copy of self."""
        sn = type(self)(self.structure,
                        self.static_mask,
                        self.mobile_mask)

        if not self._centers is None:
            sn.centers = self._centers.copy()
            if not self._vertices is None:
                sn.vertices = list(self._vertices)
            if not self._types is None:
                sn.site_types = self._types.copy()

        return sn

    def __len__(self):
        return self.n_sites

    def __getitem__(self, key):
        if self._centers is None:
            raise ValueError("This SiteNetwork has no sites; can't slice.")

        mask = np.zeros(shape = len(self), dtype = np.bool)
        mask[key] = True # This will deal with wrong shapes and all kinds of fancy indexing

        sn = type(self)(self.structure,
                        self.static_mask,
                        self.mobile_mask)

        view = self._centers[mask]
        view.flags.writeable = False
        sn.centers = view

        if not self._vertices is None:
            sn.vertices = [v for i, v in enumerate(self._vertices) if mask[i]]

        if not self._types is None:
            view = self._types[mask]
            view.flags.writeable = False
            sn.site_types = view

        return sn

    def of_type(self, stype):
        """Returns a "view" to this SiteNetwork with only sites of a certain type."""
        if self._types is None:
            raise ValueError("This SiteNetwork has no type information.")

        if not stype in self._types:
            raise ValueError("This SiteNetwork has no sites of type %i" % stype)

        return self[self._types == stype]


    @property
    def n_sites(self):
        return len(self._centers)

    @property
    def n_total(self):
        return len(self.static_mask)

    @property
    def centers(self):
        view = self._centers.view()
        view.flags.writeable = False
        return view

    @centers.setter
    def centers(self, value):
        self._centers = value
        self._vertices = None
        self._types = None

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        if not len(value) == len(self._centers):
            raise ValueError("Wrong # of vertices %i; expected %i" % (len(value), len(self._centers)))
        self._vertices = value

    @property
    def site_types(self):
        if self._types is None:
            return None
        view = self._types.view()
        view.flags.writeable = False
        return view

    @site_types.setter
    def site_types(self, value):
        if not value.shape == (len(self._centers),):
            raise ValueError("Wrong # of types %i; expected %i" % (value.shape, len(self._centers)))
        self._types = value

    @property
    def n_types(self):
        return len(np.unique(self.site_types))

    @property
    def types(self):
        return np.unique(self.site_types)

    @plotter(is3D = True)
    def plot(self, **kwargs):
        pts_params = {'points' : self.centers}
        if not self._types is None:
            pts_params['color'] = [DEFAULT_COLORS[t] for t in self._types]
        else:
            pts_params['color'] = 'k'
        layers((plot_atoms,  {'atoms' : self.static_structure}),
               (plot_points, pts_params), **kwargs)
