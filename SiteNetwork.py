from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

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

        self.structure = structure
        self.static_mask = static_mask
        self.n_static = np.sum(static_mask)
        self.mobile_mask = mobile_mask
        self.n_mobile = np.sum(mobile_mask)

        # Create static structure
        self.static_structure = structure.copy()
        del self.static_structure[(~static_mask) & mobile_mask]
        assert len(self.static_structure) == self.n_static

        # Set variables
        self._centers = None
        self._vertices = None # Internally, keep vertices in
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
                sn.types = self._types.copy()

        return sn

    def __len__(self):
        return self.n_sites

    @property
    def n_sites(self):
        return len(self._centers)

    @property
    def n_total(self):
        return len(self.static_mask)

    @property
    def centers(self):
        return self._centers

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
        return self._types

    @site_types.setter
    def site_types(self, value):
        if not value.shape == (len(self._centers),):
            raise ValueError("Wrong # of types %i; expected %i" % (value.shape, len(self._centers)))
        self._types = value
