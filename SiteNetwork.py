from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

class SiteNetwork(object):
    """A network of sites for some diffusive/mobile particle in a static lattice.

    Stores the locations of sites, their defining static atoms. Can also store
    information assigning mobile particles to sites over time.
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

        self._structure = structure
        self._static_mask = static_mask
        self._n_static = np.sum(static_mask)
        self._mobile_mask = mobile_mask
        self._n_mobile = np.sum(mobile_mask)

        # Create static structure
        self._static_structure = structure.copy()
        del self._static_structure[(~static_mask) & mobile_mask]
        assert len(self._static_structure) == self.n_static

        # Set variables
        self._centers = None
        self._vertices = None # Internally, keep vertices in

    def copy(self):
        """Returns a (deep) copy of self."""
        sn = type(self)(self._structure,
                        self._static_mask,
                        self._mobile_mask)
        if not self._centers is None:
            sn.set_sites(self._centers, self._vertices)
        if not self._particle_assignments is None:
            sn._particle_assignments = self._particle_assignments.copy()
        return sn

    def set_sites(centers, vertices = None):
        """Set centers (and vertices). Copies both arrays."""
        self._centers = centers.copy()
        if not vertices is None:
            self._vertices = vertices.copy()

    def set_particle_assignments(self, assignments):
        """Set the SiteNetwork's particle assignments. Does NOT copy `assignments`."""
        if self._centers is None:
            raise ValueError("SiteNetwork must have sites before particle assignments!")
        if assignments.ndim != 2 or assignments.shape[1] != self._n_mobile:
            raise ValueError("Assignments must be of shape (n_frames, n_mobile)")
        self._particle_assignments = assignments

    def __len__(self):
        return self.n_sites

    @property
    def n_sites(self):
        return len(self._centers)
    @property
    def n_static(self):
        return self._n_static
    @property
    def n_mobile(self):
        return self._n_mobile
    @property
    def n_total(self):
        return len(self._static_mask)
    @property
    def mobile_mask(self):
        return self._mobile_mask
    @property
    def static_mask(self):
        return self._static_mask
    @property
    def static_structure(self):
        return self._static_structure
    @property
    def centers(self):
        return self._centers
    @property
    def vertices(self):
        return self._vertices
