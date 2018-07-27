from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

SITE_UNKNOWN = -1

class SiteTrajectory(object):
    """A trajectory capturing the dynamics of particles through a SiteNetwork."""

    def __init__(self,
                 site_network,
                 particle_assignments,
                 confidences = None):
        """
        :param SiteNetwork site_network:
        :param ndarray (n_frames, n_mobile) particle_assignments:
        :param ndarray (n_frames, n_mobile) confidences (optional): the confidence
            with which each assignment was made.
        """
        if particle_assignments.ndim != 2:
            raise ValueError("particle_assignments must be 2D")
        if particle_assignments.shape[1] != site_network.n_mobile:
            raise ValueError("particle_assignments has wrong shape %s" % particle_assignments.shape)

        self._sn = site_network
        self._traj = particle_assignments.copy()

        if not confidences is None:
            if confidences.shape != particle_assignments.shape:
                raise ValueError("confidences has wrong shape %s; should be %s" % (confidences.shape, particle_assignments.shape))
            self._confs = confidences

        self._real_traj = None

    def set_real_points(self, real_traj):
        """Assocaite this SiteTrajectory with a trajectory of points in real space.

        The trajectory is not copied, and should have shape (n_frames, n_total)
        """
        expected_shape = (self.n_frames, self._sn.n_total)
        if not real_traj.shape == expected_shape:
            raise ValueError("real_traj of shape %s does not have expected shape %s" % (real_traj.shape, expected_shape))
        self._real_traj = real_traj

    def remove_real_traj(self):
        """Forget associated real trajectory."""
        del self._real_traj
        self._real_traj = None

    def trajectory_for_particle(self, i, return_confidences = False):
        """Returns the array of sites particle i is assigned to over time."""
        if return_confidences and self._confs is None:
            raise ValueError("This SiteTrajectory has no confidences")
        if return_confidences:
            return self._traj[:, i], self._confs[:, i]
        else:
            return self._traj[:, i]

    def real_positions_for_site(self, site, return_confidences = False):
        if self._real_traj is None:
            raise ValueError("This SiteTrajectory has no real trajectory")
        if return_confidences and self._confs is None:
            raise ValueError("This SiteTrajectory has no confidences")

        assert site < self._sn.n_sites
        msk = self._traj == site
        pts = self._real_traj[:, self._sn.mobile_mask][msk]

        assert pts.shape[1] == 3

        if return_confidences:
            return pts, self._confs[msk].flatten()
        else:
            return pts

    @property
    def traj(self):
        """The underlying trajectory."""
        return self._traj

    @property
    def n_frames(self):
        return len(self._traj)
