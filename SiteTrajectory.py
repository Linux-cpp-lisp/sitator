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

    @property
    def traj(self):
        """The underlying trajectory."""
        return self._traj

    @property
    def n_frames(self):
        return len(self._traj)

    @property
    def n_unassigned(self):
        return np.sum(self._traj < 0)

    @property
    def n_assigned(self):
        return self._sn.n_total - self.n_unassigned

    @property
    def percent_unassigned(self):
        return float(self.n_unassigned) / (self._sn.n_total * self.n_frames)

    @property
    def site_network(self):
        return self._sn

    @property
    def real_trajectory(self):
        return self._real_traj

    def set_real_traj(self, real_traj):
        """Assocaite this SiteTrajectory with a trajectory of points in real space.

        The trajectory is not copied, and should have shape (n_frames, n_total)
        """
        expected_shape = (self.n_frames, self._sn.n_total, 3)
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

    def get_site_occupancies(self):
        return np.true_divide(np.bincount(self._traj[self._traj >= 0]), self.n_frames)

    def assign_to_last_known_site(self, frame_threshold = 1, verbose = True):
        """Assign unassigned mobile particles to their last known site within
            `frame_threshold` frames.

        :returns: information dictionary of debugging/diagnostic information.
        """
        total_unknown = self.n_unassigned

        if verbose:
            print("%i unassigned positions (%i%%); assigning unassigned mobile particles to last known positions within %i frames..." % (total_unknown, 100.0 * self.percent_unassigned, frame_threshold))

        last_known = np.empty(shape = self._sn.n_mobile, dtype = np.int)
        last_known.fill(-1)
        time_unknown = np.zeros(shape = self._sn.n_mobile, dtype = np.int)
        avg_time_unknown = 0
        avg_time_unknown_div = 0
        max_time_unknown = 0
        total_reassigned = 0

        for i in xrange(self.n_frames):
            # All those unknown this frame
            unknown = self._traj[i] == -1
            # Update last_known for assigned sites
            last_known[~unknown] = self._traj[i][~unknown]

            times = time_unknown[~unknown]
            times = times[times != 0]

            if len(times) > 0:
                maxtime = np.max(times)
                if maxtime > frame_threshold:
                    max_time_unknown = maxtime
                avg_time_unknown += np.sum(times)
                avg_time_unknown_div += len(times)

            time_unknown[~unknown] = 0

            to_correct = unknown & (time_unknown < frame_threshold)

            self._traj[i][to_correct] = last_known[to_correct]
            total_reassigned += np.sum(to_correct)
            time_unknown[unknown] += 1

        res = None
        if avg_time_unknown_div > 0: # We corrected some unknowns
            avg_time_unknown = float(avg_time_unknown) / avg_time_unknown_div

            if verbose:
                print("  Maximum # of frames any mobile particle spent unassigned: %i" % max_time_unknown)
                print("  Avg. # of frames spent unassigned: %f" % avg_time_unknown)
                print("  Assigned %i/%i unassigned positions, leaving %i (%i%%) unknown" % (total_reassigned, total_unknown, self.n_unassigned, self.percent_unassigned))

            res = {
                'max_time_unknown' : max_time_unknown,
                'avg_time_unknown' : avg_time_unknown,
                'total_reassigned' : total_reassigned
            }
        else:
            if self.verbose:
                print("  None to correct.")

            res = {
                'max_time_unknown' : 0,
                'avg_time_unknown' : 0,
                'total_reassigned' : 0
            }

        return res
