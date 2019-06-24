import numpy as np

from sitator.util import PBCCalculator
from sitator.visualization import SiteTrajectoryPlotter
from sitator.util.progress import tqdm

import logging
logger = logging.getLogger(__name__)

class SiteTrajectory(object):
    """A trajectory capturing the dynamics of particles through a SiteNetwork."""

    SITE_UNKNOWN = -1

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
        else:
            self._confs = None

        self._real_traj = None

        self._default_plotter = None

    def __len__(self):
        return self.n_frames

    def __getitem__(self, key):
        st = type(self)(self._sn,
                        self._traj[key],
                        confidences = None if self._confs is None else self._confs[key])
        if not self._real_traj is None:
            st.set_real_traj(self._real_traj[key])
        return st

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
        return self._sn.n_mobile * self.n_frames - self.n_unassigned

    @property
    def percent_unassigned(self):
        return float(self.n_unassigned) / (self._sn.n_mobile * self.n_frames)

    @property
    def site_network(self):
        return self._sn

    @site_network.setter
    def site_network(self, value):
        # Captures len, #, and dist.
        assert np.all(value.mobile_mask == self._sn.mobile_mask)
        assert np.all(value.static_mask == self._sn.static_mask)
        self._sn = value

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


    def compute_site_occupancies(self):
        """Computes site occupancies and adds site attribute `occupancies` to site_network."""
        occ = np.true_divide(np.bincount(self._traj[self._traj >= 0], minlength = self._sn.n_sites), self.n_frames)
        self.site_network.add_site_attribute('occupancies', occ)
        return occ


    def assign_to_last_known_site(self, frame_threshold = 1):
        """Assign unassigned mobile particles to their last known site within
            `frame_threshold` frames.

        :returns: information dictionary of debugging/diagnostic information.
        """
        total_unknown = self.n_unassigned

        logger.info("%i unassigned positions (%i%%); assigning unassigned mobile particles to last known positions within %i frames..." % (total_unknown, 100.0 * self.percent_unassigned, frame_threshold))

        last_known = np.empty(shape = self._sn.n_mobile, dtype = np.int)
        last_known.fill(-1)
        time_unknown = np.zeros(shape = self._sn.n_mobile, dtype = np.int)
        avg_time_unknown = 0
        avg_time_unknown_div = 0
        max_time_unknown = 0
        total_reassigned = 0

        for i in range(self.n_frames):
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

            logger.info("  Maximum # of frames any mobile particle spent unassigned: %i" % max_time_unknown)
            logger.info("  Avg. # of frames spent unassigned: %f" % avg_time_unknown)
            logger.info("  Assigned %i/%i unassigned positions, leaving %i (%i%%) unknown" % (total_reassigned, total_unknown, self.n_unassigned, self.percent_unassigned))

            res = {
                'max_time_unknown' : max_time_unknown,
                'avg_time_unknown' : avg_time_unknown,
                'total_reassigned' : total_reassigned
            }
        else:
            logger.info("  None to correct.")

            res = {
                'max_time_unknown' : 0,
                'avg_time_unknown' : 0,
                'total_reassigned' : 0
            }

        return res


    # ---- Plotting code
    def plot_frame(self, *args, **kwargs):
        if self._default_plotter is None:
            self._default_plotter = SiteTrajectoryPlotter()
        self._default_plotter.plot_frame(self, *args, **kwargs)

    def plot_site(self, *args, **kwargs):
        if self._default_plotter is None:
            self._default_plotter = SiteTrajectoryPlotter()
        self._default_plotter.plot_site(self, *args, **kwargs)

    def plot_particle_trajectory(self, *args, **kwargs):
        if self._default_plotter is None:
            self._default_plotter = SiteTrajectoryPlotter()
        self._default_plotter.plot_particle_trajectory(self, *args, **kwargs)
