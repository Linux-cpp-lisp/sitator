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
        """The site assignments over time."""
        return self._traj

    @property
    def n_frames(self):
        """The number of frames in the trajectory."""
        return len(self._traj)

    @property
    def n_unassigned(self):
        """The total number of times a mobile particle is unassigned."""
        return np.sum(self._traj < 0)

    @property
    def n_assigned(self):
        """The total number of times a mobile particle was assigned to a site."""
        return self._sn.n_mobile * self.n_frames - self.n_unassigned

    @property
    def percent_unassigned(self):
        """Proportion of particle positions that are unassigned over all time."""
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
        """The real-space trajectory this ``SiteTrajectory`` is based on."""
        return self._real_traj

    def copy(self, with_computed = True):
        """Return a copy.

        Args:
            with_computed (bool): See ``SiteNetwork.copy()``.
        """
        st = self[:]
        st.site_network = st.site_network.copy(with_computed = with_computed)
        return st

    def set_real_traj(self, real_traj):
        """Assocaite this SiteTrajectory with a trajectory of points in real space.

        The trajectory is not copied.

        Args:
            real_traj (ndarray of shape (n_frames, n_total))
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
        """Returns the array of sites particle i is assigned to over time.

        Args:
            i (int)
            return_confidences (bool): If ``True``, also return the confidences
                with which those assignments were made.
        Returns:
            ndarray (int) of length ``n_frames``[, ndarray (float) length ``n_frames``]
        """
        if return_confidences and self._confs is None:
            raise ValueError("This SiteTrajectory has no confidences")
        if return_confidences:
            return self._traj[:, i], self._confs[:, i]
        else:
            return self._traj[:, i]


    def real_positions_for_site(self, site, return_confidences = False):
        """Get all real-space positions assocated with a site.

        Args:
            site (int)
            return_confidences (bool): If ``True``, the confidences with which
                each real-space position was assigned to ``site`` are also
                returned.

        Returns:
            ndarray (N, 3)[, ndarray (N)]
        """
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
        """Computes site occupancies.

        Adds site attribute ``occupancies`` to ``site_network``.

        In cases of multiple occupancy, this will be higher than the number of
        frames in which the site is occupied and could be over 1.0.

        Returns:
            ndarray of occupancies (length ``n_sites``)
        """
        occ = np.true_divide(np.bincount(self._traj[self._traj >= 0], minlength = self._sn.n_sites), self.n_frames)
        if self.site_network.has_attribute('occupancies'):
            self.site_network.remove_attribute('occupancies')
        self.site_network.add_site_attribute('occupancies', occ)
        return occ


    def check_multiple_occupancy(self, max_mobile_per_site = 1):
        """Count cases of "multiple occupancy" where more than one mobile share the same site at the same time.

        These cases usually indicate bad site analysis.

        Returns:
            int: the total number of multiple assignment incidents; and
            float: the average number of mobile atoms at any site at any one time.
        """
        from sitator.landmark.errors import MultipleOccupancyError
        n_more_than_ones = 0
        avg_mobile_per_site = 0
        divisor = 0
        for frame_i, site_frame in enumerate(self._traj):
            _, counts = np.unique(site_frame[site_frame >= 0], return_counts = True)
            count_msk = counts > max_mobile_per_site
            if np.any(count_msk):
                raise MultipleOccupancyError("%i mobile particles were assigned to only %i site(s) (%s) at frame %i." % (np.sum(counts[count_msk]), np.sum(count_msk), np.where(count_msk)[0], frame_i))
            n_more_than_ones += np.sum(counts > 1)
            avg_mobile_per_site += np.sum(counts)
            divisor += len(counts)
        avg_mobile_per_site /= divisor
        return n_more_than_ones, avg_mobile_per_site


    def assign_to_last_known_site(self, frame_threshold = 1):
        """Assign unassigned mobile particles to their last known site.

        Args:
            frame_threshold (int): The maximum number of frames between the last
                known site and the present frame up to which the last known site
                can be used.

        Returns:
            information dictionary of debugging/diagnostic information.
        """
        total_unknown = self.n_unassigned

        logger.info("%i unassigned positions (%i%%); assigning unassigned mobile particles to last known positions within %s frames..." % (total_unknown, 100.0 * self.percent_unassigned, frame_threshold))

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


    def jumps(self, unknown_as_jump = False):
        """Generator to iterate over all jumps in the trajectory.

        A jump is considered to occur "at the frame" when it first acheives its
        new site. For example,

         - Frame 0: Atom 1 at site 4
         - Frame 1: Atom 1 at site 5

        will yield a jump ``(1, 1, 4, 5)``.

        Args:
            unknown_as_jump (bool): If ``True``, moving from a site to unknown
                (or vice versa) is considered a jump; if ``False``, unassigned
                mobile atoms are considered to be at their last known sites.
        Yields:
            tuple: (frame_number, mobile_atom_number, from_site, to_site)
        """
        traj = self.traj
        n_mobile = self.site_network.n_mobile
        assert n_mobile == traj.shape[1]
        last_known = traj[0].copy()
        known = np.ones(shape = len(last_known), dtype = np.bool)
        jumped = np.zeros(shape = len(last_known), dtype = np.bool)
        for frame_i in range(1, self.n_frames):
            if not unknown_as_jump:
                np.not_equal(traj[frame_i], SiteTrajectory.SITE_UNKNOWN, out = known)

            np.not_equal(traj[frame_i], last_known, out = jumped)
            jumped &= known # Must be currently known to have jumped

            for atom_i in range(n_mobile):
                if jumped[atom_i]:
                    yield frame_i, atom_i, last_known[atom_i], traj[frame_i, atom_i]

            last_known[known] = traj[frame_i, known]


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
