import numpy as np

from sitator import SiteTrajectory
from sitator.errors import InsufficientSitesError

import logging
logger = logging.getLogger(__name__)


class RemoveUnoccupiedSites(object):
    """Remove unoccupied sites."""
    def __init__(self):
        pass

    def run(self, st, threshold = 0., return_kept_sites = False):
        """
        Args:
            threshold (float): The minimal percentage (0.0-1.0) of the time a
                site has to be occupied to be considered occupied and not removed.
                Defaults to 0.0, i.e., if ever occupied will be kept.
            return_kept_sites (bool): If ``True``, a list of the sites from ``st``
                that were kept will be returned.

        Returns:
            A ``SiteTrajectory``, or ``st`` itself if it has no unoccupied sites.
        """
        assert isinstance(st, SiteTrajectory)

        old_sn = st.site_network

        threshold = int(threshold * st.n_frames)

        # Allow for the -1 to affect nothing
        count_buf = np.zeros(shape = old_sn.n_sites + 1, dtype = np.int)
        seen_mask = np.zeros(shape = old_sn.n_sites, dtype = np.bool)

        for frame in st.traj:
            count_buf[frame] += 1
            np.greater_equal(count_buf[:-1], threshold, out = seen_mask)
            if np.all(seen_mask):
                return st

        logger.info("Removing unoccupied sites %s" % np.where(~seen_mask)[0])

        n_new_sites = np.sum(seen_mask)

        if n_new_sites < old_sn.n_mobile:
            raise InsufficientSitesError(
                verb = "Removing unoccupied sites",
                n_sites = n_new_sites,
                n_mobile = old_sn.n_mobile
            )

        translation = np.empty(shape = old_sn.n_sites + 1, dtype = np.int)
        translation[:-1][seen_mask] = np.arange(n_new_sites)
        translation[:-1][~seen_mask] = -4321
        translation[-1] = SiteTrajectory.SITE_UNKNOWN # Map unknown to unknown

        newtraj = translation[st.traj.reshape(-1)]
        newtraj.shape = st.traj.shape

        assert -4321 not in newtraj

        # We don't clear computed attributes since nothing is changing for other sites.
        newsn = old_sn[seen_mask]

        new_st = SiteTrajectory(
            site_network = newsn,
            particle_assignments = newtraj
        )
        if st.real_trajectory is not None:
            new_st.set_real_traj(st.real_trajectory)
        if return_kept_sites:
            return new_st, np.where(seen_mask)
        else:
            return new_st
