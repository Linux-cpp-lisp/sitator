import numpy as np

from sitator import SiteTrajectory

import logging
logger = logging.getLogger(__name__)


class RemoveUnoccupiedSites(object):
    """Remove unoccupied sites."""
    def __init__(self):
        pass

    def run(self, st, return_kept_sites = False):
        """
        Args:
            return_kept_sites (bool): If ``True``, a list of the sites from ``st``
                that were kept will be returned.

        Returns:
            A ``SiteTrajectory``, or ``st`` itself if it has no unoccupied sites.
        """
        assert isinstance(st, SiteTrajectory)

        old_sn = st.site_network

        # Allow for the -1 to affect nothing
        seen_mask = np.zeros(shape = old_sn.n_sites + 1, dtype = np.bool)

        for frame in st.traj:
            seen_mask[frame] = True
            if np.all(seen_mask[:-1]):
                return st

        seen_mask = seen_mask[:-1]

        logger.info("Removing unoccupied sites %s" % np.where(~seen_mask)[0])

        n_new_sites = np.sum(seen_mask)
        translation = np.empty(shape = old_sn.n_sites, dtype = np.int)
        translation[seen_mask] = np.arange(n_new_sites)
        translation[~seen_mask] = SiteTrajectory.SITE_UNKNOWN

        newtraj = translation[st.traj.reshape(-1)]
        newtraj.shape = st.traj.shape

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
