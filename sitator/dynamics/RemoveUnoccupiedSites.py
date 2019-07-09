import numpy as np

from sitator import SiteTrajectory

class RemoveUnoccupiedSites(object):
    def __init__(self):
        pass

    def run(self, st):
        """
        """
        assert isinstance(st, SiteTrajectory)

        old_sn = st.site_network

        seen_mask = np.zeros(shape = old_sn.n_sites, dtype = np.bool)

        for frame in st.traj:
            seen_mask[frame] = True

        n_new_sites = np.sum(seen_mask)
        translation = np.empty(shape = old_sn.n_sites, dtype = np.int)
        translation[seen_mask] = np.arange(n_new_sites)
        translation[~seen_mask] = -4321

        newtraj = translation[st.traj.reshape(-1)]
        newtraj.shape = st.traj.shape

        newsn = old_sn[seen_mask]

        new_st = SiteTrajectory(
            site_network = newsn,
            particle_assignments = newtraj
        )
        if st.real_trajectory is not None:
            new_st.set_real_traj(st.real_trajectory)
        return new_st
