import numpy as np

from analysis import SiteNetwork, SiteTrajectory
from analysis.dynamics import JumpAnalysis
from analysis.util import PBCCalculator

import markov_clustering

class MergeSitesByDynamics(object):
    """Merges sites using dynamical data.

    Given a SiteTrajectory, merges sites using Markov Clustering.

    :param float distance_threshold: Will never merge sites further than this
        in real space. Angstrom.
    :param bool check_types: If True, only sites of the same type are candidates to
        be merged; if false, type information is ignored. Merged sites will only
        be assigned types if this is True.
    """
    def __init__(self, distance_threshold = 1.0, check_types = True, verbose = True):
        self.verbose = verbose
        self.distance_threshold = distance_threshold
        self.check_types = check_types

    def run(self, st):
        """Takes a SiteTrajectory and returns a SiteTrajectory, including a new SiteNetwork."""

        if self.check_types and st.site_network.site_types is None:
            raise ValueError("Cannot run a check_types=True MergeSitesByDynamics on a SiteTrajectory without type information.")

        # Compute jump statistics
        ja = JumpAnalysis(verbose = self.verbose)
        ja.run(st)

        pbcc = PBCCalculator(st.site_network.structure.cell)
        site_centers = st.site_network.centers
        if self.check_types:
            site_types = st.site_network.site_types

        #connectivity_matrix = ja.n_ij
        connectivity_matrix = ja.n_ij / ja.total_time_spent_at_site
        assert st.site_network.n_sites == connectivity_matrix.shape[0]

        m1 = markov_clustering.run_mcl(connectivity_matrix,
                                       loop_value = 0) # Don't set new loop values
        clusters = markov_clustering.get_clusters(m1)
        new_n_sites = len(clusters)

        if self.verbose:
            print "After merge there will be %i sites" % new_n_sites

        if self.check_types:
            new_types = np.empty(shape = new_n_sites, dtype = np.int)

        new_centers = np.empty(shape = (new_n_sites, 3), dtype = st.site_network.centers.dtype)
        translation = np.empty(shape = st.site_network.n_sites, dtype = np.int)
        translation.fill(-1)

        for newsite in xrange(new_n_sites):
            mask = list(clusters[newsite])
            # Update translation table
            if np.any(translation[mask] != -1):
                # We've assigned a different cluster for this before... weird
                # degeneracy
                raise ValueError("Markov clustering tried to merge site(s) into more than one new site")
            translation[mask] = newsite

            to_merge = site_centers[mask]

            # Check distances
            dists = pbcc.distances(to_merge[0], to_merge[1:])
            assert np.all(dists < self.distance_threshold), "Markov clustering tried to merge sites more than %f apart" % self.distance_threshold

            # New site center
            new_centers[newsite] = pbcc.average(to_merge)
            if self.check_types:
                assert np.all(site_types[mask] == site_types[mask][0])
                new_types[newsite] = site_types[mask][0]

        newsn = st.site_network.copy()
        newsn.centers = new_centers
        if self.check_types:
            newsn.site_types = new_types

        newtraj = translation[st._traj]
        newtraj[st._traj == SiteTrajectory.SITE_UNKNOWN] = SiteTrajectory.SITE_UNKNOWN

        # It doesn't make sense to propagate confidence information through a
        # transform that might completely invalidate it
        newst = SiteTrajectory(newsn, newtraj, confidences = None)

        if not st.real_trajectory is None:
            newst.set_real_traj(st.real_trajectory)

        return newst
