import numpy as np

from analysis import SiteNetwork, SiteTrajectory
from analysis.dynamics import JumpAnalysis
from analysis.util import PBCCalculator

from scipy.sparse.csgraph import connected_components

class MergeSitesByDynamics(object):
    """Merges sites using dynamical data.

    Given a SiteTrajectory, merges groups sites between which jumps happen
    "faster" than some threshold.

    :params int threshold: in frames.
    :param float distance_threshold: Will never merge sites further than this
        in real space. Angstrom.
    :param bool check_types: If True, only sites of the same type are candidates to
        be merged; if false, type information is ignored. Merged sites will only
        be assigned types if this is True.
    """
    def __init__(self, threshold, distance_threshold = 1.0, check_types = True, verbose = True):
        self.threshold = threshold
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

        # Construct thresholded connectivity matrix
        pbcc = PBCCalculator(st.site_network.structure.cell)

        site_centers = st.site_network.centers
        if self.check_types:
            site_types = st.site_network.site_types

        connectivity_matrix = ja.jump_lag < self.threshold
        from_sites, to_sites = np.where(connectivity_matrix)

        distbuf = np.empty(shape = 1, dtype = np.float)

        for from_site, to_site in zip(from_sites, to_sites):
            if self.check_types and site_types[from_site] != site_types[to_site]:
                connectivity_matrix[from_site, to_site] = False
                continue # short circut around the distance computation for a little performance

            pbcc.distances(site_centers[from_site], site_centers[to_site:to_site + 1], out = distbuf)
            if distbuf[0] >= self.distance_threshold:
                connectivity_matrix[from_site, to_site] = False

        # Get strongly connected components
        # If the jump rates aren't symmetric -- i.e. components aren't strongly
        # connected -- they are perhaps physically distinct, but one is simply
        # very unfavorable compared to the other.
        new_n_sites, merge_membership = connected_components(connectivity_matrix,
                                                            directed = True,
                                                            connection = 'strong')
        if self.verbose:
            print "After merge there will be %i sites" % new_n_sites


        if self.check_types:
            new_types = np.empty(shape = new_n_sites, dtype = np.int)

        assert st.site_network.n_sites == len(merge_membership) == connectivity_matrix.shape[0]

        new_centers = np.empty(shape = (new_n_sites, 3), dtype = st.site_network.centers.dtype)

        for newsite in xrange(new_n_sites):
            mask = merge_membership == newsite
            new_centers[newsite] = pbcc.average(site_centers[mask])
            if self.check_types:
                new_types[newsite] = site_types[mask][0]

        newsn = st.site_network.copy()
        newsn.centers = new_centers
        if self.check_types:
            newsn.site_types = new_types

        newtraj = merge_membership[st._traj]
        newtraj[st._traj == SiteTrajectory.SITE_UNKNOWN] = SiteTrajectory.SITE_UNKNOWN

        # It doesn't make sense to propagate confidence information through a
        # transform that might completely invalidate it
        newst = SiteTrajectory(newsn, newtraj, confidences = None)

        if not st.real_trajectory is None:
            newst.set_real_traj(st.real_trajectory)

        return newst
