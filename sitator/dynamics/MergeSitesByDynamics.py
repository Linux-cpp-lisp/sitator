import numpy as np

from sitator import SiteNetwork, SiteTrajectory
from sitator.dynamics import JumpAnalysis
from sitator.util import PBCCalculator

class MergeSitesByDynamics(object):
    """Merges sites using dynamical data.

    Given a SiteTrajectory, merges sites using Markov Clustering.

    :param float distance_threshold: Don't merge sites further than this
        in real space.
    :param float post_check_thresh_factor: Throw an error if proposed merge sites
        are further than this * distance_threshold away. Only a sanity check; not
        a hard guerantee. Can be `None`; defaults to `1.5`. Can be loosely
        thought of as how "normally distributed" the merge sites need to be, with
        larger values allowing more and more oblong point clouds.
    :param bool check_types: If True, only sites of the same type are candidates to
        be merged; if false, type information is ignored. Merged sites will only
        be assigned types if this is True.
    :param int iterlimit: Maximum number of Markov Clustering iterations to run
        before throwing an error.
    :param dict markov_parameters: Parameters for underlying Markov Clustering.
        Valid keys are ``'inflation'``, ``'expansion'``, and ``'pruning_threshold'``.
    """
    def __init__(self,
                 distance_threshold = 1.0,
                 post_check_thresh_factor = 1.5,
                 check_types = True,
                 verbose = True,
                 iterlimit = 100,
                 markov_parameters = {}):

        self.verbose = verbose
        self.distance_threshold = distance_threshold
        self.post_check_thresh_factor = post_check_thresh_factor
        self.check_types = check_types
        self.iterlimit = iterlimit
        self.markov_parameters = markov_parameters

    def run(self, st):
        """Takes a SiteTrajectory and returns a SiteTrajectory, including a new SiteNetwork."""

        if self.check_types and st.site_network.site_types is None:
            raise ValueError("Cannot run a check_types=True MergeSitesByDynamics on a SiteTrajectory without type information.")

        # -- Compute jump statistics
        if not st.site_network.has_attribute('p_ij'):
            ja = JumpAnalysis(verbose = self.verbose)
            ja.run(st)

        pbcc = PBCCalculator(st.site_network.structure.cell)
        site_centers = st.site_network.centers
        if self.check_types:
            site_types = st.site_network.site_types

        # -- Build connectivity_matrix
        connectivity_matrix = st.site_network.n_ij.copy()
        n_sites_before = st.site_network.n_sites
        assert n_sites_before == connectivity_matrix.shape[0]

        centers_before = st.site_network.centers

        # For diagnostic purposes
        no_diag_graph = connectivity_matrix.astype(dtype = np.float, copy = True)
        np.fill_diagonal(no_diag_graph, np.nan)
        # Rather arbitrary, but this is really just an alarm for if things
        # are really, really wrong
        edge_threshold = np.nanmean(no_diag_graph) + 3 * np.nanstd(no_diag_graph)
        n_alarming_ignored_edges = 0

        # Apply distance threshold
        for i in range(n_sites_before):
            dists = pbcc.distances(centers_before[i], centers_before[i + 1:])
            js_too_far = np.where(dists > self.distance_threshold)[0]
            js_too_far += i + 1

            if np.any(connectivity_matrix[i, js_too_far] > edge_threshold) or \
               np.any(connectivity_matrix[js_too_far, i] > edge_threshold):
               n_alarming_ignored_edges += 1

            connectivity_matrix[i, js_too_far] = 0
            connectivity_matrix[js_too_far, i] = 0 # Symmetry

        if self.verbose and n_alarming_ignored_edges > 0:
            print("  At least %i site pairs with high (z-score > 3) fluxes were over the given distance cutoff.\n"
                  "  This may or may not be a problem; but if `distance_threshold` is low, consider raising it." % n_alarming_ignored_edges)

        # -- Do Markov Clustering
        clusters = self._markov_clustering(connectivity_matrix, **self.markov_parameters)

        new_n_sites = len(clusters)

        if self.verbose:
            print("After merge there will be %i sites" % new_n_sites)

        if self.check_types:
            new_types = np.empty(shape = new_n_sites, dtype = np.int)

        # -- Merge Sites
        new_centers = np.empty(shape = (new_n_sites, 3), dtype = st.site_network.centers.dtype)
        translation = np.empty(shape = st.site_network.n_sites, dtype = np.int)
        translation.fill(-1)

        for newsite in range(new_n_sites):
            mask = list(clusters[newsite])
            # Update translation table
            if np.any(translation[mask] != -1):
                # We've assigned a different cluster for this before... weird
                # degeneracy
                raise ValueError("Markov clustering tried to merge site(s) into more than one new site")
            translation[mask] = newsite

            to_merge = site_centers[mask]

            # Check distances
            if not self.post_check_thresh_factor is None:
                dists = pbcc.distances(to_merge[0], to_merge[1:])
                assert np.all(dists < self.post_check_thresh_factor * self.distance_threshold), \
                            "Markov clustering tried to merge sites more than %f * %f apart. Lower your distance_threshold?" % (self.post_check_thresh_factor, self.distance_threshold)

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

    def _markov_clustering(self,
                           transition_matrix,
                           expansion = 2,
                           inflation = 2,
                           pruning_threshold = 0.00001):
        """
        See https://micans.org/mcl/.

        Because we're dealing with matrixes that are stochastic already,
        there's no need to add artificial loop values.

        Implementation inspired by https://github.com/GuyAllard/markov_clustering
        """

        assert transition_matrix.shape[0] == transition_matrix.shape[1]

        m1 = transition_matrix.copy()

        # Normalize (though it should be close already)
        m1 /= np.sum(m1, axis = 0)

        allcols = np.arange(m1.shape[1])

        converged = False
        for i in range(self.iterlimit):
            # -- Expansion
            m2 = np.linalg.matrix_power(m1, expansion)
            # -- Inflation
            np.power(m2, inflation, out = m2)
            m2 /= np.sum(m2, axis = 0)
            # -- Prune
            to_prune = m2 < pruning_threshold
            # Exclude the max of every column
            to_prune[np.argmax(m2, axis = 0), allcols] = False
            m2[to_prune] = 0.0
            # -- Check converged
            if np.allclose(m1, m2):
                converged = True
                if self.verbose:
                    print("Markov Clustering converged in %i iterations" % i)
                break

            m1[:] = m2

        if not converged:
            raise ValueError("Markov Clustering couldn't converge in %i iterations" % self.iterlimit)

        # -- Get clusters
        attractors = m2.diagonal().nonzero()[0]

        clusters = set()

        for a in attractors:
            cluster = tuple(m2[a].nonzero()[0])
            clusters.add(cluster)

        return list(clusters)
