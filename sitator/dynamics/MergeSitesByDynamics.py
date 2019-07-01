import numpy as np

from sitator.dynamics import JumpAnalysis
from sitator.util import PBCCalculator
from sitator.network.merging import MergeSites

import logging
logger = logging.getLogger(__name__)


class MergeSitesByDynamics(MergeSites):
    """Merges sites using dynamical data.

    Given a SiteTrajectory, merges sites using Markov Clustering.

    :param float distance_threshold: Don't merge sites further than this
        in real space. Zeros out the connectivity_matrix at distances greater than
        this; a hard, step function style cutoff. For a more gentle cutoff, try
        changing `connectivity_matrix_generator` to incorporate distance.
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
                 connectivity_matrix_generator = None,
                 distance_threshold = 1.0,
                 post_check_thresh_factor = 1.5,
                 check_types = True,
                 iterlimit = 100,
                 markov_parameters = {}):

        super().__init__(
            maximum_merge_distance = post_check_thresh_factor * distance_threshold,
            check_types = check_types
        )

        if connectivity_matrix_generator is None:
            connectivity_matrix_generator = MergeSitesByDynamics.connectivity_n_ij
        assert callable(connectivity_matrix_generator)

        self.connectivity_matrix_generator = connectivity_matrix_generator
        self.distance_threshold = distance_threshold
        self.post_check_thresh_factor = post_check_thresh_factor
        self.check_types = check_types
        self.iterlimit = iterlimit
        self.markov_parameters = markov_parameters

    # Connectivity Matrix Generation Schemes:

    @staticmethod
    def connectivity_n_ij(sn):
        """Basic default connectivity scheme: uses n_ij directly as connectivity matrix.

        Works well for systems with sufficient statistics.
        """
        return sn.n_ij

    @staticmethod
    def connectivity_jump_lag_biased(jump_lag_coeff = 1.0,
                                     jump_lag_sigma = 20.0,
                                     jump_lag_cutoff = np.inf,
                                     distance_coeff = 0.5,
                                     distance_sigma = 1.0):
        """Bias the typical connectivity matrix p_ij with jump lag and distance contributions.

        The jump lag and distance are processed through Gaussian functions with
        the given sigmas (i.e. higher jump lag/larger distance => lower
        connectivity value). These matrixes are then added to p_ij, with a prefactor
        of `jump_lag_coeff` and `distance_coeff`.

        Site pairs with jump lags greater than `jump_lag_cutoff` have their bias
        set to zero regardless of `jump_lag_sigma`. Defaults to `inf`.
        """
        def cfunc(sn):
            jl = sn.jump_lag.copy()
            jl -= 1.0 # Center it around 1 since that's the minimum lag, 1 frame
            jl /= jump_lag_sigma
            np.square(jl, out = jl)
            jl *= -0.5
            np.exp(jl, out = jl) # exp correctly takes the -infs to 0

            jl[sn.jump_lag > jump_lag_cutoff] = 0.

            # Distance term
            pbccalc = PBCCalculator(sn.structure.cell)
            dists = pbccalc.pairwise_distances(sn.centers)
            dmat = dists.copy()

            # We want to strongly boost the similarity of *very* close sites
            dmat /= distance_sigma
            np.square(dmat, out = dmat)
            dmat *= -0.5
            np.exp(dmat, out = dmat)

            return (sn.p_ij + jump_lag_coeff * jl) * (distance_coeff * dmat + (1 - distance_coeff))

        return cfunc

    # Real methods

    def _get_sites_to_merge(self, st):
        # -- Compute jump statistics
        if not st.site_network.has_attribute('n_ij'):
            ja = JumpAnalysis()
            ja.run(st)

        pbcc = PBCCalculator(st.site_network.structure.cell)
        site_centers = st.site_network.centers

        # -- Build connectivity_matrix
        connectivity_matrix = self.connectivity_matrix_generator(st.site_network).copy()
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

        if n_alarming_ignored_edges > 0:
            logger.warning("  At least %i site pairs with high (z-score > 3) fluxes were over the given distance cutoff.\n"
                           "  This may or may not be a problem; but if `distance_threshold` is low, consider raising it." % n_alarming_ignored_edges)

        # -- Do Markov Clustering
        clusters = self._markov_clustering(connectivity_matrix, **self.markov_parameters)
        return clusters


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
                logger.info("Markov Clustering converged in %i iterations" % i)
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
