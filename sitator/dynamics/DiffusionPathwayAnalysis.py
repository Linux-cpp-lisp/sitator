
import numpy as np

import numbers

from scipy.sparse.csgraph import connected_components

class DiffusionPathwayAnalysis(object):
    """Find connected diffusion pathways in a SiteNetwork.

    :param float|int connectivity_threshold: The percentage of the total number of
        (non-self) jumps, or absolute number of jumps, that must occur over an edge
        for it to be considered connected.
    :param int minimum_n_sites: The minimum number of sites that must be part of
        a pathway for it to be considered as such.
    """

    NO_PATHWAY = -1

    def __init__(self,
                connectivity_threshold = 0.001,
                minimum_n_sites = 4,
                verbose = True):
        assert minimum_n_sites >= 0

        self.connectivity_threshold = connectivity_threshold
        self.minimum_n_sites = minimum_n_sites

        self.verbose = verbose

    def run(self, sn):
        """
        Expects a SiteNetwork that has had a JumpAnalysis run on it.
        """
        if not sn.has_attribute('n_ij'):
            raise ValueError("SiteNetwork has no `n_ij`; run a JumpAnalysis on it first.")

        nondiag = np.ones(shape = sn.n_ij.shape, dtype = np.bool)
        np.fill_diagonal(nondiag, False)
        n_non_self_jumps = np.sum(sn.n_ij[nondiag])

        if isinstance(self.connectivity_threshold, numbers.Integral):
            threshold = self.connectivity_threshold
        elif isinstance(self.connectivity_threshold, numbers.Real):
            threshold = self.connectivity_threshold * n_non_self_jumps
        else:
            raise TypeError("Don't know how to interpret connectivity_threshold `%s`" % self.connectivity_threshold)

        connectivity_matrix = sn.n_ij >= threshold

        n_ccs, ccs = connected_components(connectivity_matrix,
                                   directed = False, # even though the matrix is symmetric
                                   connection = 'weak') # diffusion could be unidirectional

        _, counts = np.unique(ccs, return_counts = True)

        is_pathway = counts >= self.minimum_n_sites

        if self.verbose:
            print "Taking all edges with at least %i/%i jumps..." % (threshold, n_non_self_jumps)
            print "Found %i connected components, of which %i are large enough to qualify as pathways." % (n_ccs, np.sum(is_pathway))

        translation = np.empty(n_ccs, dtype = np.int)
        translation[~is_pathway] = DiffusionPathwayAnalysis.NO_PATHWAY
        translation[is_pathway] = np.arange(np.sum(is_pathway))

        node_pathways = translation[ccs]

        outmat = np.empty(shape = (sn.n_sites, sn.n_sites), dtype = np.int)

        for i in xrange(sn.n_sites):
            rowmask = node_pathways[i] == node_pathways
            outmat[i, rowmask] = node_pathways[i]
            outmat[i, ~rowmask] = DiffusionPathwayAnalysis.NO_PATHWAY

        sn.add_site_attribute('site_diffusion_pathway', node_pathways)
        sn.add_edge_attribute('edge_diffusion_pathway', outmat)
        return sn
