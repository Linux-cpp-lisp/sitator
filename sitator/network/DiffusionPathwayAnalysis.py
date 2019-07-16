
import numpy as np

import numbers
import itertools

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from sitator import SiteNetwork
from sitator.util import PBCCalculator

import logging
logger = logging.getLogger(__name__)

class DiffusionPathwayAnalysis(object):
    """Find connected diffusion pathways in a SiteNetwork.

    :param float|int connectivity_threshold: The percentage of the total number of
        (non-self) jumps, or absolute number of jumps, that must occur over an edge
        for it to be considered connected.
    :param int minimum_n_sites: The minimum number of sites that must be part of
        a pathway for it to be considered as such.
    :param bool true_periodic_pathways: Whether only to return true periodic
        pathways that include sites and their periodic images (i.e. conductive
        in the bulk) rather than just connected components. If ``True``,
        ``minimum_n_sites`` is NOT respected.
    """

    NO_PATHWAY = -1

    def __init__(self,
                connectivity_threshold = 1,
                true_periodic_pathways = True,
                minimum_n_sites = 0):
        assert minimum_n_sites >= 0

        self.true_periodic_pathways = true_periodic_pathways
        self.connectivity_threshold = connectivity_threshold
        self.minimum_n_sites = minimum_n_sites

    def run(self, sn, return_count = False):
        """
        Expects a ``SiteNetwork`` that has had a ``JumpAnalysis`` run on it.

        Adds information to ``sn`` in place.

        Args:
            sn (SiteNetwork): Must have jump statistics from a ``JumpAnalysis``.
            return_count (bool): Return the number of connected pathways.
        Returns:
            sn, [n_pathways]
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

        if self.true_periodic_pathways:
            connectivity_matrix, mask_000 = self._build_mic_connmat(sn, connectivity_matrix)

        n_ccs, ccs = connected_components(connectivity_matrix,
                                   directed = False, # even though the matrix is symmetric
                                   connection = 'weak') # diffusion could be unidirectional

        _, counts = np.unique(ccs, return_counts = True)

        if self.true_periodic_pathways:
            # is_pathway = np.ones(shape = n_ccs, dtype = np.bool)
            # We have to check that the pathways include a site and its periodic
            # image, and throw out those that don't
            new_n_ccs = 1
            new_ccs = np.zeros(shape = len(sn), dtype = np.int)

            # Add a non-path (contains no sites, all False) so the broadcasting works
            site_masks = [np.zeros(shape = len(sn), dtype = np.bool)]
            #seen_mask = np.zeros(shape = len(sn), dtype = np.bool)

            for pathway_i in np.arange(n_ccs):
                path_mask = ccs == pathway_i

                if not np.any(path_mask & mask_000):
                    # If the pathway is entirely outside the unit cell, we don't care
                    continue

                # Sum along each site's periodic images, giving a count site-by-site
                site_counts = np.sum(path_mask.reshape((-1, sn.n_sites)).astype(np.int), axis = 0)
                if not np.any(site_counts > 1):
                    # Not percolating; doesn't contain any site and its periodic image.
                    continue

                cur_site_mask = site_counts > 0

                intersects_with = np.where(np.any(np.logical_and(site_masks, cur_site_mask), axis = 1))[0]
                # Merge them:
                if len(intersects_with) > 0:
                    path_mask = cur_site_mask | np.logical_or.reduce([site_masks[i] for i in intersects_with], axis = 0)
                else:
                    path_mask = cur_site_mask
                # Remove individual merged paths
                # Going in reverse order means indexes don't become invalid as deletes happen
                for i in sorted(intersects_with, reverse=True):
                    del site_masks[i]
                # Add new (super)path
                site_masks.append(path_mask)

                new_ccs[path_mask] = new_n_ccs
                new_n_ccs += 1

            n_ccs = new_n_ccs
            ccs = new_ccs
            # Only actually take the ones that were assigned to in the end
            # This will deal with the ones that were merged.
            is_pathway = np.in1d(np.arange(n_ccs), ccs)
            is_pathway[0] = False # Cause this was the "unassigned" value, we initialized with zeros up above
        else:
            is_pathway = counts >= self.minimum_n_sites

            logging.info("Taking all edges with at least %i/%i jumps..." % (threshold, n_non_self_jumps))
            logging.info("Found %i connected components, of which %i are large enough to qualify as pathways (%i sites)." % (n_ccs, np.sum(is_pathway), self.minimum_n_sites))

        n_pathway = np.sum(is_pathway)
        translation = np.empty(n_ccs, dtype = np.int)
        translation[~is_pathway] = DiffusionPathwayAnalysis.NO_PATHWAY
        translation[is_pathway] = np.arange(n_pathway)

        node_pathways = translation[ccs]

        outmat = np.empty(shape = (sn.n_sites, sn.n_sites), dtype = np.int)

        for i in range(sn.n_sites):
            rowmask = node_pathways[i] == node_pathways
            outmat[i, rowmask] = node_pathways[i]
            outmat[i, ~rowmask] = DiffusionPathwayAnalysis.NO_PATHWAY

        sn.add_site_attribute('site_diffusion_pathway', node_pathways)
        sn.add_edge_attribute('edge_diffusion_pathway', outmat)

        if return_count:
            return sn, n_pathway
        else:
            return sn


    def _build_mic_connmat(self, sn, connectivity_matrix):
        # We use a 3x3x3 = 27 supercell, so there are 27x as many sites
        assert len(sn) == connectivity_matrix.shape[0]

        images = np.asarray(list(itertools.product(range(-1, 2), repeat = 3)))
        image_to_idex = dict((100 * (image[0] + 1) + 10 * (image[1] + 1) + (image[2] + 1), i) for i, image in enumerate(images))
        n_images = len(images)
        assert n_images == 27

        n_sites = len(sn)
        pos = sn.centers #.copy() # TODO: copy not needed after reinstall of sitator!
        n_total_sites = len(images) * n_sites
        newmat = lil_matrix((n_total_sites, n_total_sites), dtype = np.bool)

        mask_000 = np.zeros(shape = n_total_sites, dtype = np.bool)
        index_000 = image_to_idex[111]
        mask_000[index_000:index_000 + n_sites] = True
        assert np.sum(mask_000) == len(sn)

        pbcc = PBCCalculator(sn.structure.cell)
        buf = np.empty(shape = 3)

        internal_mat = np.zeros_like(connectivity_matrix)
        external_connections = []
        for from_site, to_site in zip(*np.where(connectivity_matrix)):
            buf[:] = pos[to_site]
            if pbcc.min_image(pos[from_site], buf) == 111:
                # If we're in the main image, keep the connection: it's internal
                internal_mat[from_site, to_site] = True
                #internal_mat[to_site, from_site] = True # fake FIXME
            else:
                external_connections.append((from_site, to_site))
                #external_connections.append((to_site, from_site)) # FAKE FIXME

        for image_idex, image in enumerate(images):
            # Make the block diagonal
            newmat[image_idex * n_sites:(image_idex + 1) * n_sites,
                   image_idex * n_sites:(image_idex + 1) * n_sites] = internal_mat

            # Check all external connections from this image; add other sparse entries
            for from_site, to_site in external_connections:
                buf[:] = pos[to_site]
                to_mic = pbcc.min_image(pos[from_site], buf)
                to_in_image = image + [(to_mic // 10**(2 - i) % 10) - 1 for i in range(3)]  # FIXME: is the -1 right
                assert to_in_image is not None, "%s" % to_in_image
                assert np.max(np.abs(to_in_image)) <= 2
                if not np.any(np.abs(to_in_image) > 1):
                    to_in_image = 100 * (to_in_image[0] + 1) + 10 * (to_in_image[1] + 1) + 1 * (to_in_image[2] + 1)
                    newmat[image_idex * n_sites + from_site,
                           image_to_idex[to_in_image] * n_sites + to_site] = True

        assert np.sum(newmat) >= n_images * np.sum(internal_mat) # Lowest it can be is if every one is internal

        return newmat, mask_000
