import numpy as np

import abc

from sitator.util import PBCCalculator
from sitator import SiteNetwork, SiteTrajectory

import logging
logger = logging.getLogger(__name__)

class MergeSitesError(Exception):
    pass

class MergedSitesTooDistantError(MergeSitesError):
    pass

class TooFewMergedSitesError(MergeSitesError):
    pass


class MergeSites(abc.ABC):
    """Abstract base class for merging sites.

    :param bool check_types: If True, only sites of the same type are candidates to
        be merged; if false, type information is ignored. Merged sites will only
        be assigned types if this is True.
    :param float maximum_merge_distance: Maximum distance between two sites
        that are in a merge group, above which an error will be raised.
    :param bool set_merged_into: If True, a site attribute `"merged_into"` will
        be added to the original `SiteNetwork` indicating which new site
        each old site was merged into.
    """
    def __init__(self,
                 check_types = True,
                 maximum_merge_distance = None,
                 set_merged_into = False):
        self.check_types = check_types
        self.maximum_merge_distance = maximum_merge_distance
        self.set_merged_into = set_merged_into


    def run(self, st, **kwargs):
        """Takes a SiteTrajectory and returns a SiteTrajectory, including a new SiteNetwork."""

        if self.check_types and st.site_network.site_types is None:
            raise ValueError("Cannot run a check_types=True MergeSites on a SiteTrajectory without type information.")

        # -- Compute jump statistics
        pbcc = PBCCalculator(st.site_network.structure.cell)
        site_centers = st.site_network.centers
        if self.check_types:
            site_types = st.site_network.site_types

        clusters = self._get_sites_to_merge(st, **kwargs)

        old_n_sites = st.site_network.n_sites
        new_n_sites = len(clusters)

        logger.info("After merging %i sites there will be %i sites for %i mobile particles" % (len(site_centers), new_n_sites, st.site_network.n_mobile))

        if new_n_sites < st.site_network.n_mobile:
            raise TooFewMergedSitesError("There are %i mobile atoms in this system, but only %i sites after merge" % (np.sum(st.site_network.mobile_mask), new_n_sites))

        if self.check_types:
            new_types = np.empty(shape = new_n_sites, dtype = np.int)
        merge_verts = st.site_network.vertices is not None
        if merge_verts:
            new_verts = []

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
                raise ValueError("Site merging tried to merge site(s) into more than one new site. This shouldn't happen.")
            translation[mask] = newsite

            to_merge = site_centers[mask]

            # Check distances
            if not self.maximum_merge_distance is None:
                dists = pbcc.distances(to_merge[0], to_merge[1:])
                if not np.all(dists <= self.maximum_merge_distance):
                    raise MergedSitesTooDistantError("Markov clustering tried to merge sites more than %.2f apart. Lower your distance_threshold?" % self.maximum_merge_distance)

            # New site center
            new_centers[newsite] = pbcc.average(to_merge)
            if self.check_types:
                assert np.all(site_types[mask] == site_types[mask][0])
                new_types[newsite] = site_types[mask][0]
            if merge_verts:
                new_verts.append(set.union(*[set(st.site_network.vertices[i]) for i in mask]))

        newsn = st.site_network.copy()
        newsn.centers = new_centers
        if self.check_types:
            newsn.site_types = new_types
        if merge_verts:
            newsn.vertices = new_verts

        newtraj = translation[st._traj]
        newtraj[st._traj == SiteTrajectory.SITE_UNKNOWN] = SiteTrajectory.SITE_UNKNOWN

        # It doesn't make sense to propagate confidence information through a
        # transform that might completely invalidate it
        newst = SiteTrajectory(newsn, newtraj, confidences = None)

        if not st.real_trajectory is None:
            newst.set_real_traj(st.real_trajectory)

        if self.set_merged_into:
            if st.site_network.has_attribute("merged_into"):
                st.site_network.remove_attribute("merged_into")
            st.site_network.add_site_attribute("merged_into", translation)

        return newst

    @abc.abstractmethod
    def _get_sites_to_merge(self, st, **kwargs):
        """Get the groups of sites to merge.

        Returns a list of list/tuples each containing the numbers of sites to be merged.
        There should be no overlap, and every site should be mentioned in at most
        one site merging group.

        If not mentioned in any, the site will disappear.
        """
        pass
