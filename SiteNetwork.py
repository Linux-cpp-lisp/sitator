from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import numpy as np

import itertools

from scipy.spatial import Voronoi, ConvexHull, cKDTree

from vorozeo import Zeopy



class SiteNetwork(object):
    """A network of potential interstitial sites in a lattice.

    Calculates and represents a network of sites. Each site is represented by a
    center point and a set of vertices, which define a convex hull.
    """

    @classmethod
    def from_lattice(cls, structure, radial = True):
        z = Zeopy("/home/amusaelian/Documents/Ionic Frustration/code/lib/zeo++/trunk/network")
        nodes, edges = z.voronoi(structure, radial=radial)

        positions = np.asarray([n['coords'] for n in nodes])

        current_from = 0
        edgelist = []
        node_el = []
        for e in edges:
            if e['from'] != current_from:
                node_el.sort(key=lambda n: np.linalg.norm(positions[current_from] - positions[n]))
                edgelist.append(node_el)
                node_el = []
                current_from = e['from']
            node_el.append(e['to'])

        sn = cls(positions,
                 [set(n['region-atom-indexes']) for n in nodes],
                 structure.get_positions(),
                 structure.get_pbc(),
                 structure.get_cell())

        return sn



    def __init__(self, centers, vertices, positions, pbc, cell):
        """

        :param neighbor_matrix: An NxK integer matrix giving the K nearest neighbors, in order, of 2

        """
        if len(centers) != len(vertices):
            raise ValueError("The number of centers and vertex sets must be the same.")
        if any((not type(v) is set) or (len(v) < 4) for v in vertices):
            raise ValueError("Each site convex hull must be defined by at least 4 vertex indexes.")

        self._count = len(centers)
        self._centers = np.asarray(centers)
        self._vertices = np.asarray(vertices)
        self._hulls = []
        # for vert_set in vertices:
        #     vpts = np.array([positions[j] for j in vert_set])
        #     self._hulls.append(ConvexHull(vpts, incremental=True))

        self._pbc = pbc
        self._cell = cell

        self._dirty = False

    @property
    def count(self):
        return self._count

    @property
    def centers(self):
        return self._centers

    @property
    def vertices(self):
        return self._vertices

    def site_of_particle(self, position, prev_site = -1):
        """Returns the site to which a point at `position` belongs.

        If the point's prior site `prev_site` is known, providing it will
        likely improve performance.
        """
        raise NotImplementedError()

        if self._dirty: raise ValueError("SiteNetwork used without creating/updating hulls; did you forget to call `update_hulls`?")

        # If no previous site, choose one semi-sensibly
        if prev_site == -1:
            pass

        hulls = self._hulls
        edges = self._edges

        # Obviously, check first if it's still in the same site
        if self._in_hull(position, hulls[prev_site]): return prev_site

        check_neighbors_of = [prev_site]
        visited = set([])
        while True:
            all_neighbors = []
            print("Checking neighbors of %s" % check_neighbors_of)
            for s in check_neighbors_of:
                neighbors = edges[s]
                for neighbor in neighbors:
                    if not neighbor in visited:
                        print("Checking %i" % neighbor)
                        if self._in_hull(position, hulls[neighbor]):
                            return neighbor
                        else:
                            all_neighbors.append(neighbor)
                            visited.add(neighbor)
            check_neighbors_of = all_neighbors

    @staticmethod
    def _in_hull(pt, hull):
        """Test whether `pt` is in `hull`.

        Tests whether `pt` is in `hull` by adding `pt` to hull and seeing if
        the vertices change.
        """
        hull_pts_old = hull.points
        hull_verts_old = set(hull.vertices)
        hull.add_points([pt])
        index_of_new_point = len(hull_pts_old)

        #res = not index_of_new_point in hull.vertices
        print(hull_verts_old)
        print(hull.vertices)
        res = hull_verts_old == set(hull.vertices)

        # If the point is inside the hull, there's no need to recompute
        # the hull -- the presense of the point in the body won't affect
        # future checks.
        #
        # So, we save a call to qhull and only reset the hull if the point
        # was outside and so changed it.
        #if not res:
        if True:
            hull.add_points(hull_pts_old, restart = True)
            print(hull.vertices)

        return res

    def update_hulls(self, positions):
        """Update the hulls with new vertex position information.

        The vertex identifers are used as indexes into `positions`, which should
        give the current coordinates of that vertex.
        """
        for i in range(self.count):
            pts = [positions[j] for j in self._vertices[i]]
            self._hulls[i].add_points(pts, True)
        self._dirty = False

    def collapse_sites(self, threshold, verbose = True, n_iters = 4):
        """Collapse nearby sites (within `threshold`) into a single site.

        .. warning:: :func:`update_hulls` must be called after :func:`collapse_sites` before the SiteNetwork is used.

        :param sites: site coordinates.
        :type sites: (n, d) ndarray
        :param threshold: the distance threshold within which to collapse sites together.
        :type threshold: float

        :return: None
        """

        n_merges = 0

        i, j, d = ase.neighbors.primitive_neighbor_list('ijd', self._pbc, self._cell, self._centers, threshold)

        # Indicates which sites no longer "exist", i.e. have been merged into another.
        mask = np.ones(shape=self.count, dtype=np.bool_)

        #factor = np.sqrt(5) * 0.5
        #factor = 2




        #  -- Do a first pass with KD tree
        kd = cKDTree(self._centers)

        within_thresh = kd.query_pairs(threshold)

        # Base case
        if len(within_thresh) == 0:
            return 0

        affect_pairs = (kd.query_pairs(factor * threshold) - within_thresh)
        can_be_affected = set(sum(affect_pairs, ()))

        #sitemap = np.arange(self.count)
        #maskmap = np.where(mask)[0]

        # print(affect_pairs)
        # print(within_thresh)
        # print(can_be_affected)

        for i1, i2 in within_thresh:

            if not (mask[i1] and mask[i2]):
                continue

            if not ((i1 in can_be_affected) or (i2 in can_be_affected)):
                # Merge sites without consequence
                #assert sitemap[i1] == i1 and sitemap[i2] == i2, "%i: %i; %i: %i" % (i1, sitemap[i1], i2, sitemap[i2])
                self._centers[i1] = (self._centers[i1] + self._centers[i2]) * 0.5
                self._vertices[i1] = set.union(self._vertices[i1], self._vertices[i2])
                n_merges += 1
                mask[i2] = False
                #sitemap[i2] = i1

        if verbose:
            print("collapse_sites: did %i 1st pass merges" % n_merges)

        assert np.sum(~mask) == n_merges, "Mask ct: %i, n_merges %i" % (np.sum(~mask), n_merges)

        #Remove merged sites
        self.remove_sites(mask)

        return n_merges + self.collapse_sites(threshold * 0.5, verbose = verbose)

        # -- Second pass for annoying cases
        # pair_to_check = affect_pairs
        #
        # while True:
        #     min_dist = np.inf
        #     min_dist_pair = (None, None)
        #
        #     for i1, i2 in pair_to_check:
        #
        #         i1 = sitemap[i1]
        #         i2 = sitemap[i2]
        #
        #         if i1  == i2:
        #             continue
        #         if not (mask[i1] and mask[i2]):
        #             continue
        #
        #         dist = np.linalg.norm(self._centers[i1] - self._centers[i2])
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_dist_pair = (i1, i2)
        #
        #     print(min_dist)
        #     if min_dist > threshold: break
        #
        #     #Merge the two sites with minimum distance
        #     target, other = min_dist_pair
        #     self._centers[target] = (self._centers[target] + self._centers[other]) * 0.5
        #     self._vertices[target] = set.union(self._vertices[target], self._vertices[other])
        #     n_merges += 1
        #     #Mark the merged (second) site as such
        #     mask[other] = False
        #     sitemap[i2] = i1
        #
        #     pair_to_check = itertools.combinations(range(self.count), 2)



    def old_collapse_sites(self, threshold, verbose = True):
        """Collapse nearby sites (within `threshold`) into a single site.

        .. warning:: :func:`update_hulls` must be called after :func:`collapse_sites` before the SiteNetwork is used.

        :param sites: site coordinates.
        :type sites: (n, d) ndarray
        :param threshold: the distance threshold within which to collapse sites together.
        :type threshold: float

        :return: None
        """

        # Indicates which sites no longer "exist", i.e. have been merged into another.
        mask = np.ones(shape=self.count, dtype=np.bool_)

        n_merges = 0

        while True:
            min_dist = np.inf
            min_dist_pair = (None, None)

            for i1 in range(self.count):
                if not mask[i1]: continue
                for i2 in range(i1 + 1, self.count):
                    if not mask[i2]: continue

                    dist = np.linalg.norm(self._centers[i1] - self._centers[i2])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_pair = (i1, i2)

            if min_dist > threshold: break

            #Merge the two sites with minimum distance
            target, other = min_dist_pair
            self._centers[target] = (self._centers[target] + self._centers[other]) * 0.5
            self._vertices[target] = set.union(self._vertices[target], self._vertices[other])
            n_merges += 1
            #print("Collapsing %i and %i" % (target, other))
            #Mark the merged (second) site as such
            mask[other] = False


        if verbose:
            print("collapse_sites: did %i merges." % (n_merges))

        assert np.sum(~mask) == n_merges

        #Remove merged sites
        self.remove_sites(mask)

    def remove_sites(self, mask):
        """Remove sites where `mask` is False."""
        self._centers = self._centers[mask]
        self._vertices = self._vertices[mask]
        self._count = len(self._centers)

        self._dirty = True
        return None


    def total_volume(self):
        """The total volume contained in the network's hulls."""
        return sum(h.volume for h in self._hulls)

    def _update_neighbor_lists():
        pass
