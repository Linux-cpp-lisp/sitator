from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import numpy as np

from scipy.spatial import Voronoi, ConvexHull



class SiteNetwork(object):
    """A network of potential interstitial sites in a lattice.

    Calculates and represents a network of sites. Each site is represented by a
    center point and a set of vertices, which define a convex hull.
    """

    @classmethod
    def from_lattice(cls, points):
        vor = Voronoi(points)
        centers = vor.vertices
        vertices = []
        for i in range(len(centers)):
            vert_set = set([])
            for ridge in [j for j, e in enumerate(vor.ridge_vertices) if i in e]:
                for pt_idex in vor.ridge_points[ridge]:
                    vert_set.add(pt_idex)
            vertices.append(vert_set)

        return cls(centers, vertices, points)

    def __init__(self, centers, vertices, positions):
        """

        :param neighbor_matrix: An NxK integer matrix giving the K nearest neighbors, in order, of 2

        """
        if len(centers) != len(vertices):
            raise ValueError("The number of centers and vertex sets must be the same.")
        if any(((not type(v) is set) or (len(v) < 4)) for v in vertices):
            raise ValueError("Each site convex hull must be defined by a `set` of least 4 vertex indexes.")

        print(positions)

        self._count = len(centers)
        self._centers = np.asarray(centers)
        self._vertices = np.asarray(vertices)
        self._hulls = []
        for vert_set in vertices:
            vpts = np.array([positions[j] for j in vert_set])
            print(vpts)
            #self._hulls.append(ConvexHull(vpts, incremental=True))

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
        if self._dirty: raise ValueError("SiteNetwork used without creating/updating hulls; did you forget to call `update_hulls`?")

        def in_hull(pt, hull):
            """Test whether `pt` is in `hull`.

            Tests whether `pt` is in `hull` by adding `pt` to hull and seeing if
            the vertices change.
            """
            hull_pts_old = hull.points
            hull.add_points([pt])
            index_of_new_point = len(hull_pts_old)

            res = not index_of_new_point in hull.vertices

            # If the point is inside the hull, there's no need to recompute
            # the hull -- the presense of the point in the body won't affect
            # future checks.
            #
            # So, we save a call to qhull and only reset the hull if the point
            # was outside and so changed it.
            if not res:
                hull.add_points(hull_pts_old, True)

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

    def collapse_sites(self, threshold):
        """Collapse nearby sites (within `threshold`) into a single site.

        .. warning:: :func:`update_hulls` must be called after :func:`collapse_sites` before the SiteNetwork is used.

        :param sites: site coordinates.
        :type sites: (n, d) ndarray
        :param threshold: the distance threshold within which to collapse sites together.
        :type threshold: float

        :return: None
        """

        #Indicates which sites no longer "exist", i.e. have been merged into another.
        mask = np.ones(shape=self.count, dtype=np.bool_)

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
            self._vertices[target] = self._vertices[target].union(self._vertices[other])
            #print("Collapsing %i and %i" % (target, other))
            #Mark the merged (second) site as such
            mask[other] = False

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
