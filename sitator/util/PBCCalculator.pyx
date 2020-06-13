# cython: boundscheck=False, wraparound=False

import numpy as np

cimport cython
cimport numpy as npc

from libc.math cimport sqrt, cos, M_PI, isnan, floor, INFINITY, fmod

cdef class PBCCalculator(object):
    """Performs calculations on collections of 3D points under PBC."""

    def __init__(self, cell):
        """
        :param DxD ndarray: the unit cell -- an array of cell vectors, like the
          cell of an ASE :class:Atoms object.
        """
        cellmat = cell.T

        if not cell.shape[1] == cell.shape[0]:
            raise ValueError("Cell must be square")
        if not cell.shape[0] == 3:
            raise ValueError("Cell must be three-dimensional")

        self._cell_np = cell
        self._cell = cell
        self._cell_vec_lengths = np.linalg.norm(cell, axis = 1)

        if np.count_nonzero(self._cell_vec_lengths) < 3:
            raise ValueError("The given cell is invalid (one or more cell vectors have zero norm)")

        self._cell_mat_array = cellmat
        self._cell_mat_inverse_array = np.linalg.inv(cellmat)
        self._cell_centroid = np.sum(0.5 * cell, axis = 0)


    @property
    def cell_centroid(self):
        return np.asarray(self._cell_centroid)
    @property
    def cell_vector_lengths(self):
        return np.asarray(self._cell_vec_lengths)
    @property
    def cell(self):
        return self._cell_np


    cpdef pairwise_distances(self, pts, out = None):
        """Compute the pairwise distance matrix of ``pts`` with itself.

        :returns ndarray (len(pts), len(pts)): distances
        """
        if out is None:
            out = np.empty(shape = (len(pts), len(pts)), dtype = pts.dtype)

        buf = pts.copy()

        for i in xrange(len(pts) - 1):
            out[i, i] = 0
            self.distances(pts[i], buf[i + 1:], in_place = True, out = out[i, i + 1:])
            out[i + 1:, i] = out[i, i + 1:]
            buf[:] = pts

        out[len(pts) - 1, len(pts) - 1] = 0

        return out


    cpdef distances(self, pt1, pts2, in_place = False, out = None):
        """
        Compute the Euclidean distances from ``pt1`` to all points in
        ``pts2``, using shift-and-wrap.

        Makes a copy of ``pts2`` unless ``in_place == True``.

        :returns ndarray len(pts2): distances
        """
        pt1 = np.asarray(pt1)
        pts2 = np.asarray(pts2)

        assert pt1.ndim == 1
        assert pts2.ndim == 2
        assert pt1.shape[0] == pts2.shape[1]

        if not in_place:
            pts2 = np.copy(pts2)

        # Wrap
        offset = self._cell_centroid - pt1
        pts2 += offset
        self.wrap_points(pts2)

        # Put distance vectors in pts2

        pts2 = -pts2
        pts2 += self._cell_centroid

        # Square in place
        pts2 *= pts2

        if out is None:
            out = np.empty(shape = len(pts2), dtype = pts2.dtype)

        # Sum
        np.sum(pts2, axis = 1, out = out)

        #return np.linalg.norm(self._cell_centroid - pts2, axis = 1)
        return np.sqrt(out, out = out)


    cpdef average(self, points, weights = None):
        """Average position of a "cloud" of points using the shift-and-wrap hack.

        Copies the points.

        Assumes that the points are relatively close (within a half unit cell)
        together, and that the first point is not a particular outsider (the
        cell is centered at that point). If the average is weighted, the
        maximally weighted point will be taken as the center.

        Can be a weighted average with the semantics of :func:numpy.average.
        """
        assert points.shape[1] == 3 and points.ndim == 2

        center_about = 0
        if weights is not None:
            center_about = np.argmax(weights)

        offset = self._cell_centroid - points[center_about]

        ptbuf = points.copy()

        # Shift and wrap
        ptbuf += offset
        self.wrap_points(ptbuf)

        out = np.average(ptbuf, weights = weights, axis = 0)
        out -= offset

        self.wrap_point(out)

        del ptbuf

        return out


    cpdef time_average(self, frames):
        """Do multiple PBC correct means. Frames is n_frames x n_pts x 3.

        Returns a time average the size of one frame.
        """
        assert frames.shape[2] == 3
        out = np.empty(shape = (frames.shape[1], 3), dtype = frames.dtype)
        posbuf = np.empty(shape = (frames.shape[0], 3), dtype = frames.dtype)

        offset = np.empty(shape = (3,), dtype = frames.dtype)

        centroid = self._cell_centroid

        for i in xrange(frames.shape[1]):
            # Get all positions for that particle
            np.copyto(posbuf, frames[:, i, :])
            # Compute offset
            np.subtract(centroid, posbuf[0], out = offset)
            # Add offset
            posbuf += offset
            # Wrap to unit cell
            self.wrap_points(posbuf)
            # Take mean
            np.mean(posbuf, axis = 0, out = out[i])
            out[i] -= offset

        self.wrap_points(out)

        del posbuf
        return out


    cpdef void wrap_point(self, precision [:] pt):
        """Wrap a single point into the unit cell, IN PLACE. 3D only."""
        cdef cell_precision [:, :] cell = self._cell_mat_array
        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        # see https://stackoverflow.com/questions/11980292/how-to-wrap-around-a-range
        # re. fmod replacement. This gives the same behaviour as numpy's remainder.
        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])
            buf[dim] -= floor(buf[dim])

        pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];

        for dim in xrange(3):
            buf[dim] = (cell[dim, 0]*pt[0] + cell[dim, 1]*pt[1] + cell[dim, 2]*pt[2])

        pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];


    cpdef bint is_in_unit_cell(self, const precision [:] pt):
        cdef cell_precision [:, :] cell = self._cell_mat_array
        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        # see https://stackoverflow.com/questions/11980292/how-to-wrap-around-a-range
        # re. fmod replacement. This gives the same behaviour as numpy's remainder.
        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

        # buf is not crystal coords
        return (buf[0] < 1.0) and (buf[1] < 1.0) and (buf[2] < 1.0) and \
               (buf[0] >= 0.0) and (buf[1] >= 0.0) and (buf[2] >= 0.0)


    cpdef bint all_in_unit_cell(self, const precision [:, :] pts):
        for pt in pts:
            if not self.is_in_unit_cell(pt):
                return False
        return True


    cpdef bint is_in_image_of_cell(self, const precision [:] pt, image):
        cdef cell_precision [:, :] cell = self._cell_mat_array
        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        # see https://stackoverflow.com/questions/11980292/how-to-wrap-around-a-range
        # re. fmod replacement. This gives the same behaviour as numpy's remainder.
        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

        # buf is not crystal coords
        cdef bint out = True
        for dim in xrange(3):
            out &= (buf[dim] >= image[dim]) & (buf[dim] < (image[dim] + 1))

        return out


    cpdef void to_cell_coords(self, precision [:, :] points):
        """Convert to cell coordinates in place."""
        assert points.shape[1] == 3, "Points must be 3D"

        cdef precision buf[3]
        cdef precision pt[3]
        cdef Py_ssize_t dim, i

        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        # Iterates over points
        for i in xrange(len(points)):
            # Load into pt
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            # Row by row, do the matrix multiplication
            for dim in xrange(3):
                buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

            # Store into points
            points[i, 0] = buf[0]; points[i, 1] = buf[1]; points[i, 2] = buf[2];


    cpdef void to_cell_coords_wrapped(self, precision [:, :] points):
        """Convert cartesian coordinates to wrapped cell coordinates."""
        cdef precision buf[3]
        cdef precision pt[3]
        cdef Py_ssize_t dim, i

        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        # Iterates over points
        for i in range(len(points)):
            # Load into pt
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            # Row by row, do the matrix multiplication
            for dim in xrange(3):
                buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

            # Store into points & wrap
            points[i, 0] = fmod(buf[0], 1.0)
            points[i, 1] = fmod(buf[1], 1.0)
            points[i, 2] = fmod(buf[2], 1.0)


    cpdef int min_image(self, const precision [:] ref, precision [:] pt):
        """Find the minimum image of ``pt`` relative to ``ref``. In place in pt.

        Uses the brute force algorithm for correctness; returns the minimum image.

        Assumes that ``ref`` and ``pt`` are already in the *same* cell (though not
        necessarily the <0,0,0> cell -- any periodic image will do).

        :returns int[3] minimg: Which image was the minimum image.
        """
        # # There are 27 possible minimum images
        # buf = np.empty(shape = (27, 3), dtype = ref.dtype)
        # All possible min images
        cdef int minimg[3]
        cdef precision mindist = INFINITY
        cdef precision curdist = 0.0

        cdef precision buf[3]

        cdef int i, j, k
        for i in xrange(3):
            for j in xrange(3):
                for k in xrange(3):
                    # Copy point
                    buf[0] = pt[0]; buf[1] = pt[1]; buf[2] = pt[2]
                    # Add the right cell vectors to get this image
                    buf[0] += (i - 1) * self._cell[0, 0] + \
                              (j - 1) * self._cell[1, 0] + \
                              (k - 1) * self._cell[2, 0]
                    buf[1] += (i - 1) * self._cell[0, 1] + \
                              (j - 1) * self._cell[1, 1] + \
                              (k - 1) * self._cell[2, 1]
                    buf[2] += (i - 1) * self._cell[0, 2] + \
                              (j - 1) * self._cell[1, 2] + \
                              (k - 1) * self._cell[2, 2]
                    # Compute distance
                    buf[0] -= ref[0]; buf[1] -= ref[1]; buf[2] -= ref[2]
                    buf[0] *= buf[0]; buf[1] *= buf[1]; buf[2] *= buf[2]
                    curdist = sqrt(buf[0] + buf[1] + buf[2])

                    if curdist < mindist:
                        mindist = curdist
                        minimg[0] = i; minimg[1] = j; minimg[2] = k

        # Update pt in-place
        pt[0] += (minimg[0] - 1) * self._cell[0, 0] + \
                 (minimg[1] - 1) * self._cell[1, 0] + \
                 (minimg[2] - 1) * self._cell[2, 0]
        pt[1] += (minimg[0] - 1) * self._cell[0, 1] + \
                 (minimg[1] - 1) * self._cell[1, 1] + \
                 (minimg[2] - 1) * self._cell[2, 1]
        pt[2] += (minimg[0] - 1) * self._cell[0, 2] + \
                 (minimg[1] - 1) * self._cell[1, 2] + \
                 (minimg[2] - 1) * self._cell[2, 2]

        return 100 * minimg[0] + 10 * minimg[1] + 1 * minimg[2]


    cpdef void to_real_coords(self, precision [:, :] points):
        """Convert to real coords from crystal coords in place."""
        assert points.shape[1] == 3, "Points must be 3D"

        cdef precision buf[3]
        cdef precision pt[3]

        cdef cell_precision [:, :] cell = self._cell_mat_array

        # Iterates over points
        for i in xrange(len(points)):
            # Load into pt
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            # Row by row, do the matrix multiplication
            for dim in xrange(3):
                buf[dim] = (cell[dim, 0]*pt[0] + cell[dim, 1]*pt[1] + cell[dim, 2]*pt[2])

            # Store into points
            points[i, 0] = buf[0]; points[i, 1] = buf[1]; points[i, 2] = buf[2];


    cpdef void wrap_points(self, precision [:, :] points):
        """Wrap ``points`` into a unit cell, IN PLACE. 3D only.
        """

        assert points.shape[1] == 3, "Points must be 3D"

        cdef cell_precision [:, :] cell = self._cell_mat_array
        cdef cell_precision [:, :] cell_I = self._cell_mat_inverse_array

        cdef precision buf[3]
        cdef precision pt[3]

        # Iterates over points
        for i in xrange(len(points)):
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            for dim in xrange(3):
                buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])
                buf[dim] -= floor(buf[dim])

            pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];

            for dim in xrange(3):
                buf[dim] = (cell[dim, 0]*pt[0] + cell[dim, 1]*pt[1] + cell[dim, 2]*pt[2])

            points[i, 0] = buf[0]; points[i, 1] = buf[1]; points[i, 2] = buf[2];
