# cython: boundscheck=False, wraparound=False

import numpy as np

cimport cython

from libc.math cimport sqrt, cos, M_PI, isnan, floor

ctypedef double precision
ctypedef double cell_precision

cdef class PBCCalculator(object):
    """Performs calculations on collections of 3D points under PBC"""

    cdef cell_precision [:, :] _cell_array
    cdef cell_precision [:, :] _cell_inverse_array
    cdef cell_precision [:] _cell_centroid

    def __init__(self, cell):
        """
        :param DxD ndarray: the unit cell -- an array of cell vectors, like the
          cell of an ASE :class:Atoms object.
        """
        cellmat = np.matrix(cell).T

        assert cell.shape[1] == cell.shape[0], "Cell must be square"

        self._cell_array = np.asarray(cellmat)
        self._cell_inverse_array = np.asarray(cellmat.I)
        self._cell_centroid = np.sum(0.5 * cell, axis = 0)

    @property
    def cell_centroid(self):
      return self._cell_centroid

    cpdef distances(self, pt1, pts2, in_place = False):
        """Compute the Euclidean distances from pt1 to all points in pts2, using
        shift-and-wrap.

        Makes a copy of pts2 unless in_place == True.

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

        return np.linalg.norm(self._cell_centroid - pts2, axis = 1)

    cpdef average(self, points, weights = None):
        """Average position of a "cloud" of points using the shift-and-wrap hack.

        Copies the points.

        Assumes that the points are relatively close (within a half unit cell)
        together, and that the first point is not a particular outsider (the
        cell is centered at that point).

        Can be a weighted average with the semantics of :func:numpy.average.
        """
        assert points.shape[1] == 3 and points.ndim == 2

        offset = self._cell_centroid - points[0]

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
        cdef cell_precision [:, :] cell = self._cell_array
        cdef cell_precision [:, :] cell_I = self._cell_inverse_array

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

    cpdef bint is_in_unit_cell(self, precision [:] pt):
        cdef cell_precision [:, :] cell = self._cell_array
        cdef cell_precision [:, :] cell_I = self._cell_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        # see https://stackoverflow.com/questions/11980292/how-to-wrap-around-a-range
        # re. fmod replacement. This gives the same behaviour as numpy's remainder.
        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

        # buf is not crystal coords
        return (buf[0] < 1.0) and (buf[1] < 1.0) and (buf[2] < 1.0) and \
               (buf[0] >= 0.0) and (buf[1] >= 0.0) and (buf[2] >= 0.0)

    cpdef bint is_in_image_of_cell(self, precision [:] pt, image):
        cdef cell_precision [:, :] cell = self._cell_array
        cdef cell_precision [:, :] cell_I = self._cell_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        # see https://stackoverflow.com/questions/11980292/how-to-wrap-around-a-range
        # re. fmod replacement. This gives the same behaviour as numpy's remainder.
        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

        # buf is not crystal coords
        cdef bint out = True
        for dim in xrange(3):
            out &= (buf[dim] >= image[dim]) and (buf[dim] < (image[dim] + 1))

        return out

    cpdef void to_cell_coords(self, precision [:, :] points):
        """Convert to cell coordinates in place."""
        assert points.shape[1] == 3, "Points must be 3D"

        cdef precision buf[3]
        cdef precision pt[3]

        cdef cell_precision [:, :] cell = self._cell_array
        cdef cell_precision [:, :] cell_I = self._cell_inverse_array

        # Iterates over points
        for i in xrange(len(points)):
            # Load into pt
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            # Row by row, do the matrix multiplication
            for dim in xrange(3):
                buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

            # Store into points
            points[i, 0] = buf[0]; points[i, 1] = buf[1]; points[i, 2] = buf[2];

    cpdef void wrap_points(self, precision [:, :] points):
        """Wrap `points` into a unit cell, IN PLACE. 3D only.
        """

        assert points.shape[1] == 3, "Points must be 3D"

        cdef cell_precision [:, :] cell = self._cell_array
        cdef cell_precision [:, :] cell_I = self._cell_inverse_array

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
