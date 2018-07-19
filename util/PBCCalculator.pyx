# cython: boundscheck=False wraparound=False

import numpy as np

cimport cython

ctypedef fused precision:
    float
    double
    long double

class PBCCalculator(object):
    """Performs calculations on collections of 3D points inside
    """

    def __init__(self, cell):
        """
        :param DxD ndarray: the unit cell -- an array of cell vectors, like the
          cell of an ASE :class:Atoms object.
        """
        cellmat = np.matrix(cell).T

        assert cell.shape[1] == cell.shape[0], "Cell must be square"

        self._cell_array = np.asarray(cellmat)
        self._cell_inverse_array = np.asarray(cellmat.I)

    cpdef void wrap_point(precision [:] pt):
        """Wrap a single point into the unit cell, IN PLACE. 3D only."""
        precision [:, :] cell = self._cell_array
        precision [:, :] cell_I = self._cell_inverse_array

        assert len(pt) == 3, "Points must be 3D"

        cdef precision buf[3]
        for dim in xrange(3):
            buf[dim] = (cell[dim, 0]*pt[0] + cell[dim, 1]*pt[1] + cell[dim, 2]*pt[2]) % 1.0

        pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];

        for dim in xrange(3):
            buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

        pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];

    cpdef void wrap_points(precision [:, :] points):
        """Wrap `points` into a unit cell, IN PLACE. 3D only.
        """

        assert points.shape[1] == 3, "Points must be 3D"

        precision [:, :] cell = self._cell_array
        precision [:, :] cell_I = self._cell_inverse_array

        cdef precision buf[3]
        cdef precision pt[3]

        # Iterates over points
        for i in xrange(len(points)):
            pt[0] = points[i, 0]; pt[1] = points[i, 1]; pt[2] = points[i, 2];

            for dim in xrange(3):
                buf[dim] = (cell[dim, 0]*pt[0] + cell[dim, 1]*pt[1] + cell[dim, 2]*pt[2]) % 1.0

            pt[0] = buf[0]; pt[1] = buf[1]; pt[2] = buf[2];

            for dim in xrange(3):
                buf[dim] = (cell_I[dim, 0]*pt[0] + cell_I[dim, 1]*pt[1] + cell_I[dim, 2]*pt[2])

            points[i, 0] = buf[0]; points[i, 1] = buf[1]; points[i, 2] = buf[2];

    cpdef void wrap_coords_general(coords):
        """Wrap `coords` into a unit cell, IN PLACE.

        For 3D data, use the 2x as fast :func:wrap_points

        :param NxD ndarray coords: coordinates.
        :param DxD ndarray cell: the unit cell vectors
        :param DxD ndarray cell_I: the inverse of the unit cell.

        Equivalent to pure Python:

          for pt in range(len(coords), ):
              # Get wrapped lattice coords
              coords[pt] = np.dot(cell_I, coords[pt]) % 1.0
              # Restore to real coords
              coords[pt] = np.dot(cell, coords[pt])

        :returns void:
        """
        precision [:, :] cell = self._cell_array
        precision [:, :] cell_I = self._cell_inverse_array

        # Wrap all positions into the unit cell
        buf = np.empty(shape = coords.shape, dtype = coords.dtype)

        # vectorized multiply by each row
        for dim in xrange(3):
            buf[:,dim] = np.remainder(np.sum(np.multiply(cell_I[dim], coords), axis = 1), 1.0)

        coords[:] = buf

        # vectorized multiply by each row
        for dim in xrange(3):
            buf[:,dim] = np.sum(np.multiply(cell[dim], coords), axis = 1)

        coords[:] = buf
