# cython: boundscheck=False, wraparound=False

cimport cython
from libc.math cimport sqrt, floor, ceil

import numpy as np

ctypedef double precision

@cython.cdivision
cdef inline Py_ssize_t to_linear_index_3D(const Py_ssize_t* idex, const Py_ssize_t* shape) nogil:
    return idex[0]*shape[1]*shape[2] + idex[1]*shape[2] + idex[2]


@cython.cdivision
cdef inline void from_linear_index_3D(Py_ssize_t lin_idex, const Py_ssize_t* shape, Py_ssize_t* out) nogil:
    out[0] = lin_idex / (shape[1]*shape[2]) # This is C-style floor division
    lin_idex -= out[0]*shape[1]*shape[2]
    out[1] = lin_idex / shape[2]
    lin_idex -= out[1]*shape[2]
    out[2] = lin_idex


cpdef gridded_density_periodic(points,
                               int n_boxes_max,
                               pbcc,
                               which_pts = None,
                               out = None,
                               output_box_assignments = None,
                               precision d_cutoff = 0.0):
    """Compute the number of points in each triclinic subcell in grid covering a triclinic cell.

    The shortest crystal direction will have `n_boxes_min` gridboxes, and
    the other dimensions will have the number of gridboxes that makes as
    regular (same side lengths) as possible.

    Params:
     - which_pts (default None): list of indexes of the points in each frame
        that should be considered. If None, all points considered.
     - output_box_assignments: If present, an array in which to put the linear
        index of the box each point is placed into.
    """
    # == Some variable set-ups
    cdef Py_ssize_t i, j
    cdef int x, y, z

    # == Set-up input data
    # TODO: extend to arbitrary dimensional case.
    frames = None
    if len(points.shape) == 3:
        assert points.shape[2] == 3, "Data points must currently be in R^3"
        frames = points
    else:
        raise ValueError("Invalid shape for `points`, it must be either 2- or 3-dimensional")

    # == Set up the cell
    cdef precision [:] cell_vec_lengths = pbcc.cell_vector_lengths

    # == Determine number of boxes
    cdef Py_ssize_t n_boxes[3]
    cdef precision box_dims[3]

    cdef Py_ssize_t longest_dim = np.argmax(cell_vec_lengths)
    cdef precision box_side_length = cell_vec_lengths[longest_dim] / n_boxes_max
    for i in range(3):
        n_boxes[i] = <Py_ssize_t>(cell_vec_lengths[i] // box_side_length)
        box_dims[i] = cell_vec_lengths[i] / n_boxes[i]

    # == Build output array
    if out is None:
        out = np.empty(shape = (n_boxes[0], n_boxes[1], n_boxes[2]), dtype = frames.dtype)
    cdef precision [:, :, :] out_c  = out
    out_c[:, :, :] = 0

    # == Make the weight mapping
    cdef int weight_width = <int> max([floor(d_cutoff / box_dims[i]) for i in range(3)])
    # One octant. +1 larger because there's always the center point
    weight_grid = np.zeros(shape = (weight_width + 1,) * 3, dtype = frames.dtype)
    cdef precision [:, :, :] weight_grid_c = weight_grid
    cdef precision temp_dist
    for x in range(weight_width + 1):
        for y in range(weight_width + 1):
            for z in range(weight_width + 1):
                temp_dist = sqrt(
                    (x * box_dims[0])*(x * box_dims[0]) +
                    (y * box_dims[0])*(y * box_dims[1]) +
                    (z * box_dims[0])*(z * box_dims[2])
                )
                if temp_dist <= d_cutoff:
                    weight_grid_c[x, y, z] = 1.0

    # == Frame-by-frame
    cdef Py_ssize_t [:] which_pts_c
    if which_pts is None:
        which_pts_c = np.arange(frames.shape[1])
    else:
        assert np.min(which_pts) >= 0
        assert np.max(which_pts) < frames.shape[1]
        which_pts_c = which_pts
    cdef Py_ssize_t n_which_pts = len(which_pts_c)

    cdef bint do_output_assign = 0
    cdef Py_ssize_t [:, :] output_box_assignments_c
    if output_box_assignments is not None:
        assert output_box_assignments.shape == (frames.shape[0], len(which_pts_c))
        output_box_assignments_c = output_box_assignments
        do_output_assign = 1

    cdef precision [:, :, :] frames_c = frames
    cdef precision [:, :] buf = np.empty(shape = (len(which_pts_c), 3), dtype = frames.dtype)
    cdef Py_ssize_t [3] idex
    cdef Py_ssize_t buf_idex
    for i in range(len(frames)):
        # Copy inputs into the buffer
        buf_idex = 0
        for j in range(n_which_pts):
            buf[buf_idex] = frames_c[i, which_pts_c[j]]
            buf_idex += 1

        # Bring points into cell coordinates
        pbcc.to_cell_coords_wrapped(buf)

        # Assign points to boxes an accumulate density
        for j in range(len(buf)):
            for k in range(3):
                idex[k] = <Py_ssize_t>floor(buf[j, k] * n_boxes[k])
            for x in range(-weight_width, weight_width + 1):
                for y in range(-weight_width, weight_width + 1):
                    for z in range(-weight_width, weight_width + 1):
                        out_c[
                            (idex[0] + x) % n_boxes[0],
                            (idex[1] + y) % n_boxes[1],
                            (idex[2] + z) % n_boxes[2],
                        ] += weight_grid_c[abs(x), abs(y), abs(z)]
            if do_output_assign:
                output_box_assignments_c[i, j] = to_linear_index_3D(idex, n_boxes)

    return out, box_dims
